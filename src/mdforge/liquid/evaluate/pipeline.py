"""Run a water-model evaluation: ingest → compute → assemble results.

Capability-driven: each property is computed only for the legs whose data
support it, so the pipeline degrades gracefully from a full campaign
(heat→NVT→NPT→NVT2→NVE) down to a single NPT trajectory. It wires the existing
``mdforge.liquid`` kernels — it introduces no new physics.

Which leg feeds the *scored* value follows the campaign's reasoning:
thermodynamics (density, ΔHvap, Cp, α, κ) come from the NPT leg (the model's own
equilibrium density); structure, diffusion and the dielectric prefer an NVT leg
run at the experimental density (a fair comparison to experiment) and fall back
to whatever leg carries the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import compute_bulk_properties
from ..stats import bootstrap_error, mean_stderr
from ..structure import (
    coordination_number,
    hydrogen_bonds,
    rdf,
    tetrahedral_order,
)
from ..thermo import (
    clausius_mossotti_eps_inf,
    dielectric_constant,
    equilibrate,
    heat_capacity,
    heat_of_vaporization,
)
from ..transport import msd, self_diffusion, unwrap_com, yeh_hummer_correction
from .config import EvalConfig, state_guard
from .ingest import LegData, ingest_leg, water_profile_from_topology
from .profiles import WaterProfile
from .reference import load_experimental_rdf, load_skinner_rdf

_E_ANG_TO_DEBYE = 4.803  # 1 e·Å = 4.803 D
_AVOGADRO = 6.02214076e23


@dataclass
class EvalResult:
    """Assembled, JSON-serialisable results for one model evaluation."""

    meta: dict = field(default_factory=dict)
    thermo: dict = field(default_factory=dict)       # {leg: {...}}
    structure: dict = field(default_factory=dict)    # {leg: {...}}
    diffusion: dict = field(default_factory=dict)    # {leg: {...}}
    dielectric: dict = field(default_factory=dict)   # {leg: {...}}
    series: dict = field(default_factory=dict)        # {leg: {dt_ps, equil, columns}}
    rdf_exp: dict = field(default_factory=dict)             # Soper (2013) neutron
    rdf_exp_skinner: dict = field(default_factory=dict)     # Skinner (2014) X-ray g_OO
    scoring_inputs: dict = field(default_factory=dict)          # key -> (value, unit)
    scoring_uncertainties: dict = field(default_factory=dict)   # key -> float
    scoring_sources: dict = field(default_factory=dict)         # key -> leg name
    warnings: list = field(default_factory=list)

    def to_json_dict(self) -> dict:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj
        return clean({
            "meta": self.meta, "thermo": self.thermo, "structure": self.structure,
            "diffusion": self.diffusion, "dielectric": self.dielectric,
            "series": self.series,
            "rdf_exp": self.rdf_exp,
            "rdf_exp_skinner": self.rdf_exp_skinner,
            "scoring_inputs": {k: list(v) for k, v in self.scoring_inputs.items()},
            "scoring_sources": self.scoring_sources, "warnings": self.warnings,
        })


# ---------------------------------------------------------------------------
# Per-leg compute
# ---------------------------------------------------------------------------

def _first_minimum(r: np.ndarray, g: np.ndarray,
                   *, lo: float = 3.0, hi: float = 3.7, smooth: int = 5) -> tuple[float, float]:
    """Robust O–O first-shell minimum: the deepest point of a lightly-smoothed
    g(r) within the physical first-minimum window ``[lo, hi]`` Å (water O–O first
    minimum ≈ 3.3 Å).

    Taking the global minimum inside a fixed physical bracket — rather than the
    *first local* minimum after the peak — is robust to the spurious noise dips
    on the peak's descending flank that otherwise collapse the coordination
    number on short/noisy RDFs, and to unstructured RDFs with no clear first
    shell (which fooled the old detector into walking outward).

    Returns ``(r_min, g_min)`` — the minimum's radius and the smoothed g there.
    A ``g_min`` well above ~1 signals a poorly-formed first shell (short/noisy or
    unstructured RDF), i.e. an unreliable coordination number.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    gs = (np.convolve(g, np.ones(smooth) / smooth, mode="same")
          if smooth > 1 and len(g) >= smooth else g)
    mask = (r >= lo) & (r <= hi)
    if not np.any(mask):
        return float(0.5 * (lo + hi)), float("nan")
    rw, gw = r[mask], gs[mask]
    i = int(np.argmin(gw))
    return float(rw[i]), float(gw[i])


# Per-frame log columns retained for the optional per-leg timeseries (raw key ->
# axis label); only those present in a given log are kept.
_SERIES_COLUMNS = (
    ("density_gcc", "density (g/cm³)"),
    ("temp_K", "temperature (K)"),
    ("pressure_atm", "pressure (atm)"),
    ("e_total", "total energy (kcal/mol)"),
    ("pe", "potential energy (kcal/mol)"),
    ("volume_ang3", "volume (Å³)"),
)


def _collect_series(leg: LegData) -> dict | None:
    """Per-frame thermodynamic timeseries for the optional per-leg plots/JSON.

    Reads whichever of :data:`_SERIES_COLUMNS` the leg's log carries. The time
    step is taken from a ``time_ps`` column when present, else the trajectory's.
    """
    raw = leg.raw_columns
    if not raw:
        return None
    cols = {label: np.asarray(raw[key], dtype=float).tolist()
            for key, label in _SERIES_COLUMNS if key in raw}
    if not cols:
        return None
    n = len(next(iter(cols.values())))
    t_ps = None
    if "time_ps" in raw and len(raw["time_ps"]) == n:
        t = np.asarray(raw["time_ps"], dtype=float)
        t_ps = t.tolist()
        dt_ps = float(np.median(np.diff(t))) if n > 1 else 1.0
    else:
        dt_ps = float(leg.traj.dt_ps) if (leg.traj and leg.traj.dt_ps) else 1.0
    return {
        "ensemble": leg.ensemble, "n_frames": int(n),
        "equil": int(leg.equil_frames(n)), "dt_ps": dt_ps,
        "t_ps": t_ps, "columns": cols,
    }


def _compute_thermo(leg: LegData, config: EvalConfig, n_molecules: int) -> dict | None:
    traj = leg.traj
    if traj is None or traj.n_frames == 0:
        return None
    seed = config.analysis.seed
    n = traj.n_frames
    eq = leg.equil_frames(n)
    if eq >= n:
        leg.warnings.append(f"{leg.name}: equil ({eq}) >= frames ({n}); thermo skipped")
        return None

    out: dict[str, Any] = {"n_frames": int(n), "equil": int(eq),
                           "ensemble": leg.ensemble, "dt_ps": float(traj.dt_ps)}
    raw = leg.raw_columns

    # temperature
    if "temp_K" in raw:
        Tm, Te, _ = mean_stderr(raw["temp_K"][eq:])
        out["temperature_K"], out["temperature_err"] = float(Tm), float(Te)

    has_vol = traj.volume is not None
    has_enth = traj.enthalpy is not None

    if leg.ensemble in ("NPT",) and has_vol and has_enth:
        bp = compute_bulk_properties(
            traj, equil=eq, molpol=config.system.molecular_polarizability,
            gas_pe_per_molecule=config.system.gas_pe_per_molecule,
            n_molecules=n_molecules, bootstrap=True, seed=seed)
        m, err = bp.metadata, bp.metadata.get("errors", {})
        # density (prefer the logged per-frame density; else the kernel value)
        if "density_gcc" in raw:
            rho_m, rho_e, _ = mean_stderr(raw["density_gcc"][eq:])
            out["density_g_cm3"], out["density_err"] = float(rho_m), float(rho_e)
        else:
            out["density_g_cm3"] = float(m.get("density_g_cm3"))
            out["density_err"] = float(err.get("density_g_cm3", 0.0))
        out["alpha_T_1e4_K"] = float(m["alpha_T"]) * 1e4
        out["alpha_T_err"] = float(err.get("alpha_T", 0.0)) * 1e4
        out["kappa_T_1e6_bar"] = float(m["kappa_T"])       # kernel unit == 1e-6/bar
        out["kappa_T_err"] = float(err.get("kappa_T", 0.0))
        if "cp" in m:
            h = equilibrate(traj.enthalpy, eq)
            out["cp_cal_mol_K"] = float(m["cp"])
            out["cp_err"] = float(bootstrap_error(
                lambda idx: heat_capacity(h[idx], n_molecules, traj.temperature_K),
                len(h), seed=seed))
        if "delta_hvap_kcal_mol" in m:
            pe = equilibrate(traj.potential_energy, eq)
            out["delta_hvap_kcal_mol"] = float(m["delta_hvap_kcal_mol"])
            gpe = config.system.gas_pe_per_molecule
            out["delta_hvap_err"] = float(bootstrap_error(
                lambda idx: heat_of_vaporization(
                    gpe, float(pe[idx].mean()) / n_molecules, traj.temperature_K),
                len(pe), seed=seed))
        if "pressure_atm" in raw:
            Pm, Pe, _ = mean_stderr(raw["pressure_atm"][eq:])
            out["pressure_atm"], out["pressure_err"] = float(Pm), float(Pe)
        # neutral energy components (present only for some engines)
        comps = {k: raw[k] for k in ("pe_rigg", "pe_ewald", "pe_pppm", "pe") if k in raw}
        if comps:
            out["energy_components_per_molecule"] = {
                k: float(v[eq:].mean()) / n_molecules for k, v in comps.items()}

    if leg.ensemble in ("NVT", "NVE") and has_enth:
        h = equilibrate(traj.enthalpy, eq)
        cv = heat_capacity(h, n_molecules, traj.temperature_K)
        out["cv_cal_mol_K"] = float(cv)
        out["cv_err"] = float(bootstrap_error(
            lambda idx: heat_capacity(h[idx], n_molecules, traj.temperature_K),
            len(h), seed=seed))
        if traj.total_energy is not None:
            et = equilibrate(traj.total_energy, eq)
            out["e_total_mean"] = float(et.mean())
            out["e_total_std"] = float(et.std())
            # energy-conservation drift for NVE (kcal/mol per ns)
            if leg.ensemble == "NVE" and len(et) > 2 and traj.dt_ps:
                t = np.arange(len(et)) * traj.dt_ps
                slope = float(np.polyfit(t, et, 1)[0])   # kcal/mol per ps
                out["energy_drift_per_ns"] = slope * 1000.0
        if "density_gcc" in raw:
            out["density_g_cm3"] = float(raw["density_gcc"][eq:].mean())
    return out


def _compute_structure(leg: LegData, config: EvalConfig, n_molecules: int) -> dict | None:
    if leg.atoms is None or leg.box is None:
        return None
    n = leg.atoms.shape[0]
    eq = leg.equil_frames(n)
    if eq >= n:
        return None
    knobs = config.analysis
    oxy = leg.atoms[:, leg.o_idx, :]
    hyd = leg.atoms[:, leg.h_idx, :]
    box = leg.box
    sel = range(eq, n, max(1, knobs.rdf.stride))
    sel_s = range(eq, n, max(1, knobs.structure_stride))
    r_max, n_bins = knobs.rdf.r_max, knobs.rdf.n_bins

    r, gOO = rdf(oxy, box, frames=sel, r_max=r_max, n_bins=n_bins)
    # g_OH / g_HH exclude intramolecular pairs (same-molecule O–H and H–H) so the
    # partials are inter-molecular only, matching the Soper (2013) reference —
    # which carries no covalent/rigid intramolecular peaks. Plotted only, never
    # scored.
    _, gOH = rdf(oxy, box, positions_b=hyd, mol_a=leg.mol_o, mol_b=leg.mol_h,
                 frames=sel, r_max=r_max, n_bins=n_bins)
    _, gHH = rdf(hyd, box, mol_a=leg.mol_h, frames=sel, r_max=r_max, n_bins=n_bins)

    V = float(np.mean(leg.volume[eq:])) if leg.volume is not None else float(
        np.prod(box[eq:].mean(axis=0)))
    ndens_O = len(leg.o_idx) / V
    ipk = int(np.argmax(gOO[r > 2.0]) + np.searchsorted(r, 2.0))
    r_min, g_min = _first_minimum(r, gOO)
    coord = coordination_number(r, gOO, ndens_O, r_min)
    if not (g_min == g_min) or g_min > 1.0:   # NaN (empty window) or weak dip
        leg.warnings.append(
            f"{leg.name}: weak g_OO first-shell minimum (g≈{g_min:.2f} at "
            f"{r_min:.2f} Å); coordination number may be unreliable "
            "(short or noisy RDF — try a longer trajectory)")
    q, q_mean = tetrahedral_order(oxy, box, frames=sel_s)
    hb, hb_info = hydrogen_bonds(oxy, hyd, box, frames=sel_s,
                                 r_oo=knobs.hbond.r_oo, angle_deg=knobs.hbond.angle_deg)
    density = n_molecules * config.system.molar_mass_g_mol / _AVOGADRO / (V * 1e-24)

    return {
        "ensemble": leg.ensemble, "n_frames": int(n), "equil": int(eq),
        "density_g_cm3": float(density),
        "r": r.tolist(), "g_OO": gOO.tolist(), "g_OH": gOH.tolist(), "g_HH": gHH.tolist(),
        "gOO_peak_r": float(r[ipk]), "gOO_peak_g": float(gOO[ipk]),
        "gOO_first_min_r": float(r_min), "coordination_number": float(coord),
        "number_density_O_inv_A3": float(ndens_O),
        "tetrahedral_q_mean": float(q_mean),
        "tetrahedral_q_hist": np.histogram(q, bins=40, range=(-0.2, 1.0),
                                           density=True)[0].tolist(),
        "hbonds_per_molecule": float(hb),
        "hbond_criteria": {"r_oo": hb_info["r_oo"], "angle_deg": hb_info["angle_deg"]},
    }


def _compute_diffusion(leg: LegData, config: EvalConfig) -> dict | None:
    if leg.com is None or leg.box is None:
        return None
    n = leg.com.shape[0]
    eq = leg.equil_frames(n)
    if eq >= n - 2:
        return None
    dk = config.analysis.diffusion
    dt_ps = dk.dt_ps
    if dt_ps is None:
        dt_ps = float(leg.traj.dt_ps) if (leg.traj and leg.traj.dt_ps) else None
    if not dt_ps:
        leg.warnings.append(f"{leg.name}: no dt_ps for diffusion; set analysis.diffusion.dt_ps")
        return None
    # dt_ps is the spacing of *saved* frames; read-time striding widened the gap
    # between the frames actually loaded, so scale the MSD time axis to match.
    dt_ps *= leg.frame_stride
    u = unwrap_com(leg.com[eq:], leg.box[eq:])
    m = msd(u)
    fit = self_diffusion(m, dt_ps, fit_lo=dk.fit_lo, fit_hi=dk.fit_hi)
    L = float(np.cbrt(np.mean(leg.volume[eq:]))) if leg.volume is not None else float(
        np.cbrt(np.prod(leg.box[eq:].mean(axis=0))))
    D5 = fit["D_cm2_s"] * 1e5
    out = {
        "ensemble": leg.ensemble, "n_frames": int(n), "equil": int(eq),
        "dt_ps": float(dt_ps), "box_L_ang": L,
        "D_1e5_cm2_s": float(D5), "D_ang2_ps": float(fit["D_ang2_ps"]),
        "msd": m.tolist(), "t_ps": fit["t"].tolist(),
        "fit_slice": list(fit["fit_slice"]),
    }
    if dk.finite_size_correction:
        dD = yeh_hummer_correction(dk.reference_viscosity_pa_s, L,
                                   config.state.temperature_K) * 1e5
        out["fs_correction_1e5"] = float(dD)
        out["D_corr_1e5_cm2_s"] = float(D5 + dD)
        out["reference_viscosity_pa_s"] = dk.reference_viscosity_pa_s
    return out


def _cell_dipole(leg: LegData, profile: WaterProfile, eq: int) -> np.ndarray:
    """Atomistic cell dipole ``M(t)`` (e·Å), molecules made whole about O."""
    n_mol = leg.n_molecules
    apm = profile.atoms_per_molecule
    a = leg.atoms[eq:].reshape(-1, n_mol, apm, 3)
    box = leg.box[eq:]
    oxy = a[:, :, profile.oxygen_local_index, :]
    rel = a - oxy[:, :, None, :]
    L = box[:, None, None, :]
    rel -= L * np.round(rel / L)
    q = profile.per_molecule_charges()[None, None, :, None]
    mu = (q * rel).sum(axis=2)              # (T, n_mol, 3)
    return mu.sum(axis=1), mu               # M(t) (T,3), per-molecule mu (T,n_mol,3)


_DIPOLE_COLS = ("dipole_x", "dipole_y", "dipole_z")


def _compute_dielectric(leg: LegData, config: EvalConfig, profile: WaterProfile) -> dict | None:
    """Static dielectric ε₀ from cell-dipole fluctuations.

    The dipole series M(t) is sourced flexibly (:func:`dielectric_constant` only
    needs M in **Debye** and a per-frame volume, so it is agnostic to the model):

    - **engine dipole** — when the log carries a per-frame cell dipole (Debye)
      under ``dipole_{x,y,z}`` (e.g. from a Tinker ``analyze`` log via the
      pre-converter). This includes induced dipoles, so it is the correct source
      for **polarizable** models (HIPPO/AMOEBA).
    - **point-charge dipole** — otherwise M = Σ qᵢ rᵢ from coordinates + charges
      (:func:`_cell_dipole`, e·Å → converted to Debye). Complete for
      **non-polarizable** fixed-charge models; approximate for polarizable ones
      (it omits the induced part).

    ε_∞ comes from the molecular polarizability (Clausius–Mossotti) when given,
    else 1.
    """
    raw = leg.raw_columns
    mu = None
    if all(c in raw for c in _DIPOLE_COLS) and "volume_ang3" in raw:
        # engine-reported total cell dipole (Debye), volume from the same log
        Md = np.column_stack([np.asarray(raw[c], dtype=float) for c in _DIPOLE_COLS])
        vol_all = np.asarray(raw["volume_ang3"], dtype=float)
        n = min(len(Md), len(vol_all))
        eq = leg.equil_frames(n)
        if eq >= n:
            return None
        M, vol = Md[eq:n], vol_all[eq:n]
        source = "engine_dipole"
    elif leg.atoms is not None and leg.box is not None and leg.volume is not None:
        n = leg.atoms.shape[0]
        eq = leg.equil_frames(n)
        if eq >= n:
            return None
        M_eA, mu = _cell_dipole(leg, profile, eq)
        M = M_eA * _E_ANG_TO_DEBYE          # e·Å → Debye (kernel expects Debye)
        vol = leg.volume[eq:]
        source = "point_charge"
    else:
        return None

    eps_inf = 1.0
    if config.system.molecular_polarizability:
        vmol = float(np.mean(vol)) / leg.n_molecules
        eps_inf = clausius_mossotti_eps_inf(config.system.molecular_polarizability, vmol)
    eps = dielectric_constant(M, vol, config.state.temperature_K, eps_inf=eps_inf)
    out = {
        "ensemble": leg.ensemble, "n_frames": int(n), "equil": int(eq),
        "epsilon_0": float(eps), "eps_inf": float(eps_inf),
        "dipole_source": source, "net_charge_e": float(profile.net_charge()),
    }
    if mu is not None:
        mu_mag = float(np.mean(np.linalg.norm(mu, axis=-1)))
        out["mu_molecule_eA"] = mu_mag
        out["mu_molecule_debye"] = mu_mag * _E_ANG_TO_DEBYE
    else:
        out["cell_dipole_debye_mean"] = float(np.mean(np.linalg.norm(M, axis=1)))
    return out


# ---------------------------------------------------------------------------
# Scoring-source selection
# ---------------------------------------------------------------------------

def _pick(by_leg: dict, legs_meta: dict, prefer: list[str]):
    for ens in prefer:
        for name, res in by_leg.items():
            if legs_meta.get(name) == ens:
                return name, res
    for name, res in by_leg.items():
        return name, res
    return None, None


def _select_scoring_inputs(result: EvalResult, legs_meta: dict, config: EvalConfig) -> None:
    inputs: dict[str, tuple[float, str]] = {}
    unc: dict[str, float] = {}
    sources: dict[str, str] = {}

    # Thermodynamics from NPT.
    tname, tset = _pick(result.thermo, legs_meta, ["NPT"])
    if tset:
        thermo_map = {
            "density": ("density_g_cm3", "density_err", "g/cm3"),
            "delta_hvap": ("delta_hvap_kcal_mol", "delta_hvap_err", "kcal/mol"),
            "cp": ("cp_cal_mol_K", "cp_err", "cal/mol/k"),
            "alpha_T": ("alpha_T_1e4_K", "alpha_T_err", "1e-4/k"),
            "kappa_T": ("kappa_T_1e6_bar", "kappa_T_err", "1e-6/bar"),
        }
        for key, (field_, errf, unit) in thermo_map.items():
            if field_ in tset:
                inputs[key] = (tset[field_], unit)
                sources[key] = tname
                if errf in tset:
                    unc[key] = float(tset[errf])

    # Structure / diffusion / dielectric prefer NVT (experimental density).
    sname, sset = _pick(result.structure, legs_meta, ["NVT", "NVE", "NPT"])
    if sset:
        for key, field_ in {"gOO_peak_r": "gOO_peak_r", "gOO_peak_g": "gOO_peak_g",
                            "tetrahedral_q": "tetrahedral_q_mean",
                            "hbonds_per_molecule": "hbonds_per_molecule",
                            "coordination_number": "coordination_number"}.items():
            unit = "angstrom" if key == "gOO_peak_r" else "dimensionless"
            inputs[key] = (sset[field_], unit)
            sources[key] = sname
        if legs_meta.get(sname) != "NVT":
            result.warnings.append(
                f"structure scored from the {sname} ({legs_meta.get(sname)}) leg at the "
                "model's own density; g(r)/coordination are not a fair comparison to "
                "experiment (no experimental-density NVT leg present)")

    dname, dset = _pick(result.diffusion, legs_meta, ["NVT", "NVE", "NPT"])
    if dset:
        val = dset.get("D_corr_1e5_cm2_s", dset["D_1e5_cm2_s"])
        inputs["self_diffusion"] = (val, "1e-5 cm2/s")
        sources["self_diffusion"] = dname

    ename, eset = _pick(result.dielectric, legs_meta, ["NVT", "NVE", "NPT"])
    if eset:
        inputs["dielectric"] = (eset["epsilon_0"], "dimensionless")
        sources["dielectric"] = ename

    result.scoring_inputs = inputs
    result.scoring_uncertainties = unc
    result.scoring_sources = sources


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(config: EvalConfig, *, enforce_state: bool = True,
                   max_frames: int | None = None, stride: int = 1,
                   record_timeseries: bool = False) -> EvalResult:
    """Run the full evaluation and return an :class:`EvalResult`.

    Set ``record_timeseries`` to retain each leg's per-frame thermodynamic
    series (density, T, P, energies, volume) under :attr:`EvalResult.series`
    for the optional per-leg timeseries plots.

    ``stride`` keeps every ``stride``-th trajectory frame at read time (DCD only)
    so a long run can be spanned cheaply; the diffusion time axis rescales
    automatically.
    """
    if enforce_state:
        state_guard(config.state)

    profile, n_molecules = water_profile_from_topology(config)
    result = EvalResult(meta={
        "model": config.model.name, "engine": config.model.engine,
        "species": config.species,
        "temperature_K": config.state.temperature_K,
        "pressure_atm": config.state.pressure_atm,
        "n_molecules": n_molecules,
        "molar_mass_g_mol": config.system.molar_mass_g_mol,
        "charges_e": profile.charges_e,
        "virtual_sites": list(config.system.virtual_sites or []),
        "gas_pe_per_molecule": config.system.gas_pe_per_molecule,
        "legs": [leg.name for leg in config.legs],
    })
    net_q = profile.net_charge()
    if abs(net_q) > 1e-3:
        # A non-neutral molecule makes the point-charge cell dipole M(t)=Σ qᵢrᵢ
        # origin-dependent, so the dielectric it feeds is physically meaningless.
        result.warnings.append(
            f"model molecule net charge is {net_q:+.4f} e (not ~0); the dielectric "
            "assumes a net-neutral molecule — check system.charges_e (a 4-site "
            "model needs the ghost-site charge, e.g. M = -2·q_H)"
        )
    legs_meta = {leg.name: leg.ensemble.upper() for leg in config.legs}

    for leg_spec in config.legs:
        leg = ingest_leg(leg_spec, config, profile, n_molecules,
                         max_frames=max_frames, stride=stride)

        thermo = _compute_thermo(leg, config, n_molecules)
        if thermo:
            result.thermo[leg.name] = thermo
        structure = _compute_structure(leg, config, n_molecules)
        if structure:
            result.structure[leg.name] = structure
        diffusion = _compute_diffusion(leg, config)
        if diffusion:
            result.diffusion[leg.name] = diffusion
        dielectric = _compute_dielectric(leg, config, profile)
        if dielectric:
            result.dielectric[leg.name] = dielectric
        if record_timeseries:
            series = _collect_series(leg)
            if series:
                result.series[leg.name] = series
        # collect warnings after compute so per-property warnings surface too
        result.warnings.extend(leg.warnings)

    # experimental partial RDFs for the report (298 K / 1 atm only)
    try:
        result.rdf_exp = load_experimental_rdf(
            config.state.temperature_K, config.state.pressure_atm)
    except FileNotFoundError:
        result.rdf_exp = {}
    # second, independent O-O reference (Skinner 2014 X-ray), plotted by default
    try:
        result.rdf_exp_skinner = load_skinner_rdf(
            config.state.temperature_K, config.state.pressure_atm)
    except FileNotFoundError:
        result.rdf_exp_skinner = {}

    _select_scoring_inputs(result, legs_meta, config)
    return result


__all__ = ["EvalResult", "run_evaluation"]
