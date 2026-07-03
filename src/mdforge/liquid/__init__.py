"""mdforge.liquid — liquid-phase property computation (goal e).

Parsing is decoupled from computation:

    Tinker logs ──(parse)──▶ Trajectory (numpy arrays) ──(kernels)──▶ BulkProperties

The kernels in :mod:`~mdforge.liquid.thermo`, :mod:`~mdforge.liquid.transport`,
and :mod:`~mdforge.liquid.stats` take arrays and never touch a file.
:func:`compute_bulk_properties` is the one-call convenience that runs the
thermodynamic kernels over a :class:`~mdforge.core.records.Trajectory`.
"""

from __future__ import annotations

import numpy as np

from ..core.records import BulkProperties, Trajectory
from . import parse, plots, stats, structure, thermo, transport
from .constants import R_KCAL
from .stats import bootstrap_error, bzavg, mean_stderr, statistical_inefficiency
from .structure import (
    angular_rdf,
    coordination_number,
    normals_from_orientations,
    plane_normal_from_points,
    rdf,
)
from .thermo import (
    clausius_mossotti_eps_inf,
    density,
    dielectric_constant,
    equilibrate,
    heat_capacity,
    heat_of_vaporization,
    isothermal_compressibility,
    thermal_expansion,
)
from .transport import (
    msd,
    pressure_tensor,
    self_diffusion,
    unwrap_com,
    viscosity_einstein,
    viscosity_green_kubo,
    yeh_hummer_correction,
)

# g/cm³ → kg/m³
_G_CM3_TO_KG_M3 = 1000.0


def compute_bulk_properties(
    traj: Trajectory,
    *,
    equil: int = 0,
    molpol: float | None = None,
    gas_pe_per_molecule: float | None = None,
    n_molecules: int | None = None,
    bootstrap: bool = False,
    seed: int | None = None,
) -> BulkProperties:
    """Compute bulk-phase properties from a :class:`Trajectory`.

    Runs whichever kernels the trajectory has data for: density needs volume +
    masses; α/Cp need enthalpy + volume; κ_T needs volume; ε₀ needs dipole +
    volume; ΔHvap needs ``gas_pe_per_molecule``. Missing inputs leave the
    corresponding field ``None``.

    Parameters
    ----------
    traj:
        Source trajectory (native units: kcal/mol, Å³, e·Å).
    equil:
        Number of leading frames to discard as equilibration.
    molpol:
        Molecular polarizability (Å³). If given with a known volume, sets
        ε_∞ via Clausius-Mossotti; otherwise ε_∞ = 1.
    gas_pe_per_molecule:
        Gas-phase per-molecule potential average (kcal/mol) for ΔHvap.
    n_molecules:
        Molecule count (defaults to ``traj.n_molecules``).
    bootstrap:
        If True, attach bootstrap standard errors to ``metadata['errors']``.
    seed:
        Seed for the bootstrap (for reproducibility).

    Returns
    -------
    :class:`BulkProperties` with a ``metadata`` dict carrying the point
    estimates (and errors, if requested).
    """
    T = traj.temperature_K
    nmol = n_molecules if n_molecules is not None else traj.n_molecules
    meta: dict = {"equil": equil, "temperature_K": T}
    errors: dict = {}

    vol = equilibrate(traj.volume, equil) if traj.volume is not None else None
    enth = equilibrate(traj.enthalpy, equil) if traj.enthalpy is not None else None
    dip = equilibrate(traj.dipole, equil) if traj.dipole is not None else None

    props = BulkProperties(temperature_K=T, metadata=meta)

    # Density --------------------------------------------------------------
    if vol is not None and traj.total_mass is not None:
        rho = density(vol, traj.total_mass)  # g/cm³
        rho_mean, rho_err, _ = mean_stderr(rho)
        props.density_kg_m3 = rho_mean * _G_CM3_TO_KG_M3
        meta["density_g_cm3"] = rho_mean
        if bootstrap:
            errors["density_g_cm3"] = bootstrap_error(
                lambda idx: float(rho[idx].mean()), len(rho), seed=seed
            )

    # Thermal expansion and heat capacity (need enthalpy + volume) ---------
    if enth is not None and vol is not None:
        L = min(len(enth), len(vol))
        h, v = enth[:L], vol[:L]
        props.alpha_T = thermal_expansion(h, v, T)
        meta["alpha_T"] = props.alpha_T
        if nmol:
            props.metadata["cp"] = heat_capacity(h, nmol, T)
        if bootstrap:
            errors["alpha_T"] = bootstrap_error(
                lambda idx: thermal_expansion(h[idx], v[idx], T), L, seed=seed
            )

    # Isothermal compressibility (needs volume) ----------------------------
    if vol is not None:
        props.kappa_T = isothermal_compressibility(vol, T)
        meta["kappa_T"] = props.kappa_T
        if bootstrap:
            errors["kappa_T"] = bootstrap_error(
                lambda idx: isothermal_compressibility(vol[idx], T), len(vol), seed=seed
            )

    # Static dielectric constant (needs dipole + volume) -------------------
    if dip is not None and dip.size and vol is not None:
        eps_inf = 1.0
        if molpol is not None and molpol > 0 and nmol:
            vmol = float(np.mean(vol)) / nmol
            eps_inf = clausius_mossotti_eps_inf(molpol, vmol)
        props.dielectric = dielectric_constant(dip, vol, T, eps_inf=eps_inf)
        meta["dielectric"] = props.dielectric
        meta["eps_inf"] = eps_inf

    # Heat of vaporization (needs gas reference) ---------------------------
    if gas_pe_per_molecule is not None and traj.potential_energy is not None and nmol:
        liquid_pe = equilibrate(traj.potential_energy, equil)
        liquid_pe_per_mol = float(liquid_pe.mean()) / nmol
        props.delta_hvap_kcal_mol = heat_of_vaporization(gas_pe_per_molecule, liquid_pe_per_mol, T)
        meta["delta_hvap_kcal_mol"] = props.delta_hvap_kcal_mol

    if bootstrap:
        meta["errors"] = errors
    return props


def __getattr__(name: str):
    """Lazily expose the ``evaluate`` subpackage without eager (circular) import.

    ``mdforge.liquid.evaluate`` pulls in the config/ingest/report layer (and its
    optional pyyaml/gsd deps only when actually used); importing it lazily keeps
    ``import mdforge.liquid`` light and avoids a compute_bulk_properties import cycle.
    """
    if name == "evaluate":
        import importlib

        module = importlib.import_module(".evaluate", __name__)
        globals()["evaluate"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # subpackages
    "evaluate",
    # submodules
    "parse", "thermo", "transport", "stats", "plots", "structure",
    # records
    "Trajectory", "BulkProperties",
    # orchestrator
    "compute_bulk_properties",
    # stats
    "bzavg", "statistical_inefficiency", "mean_stderr", "bootstrap_error",
    # thermo kernels
    "equilibrate", "density", "thermal_expansion", "isothermal_compressibility",
    "heat_capacity", "dielectric_constant", "clausius_mossotti_eps_inf",
    "heat_of_vaporization",
    # transport kernels
    "pressure_tensor", "viscosity_einstein", "viscosity_green_kubo",
    "unwrap_com", "msd", "self_diffusion", "yeh_hummer_correction",
    # structure kernels
    "rdf", "coordination_number",
    "angular_rdf", "normals_from_orientations", "plane_normal_from_points",
    # misc
    "R_KCAL",
]
