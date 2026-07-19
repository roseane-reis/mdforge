"""Tests for the capability-driven pipeline and graceful degradation."""

from __future__ import annotations

import math

import pytest

from mdforge.liquid.evaluate.config import EvalConfig
from mdforge.liquid.evaluate.pipeline import run_evaluation

pytest.importorskip("gsd.hoomd")


def _cfg(tmp_path, legs, **over):
    base = {
        "model": {"name": "SynthWater"},
        "system": {"n_molecules": 27, "charges_e": {"O": -0.68, "H": 0.34}},
        "topology": {"pdb": str(tmp_path / "liquid.pdb")},
        "legs": legs,
        "analysis": {"diffusion": {"dt_ps": 5.0}, "rdf": {"r_max": 6.0, "n_bins": 60}},
    }
    base.update(over)
    return EvalConfig.from_dict(base)


@pytest.fixture
def synth_run(tmp_path, make_water_gsd, make_water_pdb, make_hoomd_npy):
    make_water_pdb(tmp_path / "liquid.pdb", n_mol=27, box_L=12.0)
    make_water_gsd(tmp_path / "npt.gsd", n_mol=27, box_L=12.0, n_frames=12, seed=1)
    make_water_gsd(tmp_path / "nvt.gsd", n_mol=27, box_L=12.0, n_frames=12, seed=2)
    make_hoomd_npy(tmp_path / "npt.npy", ensemble="npt", n_molecules=27)
    make_hoomd_npy(tmp_path / "nvt.npy", ensemble="nvt", n_molecules=27)
    return tmp_path


def test_single_npt_produces_all_computable(synth_run):
    cfg = _cfg(synth_run, [{"name": "npt", "ensemble": "NPT",
                            "trajectory": str(synth_run / "npt.gsd"),
                            "log": str(synth_run / "npt.npy")}])
    res = run_evaluation(cfg)
    # thermo + structure + diffusion + dielectric all from the one NPT leg
    for prop in ("density", "delta_hvap", "cp", "alpha_T", "kappa_T",
                 "gOO_peak_r", "tetrahedral_q", "hbonds_per_molecule",
                 "coordination_number", "self_diffusion", "dielectric"):
        assert prop in res.scoring_inputs, prop
        assert math.isfinite(res.scoring_inputs[prop][0]), prop
    assert any("model's own density" in w for w in res.warnings)


def test_two_leg_prefers_nvt_for_structure(synth_run):
    cfg = _cfg(synth_run, [
        {"name": "npt", "ensemble": "NPT", "trajectory": str(synth_run / "npt.gsd"),
         "log": str(synth_run / "npt.npy")},
        {"name": "nvt", "ensemble": "NVT", "trajectory": str(synth_run / "nvt.gsd"),
         "log": str(synth_run / "nvt.npy")},
    ])
    res = run_evaluation(cfg)
    assert set(res.structure) == {"npt", "nvt"}
    # thermo (density) from NPT; structure/diffusion/dielectric from NVT
    assert res.scoring_sources["density"] == "npt"
    assert res.scoring_sources["gOO_peak_r"] == "nvt"
    assert res.scoring_sources["self_diffusion"] == "nvt"
    assert res.scoring_sources["dielectric"] == "nvt"


def test_degrade_log_only_no_trajectory(synth_run):
    cfg = _cfg(synth_run, [{"name": "npt", "ensemble": "NPT",
                            "log": str(synth_run / "npt.npy")}])
    res = run_evaluation(cfg)
    assert "density" in res.scoring_inputs           # thermo from the log
    assert "gOO_peak_r" not in res.scoring_inputs    # no trajectory -> no structure
    assert "self_diffusion" not in res.scoring_inputs


def test_degrade_trajectory_only_no_log(synth_run):
    cfg = _cfg(synth_run, [{"name": "npt", "ensemble": "NPT",
                            "trajectory": str(synth_run / "npt.gsd")}])
    res = run_evaluation(cfg)
    assert "gOO_peak_r" in res.scoring_inputs        # structure from the GSD
    assert "self_diffusion" in res.scoring_inputs    # diffusion from COM
    assert "density" not in res.scoring_inputs       # no thermo log


def test_dielectric_prefers_engine_dipole():
    import numpy as np

    from mdforge.liquid.evaluate.config import EvalConfig
    from mdforge.liquid.evaluate.ingest import LegData
    from mdforge.liquid.evaluate.pipeline import _compute_dielectric
    from mdforge.liquid.evaluate.profiles.water import get_profile

    cfg = EvalConfig.from_dict({
        "system": {"n_molecules": 100, "charges_e": {"O": -0.8, "H": 0.4}},
        "topology": {"pdb": "x.pdb"},
        "legs": [{"name": "npt", "ensemble": "NPT", "log": "l.npy"}],
    })
    profile = get_profile("water", charges_e={"O": -0.8, "H": 0.4})

    rng = np.random.default_rng(0)
    nfr = 400
    M = rng.normal(0.0, 40.0, (nfr, 3))          # Debye
    vol = np.full(nfr, 30000.0)
    raw = {"dipole_x": M[:, 0], "dipole_y": M[:, 1], "dipole_z": M[:, 2],
           "volume_ang3": vol}
    leg = LegData(name="npt", ensemble="NPT", equil_frac=0.1, source="log-only",
                  n_molecules=100, raw_columns=raw)

    out = _compute_dielectric(leg, cfg, profile)
    assert out["dipole_source"] == "engine_dipole"
    assert out["epsilon_0"] > 1.0

    # variance drives ε: doubling the dipole quadruples (ε − ε∞)
    raw2 = {k: (2 * v if k.startswith("dipole") else v) for k, v in raw.items()}
    leg2 = LegData(name="npt", ensemble="NPT", equil_frac=0.1, source="log-only",
                   n_molecules=100, raw_columns=raw2)
    out2 = _compute_dielectric(leg2, cfg, profile)
    assert (out2["epsilon_0"] - 1.0) == pytest.approx(4.0 * (out["epsilon_0"] - 1.0), rel=1e-6)


def test_4site_gsd_end_to_end(tmp_path, make_water_4site_gsd, make_water_pdb, make_hoomd_npy):
    # A HOOMD 4-site (M-site) rigid-body run: the GSD reconstructs O,H,H,M per
    # molecule; the PDB topology lists only the 3 real atoms. With
    # atoms_per_molecule=4 + virtual_sites=[M], structure uses O/H only and the
    # dielectric includes the M-site (per-molecule dipole ~2.3 D).
    make_water_4site_gsd(tmp_path / "npt.gsd", n_mol=27, box_L=14.0, n_frames=12)
    make_water_pdb(tmp_path / "liquid.pdb", n_mol=27, box_L=14.0)   # 3-site, no M
    make_hoomd_npy(tmp_path / "npt.npy", ensemble="npt", n_molecules=27, box_L=14.0)

    base = {
        "model": {"name": "msite"},
        "system": {"n_molecules": 27, "atoms_per_molecule": 4, "virtual_sites": ["M"],
                   "charges_e": {"O": 0.0, "H": 0.55975, "M": -1.1195}},
        "topology": {"pdb": str(tmp_path / "liquid.pdb")},
        "legs": [{"name": "npt", "ensemble": "NPT", "trajectory": str(tmp_path / "npt.gsd"),
                  "log": str(tmp_path / "npt.npy")}],
        "analysis": {"rdf": {"r_max": 6.0, "n_bins": 60}, "diffusion": {"dt_ps": 5.0}},
    }
    res = run_evaluation(EvalConfig.from_dict(base))
    st = res.structure["npt"]
    assert len(st["g_OO"]) == 60 and math.isfinite(st["gOO_peak_r"])
    di = res.dielectric["npt"]
    assert di["dipole_source"] == "point_charge"
    assert di["net_charge_e"] == pytest.approx(0.0, abs=1e-9)
    assert 2.0 < di["mu_molecule_debye"] < 2.6          # M-site enters the dipole

    # the old 3-site config on this 4-site GSD must be rejected, not silently wrong
    wrong = {**base, "system": {"n_molecules": 27, "atoms_per_molecule": 3,
                                "charges_e": {"O": -0.70, "H": 0.35}}}
    with pytest.raises(ValueError, match="sites per molecule"):
        run_evaluation(EvalConfig.from_dict(wrong))


def test_first_minimum_robust_to_noise_dip():
    import numpy as np

    from mdforge.liquid.evaluate.pipeline import _first_minimum

    r = np.linspace(0.0, 8.0, 400)
    g = (1.0
         + 2.0 * np.exp(-((r - 2.8) / 0.18) ** 2)      # first peak ~2.8
         - 0.6 * np.exp(-((r - 3.4) / 0.30) ** 2)      # first-minimum well ~3.4
         + 0.4 * np.exp(-((r - 4.5) / 0.45) ** 2))     # second peak ~4.5
    g[r < 2.4] = 0.0

    r_min, g_min = _first_minimum(r, g)
    assert 3.2 <= r_min <= 3.6 and g_min < 1.0          # true first minimum ~3.4

    # a shallow spurious dip on the descending flank (higher than the true well)
    # must NOT be picked — the global min in the physical window is still ~3.4
    g_noisy = g.copy()
    g_noisy[(r > 3.05) & (r < 3.15)] -= 0.4
    r_min_n, _ = _first_minimum(r, g_noisy)
    assert 3.2 <= r_min_n <= 3.6


def test_record_timeseries_populates_series(synth_run):
    cfg = _cfg(synth_run, [{"name": "npt", "ensemble": "NPT",
                            "trajectory": str(synth_run / "npt.gsd"),
                            "log": str(synth_run / "npt.npy")}])
    off = run_evaluation(cfg)
    assert off.series == {}                          # opt-in: nothing by default

    res = run_evaluation(cfg, record_timeseries=True)
    assert set(res.series) == {"npt"}
    s = res.series["npt"]
    assert s["dt_ps"] == pytest.approx(5.0)          # from the log's time_ps
    assert s["ensemble"] == "NPT"
    assert s["t_ps"][0] == pytest.approx(0.0)        # real clock retained (starts at 0 here)
    assert len(s["t_ps"]) == s["n_frames"]
    for label in ("density (g/cm³)", "temperature (K)", "pressure (atm)"):
        assert label in s["columns"]
        assert len(s["columns"][label]) == s["n_frames"]
    assert 0 <= s["equil"] < s["n_frames"]
    assert res.to_json_dict()["series"]["npt"]["columns"]  # serialisable


def test_state_guard_blocks_off_state(synth_run):
    cfg = _cfg(synth_run, [{"name": "npt", "ensemble": "NPT",
                            "trajectory": str(synth_run / "npt.gsd")}],
               state={"temperature_K": 350.0})
    from mdforge.liquid.evaluate.config import EvalStateError
    with pytest.raises(EvalStateError):
        run_evaluation(cfg)
    # bypass works
    res = run_evaluation(cfg, enforce_state=False)
    assert "gOO_peak_r" in res.scoring_inputs
