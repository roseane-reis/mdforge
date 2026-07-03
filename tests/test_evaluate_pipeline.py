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
