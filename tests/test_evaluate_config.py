"""Tests for the evaluation config loader, validation, and state guard."""

from __future__ import annotations

import pytest

from mdforge.liquid.evaluate.config import (
    EvalConfig,
    EvalConfigError,
    EvalStateError,
    StateSpec,
    state_guard,
)


def _min_config(**over):
    base = {
        "model": {"name": "M"},
        "system": {"n_molecules": 10, "charges_e": {"O": -0.68, "H": 0.34}},
        "topology": {"pdb": "liquid.pdb"},
        "legs": [{"name": "npt", "ensemble": "NPT", "trajectory": "npt.gsd", "log": "npt.npy"}],
    }
    base.update(over)
    return base


def test_from_dict_defaults():
    cfg = EvalConfig.from_dict(_min_config())
    assert cfg.model.name == "M"
    assert cfg.state.temperature_K == 298.15
    assert cfg.legs[0].ensemble == "NPT"
    assert cfg.analysis.rdf.n_bins == 200
    assert cfg.system.gas_pe_per_molecule == 0.0


def test_output_timeseries_flag():
    assert EvalConfig.from_dict(_min_config()).output.timeseries is False
    cfg = EvalConfig.from_dict(_min_config(output={"timeseries": True}))
    assert cfg.output.timeseries is True


def test_default_equil_frac_by_ensemble():
    cfg = EvalConfig.from_dict(_min_config(legs=[
        {"name": "npt", "ensemble": "NPT", "trajectory": "a.gsd"},
        {"name": "nvt", "ensemble": "NVT", "log": "b.npy"},
        {"name": "nve", "ensemble": "NVE", "log": "c.npy"},
    ]))
    fracs = {leg.name: leg.resolved_equil_frac() for leg in cfg.legs}
    assert fracs == {"npt": 0.5, "nvt": 0.2, "nve": 0.2}


def test_per_leg_equil_frac_overrides():
    cfg = EvalConfig.from_dict(_min_config(legs=[
        {"name": "npt", "ensemble": "NPT", "trajectory": "a.gsd", "equil_frac": 0.7},
    ]))
    assert cfg.legs[0].resolved_equil_frac() == 0.7


def test_unknown_key_raises():
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(system={"n_molecules": 10, "typo_field": 1}))


def test_no_legs_raises():
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(legs=[]))


def test_leg_needs_trajectory_or_log():
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(legs=[{"name": "x", "ensemble": "NPT"}]))


def test_bad_ensemble_raises():
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(legs=[
            {"name": "x", "ensemble": "REPLICA", "trajectory": "a.gsd"}]))


def test_topology_required():
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(topology={}))


def test_path_resolution_relative(tmp_path):
    cfg = EvalConfig.from_dict(_min_config(), base_dir=tmp_path)
    assert cfg.resolve("npt.gsd") == tmp_path / "npt.gsd"
    assert cfg.resolve("/abs/x.gsd").is_absolute()
    assert cfg.resolve(None) is None


def test_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_min_config()))
    cfg = EvalConfig.from_yaml(p)
    assert cfg.model.name == "M"
    assert cfg.base_dir == tmp_path


def test_cli_overrides_rdf_knobs(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    from mdforge.liquid.evaluate.__main__ import _build_parser, _load_config

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_min_config(analysis={"rdf": {"r_max": 8.0, "n_bins": 200}})))

    base = _load_config(_build_parser().parse_args(["--config", str(cfg_path)]))
    assert base.analysis.rdf.r_max == 8.0 and base.analysis.rdf.n_bins == 200

    over = _load_config(_build_parser().parse_args(
        ["--config", str(cfg_path), "--r-max", "6.5", "--rdf-bins", "130"]))
    assert over.analysis.rdf.r_max == 6.5
    assert over.analysis.rdf.n_bins == 130


def test_virtual_sites_4site_builds_profile():
    # A parsed 4-site config must build a profile with the M-site massless,
    # placed last, and net-neutral (not just echo back the stored fields).
    from mdforge.liquid.evaluate.profiles.water import water_profile
    cfg = EvalConfig.from_dict(_min_config(system={
        "n_molecules": 10, "atoms_per_molecule": 4, "virtual_sites": ["M"],
        "charges_e": {"O": 0.0, "H": 0.52422, "M": -1.04844},
    }))
    prof = water_profile(charges_e=cfg.system.charges_e,
                         atoms_per_molecule=cfg.system.atoms_per_molecule,
                         virtual_sites=cfg.system.virtual_sites)
    assert prof.element_order == ("O", "H", "H", "M")
    assert prof.virtual_local_indices == (3,)
    assert prof.per_molecule_masses().tolist()[3] == 0.0
    assert prof.net_charge() == pytest.approx(0.0, abs=1e-9)


def test_virtual_sites_apm_mismatch_raises():
    # virtual_sites=["M"] implies atoms_per_molecule=4; leaving it at 3 is an error
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(system={
            "n_molecules": 10, "atoms_per_molecule": 3, "virtual_sites": ["M"],
            "charges_e": {"O": 0.0, "H": 0.52422, "M": -1.04844},
        }))


def test_atoms_per_molecule_without_virtual_sites_raises():
    # a bare atoms_per_molecule != 3 (no virtual_sites) must fail at config time,
    # not slip through to a confusing profile-build / ingest error later
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(system={
            "n_molecules": 10, "atoms_per_molecule": 4,
            "charges_e": {"O": 0.0, "H": 0.52422}}))


def test_virtual_sites_must_be_string_list():
    with pytest.raises(EvalConfigError):
        EvalConfig.from_dict(_min_config(system={
            "n_molecules": 10, "atoms_per_molecule": 4, "virtual_sites": ["M", ""],
            "charges_e": {"O": 0.0, "H": 0.52422, "M": -1.04844},
        }))


def test_state_guard_pass():
    state_guard(StateSpec(temperature_K=298.15, pressure_atm=1.0))
    state_guard(StateSpec(temperature_K=298.6, pressure_atm=1.3))  # within tol


def test_state_guard_off_state_raises():
    with pytest.raises(EvalStateError):
        state_guard(StateSpec(temperature_K=350.0, pressure_atm=1.0))
    with pytest.raises(EvalStateError):
        state_guard(StateSpec(temperature_K=298.15, pressure_atm=1000.0))
