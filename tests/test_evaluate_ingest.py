"""Tests for topology→profile, log parsing, and campaign auto-discovery."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from mdforge.liquid.evaluate.config import EvalConfig
from mdforge.liquid.evaluate.ingest import (
    _csv_to_structured,
    ingest_leg,
    legs_from_campaign,
    water_profile_from_topology,
)


def _config(tmp_path, **over):
    base = {
        "model": {"name": "M"},
        "system": {"charges_e": {"O": -0.68, "H": 0.34}},
        "topology": {"pdb": "liquid.pdb"},
        "legs": [{"name": "npt", "ensemble": "NPT", "trajectory": "npt.gsd", "log": "npt.npy"}],
    }
    base.update(over)
    return EvalConfig.from_dict(base, base_dir=tmp_path)


def test_profile_and_count_from_pdb(tmp_path, make_water_pdb):
    make_water_pdb(tmp_path / "liquid.pdb", n_mol=27)
    cfg = _config(tmp_path)
    profile, n_mol = water_profile_from_topology(cfg)
    assert n_mol == 27
    assert profile.oxygen_local_index == 0
    assert profile.charges_e == {"O": -0.68, "H": 0.34}
    assert profile.net_charge() == pytest.approx(0.0)


def test_profile_rejects_non_water_topology(tmp_path):
    from mdforge.formats.pdb import to_pdb_string
    # H, H, O order (wrong) should fail element validation
    txt = to_pdb_string(np.zeros((3, 3)), ["H", "H", "O"], res_name="HOH")
    (tmp_path / "bad.pdb").write_text(txt)
    cfg = _config(tmp_path, topology={"pdb": "bad.pdb"}, system={"n_molecules": 1})
    with pytest.raises(ValueError):
        water_profile_from_topology(cfg)


def test_csv_to_structured_matches_npy(tmp_path, make_hoomd_npy):
    npy = make_hoomd_npy(tmp_path / "npt.npy")
    arr = np.load(npy)
    # write an equivalent CSV
    csv_path = tmp_path / "npt.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(arr.dtype.names)
        for row in arr:
            w.writerow([row[n] for n in arr.dtype.names])
    got = _csv_to_structured(csv_path)
    assert got.dtype.names == arr.dtype.names
    for name in arr.dtype.names:
        assert np.allclose(got[name].astype(float), arr[name].astype(float))


def test_ingest_gsd_leg(tmp_path, make_water_gsd, make_water_pdb, make_hoomd_npy):
    pytest.importorskip("gsd.hoomd")
    make_water_pdb(tmp_path / "liquid.pdb", n_mol=27)
    make_water_gsd(tmp_path / "npt.gsd", n_mol=27, n_frames=6)
    make_hoomd_npy(tmp_path / "npt.npy", n_molecules=27)
    cfg = _config(tmp_path)
    profile, n_mol = water_profile_from_topology(cfg)
    leg = ingest_leg(cfg.legs[0], cfg, profile, n_mol)
    assert leg.source == "gsd"
    assert leg.atoms.shape == (6, 27 * 3, 3)
    assert leg.com.shape == (6, 27, 3)
    assert leg.o_idx.tolist() == list(range(0, 81, 3))
    assert leg.traj is not None and leg.traj.n_frames == 60
    # O-H bond lengths physical (~0.96 Å) after reconstruction
    L = leg.box[0]
    oh = leg.atoms[0, 1] - leg.atoms[0, 0]
    oh -= L * np.round(oh / L)
    assert 0.9 < np.linalg.norm(oh) < 1.05


def test_legs_from_campaign(tmp_path):
    run = tmp_path / "run"
    (run / "heat").mkdir(parents=True)
    for sub, stem in (("npt", "npt"), ("nvt2", "nvt"), ("nve", "nve")):
        d = run / sub
        d.mkdir()
        (d / f"{stem}.gsd").write_bytes(b"")
        (d / f"{stem}.npy").write_bytes(b"")
    (run / "meta.json").write_text('{"t_target_K": 298}')
    legs = legs_from_campaign(run)
    names = {leg.name: leg.ensemble for leg in legs}
    assert names == {"npt": "NPT", "nvt2": "NVT", "nve": "NVE"}   # heat skipped
    nvt2 = next(leg for leg in legs if leg.name == "nvt2")
    assert nvt2.trajectory.endswith("nvt2/nvt.gsd")               # actual filename
