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


_TIP4P_CHARGES = {"O": 0.0, "H": 0.52422, "M": -1.04844}


def _config_4site(tmp_path, **over):
    base = {
        "model": {"name": "TIP4P"},
        "system": {"n_molecules": 27, "atoms_per_molecule": 4,
                   "virtual_sites": ["M"], "charges_e": _TIP4P_CHARGES},
        "topology": {"txyz": "liquid.xyz"},
        "legs": [{"name": "npt", "ensemble": "NPT", "trajectory": "liquid.dcd",
                  "equil_frac": 0.0}],
        # box is 14 Å (fixture) → r_max must stay ≤ 7 Å (minimum-image limit)
        "analysis": {"rdf": {"r_max": 6.0, "n_bins": 60}, "structure_stride": 1},
    }
    base.update(over)
    return EvalConfig.from_dict(base, base_dir=tmp_path)


def test_profile_4site_from_txyz(tmp_path, make_water_4site):
    make_water_4site(tmp_path, n_mol=27)
    cfg = _config_4site(tmp_path)
    profile, n_mol = water_profile_from_topology(cfg)
    assert n_mol == 27
    assert profile.atoms_per_molecule == 4
    assert profile.element_order == ("O", "H", "H", "M")
    assert profile.virtual_local_indices == (3,)
    assert profile.oxygen_local_index == 0
    assert profile.hydrogen_local_indices == (1, 2)
    # M-site is massless; net charge ~0
    assert profile.per_molecule_masses().tolist()[3] == 0.0
    assert profile.net_charge() == pytest.approx(0.0, abs=1e-9)


def test_profile_4site_missing_ghost_charge_raises(tmp_path, make_water_4site):
    make_water_4site(tmp_path, n_mol=8)
    # charges_e omits the M-site charge → the profile builder must reject it
    cfg = _config_4site(tmp_path, system={
        "n_molecules": 8, "atoms_per_molecule": 4, "virtual_sites": ["M"],
        "charges_e": {"O": 0.0, "H": 0.52422}})
    with pytest.raises(ValueError):
        water_profile_from_topology(cfg)


def test_ingest_4site_dcd_excludes_m_from_com(tmp_path, make_water_4site):
    make_water_4site(tmp_path, n_mol=27, n_frames=8)
    cfg = _config_4site(tmp_path)
    profile, n_mol = water_profile_from_topology(cfg)
    leg = ingest_leg(cfg.legs[0], cfg, profile, n_mol)
    assert leg.source == "dcd"
    assert leg.atoms.shape == (8, 27 * 4, 3)
    assert leg.com.shape == (8, 27, 3)
    # O/H selections skip the M-site (every 4th atom starting at index 3)
    assert leg.o_idx.tolist() == list(range(0, 27 * 4, 4))
    assert leg.h_idx.tolist()[:4] == [1, 2, 5, 6]
    assert 3 not in set(leg.o_idx.tolist()) | set(leg.h_idx.tolist())
    # COM must equal the mass-weighted O,H,H centre (M contributes nothing)
    mol0 = leg.atoms[0, :4]
    m = np.array([15.9994, 1.00794, 1.00794])
    com_ref = (m[:, None] * mol0[:3]).sum(0) / m.sum()
    assert np.allclose(leg.com[0, 0], com_ref, atol=1e-4)


def test_dielectric_4site_includes_msite(tmp_path, make_water_4site):
    # Contract: the M-site MUST enter the cell dipole. Verify the per-molecule
    # dipole is the physical TIP4P value (~2.2 D) and that the ghost site actually
    # matters — collapsing its charge onto O (still neutral) shifts the dipole.
    from mdforge.liquid.evaluate.pipeline import _cell_dipole, _compute_dielectric
    from mdforge.liquid.evaluate.profiles.water import water_profile

    make_water_4site(tmp_path, n_mol=27, n_frames=10)
    cfg = _config_4site(tmp_path)
    profile, n_mol = water_profile_from_topology(cfg)
    leg = ingest_leg(cfg.legs[0], cfg, profile, n_mol)

    out = _compute_dielectric(leg, cfg, profile)
    assert out["dipole_source"] == "point_charge"       # no engine dipole in a DCD
    assert out["net_charge_e"] == pytest.approx(0.0, abs=1e-9)
    assert 2.0 < out["mu_molecule_debye"] < 2.5         # physical TIP4P dipole

    _, mu_4site = _cell_dipole(leg, profile, eq=0)
    prof_collapsed = water_profile(
        charges_e={"O": -1.04844, "H": 0.52422, "M": 0.0},
        atoms_per_molecule=4, virtual_sites=["M"])
    _, mu_collapsed = _cell_dipole(leg, prof_collapsed, eq=0)
    d4 = np.mean(np.linalg.norm(mu_4site, axis=-1)) * 4.803
    dc = np.mean(np.linalg.norm(mu_collapsed, axis=-1)) * 4.803
    # the ~0.15 Å M offset changes the molecular dipole by ~0.7 D
    assert dc > d4 + 0.3


def test_net_charge_warning_for_bad_4site_charges(tmp_path, make_water_4site):
    # A non-neutral molecule (mis-entered ghost charge) must surface a warning,
    # because the point-charge dielectric assumes net neutrality.
    from mdforge.liquid.evaluate.pipeline import run_evaluation

    make_water_4site(tmp_path, n_mol=27, n_frames=8)
    cfg = _config_4site(tmp_path, system={
        "n_molecules": 27, "atoms_per_molecule": 4, "virtual_sites": ["M"],
        "charges_e": {"O": 0.0, "H": 0.52422, "M": -0.5}})   # net +0.548
    res = run_evaluation(cfg, enforce_state=False)
    assert any("net charge" in w for w in res.warnings)


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
