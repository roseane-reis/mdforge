"""Tests for the Tinker-log → HOOMD-thermo pre-converter."""

from __future__ import annotations

import numpy as np
import pytest

from mdforge.liquid.parse import trajectory_from_hoomd_npy
from mdforge.liquid.tinker_thermo import build_thermo_array, write_thermo

_KB = 0.0019872041


def _dyn_log(tmp_path):
    p = tmp_path / "liquid.log"
    p.write_text(
        " Current Time                 1.0 Picosecond\n"
        " Current Potential       -100.0 Kcal/mole\n"
        " Current Kinetic           50.0 Kcal/mole\n"
        " Lattice Lengths        20.0 20.0 20.0\n"
        " Current Time                 2.0 Picosecond\n"
        " Current Potential       -101.0 Kcal/mole\n"
        " Current Kinetic           51.0 Kcal/mole\n"
        " Lattice Lengths        20.0 20.0 20.0\n"
    )
    return p


def _analyze_log(tmp_path):
    p = tmp_path / "analysis.log"
    p.write_text(
        " Number of Atoms                            6\n"
        " Total System Mass :    36.03\n"
        " Total Potential Energy :    -100.0 Kcal/mole\n"
        " Dipole X,Y,Z-Components :    1.0 2.0 3.0\n"
        " Total Potential Energy :    -101.0 Kcal/mole\n"
        " Dipole X,Y,Z-Components :    1.1 2.1 3.1\n"
    )
    return p


def test_build_thermo_with_analyze(tmp_path):
    arr = build_thermo_array(_dyn_log(tmp_path), analyze_log=_analyze_log(tmp_path))
    assert set(arr.dtype.names) == {
        "step", "time_ps", "temp_K", "pe", "ke", "e_total",
        "volume_ang3", "density_gcc", "dipole_x", "dipole_y", "dipole_z"}
    assert np.allclose(arr["time_ps"], [1.0, 2.0])
    assert np.allclose(arr["pe"], [-100.0, -101.0])
    assert np.allclose(arr["e_total"], arr["pe"] + arr["ke"])
    assert np.allclose(arr["volume_ang3"], 8000.0)
    # density from the system mass; temperature from KE and (3N-3) DOF
    assert arr["density_gcc"][0] == pytest.approx(36.03 / (0.602214076 * 8000.0))
    assert arr["temp_K"][0] == pytest.approx(2.0 * 50.0 / ((3 * 6 - 3) * _KB))
    # per-frame cell dipole (Debye) carried through for the dielectric
    assert "dipole_x" in arr.dtype.names
    assert np.allclose(arr["dipole_x"], [1.0, 1.1])
    assert np.allclose(arr["dipole_z"], [3.0, 3.1])


def test_build_thermo_dynamics_only_omits_derived(tmp_path):
    # No analyze log and no atom count / mass -> no temp_K, no density_gcc.
    arr = build_thermo_array(_dyn_log(tmp_path))
    assert "temp_K" not in arr.dtype.names
    assert "density_gcc" not in arr.dtype.names
    assert set(arr.dtype.names) == {"step", "time_ps", "pe", "ke", "e_total", "volume_ang3"}


def test_written_npy_is_ingestible(tmp_path):
    arr = build_thermo_array(_dyn_log(tmp_path), analyze_log=_analyze_log(tmp_path))
    out = write_thermo(tmp_path / "thermo.npy", arr)
    traj = trajectory_from_hoomd_npy(out, n_molecules=2, molar_mass_g_mol=18.0)
    assert traj.volume is not None and np.allclose(traj.volume, 8000.0)
    assert traj.enthalpy is not None                     # PE + KE mapped through
    assert traj.potential_energy is not None


def test_write_csv_has_header(tmp_path):
    arr = build_thermo_array(_dyn_log(tmp_path), analyze_log=_analyze_log(tmp_path))
    out = write_thermo(tmp_path / "thermo.csv", arr)
    header = out.read_text().splitlines()[0]
    assert header.split(",") == list(arr.dtype.names)
