"""Tests for mdforge.simulate — box math, replication, jobs, and end-to-end NPT.

Box/density math, lattice replication, and the job dispatcher are pure and
always run. The end-to-end NPT runs (build box → engine.dynamics → Trajectory →
liquid density) are backend-gated and skip when Tinker/OpenMM are absent.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from mdforge.formats.txyz import TinkerXYZ
from mdforge.simulate import (
    box,
    box_edge_for_density,
    density_of_box,
    molar_mass,
    n_molecules_for_box,
    replicate_cubic,
    run_jobs,
)

_TINKER_BIN = Path(os.environ.get("MDFORGE_TINKER_BIN", "/opt/tinker/bin"))
_TINKER_EXAMPLE = Path(os.environ.get("MDFORGE_TINKER_EXAMPLE", "/opt/tinker/example"))
_HAS_TINKER = (_TINKER_BIN / "dynamic").exists() and (_TINKER_EXAMPLE / "hipposmall.xyz").exists()
_HAS_OPENMM = importlib.util.find_spec("openmm") is not None

tinker_only = pytest.mark.skipif(not _HAS_TINKER, reason="local Tinker unavailable")
openmm_only = pytest.mark.skipif(not _HAS_OPENMM, reason="openmm unavailable")


def _water() -> TinkerXYZ:
    return TinkerXYZ(
        names=["O", "H", "H"],
        coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        types=np.array([1, 2, 2]),
        connectivity=[[2, 3], [1], [1]],
    )


# ---------------------------------------------------------------------------
# box / density math
# ---------------------------------------------------------------------------

class TestBoxMath:
    def test_molar_mass_water(self):
        assert molar_mass(_water()) == pytest.approx(18.015, abs=0.01)

    def test_density_box_n_roundtrip(self):
        mw, dens = 18.015, 0.997
        edge = box_edge_for_density(216, mw, dens)
        assert edge == pytest.approx(18.64, abs=0.1)
        assert n_molecules_for_box(edge, mw, dens) == 216
        assert density_of_box(216, mw, edge) == pytest.approx(dens, rel=1e-6)

    def test_density_of_known_box(self):
        # 216 HIPPO waters in an 18.655 Å cube ≈ 0.995 g/cm³
        assert density_of_box(216, 18.015, 18.655) == pytest.approx(0.995, abs=0.01)


class TestReplicate:
    def test_replicate_counts_and_box(self):
        boxed = replicate_cubic(_water(), n_copies=8, box_edge=12.0)
        assert boxed.n_atoms == 24
        assert np.allclose(boxed.box, [12.0, 12.0, 12.0, 90, 90, 90])
        assert list(boxed.types) == [1, 2, 2] * 8

    def test_replicate_connectivity_offset(self):
        boxed = replicate_cubic(_water(), n_copies=3, box_edge=15.0)
        # copy 0: O bonded to 2,3 ; copy 1: O is atom 4, bonded to 5,6
        assert boxed.connectivity[0] == [2, 3]
        assert boxed.connectivity[3] == [5, 6]
        assert boxed.connectivity[6] == [8, 9]

    def test_replicate_copies_separated(self):
        boxed = replicate_cubic(_water(), n_copies=8, box_edge=12.0)
        # the two O atoms of copies 0 and 1 should be ~spacing apart, not overlapping
        o0, o1 = boxed.coords[0], boxed.coords[3]
        assert np.linalg.norm(o1 - o0) > 1.0


# ---------------------------------------------------------------------------
# jobs dispatch
# ---------------------------------------------------------------------------

class TestJobs:
    def test_parallel_results_in_order(self):
        jobs = {"a": lambda: 1, "b": lambda: 2, "c": lambda: 3}
        results = run_jobs(jobs, max_parallel=3)
        assert [r.name for r in results] == ["a", "b", "c"]
        assert [r.value for r in results] == [1, 2, 3]
        assert all(r.ok for r in results)

    def test_failure_isolated(self):
        def boom():
            raise RuntimeError("kaboom")
        results = run_jobs({"ok": lambda: 42, "bad": boom}, max_parallel=2)
        by_name = {r.name: r for r in results}
        assert by_name["ok"].ok and by_name["ok"].value == 42
        assert not by_name["bad"].ok
        assert "kaboom" in by_name["bad"].error

    def test_list_jobs_indexed(self):
        results = run_jobs([lambda: 10, lambda: 20], max_parallel=2)
        assert [r.value for r in results] == [10, 20]
        assert [r.name for r in results] == ["0", "1"]

    def test_elapsed_recorded(self):
        results = run_jobs({"x": lambda: sum(range(1000))}, max_parallel=1)
        assert results[0].elapsed_s >= 0.0


# ---------------------------------------------------------------------------
# end-to-end NPT (backend-gated)
# ---------------------------------------------------------------------------

@openmm_only
class TestOpenMMEndToEnd:
    def test_water_box_npt_density(self):
        from mdforge.liquid import compute_bulk_properties
        from mdforge.simulate import build_openmm_water_box, run_npt

        eng = build_openmm_water_box(n_waters=150, nonbonded_cutoff_nm=0.6)
        traj = run_npt(eng, None, nsteps=400, dt_fs=2.0, temperature=298.15,
                       pressure=1.0, report_interval=200, minimize=True)
        assert traj.n_frames == 2
        assert traj.masses is not None and traj.volume is not None
        props = compute_bulk_properties(traj, equil=0, n_molecules=150)
        # short, unequilibrated TIP3P → loose but physical density window
        assert 500 < props.density_kg_m3 < 1500


@tinker_only
class TestTinkerEndToEnd:
    def test_hippo_water_box_npt_density(self):
        from mdforge.engine import get_engine
        from mdforge.formats import txyz
        from mdforge.liquid import compute_bulk_properties
        from mdforge.simulate import run_npt

        boxed = txyz.read_txyz(_TINKER_EXAMPLE / "hipposmall.xyz")  # 216 HIPPO waters
        eng = get_engine("tinker", bin_dir=_TINKER_BIN,
                         key_file=_TINKER_EXAMPLE / "hipposmall.key")
        traj = run_npt(eng, boxed, nsteps=4, dt_fs=1.0, temperature=298.0,
                       pressure=1.0, save_ps=0.001, minimize=False)
        assert traj.n_frames == 4
        assert traj.masses is not None
        props = compute_bulk_properties(traj, equil=0, n_molecules=216)
        # equilibrated HIPPO water box → ~996 kg/m³
        assert 900 < props.density_kg_m3 < 1100


def test_simulate_submodules_importable():
    assert hasattr(box, "replicate_cubic")
