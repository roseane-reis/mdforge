"""Tests for mdforge.engine — interface, registry, runner, Tinker, OpenMM.

Backend-dependent tests skip gracefully when a Tinker binary or the openmm
package is absent (per the Phase 2 acceptance criterion). On a machine with both
(the author's), they validate against real local CPU Tinker and local OpenMM,
including a cross-engine point-charge agreement.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mdforge.engine import (
    Capabilities,
    EngineResult,
    LocalRunner,
    UnsupportedOperation,
    available,
    get_engine,
    register,
    require,
    unregister,
)

# --- environment detection --------------------------------------------------
_TINKER_BIN = Path(os.environ.get("MDFORGE_TINKER_BIN", "/opt/tinker/bin"))
_TINKER_EXAMPLE = Path(os.environ.get("MDFORGE_TINKER_EXAMPLE", "/opt/tinker/example"))
_HAS_TINKER = (_TINKER_BIN / "analyze").exists() and (_TINKER_EXAMPLE / "ammonia.xyz").exists()
_HAS_OPENMM = importlib.util.find_spec("openmm") is not None

tinker_only = pytest.mark.skipif(not _HAS_TINKER, reason="local Tinker unavailable")
openmm_only = pytest.mark.skipif(not _HAS_OPENMM, reason="openmm unavailable")


# ---------------------------------------------------------------------------
# base interface
# ---------------------------------------------------------------------------

class TestBase:
    def test_engine_result_energy_normalization(self):
        r = EngineResult(energy=np.array([-463.118]), energy_unit="kJ/mol")
        # -463.118 kJ/mol → kcal/mol
        assert r.energy_in("kcal/mol")[0] == pytest.approx(-110.688, abs=1e-2)

    def test_engine_result_gradient_normalization(self):
        r = EngineResult(energy=np.array([0.0]), energy_unit="kcal/mol",
                         gradient=np.ones((1, 2, 3)), force_unit="kcal/mol/Angstrom")
        out = r.gradient_in("kJ/mol/Angstrom")
        assert np.allclose(out, 4.184)

    def test_engine_result_center_coords_optional(self):
        # Defaults to None; center-based engines may populate it (M,C,3).
        r = EngineResult(energy=np.array([0.0]), energy_unit="kcal/mol")
        assert r.center_coords is None
        cc = np.zeros((1, 2, 3))
        r2 = EngineResult(energy=np.array([0.0]), energy_unit="kcal/mol", center_coords=cc)
        assert r2.center_coords.shape == (1, 2, 3)

    def test_require_passes_and_raises(self):
        class Dummy:
            capabilities = Capabilities(single_point=True)
        require(Dummy(), "single_point")  # ok
        with pytest.raises(UnsupportedOperation, match="dynamics"):
            require(Dummy(), "single_point", "dynamics")


# ---------------------------------------------------------------------------
# registry + plugin seam
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_builtins_registered(self):
        assert "tinker" in available()
        assert "openmm" in available()

    def test_register_and_get_plugin(self):
        class FakeEngine:
            capabilities = Capabilities(single_point=True)

            def __init__(self, **cfg):
                self.cfg = cfg

        register("fake_engine", lambda **cfg: FakeEngine(**cfg))
        try:
            eng = get_engine("fake_engine", foo=1)
            assert isinstance(eng, FakeEngine)
            assert eng.cfg == {"foo": 1}
        finally:
            unregister("fake_engine")
        assert "fake_engine" not in available()

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError):
            get_engine("nonexistent_engine")

    def test_duplicate_register_raises(self):
        register("dup_test", lambda **c: None)
        try:
            with pytest.raises(ValueError):
                register("dup_test", lambda **c: None)
            register("dup_test", lambda **c: None, overwrite=True)  # ok
        finally:
            unregister("dup_test")


class TestLocalRunner:
    def test_run_echo(self, tmp_path):
        res = LocalRunner().run_in(tmp_path, [sys.executable, "-c", "print('hello')"])
        assert res.returncode == 0
        assert "hello" in res.stdout

    def test_nonzero_returncode(self, tmp_path):
        res = LocalRunner().run_in(tmp_path, [sys.executable, "-c", "import sys; sys.exit(3)"])
        assert res.returncode == 3


# ---------------------------------------------------------------------------
# Tinker engine (real local CPU Tinker)
# ---------------------------------------------------------------------------

@tinker_only
class TestTinkerEngine:
    def _engine(self):
        return get_engine("tinker", bin_dir=_TINKER_BIN,
                          key_file=_TINKER_EXAMPLE / "ammonia.key")

    def _ammonia(self):
        from mdforge.formats import txyz
        return txyz.read_txyz(_TINKER_EXAMPLE / "ammonia.xyz")

    def test_single_point_matches_analyze(self):
        r = self._engine().single_point(self._ammonia(), breakdown=True)
        assert r.energy[0] == pytest.approx(-3.0843, abs=2e-3)
        assert r.intermolecular[0] == pytest.approx(-3.0921, abs=2e-3)
        assert r.components["Atomic Multipoles"][0] == pytest.approx(-3.3592, abs=2e-3)
        assert r.energy_unit == "kcal/mol"

    def test_gradient_matches_testgrad(self):
        r = self._engine().gradient(self._ammonia())
        assert r.gradient.shape == (1, 8, 3)
        assert np.allclose(r.gradient[0, 0], [0.0, -0.0005, -0.0006], atol=1e-3)
        assert r.force_unit == "kcal/mol/Angstrom"

    def test_minimize(self):
        r = self._engine().minimize(self._ammonia(), tol=1.0)
        assert np.isfinite(r.energy[0])
        assert r.extra["minimized"].n_atoms == 8

    def test_batch_single_point(self):
        from mdforge.formats import arc, txyz
        xyz = self._ammonia()
        perturbed = txyz.TinkerXYZ(names=xyz.names, coords=xyz.coords + 0.01,
                                   types=xyz.types, connectivity=xyz.connectivity, box=xyz.box)
        traj = self._engine()._coerce_to_arc([xyz, perturbed])
        assert isinstance(traj, arc.ArcTrajectory) and traj.n_frames == 2
        r = self._engine().batch_single_point(traj)
        assert r.energy.shape == (2,)
        assert r.energy[0] == pytest.approx(-3.0843, abs=2e-3)


# ---------------------------------------------------------------------------
# OpenMM engine (local OpenMM)
# ---------------------------------------------------------------------------

def _two_ion_openmm(periodic: bool = False):
    """Build a 2-ion (+1/-1) OpenMM system + topology for testing."""
    import openmm
    import openmm.app as app
    system = openmm.System()
    system.addParticle(22.99)
    system.addParticle(35.45)
    nb = openmm.NonbondedForce()
    if periodic:
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2))
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(0.9)
    else:
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb.addParticle(1.0, 0.1, 0.0)
    nb.addParticle(-1.0, 0.1, 0.0)
    system.addForce(nb)
    top = app.Topology()
    ch = top.addChain()
    rs = top.addResidue("ION", ch)
    top.addAtom("NA", app.Element.getBySymbol("Na"), rs)
    top.addAtom("CL", app.Element.getBySymbol("Cl"), rs)
    return system, top


@openmm_only
class TestOpenMMEngine:
    def _engine(self, periodic=False):
        from mdforge.engine import OpenMMEngine
        system, top = _two_ion_openmm(periodic=periodic)
        return OpenMMEngine(system=system, topology=top)

    def test_single_point_coulomb(self):
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # 3 Å
        r = self._engine().single_point(coords, gradient=True)
        expected = 138.935456 * (1.0 * -1.0) / 0.3  # kJ/mol
        assert r.energy[0] == pytest.approx(expected, abs=0.5)
        assert r.gradient.shape == (1, 2, 3)
        assert r.energy_unit == "kJ/mol"

    def test_batch_reuses_context_same_result(self):
        eng = self._engine()
        frames = np.array([[[0.0, 0, 0], [3.0, 0, 0]],
                           [[0.0, 0, 0], [4.0, 0, 0]]])
        batch = eng.batch_single_point(frames)
        assert batch.energy.shape == (2,)
        # per-frame singles must match the batched result
        e0 = eng.single_point(frames[0]).energy[0]
        e1 = eng.single_point(frames[1]).energy[0]
        assert batch.energy[0] == pytest.approx(e0)
        assert batch.energy[1] == pytest.approx(e1)
        # closer charges → lower (more negative) energy
        assert batch.energy[0] < batch.energy[1]

    def test_minimize_lowers_energy(self):
        eng = self._engine()
        start = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        e_start = eng.single_point(start).energy[0]
        r = eng.minimize(start, tol=1.0)
        # opposite charges attract → minimized energy is lower
        assert r.energy[0] <= e_start
        assert r.extra["minimized_positions"].shape == (2, 3)

    def test_dynamics_nvt_returns_trajectory(self):
        from mdforge.core.records import Trajectory
        eng = self._engine()
        start = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        traj = eng.dynamics(start, nsteps=200, dt_fs=0.5, ensemble="nvt",
                            temperature=300.0, report_interval=50)
        assert isinstance(traj, Trajectory)
        assert traj.n_frames == 4
        assert traj.positions.shape == (4, 2, 3)
        assert traj.potential_energy.shape == (4,)

    def test_dynamics_npt_adds_barostat(self):
        eng = self._engine(periodic=True)
        start = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        traj = eng.dynamics(start, nsteps=100, dt_fs=1.0, ensemble="npt",
                            temperature=300.0, pressure=1.0, report_interval=50)
        assert traj.n_frames == 2
        assert traj.volume is not None and np.all(traj.volume > 0)


# ---------------------------------------------------------------------------
# cross-engine agreement (both backends)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_HAS_TINKER and _HAS_OPENMM), reason="needs both backends")
class TestCrossEngine:
    def test_point_charge_coulomb_agreement(self, tmp_path):
        """Tinker and OpenMM must agree on 2-ion Coulomb after unit normalization."""
        from mdforge.formats import txyz

        # --- Tinker: fixed-charge prm + 2 unbonded ions ---
        prm = tmp_path / "ions.prm"
        prm.write_text(
            'atom          1    1    NA    "Sodium Ion"        11    22.990    0\n'
            'atom          2    2    CL    "Chloride Ion"      17    35.453    0\n'
            "charge        1           1.0000\n"
            "charge        2          -1.0000\n"
            "vdw           1          3.0000     0.0000\n"
            "vdw           2          4.0000     0.0000\n"
        )
        (tmp_path / "ions.key").write_text(f"parameters {prm}\n")
        ions = txyz.TinkerXYZ(
            names=["NA", "CL"],
            coords=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            types=np.array([1, 2]), connectivity=[[], []],
        )
        tk = get_engine("tinker", bin_dir=_TINKER_BIN, key_file=tmp_path / "ions.key")
        e_tinker = tk.single_point(ions).energy_in("kcal/mol")[0]

        # --- OpenMM: same 2 point charges, 3 Å apart ---
        from mdforge.engine import OpenMMEngine
        system, top = _two_ion_openmm()
        om = OpenMMEngine(system=system, topology=top)
        e_openmm = om.single_point(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])).energy_in("kcal/mol")[0]

        assert e_tinker == pytest.approx(-110.688, abs=1e-2)
        assert e_tinker == pytest.approx(e_openmm, abs=1e-2)  # cross-engine agreement


# ---------------------------------------------------------------------------
# remote execution via SSHRunner (opt-in: set MDFORGE_TEST_REMOTE=1 + host env vars)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("MDFORGE_TEST_REMOTE"),
                    reason="set MDFORGE_TEST_REMOTE=1 (+ MDFORGE_REMOTE_HOST) to run remote tests")
class TestRemoteTinker:
    """End-to-end SSHRunner validation against a remote host.

    Exercises remote CPU Tinker and GPU tinker9 (both should return the
    ammonia-dimer energy, −3.0843 kcal/mol). The host and paths are read from the
    environment; gated behind an env var so CI/other machines skip it.
    """

    def test_remote_cpu_and_gpu_tinker(self):
        from mdforge.engine import SSHRunner
        from mdforge.formats import txyz

        host = os.environ.get("MDFORGE_REMOTE_HOST", "localhost")
        remote_dir = os.environ.get("MDFORGE_REMOTE_DIR", "/tmp/mdforge-remote-tests")
        remote_bin = os.environ.get("MDFORGE_REMOTE_TINKER_BIN", "/usr/local/bin")
        xyz = txyz.read_txyz(_TINKER_EXAMPLE / "ammonia.xyz")
        runner = SSHRunner(host=host, remote_dir=remote_dir)
        for tinker9 in (False, True):
            eng = get_engine("tinker", bin_dir=remote_bin, tinker9=tinker9,
                             key_file=_TINKER_EXAMPLE / "ammonia.key", runner=runner)
            r = eng.single_point(xyz)
            assert r.energy[0] == pytest.approx(-3.0843, abs=5e-2), f"tinker9={tinker9}"
