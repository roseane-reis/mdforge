"""Tests for mdforge.fit — parameters, fitter, targets, workflow.

Pure machinery (param↔vector, optimizer wrappers) and mock-engine target/workflow
tests always run. A real-Tinker self-consistent polarizability fit runs when local
Tinker + the HIPPO water21 params are available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mdforge.engine.base import Capabilities, EngineResult
from mdforge.fit import (
    BulkPropertyTarget,  # noqa: F401  (exported surface)
    DimerInteractionTarget,
    FitProblem,
    ParameterSpace,
    PolarizabilityTarget,
    differential_evolution_fit,
    least_squares_fit,
    project_components_to_sapt,
    run_sequential_fit,
    split_dimer,
    tinker_engine_factory,
)
from mdforge.formats.arc import ArcTrajectory
from mdforge.formats.txyz import TinkerXYZ

_TINKER_BIN = Path(os.environ.get("MDFORGE_TINKER_BIN", "/opt/tinker/bin"))
_WATER21 = Path(os.environ.get("MDFORGE_TINKER_PARAMS", "/opt/tinker/params")) / "water21.prm"
_HIPPOSMALL = Path(os.environ.get("MDFORGE_TINKER_EXAMPLE", "/opt/tinker/example")) / "hipposmall.xyz"
_PRMDIR = Path(os.environ.get("MDFORGE_REFDATA", "/opt/mdforge/reference-data")) / "prmfiles"
_HAS_TINKER_FIT = (_TINKER_BIN / "polarize").exists() and _WATER21.is_file() and _HIPPOSMALL.is_file()
tinker_fit_only = pytest.mark.skipif(not _HAS_TINKER_FIT, reason="Tinker + water21 params unavailable")


def _synthetic_prmdict() -> dict:
    return {
        "types": [401, 402],
        "chgpen": np.array([[4.0, 3.5], [1.0, 2.0]]),
        "dispersion": np.array([8.0, 2.0]),
        "repulsion": np.array([[5.0, 4.0, 3.0], [1.0, 2.0, 1.5]]),
        "polarize": [[0.8, 0.4], [[402], [401]]],
        "chgtrn": np.array([[3.0, 4.0], [0.0, 0.0]]),  # second row zeros → skipped
        "bond": [[["401", "402"]], [500.0], [0.957]],
        "angle": [[[402, 401, 402]], [50.0], [104.5], [""]],
        "multipole": [[], []],
    }


# ---------------------------------------------------------------------------
# ParameterSpace
# ---------------------------------------------------------------------------

class TestParameterSpace:
    def test_to_vector_terms(self):
        ps = ParameterSpace(_synthetic_prmdict(), ["chgpen", "dispersion", "repulsion", "polarize"])
        # chgpen[:,1]=[3.5,2.0], dispersion=[8,2], repulsion flat=[5,4,3,1,2,1.5], polarize=[0.8,0.4]
        assert np.allclose(ps.to_vector(), [3.5, 2.0, 8.0, 2.0, 5.0, 4.0, 3.0, 1.0, 2.0, 1.5, 0.8, 0.4])

    def test_roundtrip(self):
        d = _synthetic_prmdict()
        ps = ParameterSpace(d, ["chgpen", "dispersion", "repulsion", "polarize", "chgtrn"])
        d2 = ps.from_vector(ps.to_vector())
        assert np.allclose(d2["chgpen"], d["chgpen"])
        assert np.allclose(d2["dispersion"], d["dispersion"])
        assert np.allclose(d2["repulsion"], d["repulsion"])
        assert np.allclose(d2["polarize"][0], d["polarize"][0])
        assert np.allclose(d2["chgtrn"], d["chgtrn"])

    def test_from_vector_applies(self):
        ps = ParameterSpace(_synthetic_prmdict(), ["dispersion"])
        d2 = ps.from_vector([9.0, 5.0])
        assert np.allclose(d2["dispersion"], [9.0, 5.0])

    def test_chgtrn_skips_zeros(self):
        ps = ParameterSpace(_synthetic_prmdict(), ["chgtrn"])
        assert np.allclose(ps.to_vector(), [3.0, 4.0])      # only the nonzero row
        d2 = ps.from_vector([3.1, 4.1])
        assert np.allclose(d2["chgtrn"], [[3.1, 4.1], [0.0, 0.0]])  # zeros preserved

    def test_bounds(self):
        ps = ParameterSpace(_synthetic_prmdict(), ["dispersion"])
        lo, hi = ps.bounds(relative=0.3)
        assert np.all(lo < ps.to_vector()) and np.all(hi > ps.to_vector())

    def test_unsupported_term_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ParameterSpace(_synthetic_prmdict(), ["multipole"])

    @pytest.mark.skipif(not _PRMDIR.is_dir(), reason="reference prm files unavailable")
    def test_roundtrip_real_prm(self):
        from mdforge.formats.prm import process_prm
        prmdir = _PRMDIR
        src = sorted(prmdir.glob("*.prm"))[0]
        d = process_prm(src)
        terms = [t for t in ["chgpen", "dispersion", "repulsion", "polarize", "chgtrn"]
                 if np.sum(np.abs(d[t] if t != "polarize" else d["polarize"][0])) > 0]
        ps = ParameterSpace(d, terms)
        d2 = ps.from_vector(ps.to_vector())
        for t in terms:
            if t == "polarize":
                assert np.allclose(d2["polarize"][0], d["polarize"][0]), src.name
            else:
                assert np.allclose(d2[t], d[t]), (src.name, t)


# ---------------------------------------------------------------------------
# fitter
# ---------------------------------------------------------------------------

class TestFitter:
    def test_least_squares_recovers_minimum(self):
        target = np.array([3.0, -1.0])
        res = least_squares_fit(lambda x: x - target, [0.0, 0.0])
        assert res.success and np.allclose(res.x, target, atol=1e-6)

    def test_least_squares_with_bounds(self):
        target = np.array([3.0])
        res = least_squares_fit(lambda x: x - target, [1.0], bounds=([0.0], [5.0]))
        assert np.allclose(res.x, target, atol=1e-5)

    def test_differential_evolution(self):
        res = differential_evolution_fit(lambda x: x - np.array([1.5]), [(-5.0, 5.0)],
                                         seed=0, tol=1e-7)
        assert abs(res.x[0] - 1.5) < 1e-2


# ---------------------------------------------------------------------------
# targets (mock engine)
# ---------------------------------------------------------------------------

class MockEngine:
    capabilities = Capabilities(single_point=True, components=True, batched=True, polarizability=True)

    def __init__(self, eig=None):
        self.eig = eig

    def batch_single_point(self, structure, *, breakdown=False, **kw):
        n = structure.n_frames
        title = getattr(structure, "title", "") or ""
        if "mol1" in title:
            comp = {"Atomic Multipoles": np.full(n, 2.0), "Repulsion": np.full(n, 1.0),
                    "Dispersion": np.full(n, -0.5), "Polarization": np.full(n, -0.2),
                    "Charge Transfer": np.zeros(n)}
        elif "mol2" in title:
            comp = {"Atomic Multipoles": np.full(n, 3.0), "Repulsion": np.full(n, 1.0),
                    "Dispersion": np.full(n, -0.5), "Polarization": np.full(n, -0.3),
                    "Charge Transfer": np.zeros(n)}
        else:
            comp = {"Atomic Multipoles": np.full(n, 10.0), "Repulsion": np.full(n, 6.0),
                    "Dispersion": np.full(n, -2.0), "Polarization": np.full(n, -1.0),
                    "Charge Transfer": np.full(n, -0.5)}
        return EngineResult(energy=np.zeros(n), energy_unit="kcal/mol", components=comp)

    def polarizability(self, structure, **kw):
        return EngineResult(energy=np.array([np.nan]), energy_unit="kcal/mol",
                            extra={"polarizability_eigenvalues": self.eig})


def _dimer() -> ArcTrajectory:
    return ArcTrajectory(coords=np.zeros((2, 4, 3)), names=["O", "H", "C", "H"],
                         types=np.array([1, 2, 3, 4]), connectivity=[[2], [1], [4], [3]],
                         title="testdimer")


class TestTargets:
    def test_split_dimer(self):
        mon1, mon2 = split_dimer(_dimer(), 2)
        assert mon1.n_atoms == 2 and mon2.n_atoms == 2
        assert mon1.coords.shape == (2, 2, 3)
        assert "mol1" in mon1.title and "mol2" in mon2.title
        assert mon2.connectivity == [[2], [1]]  # renumbered from [4],[3]

    def test_project_components_to_sapt(self):
        diff = {"Atomic Multipoles": np.array([5.0]), "Repulsion": np.array([4.0]),
                "Polarization": np.array([-0.5]), "Charge Transfer": np.array([-0.5]),
                "Dispersion": np.array([-1.0])}
        out = project_components_to_sapt(diff, 1)
        assert np.allclose(out[0], [5.0, 4.0, -1.0, -1.0, 7.0])  # es,exch,ind,disp,total

    def test_dimer_interaction_model_and_residual(self):
        target = DimerInteractionTarget("t", _dimer(), n_atoms_mol1=2,
                                        qm_components=np.array([[5.0, 4.0, -1.0, -1.0, 7.0],
                                                                [5.0, 4.0, -1.0, -1.0, 7.0]]))
        model = target.model_components(MockEngine())
        assert np.allclose(model[0], [5.0, 4.0, -1.0, -1.0, 7.0])  # 10-2-3, 6-1-1, ...
        assert np.allclose(target.residual(MockEngine()), 0.0)     # matches reference

    def test_dimer_interaction_residual_offset(self):
        qm = np.array([[4.0, 4.0, -1.0, -1.0, 7.0], [4.0, 4.0, -1.0, -1.0, 7.0]])
        t = DimerInteractionTarget("t", _dimer(), 2, qm, components=(0,))  # electrostatics only
        assert np.allclose(t.residual(MockEngine()), [1.0, 1.0])    # model es 5 vs qm 4

    def test_polarizability_target(self):
        eng = MockEngine(eig=np.array([3.5, 3.9, 4.1]))
        t = PolarizabilityTarget(TinkerXYZ(names=["N"], coords=np.zeros((1, 3)), types=np.array([1])),
                                 reference_eigenvalues=np.array([3.5, 3.9, 4.1]))
        assert np.allclose(t.residual(eng), 0.0)
        t2 = PolarizabilityTarget(t.structure, np.array([3.0, 3.9, 4.1]))
        assert np.allclose(np.sort(t2.residual(eng)), np.sort([0.5, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# workflow (mock engine analytic recovery)
# ---------------------------------------------------------------------------

class _HoldEngine:
    capabilities = Capabilities()

    def __init__(self, prmdict):
        self.prmdict = prmdict


class _TermTarget:
    """Residual = (current term values) − target, reading the held prmdict."""

    def __init__(self, getter, target):
        self.getter = getter
        self.target = np.asarray(target, dtype=float)

    def residual(self, engine):
        return np.asarray(self.getter(engine.prmdict), dtype=float) - self.target


class TestWorkflow:
    def test_fit_problem_recovers_params(self):
        ps = ParameterSpace(_synthetic_prmdict(), ["dispersion"])
        target = _TermTarget(lambda d: d["dispersion"], [6.0, 3.0])
        problem = FitProblem(ps, [target], lambda d: _HoldEngine(d))
        res = problem.fit(method="least_squares", bounds=([0.1, 0.1], [80.0, 80.0]))
        assert np.allclose(res.x, [6.0, 3.0], atol=1e-4)

    def test_sequential_fit(self):
        # targets chosen within the default ±30% relative bounds of each term's
        # initial values (dispersion init [8,2]; chgpen[:,1] init [3.5,2.0]).
        def target_factory(prmdict, termfit):
            if "dispersion" in termfit:
                return [_TermTarget(lambda d: d["dispersion"], [7.0, 2.4])]
            return [_TermTarget(lambda d: d["chgpen"][:, 1], [3.0, 2.4])]

        final, results = run_sequential_fit(
            _synthetic_prmdict(), [["dispersion"], ["chgpen"]],
            target_factory, lambda d: _HoldEngine(d), method="least_squares",
        )
        assert len(results) == 2
        assert np.allclose(final["dispersion"], [7.0, 2.4], atol=1e-3)
        assert np.allclose(final["chgpen"][:, 1], [3.0, 2.4], atol=1e-3)


# ---------------------------------------------------------------------------
# real Tinker: self-consistent polarizability fit (end-to-end through the engine)
# ---------------------------------------------------------------------------

@tinker_fit_only
class TestRealTinkerFit:
    def test_polarizability_self_consistent_recovery(self, tmp_path):
        from mdforge.formats import txyz
        from mdforge.formats.prm import process_prm

        box = txyz.read_txyz(_HIPPOSMALL)                     # 216 HIPPO waters
        mono = txyz.TinkerXYZ(names=box.names[:3], coords=box.coords[:3],
                              types=box.types[:3], connectivity=[[2, 3], [1], [1]])
        prmdict0 = process_prm(_WATER21)
        factory = tinker_engine_factory(_TINKER_BIN, tmp_path)

        ps0 = ParameterSpace(prmdict0, ["polarize"])
        x_true = ps0.to_vector()
        ref_eig = factory(ps0.from_vector(x_true)).polarizability(mono).extra["polarizability_eigenvalues"]
        assert ref_eig is not None and len(ref_eig) == 3

        # start from a perturbed guess; fit must recover the reference eigenvalues
        ps = ParameterSpace(ps0.from_vector(x_true * 1.3), ["polarize"])
        problem = FitProblem(ps, [PolarizabilityTarget(mono, ref_eig)], factory)
        res = problem.fit(method="least_squares",
                          bounds=(x_true * 0.3, x_true * 3.0), max_nfev=40)

        fit_eig = np.sort(factory(ps.from_vector(res.x)).polarizability(mono).extra["polarizability_eigenvalues"])
        assert np.allclose(fit_eig, np.sort(ref_eig), atol=0.05)
