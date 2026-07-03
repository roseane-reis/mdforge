"""mdforge.fit — force-field parametrization (goal d).

The ``Auxfit`` science refactored onto the :class:`~mdforge.engine.base.Engine`
interface: every per-evaluation energy/gradient/property comes from an engine
(default Tinker), so the fitter is engine-agnostic.

    parameters  ParameterSpace — HIPPO prmdict ↔ flat optimization vector
    targets     DimerInteractionTarget / PolarizabilityTarget / BulkPropertyTarget
                — each yields a residual from an engine evaluation
    fitter      least_squares (soft_l1) / differential_evolution wrappers
    workflow    FitProblem (vector → prmdict → engine → residual) + sequential recipe

Example (one term, dimer SAPT targets, Tinker engine)::

    from mdforge.formats.prm import process_prm
    from mdforge.fit import ParameterSpace, FitProblem, tinker_engine_factory
    space = ParameterSpace(process_prm("mol.prm"), termfit=["dispersion"])
    factory = tinker_engine_factory("/path/to/tinker/bin", "/tmp/fit")
    result = FitProblem(space, [dimer_target], factory).fit(method="least_squares")
"""

from __future__ import annotations

from . import fitter, parameters, targets, workflow
from .fitter import FitResult, differential_evolution_fit, least_squares_fit
from .parameters import ParameterSpace
from .targets import (
    BulkPropertyTarget,
    DimerInteractionTarget,
    PolarizabilityTarget,
    project_components_to_sapt,
    split_dimer,
)
from .workflow import FitProblem, run_sequential_fit, tinker_engine_factory

__all__ = [
    # submodules
    "parameters", "targets", "fitter", "workflow",
    # parameters
    "ParameterSpace",
    # targets
    "DimerInteractionTarget", "PolarizabilityTarget", "BulkPropertyTarget",
    "split_dimer", "project_components_to_sapt",
    # fitter
    "FitResult", "least_squares_fit", "differential_evolution_fit",
    # workflow
    "FitProblem", "tinker_engine_factory", "run_sequential_fit",
]
