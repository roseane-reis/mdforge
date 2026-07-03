"""Water-model evaluation: ingest a trajectory, compute liquid properties, and
judge quality against experiment with the TIP3P model as the good/bad bar.

Config-driven and model/engine-agnostic. High-level use::

    from mdforge.liquid.evaluate import EvalConfig, run_evaluation, build_evaluation_report

    config = EvalConfig.from_yaml("water.yaml")
    result = run_evaluation(config)
    artifacts = build_evaluation_report(result, outdir="analysis")
    print(artifacts["rating"].overall_label)

or from the command line::

    python -m mdforge.liquid.evaluate --config water.yaml
    python -m mdforge.liquid.evaluate --campaign /path/to/run_dir

The ladder: within 1 % of experiment (or its uncertainty) → *excellent*; else no
worse than TIP3P's deviation → *good*; else *bad*. Structural metrics without a
TIP3P baseline are *unrated*. The overall rating is a weighted score over the
rated core properties.
"""

from __future__ import annotations

from .config import (
    EvalConfig,
    EvalConfigError,
    EvalStateError,
    LegSpec,
    state_guard,
)
from .pipeline import EvalResult, run_evaluation
from .reference import (
    PropertyReference,
    ReferenceSet,
    available_reference_sets,
    load_experimental_rdf,
    load_reference_set,
)
from .report import build_evaluation_report
from .score import ModelRating, PropertyVerdict, score_all, score_property

__all__ = [
    # config
    "EvalConfig", "LegSpec", "EvalConfigError", "EvalStateError", "state_guard",
    # pipeline
    "EvalResult", "run_evaluation",
    # reference
    "ReferenceSet", "PropertyReference", "load_reference_set",
    "available_reference_sets", "load_experimental_rdf",
    # scoring
    "PropertyVerdict", "ModelRating", "score_property", "score_all",
    # report
    "build_evaluation_report",
]
