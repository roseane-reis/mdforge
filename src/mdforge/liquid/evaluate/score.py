"""Quality scoring: deviation from experiment, judged against the TIP3P bar.

Per-property ladder (the user's rule, verbatim):

1. ``|deviation| <= excellent_tol_pct`` (or within the experimental uncertainty)
   → **excellent**  (this gate overrides good/bad).
2. else ``|deviation| <= |TIP3P deviation|`` → **good** (at least as good as TIP3P).
3. else → **bad** (worse than TIP3P).
4. A property with no TIP3P baseline can still earn **excellent**; otherwise it
   is **unrated** (no defensible bar) — never a fabricated good/bad.

Overall rating = **weighted score**: excellent=2, good=1, bad=0, averaged over
the rated *core* properties, then labelled (grade ≥ 1.5 excellent, ≥ 0.5 good,
else bad). Weights/thresholds/core set come from the reference dataset and are
overridable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from .reference import PropertyReference, ReferenceSet
from .units_liquid import convert

Verdict = Literal["excellent", "good", "bad", "unrated", "report"]


@dataclass(frozen=True)
class PropertyVerdict:
    key: str
    label: str
    model_value: float           # aligned to the reference unit
    unit: str
    model_uncertainty: float | None
    exp_value: float
    exp_uncertainty: float | None
    baseline_model: str | None
    baseline_value: float | None
    dev_model_pct: float         # signed % deviation from experiment
    dev_baseline_pct: float | None
    within_experimental_uncertainty: bool
    verdict: Verdict
    reason: str
    rated: bool


@dataclass(frozen=True)
class ModelRating:
    overall_label: Verdict
    grade: float                 # weighted mean over rated-core properties
    grade_max: float
    grade_pct: float
    rule: str
    counts: dict[str, int]
    rated_core_keys: list[str]
    per_property: dict[str, PropertyVerdict] = field(default_factory=dict)


def signed_pct_deviation(value: float, exp: float) -> float:
    """Signed percentage deviation ``100 (value - exp) / exp`` (nan if exp==0)."""
    if exp == 0:
        return float("nan")
    return 100.0 * (value - exp) / exp


def score_property(
    computed_value: float,
    computed_unit: str | None,
    ref: PropertyReference,
    *,
    baseline_model: str = "tip3p",
    computed_uncertainty: float | None = None,
    excellent_tol_pct: float = 1.0,
    within_uncertainty_is_excellent: bool = True,
) -> PropertyVerdict:
    """Score one computed property against its reference record."""
    v = convert(computed_value, computed_unit, ref.computed_unit, aliases=ref.unit_aliases)
    exp = ref.exp_value

    def make(verdict: Verdict, reason: str, dev_model: float,
             dev_base: float | None, base_val: float | None,
             within_unc: bool) -> PropertyVerdict:
        return PropertyVerdict(
            key=ref.key, label=ref.label, model_value=v, unit=ref.computed_unit,
            model_uncertainty=computed_uncertainty, exp_value=exp,
            exp_uncertainty=ref.exp_uncertainty, baseline_model=baseline_model,
            baseline_value=base_val, dev_model_pct=dev_model,
            dev_baseline_pct=dev_base, within_experimental_uncertainty=within_unc,
            verdict=verdict, reason=reason, rated=ref.rated,
        )

    within_unc = (
        ref.exp_uncertainty is not None and abs(v - exp) <= ref.exp_uncertainty
    )
    # report-only: show the value + deviation, but assign no verdict and never grade
    # (e.g. the g_OO peak height, whose experimental value is too source-dependent).
    if ref.report_only:
        return make("report", "reported only (not graded)",
                    signed_pct_deviation(v, exp), None, None, within_unc)

    if exp == 0:
        return make("unrated", "experimental value is zero", float("nan"), None, None, False)

    dev_model = signed_pct_deviation(v, exp)

    if abs(dev_model) <= excellent_tol_pct:
        return make("excellent", f"|dev|={abs(dev_model):.2f}% ≤ {excellent_tol_pct:g}%",
                    dev_model, None, ref.baseline(baseline_model), within_unc)
    if within_uncertainty_is_excellent and within_unc:
        return make("excellent",
                    f"within experimental uncertainty (±{ref.exp_uncertainty:g})",
                    dev_model, None, ref.baseline(baseline_model), within_unc)

    base_val = ref.baseline(baseline_model)
    if base_val is None:
        return make("unrated",
                    f"no {baseline_model} baseline and |dev|={abs(dev_model):.2f}% "
                    f"> {excellent_tol_pct:g}%",
                    dev_model, None, None, within_unc)

    dev_base = signed_pct_deviation(base_val, exp)
    if abs(dev_model) <= abs(dev_base):
        return make("good",
                    f"|dev|={abs(dev_model):.2f}% ≤ |{baseline_model} dev|="
                    f"{abs(dev_base):.2f}%",
                    dev_model, dev_base, base_val, within_unc)
    return make("bad",
                f"|dev|={abs(dev_model):.2f}% > |{baseline_model} dev|="
                f"{abs(dev_base):.2f}%",
                dev_model, dev_base, base_val, within_unc)


def _weighted_score(
    per_property: dict[str, PropertyVerdict],
    core_keys: list[str],
    weights: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[Verdict, float, float]:
    """Weighted-score aggregation over rated-core properties present in results."""
    scored = [
        weights[v.verdict]
        for k, v in per_property.items()
        if k in core_keys and v.rated and v.verdict in weights
    ]
    grade_max = max(weights.values()) if weights else 0.0
    if not scored:
        return "unrated", float("nan"), grade_max
    grade = sum(scored) / len(scored)
    exc_t = thresholds.get("excellent", 1.5)
    good_t = thresholds.get("good", 0.5)
    label: Verdict = "excellent" if grade >= exc_t else ("good" if grade >= good_t else "bad")
    return label, grade, grade_max


def score_all(
    computed: Mapping[str, tuple[float, str | None]],
    ref: ReferenceSet,
    *,
    baseline_model: str = "tip3p",
    uncertainties: Mapping[str, float] | None = None,
    aggregation: str | Callable[[dict[str, PropertyVerdict]], tuple[str, float]] | None = None,
) -> ModelRating:
    """Score every computed property and aggregate to an overall model rating.

    ``computed`` maps a reference property key to ``(value, unit)``. Keys not in
    the reference set are ignored (with no error) so a pipeline may hand over
    extra diagnostics.
    """
    agg = ref.aggregation_defaults
    tol = float(agg.get("excellent_tolerance_pct", 1.0))
    within_unc = bool(agg.get("within_uncertainty_is_excellent", True))
    uncertainties = uncertainties or {}

    per_property: dict[str, PropertyVerdict] = {}
    for key, (value, unit) in computed.items():
        if key not in ref.properties:
            continue
        per_property[key] = score_property(
            value, unit, ref.get(key), baseline_model=baseline_model,
            computed_uncertainty=uncertainties.get(key),
            excellent_tol_pct=tol, within_uncertainty_is_excellent=within_unc,
        )

    counts = {"excellent": 0, "good": 0, "bad": 0, "unrated": 0, "report": 0}
    for v in per_property.values():
        counts[v.verdict] += 1

    core_keys = ref.core_keys()
    rule = aggregation if isinstance(aggregation, str) else agg.get("rule", "weighted_score")

    if callable(aggregation):
        label, grade = aggregation(per_property)
        grade_max = max((agg.get("weights") or {"excellent": 2.0}).values())
        rule = getattr(aggregation, "__name__", "custom")
    elif rule == "weighted_score":
        weights = {k: float(w) for k, w in (agg.get("weights")
                   or {"excellent": 2.0, "good": 1.0, "bad": 0.0}).items()}
        thresholds = agg.get("label_thresholds", {"excellent": 1.5, "good": 0.5})
        label, grade, grade_max = _weighted_score(per_property, core_keys, weights, thresholds)
    else:
        raise ValueError(f"unknown aggregation rule {rule!r}")

    grade_pct = (100.0 * grade / grade_max) if grade_max and grade == grade else float("nan")
    return ModelRating(
        overall_label=label, grade=grade, grade_max=grade_max, grade_pct=grade_pct,
        rule=rule, counts=counts,
        rated_core_keys=[k for k in core_keys if k in per_property and per_property[k].rated],
        per_property=per_property,
    )


__all__ = [
    "Verdict", "PropertyVerdict", "ModelRating",
    "signed_pct_deviation", "score_property", "score_all",
]
