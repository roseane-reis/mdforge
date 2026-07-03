"""Tests for the quality-scoring ladder, unit alignment, and aggregation."""

from __future__ import annotations

import pytest

from mdforge.liquid.evaluate.reference import load_reference_set
from mdforge.liquid.evaluate.score import score_all, score_property
from mdforge.liquid.evaluate.units_liquid import UnitError, convert


@pytest.fixture
def ref():
    return load_reference_set("water", 298.15)


# --- ladder (density: exp 0.99705, TIP3P 0.980 -> |TIP3P dev| = 1.71%) -------

def test_excellent_within_1pct(ref):
    v = score_property(1.000, "g/cm3", ref.get("density"))
    assert v.verdict == "excellent"


def test_excellent_overrides_good(ref):
    # 0.998 is 0.19% off AND better than TIP3P — excellent gate wins.
    v = score_property(0.998, "g/cm3", ref.get("density"))
    assert v.verdict == "excellent"


def test_good_between_1pct_and_tip3p(ref):
    # 0.982 -> ~1.51% off, still <= TIP3P's 1.71% -> good.
    v = score_property(0.982, "g/cm3", ref.get("density"))
    assert v.verdict == "good"


def test_bad_worse_than_tip3p(ref):
    v = score_property(1.178, "g/cm3", ref.get("density"))
    assert v.verdict == "bad"
    assert v.dev_baseline_pct == pytest.approx(-1.7100, abs=1e-3)


def test_boundary_equal_to_tip3p_is_good(ref):
    # Mirror TIP3P's deviation exactly on the other side -> |dev| == |tip3p dev| -> good.
    d = ref.get("density")
    tip3p_dev = abs(d.baseline("tip3p") - d.exp_value)
    v = score_property(d.exp_value + tip3p_dev, "g/cm3", d)
    assert v.verdict == "good"


def test_signed_deviation_direction(ref):
    over = score_property(1.178, "g/cm3", ref.get("density"))
    under = score_property(0.80, "g/cm3", ref.get("density"))
    assert over.dev_model_pct > 0 and under.dev_model_pct < 0


def test_within_uncertainty_is_excellent(ref):
    # cp exp 18.0 ± 0.05; 18.04 is >1% off? no (0.22%); use a value >1% but in-band.
    # kappa_T exp 45.25 ± 1.0; 45.9 is 1.44% off but within ±1.0 -> excellent when flag on.
    on = score_property(45.9, "1e-6/bar", ref.get("kappa_T"),
                        within_uncertainty_is_excellent=True)
    off = score_property(45.9, "1e-6/bar", ref.get("kappa_T"),
                         within_uncertainty_is_excellent=False)
    assert on.verdict == "excellent"
    assert off.verdict == "good"        # still no worse than TIP3P


# --- no-baseline structural metrics ------------------------------------------

def test_no_baseline_excellent(ref):
    v = score_property(0.578, "dimensionless", ref.get("tetrahedral_q"))
    assert v.verdict == "excellent"


def test_no_baseline_unrated(ref):
    v = score_property(0.45, "dimensionless", ref.get("tetrahedral_q"))
    assert v.verdict == "unrated"
    assert "no tip3p baseline" in v.reason


# --- unit alignment ----------------------------------------------------------

def test_density_kg_m3_aligned(ref):
    v = score_property(997.05, "kg/m3", ref.get("density"))
    assert v.model_value == pytest.approx(0.99705, abs=1e-4)
    assert v.verdict == "excellent"


def test_kappa_unit_aliases_equivalent():
    # 1e-6/bar == 1e-11/Pa; 1/Pa scales by 1e11.
    assert convert(45.25, "1e-11/Pa", "1e-6/bar") == pytest.approx(45.25)
    assert convert(4.525e-10, "1/Pa", "1e-6/bar") == pytest.approx(45.25, rel=1e-9)


def test_unknown_unit_raises():
    with pytest.raises(UnitError):
        convert(1.0, "furlongs", "g/cm3")


def test_diffusion_identity_1e5():
    assert convert(2.3, "1e-5 cm2/s", "1e-5 cm2/s") == pytest.approx(2.3)
    assert convert(2.3e-5, "cm2/s", "1e-5 cm2/s") == pytest.approx(2.3)


# --- aggregation (weighted score) --------------------------------------------

def test_weighted_score_all_excellent(ref):
    computed = {k: (ref.get(k).exp_value, ref.get(k).computed_unit) for k in ref.core_keys()}
    r = score_all(computed, ref)
    assert r.overall_label == "excellent"
    assert r.grade == pytest.approx(2.0)
    assert r.grade_pct == pytest.approx(100.0)


def test_weighted_score_mixed_is_between(ref):
    core = ref.core_keys()
    computed = {}
    for i, k in enumerate(core):
        p = ref.get(k)
        computed[k] = (p.exp_value if i % 2 == 0 else p.baseline("tip3p") * 5,
                       p.computed_unit)
    r = score_all(computed, ref)
    assert 0.0 <= r.grade <= 2.0
    assert r.counts["excellent"] >= 1 and r.counts["bad"] >= 1


def test_unrated_excluded_from_grade(ref):
    # Only an unrated structural metric present -> no rated-core -> grade is nan.
    r = score_all({"tetrahedral_q": (0.45, "dimensionless")}, ref)
    assert r.overall_label == "unrated"
    assert r.counts["unrated"] == 1


def test_counts_total_matches(ref):
    computed = {k: (ref.get(k).exp_value, ref.get(k).computed_unit)
                for k in list(ref.properties)}
    r = score_all(computed, ref)
    assert sum(r.counts.values()) == len(ref.properties)
