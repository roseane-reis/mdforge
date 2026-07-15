"""Tests for the packaged reference-data loader."""

from __future__ import annotations

import json

import pytest

from mdforge.liquid.evaluate.reference import (
    available_reference_sets,
    load_experimental_rdf,
    load_reference_set,
    load_skinner_rdf,
)


def test_load_packaged_water():
    ref = load_reference_set("water", 298.15)
    assert ref.liquid == "water"
    d = ref.get("density")
    assert d.exp_value == pytest.approx(0.99705)
    assert d.baseline("tip3p") == pytest.approx(0.980)
    assert d.rated is True


def test_core_keys_all_have_tip3p_baseline():
    ref = load_reference_set("water", 298.15)
    for key in ref.core_keys():
        assert ref.get(key).baseline("tip3p") is not None, key
        assert ref.get(key).rated


def test_structural_keys_unrated_no_baseline():
    ref = load_reference_set("water", 298.15)
    for key in ("tetrahedral_q", "hbonds_per_molecule", "coordination_number"):
        p = ref.get(key)
        assert p.rated is False
        assert p.baseline("tip3p") is None


def test_citations_present():
    ref = load_reference_set("water", 298.15)
    dois = {c.doi for c in ref.citations.values()}
    assert "10.1063/1.4960175" in dois
    assert "10.1021/acs.jctc.1c00628" in dois


def test_available_reference_sets():
    assert ("water", 298.0) in available_reference_sets()


def test_missing_reference_raises():
    with pytest.raises(FileNotFoundError):
        load_reference_set("water", 350.0)


def test_override_path(tmp_path):
    ref = load_reference_set("water", 298.15)
    # round-trip through an out-of-tree JSON with a tweaked value
    raw = {
        "schema_version": 1, "liquid": "water",
        "state_point": {"temperature_K": 298.15},
        "aggregation_defaults": ref.aggregation_defaults,
        "citations": {},
        "properties": {
            "density": {"label": "Density", "unit": "g/cm3", "computed_unit": "g/cm3",
                        "rated": True, "experimental": {"value": 1.234, "source": "x"},
                        "baseline_models": {"tip3p": 1.0}},
        },
    }
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(raw))
    ref2 = load_reference_set(path=p)
    assert ref2.get("density").exp_value == 1.234


def test_experimental_rdf_loads():
    rdf = load_experimental_rdf(298.15, 1.0)
    assert set(rdf) == {"gOO", "gOH", "gHH"}
    assert 2.5 < rdf["gOO"]["peak_r"] < 3.0     # O-O first peak near 2.7-2.8 Å
    assert len(rdf["gOO"]["r"]) == len(rdf["gOO"]["g"])


def test_skinner_rdf_loads():
    sk = load_skinner_rdf(298.15, 1.0)
    assert set(sk) == {"gOO"}                   # X-ray reference is O-O only
    g = sk["gOO"]
    assert g["peak_r"] == pytest.approx(2.80, abs=0.1)   # Skinner first peak ~2.80 Å
    assert g["peak_g"] == pytest.approx(2.57, abs=0.15)  # height ~2.57
    assert len(g["r"]) == len(g["g"])


def test_skinner_rdf_off_state_raises():
    with pytest.raises(FileNotFoundError):
        load_skinner_rdf(350.0, 1.0)


def test_gOO_peak_g_is_report_only():
    from mdforge.liquid.evaluate.score import score_all

    ref = load_reference_set("water", 298.15)
    assert ref.get("gOO_peak_g").report_only is True
    # scored far from experiment + with a baseline present, yet must NOT be graded
    r = score_all({"gOO_peak_g": (3.1, None), "density": (0.997, "g/cm3")},
                  ref, baseline_model="hippo")
    pv = r.per_property["gOO_peak_g"]
    assert pv.verdict == "report"                       # not excellent/good/bad
    assert "gOO_peak_g" not in r.rated_core_keys        # never contributes to the grade
    assert r.counts["report"] == 1
    assert r.counts["bad"] == 0
