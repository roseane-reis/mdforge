"""Tests for the report/export layer (results.json, tables, REPORT.md, plots)."""

from __future__ import annotations

import json

import pytest

from mdforge.liquid.evaluate.pipeline import EvalResult
from mdforge.liquid.evaluate.reference import load_reference_set
from mdforge.liquid.evaluate.report import build_evaluation_report, format_console_summary


def _fake_result():
    """A hand-built EvalResult standing in for a pipeline run (no trajectories)."""
    return EvalResult(
        meta={"model": "FakeWater", "temperature_K": 298.15, "pressure_atm": 1.0},
        thermo={"npt": {"ensemble": "NPT"}},
        structure={"nvt": {"ensemble": "NVT", "r": [1, 2, 3],
                           "g_OO": [0, 1, 2], "g_OH": [0, 2, 1], "g_HH": [0, 1, 1],
                           "gOO_peak_r": 2.06, "gOO_peak_g": 10.0}},
        rdf_exp={"gOO": {"r": [1, 2, 3], "g": [0, 1, 2], "peak_r": 2.73, "peak_g": 2.75},
                 "gOH": {"r": [1, 2, 3], "g": [0, 2, 1], "peak_r": 1.0, "peak_g": 12.7},
                 "gHH": {"r": [1, 2, 3], "g": [0, 1, 1], "peak_r": 1.5, "peak_g": 1.7}},
        scoring_inputs={
            "density": (1.178, "g/cm3"),
            "self_diffusion": (0.28, "1e-5 cm2/s"),
            "tetrahedral_q": (0.51, "dimensionless"),
        },
        scoring_uncertainties={"density": 0.0017},
        scoring_sources={"density": "npt", "self_diffusion": "nvt",
                         "tetrahedral_q": "nvt", "gOO_peak_r": "nvt"},
        warnings=["structure scored at the model's own density"],
    )


def test_build_report_return_keys(tmp_path):
    out = build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=False)
    for key in ("rating", "rows", "reference", "results_json", "properties_csv",
                "properties_md", "report_md"):
        assert key in out


def test_console_summary_shows_values_before_evaluation():
    out = build_evaluation_report(_fake_result(), outdir=None)
    text = format_console_summary(_fake_result(), out["rating"], out["reference"])
    # the computed-properties block must come before the evaluation block
    assert "computed properties" in text
    i_props = text.index("computed properties")
    i_eval = text.index("evaluation (quality bar")
    assert i_props < i_eval
    # a raw model number and its verdict both appear
    assert "1.178" in text                     # density model value
    assert "Density" in text and "bad" in text
    assert "Overall:" in text


def test_results_json_has_evaluation_block(tmp_path):
    build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=False)
    blob = json.loads((tmp_path / "results.json").read_text())
    assert "evaluation" in blob
    ev = blob["evaluation"]
    assert ev["overall_label"] in ("excellent", "good", "bad", "unrated")
    assert ev["aggregation_rule"] == "weighted_score"
    assert "density" in ev["per_property"]
    assert ev["per_property"]["density"]["verdict"] == "bad"
    # pipeline blocks preserved alongside the new evaluation block
    assert "structure" in blob and "thermo" in blob


def test_properties_csv_has_verdict_columns(tmp_path):
    build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=False)
    header = (tmp_path / "properties_table.csv").read_text().splitlines()[0]
    for col in ("tip3p_threshold", "tip3p_dev_pct", "verdict"):
        assert col in header


def test_report_md_has_banner_and_dois(tmp_path):
    build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=False)
    text = (tmp_path / "REPORT.md").read_text()
    assert "Overall rating:" in text
    assert "10.1063/1.4960175" in text          # Izadi & Onufriev DOI
    assert "TIP3P" in text


def test_no_pdf_produced(tmp_path):
    out = build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=False)
    assert "report_pdf" not in out
    assert not list(tmp_path.glob("*.pdf"))


def test_no_outdir_returns_rating_only():
    out = build_evaluation_report(_fake_result(), outdir=None)
    assert "rating" in out and "results_json" not in out
    assert out["rating"].per_property["density"].verdict == "bad"


def test_plots_created_when_requested(tmp_path):
    pytest.importorskip("matplotlib")
    out = build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=True,
                                  reference=load_reference_set("water", 298.15))
    assert "figures" in out
    assert "rdf_partials" in out["figures"]


def test_timeseries_plot_emitted_per_leg(tmp_path):
    pytest.importorskip("matplotlib")
    result = _fake_result()
    result.series = {"npt": {
        "ensemble": "NPT", "n_frames": 5, "equil": 1, "dt_ps": 5.0,
        "columns": {
            "density (g/cm³)": [1.0, 1.01, 0.99, 1.0, 1.0],
            "temperature (K)": [298, 299, 297, 298, 298],
        },
    }}
    out = build_evaluation_report(result, outdir=tmp_path, make_plots=True,
                                  reference=load_reference_set("water", 298.15))
    assert "timeseries_npt" in out["figures"]
    assert (tmp_path / "timeseries_npt.png").is_file()


def test_no_timeseries_plot_without_series(tmp_path):
    pytest.importorskip("matplotlib")
    out = build_evaluation_report(_fake_result(), outdir=tmp_path, make_plots=True,
                                  reference=load_reference_set("water", 298.15))
    assert not any(k.startswith("timeseries_") for k in out["figures"])
    assert not list(tmp_path.glob("timeseries_*.png"))
