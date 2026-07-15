"""Render an evaluation into results.json, tables, REPORT.md, and plots.

Mirrors the return-dict style of :func:`mdforge.qm.report.compare_records`:
scoring/table building need only numpy + stdlib; plots are produced only when an
``outdir`` is given and matplotlib is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...qm.report import save_metrics_csv
from ..plots import plot_running_average
from .pipeline import EvalResult
from .reference import ReferenceSet, load_reference_set
from .score import ModelRating, score_all


def _fmt(x: Any, sig: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    if isinstance(x, str):
        return x
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.3e}"
    return f"{x:.{sig}g}"


def _pct(x: float | None) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else f"{x:+.1f}%"


def _build_rows(result: EvalResult, ref: ReferenceSet, rating: ModelRating) -> list[dict]:
    """One row per reference property, in reference order."""
    rows: list[dict] = []
    for key, pref in ref.properties.items():
        pv = rating.per_property.get(key)
        model_val = pv.model_value if pv else result.scoring_inputs.get(key, (None,))[0]
        rows.append({
            "property": pref.label,
            "unit": pref.unit,
            "model": _fmt(model_val),
            "uncertainty": _fmt(result.scoring_uncertainties.get(key)),
            "experimental": _fmt(pref.exp_value),
            "tip3p_threshold": _fmt(pref.baseline("tip3p")),
            "hippo": _fmt(pref.baselines.get("hippo")),
            "dev_vs_exp_pct": _pct(pv.dev_model_pct if pv else None),
            "tip3p_dev_pct": _pct(pv.dev_baseline_pct if pv else None),
            "verdict": pv.verdict if pv else ("n/a" if model_val is None else "unrated"),
            "note": pref.note or "",
        })
    return rows


def _primary_structure(result: EvalResult) -> dict:
    """The structure block that fed scoring (the NVT leg when present)."""
    name = (result.scoring_sources.get("gOO_peak_r")
            or result.scoring_sources.get("tetrahedral_q"))
    if name and name in result.structure:
        return result.structure[name]
    return next(iter(result.structure.values()), {})


def _banner(rating: ModelRating) -> str:
    c = rating.counts
    grade = "" if rating.grade != rating.grade else f"{rating.grade:.2f}/{rating.grade_max:.0f}"
    pct = "" if rating.grade_pct != rating.grade_pct else f" ({rating.grade_pct:.0f}%)"
    return (f"**{rating.overall_label.upper()}** — grade {grade}{pct} "
            f"[rule: {rating.rule}] · "
            f"E:{c['excellent']} G:{c['good']} B:{c['bad']} unrated:{c['unrated']}")


def _report_markdown(result: EvalResult, ref: ReferenceSet, rating: ModelRating,
                     rows: list[dict]) -> str:
    meta = result.meta
    L = [
        f"# {meta.get('model', 'model')} — Water Model Evaluation "
        f"({meta.get('temperature_K')} K, {meta.get('pressure_atm')} atm)",
        "",
        f"> Overall rating: {_banner(rating)}",
        "",
        "## Method",
        "",
        "- **Bar = TIP3P.** A property is **excellent** if within "
        f"{ref.aggregation_defaults.get('excellent_tolerance_pct', 1.0):g}% of experiment "
        "(or within the experimental uncertainty); else **good** if its deviation is no "
        "worse than TIP3P's; else **bad**. Overall rating is a weighted score "
        "(excellent=2, good=1, bad=0) over the rated core properties.",
        "- Structural metrics without a TIP3P baseline are **unrated** (still shown; "
        "excluded from the headline).",
        f"- Reference state {ref.state_point.get('temperature_K')} K / "
        f"{ref.state_point.get('pressure_atm', '1')} atm. Self-diffusion uses the "
        "Yeh–Hummer finite-size-corrected value; ΔHvap is classical (no NQE).",
        "",
        "## Per-property verdicts",
        "",
        "| Property | Unit | Model | ±Unc | Exp | TIP3P | Dev vs exp | TIP3P dev | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append("| {property} | {unit} | {model} | {uncertainty} | {experimental} | "
                 "{tip3p_threshold} | {dev_vs_exp_pct} | {tip3p_dev_pct} | {verdict} |".format(**r))

    # RDF comparison
    rr = result.rdf_exp.get("gOO", {})
    struct = _primary_structure(result)
    if rr and struct:
        L += ["", "## O–O radial distribution function", "",
              f"- Model g_OO first peak: {_fmt(struct.get('gOO_peak_r'))} Å, "
              f"height {_fmt(struct.get('gOO_peak_g'))} "
              f"(from the {struct.get('ensemble')} leg).",
              f"- Experimental (Soper 2013): {_fmt(rr.get('peak_r'))} Å, "
              f"height {_fmt(rr.get('peak_g'))}."]

    # caveats from warnings + per-property correction notes
    caveats = list(result.warnings)
    for key, pref in ref.properties.items():
        if pref.correction and key in rating.per_property:
            caveats.append(f"{pref.label}: {pref.correction}")
    if caveats:
        L += ["", "## Notes & caveats", ""] + [f"- {c}" for c in caveats]

    # references
    L += ["", "## References", ""]
    for cit in ref.citations.values():
        doi = f" DOI: {cit.doi}" if cit.doi else ""
        L.append(f"- {cit.text}{doi}")
    L.append("")
    return "\n".join(L)


def _make_plots(result: EvalResult, outdir: Path) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    figs: dict[str, str] = {}
    struct = _primary_structure(result) or None
    # partial RDFs: (model key, experimental key, subscript, title). All panels
    # share a fixed 0–3 y-range so the inter-molecular structure is comparable
    # across partials (model and Soper 2013 are both inter-molecular only).
    partials = [("g_OO", "gOO", "OO", "O–O"),
                ("g_OH", "gOH", "OH", "O–H"),
                ("g_HH", "gHH", "HH", "H–H")]
    if struct and any(mkey in struct for mkey, *_ in partials):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        for ax, (mkey, ekey, sub, title) in zip(axes, partials):
            if mkey in struct:
                ax.plot(struct["r"], struct[mkey], label=f"model ({struct['ensemble']})")
            rr = result.rdf_exp.get(ekey)
            if rr:
                ax.plot(rr["r"], rr["g"], "k--", label="experiment (Soper 2013)")
            # second, independent O-O reference (X-ray) — dotted brown
            if ekey == "gOO":
                sk = getattr(result, "rdf_exp_skinner", {}).get("gOO")
                if sk:
                    ax.plot(sk["r"], sk["g"], color="brown", linestyle=":", linewidth=1.8,
                            label="experiment (Skinner 2014, X-ray)")
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 3)
            ax.set_xlabel("r (Å)")
            ax.set_ylabel(f"g$_{{{sub}}}$(r)")
            ax.set_title(f"{title} radial distribution function")
        # legend on the middle (O–H) panel, top-right; take handles from the O–O
        # panel so the O-O-only Skinner curve is included in the legend.
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(handles, labels, loc="upper right")
        fig.tight_layout()
        p = outdir / "rdf_partials.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["rdf_partials"] = str(p)

    # optional per-leg thermodynamic timeseries (present only when the run was
    # invoked with record_timeseries / output.timeseries)
    for leg_name, s in getattr(result, "series", {}).items():
        cols = s.get("columns") or {}
        if not cols:
            continue
        n_panels = len(cols)
        ncols = 2 if n_panels > 1 else 1
        nrows = -(-n_panels // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 2.6 * nrows),
                                 squeeze=False)
        axes_list = list(axes.flat)
        for ax, (label, series) in zip(axes_list, cols.items()):
            # per-panel legend suppressed; one shared legend is drawn below
            plot_running_average(series, dt_ps=s.get("dt_ps", 1.0),
                                 equil=s.get("equil", 0), label=label, legend=False,
                                 t=s.get("t_ps"), ax=ax)
        for ax in axes_list[n_panels:]:
            ax.axis("off")
        # single shared legend (raw / running avg [/ equil]) from the first panel's lines
        handles, labels = axes_list[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.suptitle(f"{leg_name} ({s.get('ensemble', '')}) — thermodynamic timeseries")
        p = outdir / f"timeseries_{leg_name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs[f"timeseries_{leg_name}"] = str(p)
    return figs


def format_console_summary(result: EvalResult, rating: ModelRating,
                           reference: ReferenceSet, *, baseline_model: str = "tip3p") -> str:
    """Human-readable console summary: computed values, then the verdicts, then the grade.

    The **properties** block (the raw numbers vs experiment) comes first so the
    reader sees what the model produced before the evaluation. Properties are
    listed in the reference dataset's order.
    """
    meta = result.meta
    keys = [k for k in reference.properties if k in rating.per_property]

    title = (f"{meta.get('model', 'model')} — computed properties "
             f"({meta.get('temperature_K')} K, {meta.get('pressure_atm')} atm)")
    lines = ["=" * len(title), title, "=" * len(title),
             f"  {'property':36s} {'model':>11s} {'unit':14s} {'experiment':>11s}  source"]
    for k in keys:
        pv = rating.per_property[k]
        src = result.scoring_sources.get(k, "")
        lines.append(f"  {pv.label:36s} {_fmt(pv.model_value):>11s} {pv.unit:14s} "
                     f"{_fmt(pv.exp_value):>11s}  {src}")

    lines += ["", f"--- evaluation (quality bar: {baseline_model.upper()}) ---"]
    for k in keys:
        pv = rating.per_property[k]
        lines.append(f"  {pv.label:36s} {pv.verdict:9s} dev={pv.dev_model_pct:+7.1f}%  {pv.reason}")

    grade = "" if rating.grade != rating.grade else f"{rating.grade:.2f}/{rating.grade_max:.0f}"
    pct = "" if rating.grade_pct != rating.grade_pct else f" ({rating.grade_pct:.0f}%)"
    c = rating.counts
    lines += ["", (f"Overall: {rating.overall_label.upper()}  grade {grade}{pct}  "
                   f"[E:{c['excellent']} G:{c['good']} B:{c['bad']} unrated:{c['unrated']}]")]
    return "\n".join(lines)


def build_evaluation_report(
    result: EvalResult,
    *,
    reference: ReferenceSet | None = None,
    outdir: str | Path | None = None,
    baseline_model: str = "tip3p",
    aggregation=None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Score ``result`` and (optionally) write results.json, tables, REPORT.md.

    Returns a dict with keys ``rating``, ``rows``, and — when ``outdir`` is set —
    ``results_json``, ``properties_csv``, ``properties_md``, ``report_md``,
    ``figures``.
    """
    if reference is None:
        reference = load_reference_set("water", result.meta.get("temperature_K", 298.15))

    rating = score_all(result.scoring_inputs, reference, baseline_model=baseline_model,
                       uncertainties=result.scoring_uncertainties, aggregation=aggregation)
    rows = _build_rows(result, reference, rating)
    out: dict[str, Any] = {"rating": rating, "rows": rows, "reference": reference}

    if outdir is None:
        return out

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # extended results.json (pipeline blocks + evaluation block)
    blob = result.to_json_dict()
    blob["evaluation"] = {
        "schema_version": reference.schema_version,
        "baseline_model": baseline_model,
        "aggregation_rule": rating.rule,
        "overall_label": rating.overall_label,
        "grade": rating.grade, "grade_max": rating.grade_max, "grade_pct": rating.grade_pct,
        "counts": rating.counts,
        "reference_state": reference.state_point,
        "per_property": {
            k: {
                "label": v.label, "model_value": v.model_value, "unit": v.unit,
                "model_uncertainty": v.model_uncertainty,
                "exp_value": v.exp_value, "exp_uncertainty": v.exp_uncertainty,
                "dev_model_pct": v.dev_model_pct,
                "baseline_model": v.baseline_model, "baseline_value": v.baseline_value,
                "dev_baseline_pct": v.dev_baseline_pct,
                "within_experimental_uncertainty": v.within_experimental_uncertainty,
                "verdict": v.verdict, "rated": v.rated, "reason": v.reason,
            } for k, v in rating.per_property.items()
        },
        "citations": {k: {"text": c.text, "doi": c.doi}
                      for k, c in reference.citations.items()},
    }
    results_json = outdir / "results.json"
    results_json.write_text(json.dumps(blob, indent=2))
    out["results_json"] = str(results_json)

    # tables
    out["properties_csv"] = str(save_metrics_csv(rows, outdir / "properties_table.csv"))
    md_table = ["# {} — property accuracy vs TIP3P bar".format(result.meta.get("model", "model")),
                "", f"Overall rating: {_banner(rating)}", "",
                "| Property | Unit | Model | ±Unc | Exp | TIP3P | HIPPO | Dev vs exp | TIP3P dev | Verdict |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md_table.append("| {property} | {unit} | {model} | {uncertainty} | {experimental} | "
                        "{tip3p_threshold} | {hippo} | {dev_vs_exp_pct} | {tip3p_dev_pct} | "
                        "{verdict} |".format(**r))
    properties_md = outdir / "properties_table.md"
    properties_md.write_text("\n".join(md_table) + "\n")
    out["properties_md"] = str(properties_md)

    # human-readable report
    report_md = outdir / "REPORT.md"
    report_md.write_text(_report_markdown(result, reference, rating, rows))
    out["report_md"] = str(report_md)

    if make_plots:
        out["figures"] = _make_plots(result, outdir)

    return out


__all__ = ["build_evaluation_report", "format_console_summary"]
