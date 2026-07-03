"""Reference-vs-model comparison figures + metric tables (goal f core).

Lifted from ``prior internal tooling``. The metric math lives in
:mod:`mdforge.qm.metrics` (pure numpy); the 3-panel seaborn figures here
lazy-import matplotlib / seaborn / pandas so the rest of ``qm`` works on a
core install (``pip install mdforge``). Install ``mdforge[viz]`` for plotting.

Input contract: ``dict[str, np.ndarray]`` keyed by model name + a
``reference_key``. Every non-reference entry is compared against the reference
— the literal "tagged reference vs model" design.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .metrics import (
    energy_metrics,
    frame_gradient_magnitudes,
    gradient_metrics,
    regression_summary,  # re-exported for convenience
)


def _import_viz():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - only without viz extra
        raise ImportError(
            "Comparison figures require matplotlib, seaborn, and pandas. "
            "Install them with: pip install 'mdforge[viz]'"
        ) from exc
    return plt, sns, pd


def _ordered_models(data: dict[str, np.ndarray], reference_key: str,
                    models: list[str] | None) -> tuple[list[str], list[str]]:
    names = list(data.keys()) if models is None else list(models)
    comparison = [m for m in names if m != reference_key]
    return names, comparison


def _palette_dict(models: Iterable[str], sns, palette: str = "colorblind") -> dict:
    models = list(models)
    return dict(zip(models, sns.color_palette(palette, n_colors=len(models))))


def _apply_plot_style(plt, sns, context: str = "talk") -> None:
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300


def _style_axis_text(ax, *, title=None, xlabel=None, ylabel=None,
                     title_size=17.0, label_size=14.0, tick_size=11.5) -> None:
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)


def _add_top_padding(ax, frac: float = 0.18) -> None:
    ymin, ymax = ax.get_ylim()
    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        return
    span = ymax - ymin
    if np.isclose(span, 0.0):
        span = max(abs(ymax), 1.0)
    ax.set_ylim(ymin, ymax + frac * span)


def _place_legend_top_right(ax, sns) -> None:
    if ax.get_legend() is None:
        return
    sns.move_legend(ax, "upper right", title=None, frameon=True)
    _add_top_padding(ax, frac=0.20)


def _long_distribution_df(pd, values_dict, models):
    return pd.concat(
        [pd.DataFrame({"Model": m, "Value": np.asarray(values_dict[m], float).reshape(-1)})
         for m in models],
        ignore_index=True,
    )


def _long_parity_df(pd, reference, values_dict, comparison):
    return pd.concat(
        [pd.DataFrame({"Model": m, "Reference": reference.reshape(-1),
                       "Prediction": np.asarray(values_dict[m], float).reshape(-1)})
         for m in comparison],
        ignore_index=True,
    )


def _long_error_df(pd, reference, values_dict, comparison):
    return pd.concat(
        [pd.DataFrame({"Model": m,
                       "Error": np.asarray(values_dict[m], float).reshape(-1) - reference.reshape(-1)})
         for m in comparison],
        ignore_index=True,
    )


def _draw_error_panel(ax, sns, df_error, order, palette_dict, ylabel, kind="violin") -> None:
    if kind == "violin":
        sns.violinplot(data=df_error, x="Model", y="Error", order=order, hue="Model",
                       palette=palette_dict, inner=None, cut=0, linewidth=1.0,
                       saturation=0.95, legend=False, ax=ax)
        sns.boxplot(data=df_error, x="Model", y="Error", order=order, width=0.22,
                    showfliers=False,
                    boxprops={"facecolor": "white", "edgecolor": "black", "zorder": 3},
                    whiskerprops={"color": "black", "linewidth": 1.2},
                    capprops={"color": "black", "linewidth": 1.2},
                    medianprops={"color": "black", "linewidth": 1.5}, ax=ax)
    elif kind == "boxen":
        sns.boxenplot(data=df_error, x="Model", y="Error", order=order, hue="Model",
                      palette=palette_dict, linewidth=1.0, saturation=0.95,
                      legend=False, ax=ax)
    else:
        raise ValueError("error_plot must be 'violin' or 'boxen'")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.2)
    _style_axis_text(ax, xlabel="", ylabel=ylabel)
    ax.tick_params(axis="x", rotation=35)


def _draw_distribution_panel(ax, sns, df_dist, order, palette_dict, xlabel, kind="kde") -> None:
    if kind == "kde":
        sns.kdeplot(data=df_dist, x="Value", hue="Model", hue_order=order,
                    palette=palette_dict, common_norm=False, fill=False, linewidth=2.0, ax=ax)
    elif kind == "hist":
        sns.histplot(data=df_dist, x="Value", hue="Model", hue_order=order,
                     palette=palette_dict, stat="density", common_norm=False,
                     element="step", fill=False, bins=40, linewidth=1.5, ax=ax)
    elif kind == "ecdf":
        sns.ecdfplot(data=df_dist, x="Value", hue="Model", hue_order=order,
                     palette=palette_dict, linewidth=2.0, ax=ax)
    else:
        raise ValueError("dist_plot must be 'kde', 'hist', or 'ecdf'")
    _style_axis_text(ax, xlabel=xlabel)


def _three_panel(values_dict, mag_dict, reference_key, comparison, order, *,
                 parity_title, parity_xlabel, parity_ylabel,
                 error_ylabel, dist_xlabel, error_plot, dist_plot, palette, context):
    plt, sns, pd = _import_viz()
    _apply_plot_style(plt, sns, context=context)
    palette_dict = _palette_dict(order, sns, palette=palette)
    ref_mag = np.asarray(mag_dict[reference_key], float)

    df_parity = _long_parity_df(pd, ref_mag, mag_dict, comparison)
    df_error = _long_error_df(pd, ref_mag, mag_dict, comparison)
    df_dist = _long_distribution_df(pd, mag_dict, order)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    sns.scatterplot(data=df_parity, x="Reference", y="Prediction", hue="Model",
                    hue_order=comparison, palette=palette_dict, alpha=0.55, s=28,
                    linewidth=0, ax=axes[0])
    lim_lo = min(df_parity["Reference"].min(), df_parity["Prediction"].min())
    lim_hi = max(df_parity["Reference"].max(), df_parity["Prediction"].max())
    axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="black", linewidth=1.2)
    _style_axis_text(axes[0], title=parity_title, xlabel=parity_xlabel, ylabel=parity_ylabel)
    axes[0].set_aspect("equal", adjustable="box")
    _place_legend_top_right(axes[0], sns)

    _draw_error_panel(axes[1], sns, df_error, comparison, palette_dict, error_ylabel, kind=error_plot)
    _style_axis_text(axes[1], title="Error distribution")

    _draw_distribution_panel(axes[2], sns, df_dist, order, palette_dict, dist_xlabel, kind=dist_plot)
    _style_axis_text(axes[2], title="Distribution")
    _place_legend_top_right(axes[2], sns)
    return fig


def compare_energy_models(
    energies_dict: dict[str, np.ndarray],
    reference_key: str = "spice",
    models: list[str] | None = None,
    outdir: str | Path | None = None,
    quantity_name: str = "Relative energy (kcal/mol)",
    out_name: str = "energy",
    error_plot: str = "violin",
    dist_plot: str = "kde",
    palette: str = "colorblind",
    context: str = "talk",
):
    """Compare per-frame energies across models; return ``(summary_df, fig)``.

    ``summary_df`` is a pandas DataFrame of MAE/RMSE/R²/Pearson per model
    (sorted by MAE); the figure is the 3-panel parity / error / distribution plot.
    """
    plt, sns, pd = _import_viz()
    _, comparison = _ordered_models(energies_dict, reference_key, models)
    order = [reference_key] + comparison

    summary = pd.DataFrame(energy_metrics(energies_dict, reference_key, models))
    fig = _three_panel(
        energies_dict, energies_dict, reference_key, comparison, order,
        parity_title="Parity plot",
        parity_xlabel=f"Reference {quantity_name}",
        parity_ylabel=f"Model {quantity_name}",
        error_ylabel=f"Δ {quantity_name}",
        dist_xlabel=quantity_name,
        error_plot=error_plot, dist_plot=dist_plot, palette=palette, context=context,
    )
    if outdir is not None:
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / f"{out_name}_model_comparison.png", bbox_inches="tight")
    return summary, fig


def compare_gradient_models(
    gradients_dict: dict[str, np.ndarray],
    reference_key: str = "spice",
    models: list[str] | None = None,
    outdir: str | Path | None = None,
    out_name: str = "gradient",
    error_plot: str = "violin",
    dist_plot: str = "kde",
    palette: str = "colorblind",
    context: str = "talk",
):
    """Compare per-frame gradient magnitudes across models; return ``(summary_df, fig)``."""
    plt, sns, pd = _import_viz()
    _, comparison = _ordered_models(gradients_dict, reference_key, models)
    order = [reference_key] + comparison

    summary = pd.DataFrame(gradient_metrics(gradients_dict, reference_key, models))
    mag_dict = {k: frame_gradient_magnitudes(v) for k, v in gradients_dict.items()}
    fig = _three_panel(
        gradients_dict, mag_dict, reference_key, comparison, order,
        parity_title="Gradient magnitude parity",
        parity_xlabel="Reference mean |gradient| per frame",
        parity_ylabel="Model mean |gradient| per frame",
        error_ylabel="Δ mean |gradient| per frame",
        dist_xlabel="Mean |gradient| per frame",
        error_plot=error_plot, dist_plot=dist_plot, palette=palette, context=context,
    )
    if outdir is not None:
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / f"{out_name}_model_comparison.png", bbox_inches="tight")
    return summary, fig


__all__ = [
    "regression_summary",
    "energy_metrics",
    "gradient_metrics",
    "compare_energy_models",
    "compare_gradient_models",
]
