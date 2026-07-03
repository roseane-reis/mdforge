"""Interaction-energy profile plots (goal f, plotting track 2).

A streamlined version of ``prior internal tooling`` —
plots interaction energy vs. monomer COM distance (or frame id) for one or more
models on shared axes. matplotlib is lazy-imported (``mdforge[viz]``).

Like the source, this module does **no unit conversion** — energies and
distances are plotted as given; the axis labels are informational only. Arrays
in, figure out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - only without viz extra
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install 'mdforge[viz]'"
        ) from exc
    return plt


def plot_interaction_energy_profile(
    energies_by_model: Mapping[str, np.ndarray],
    x: np.ndarray | None = None,
    *,
    x_label: str = "monomer COM distance",
    energy_label: str = "interaction energy",
    reference_key: str | None = None,
    sort_by_x: bool = True,
    ax=None,
):
    """Plot interaction energy vs. ``x`` (e.g. COM distance) for each model.

    Parameters
    ----------
    energies_by_model:
        ``{model_name: (M,) interaction energies}``.
    x:
        ``(M,)`` x-axis values (COM distance, etc.). Defaults to frame index.
    reference_key:
        If given, that model is drawn as a thick black dashed line (the QM ref).
    sort_by_x:
        Sort points by x before plotting (clean line plots for scattered frames).
    """
    plt = _import_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    n = len(next(iter(energies_by_model.values())))
    xv = np.arange(n) if x is None else np.asarray(x, dtype=float)
    order = np.argsort(xv) if sort_by_x else np.arange(len(xv))

    for model, energies in energies_by_model.items():
        e = np.asarray(energies, dtype=float)
        is_ref = (model == reference_key)
        ax.plot(
            xv[order], e[order],
            marker="o", markersize=4, linewidth=2.5 if is_ref else 1.5,
            linestyle="--" if is_ref else "-",
            color="black" if is_ref else None,
            label=f"{model} (ref)" if is_ref else model, zorder=3 if is_ref else 2,
        )
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(x_label if x is not None else "frame index")
    ax.set_ylabel(energy_label)
    ax.legend()
    return ax


def plot_interaction_energy_grid(
    profiles: Sequence[tuple[str, Mapping[str, np.ndarray], np.ndarray]],
    *,
    x_label: str = "monomer COM distance",
    energy_label: str = "interaction energy",
    reference_key: str | None = None,
    ncols: int = 3,
):
    """Grid of interaction-energy profiles, one subplot per ``(title, energies, x)``."""
    plt = _import_pyplot()
    n = len(profiles)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()
    for i, (title, energies_by_model, x) in enumerate(profiles):
        plot_interaction_energy_profile(
            energies_by_model, x, x_label=x_label, energy_label=energy_label,
            reference_key=reference_key, ax=flat[i],
        )
        flat[i].set_title(title)
    for j in range(n, len(flat)):
        flat[j].set_visible(False)
    fig.tight_layout()
    return fig


__all__ = ["plot_interaction_energy_profile", "plot_interaction_energy_grid"]
