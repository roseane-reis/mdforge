"""Plotting helpers for liquid-phase properties.

matplotlib is an optional dependency (install ``mdforge[viz]``). It is imported
lazily so the rest of :mod:`mdforge.liquid` works without it; calling a plot
function without matplotlib raises a clear error.

Every function takes pre-computed arrays/values (never a log file) and returns
the matplotlib ``Axes`` so the caller can further customise or save.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install 'mdforge[viz]'"
        ) from exc
    return plt


def plot_running_average(series, dt_ps: float = 1.0, *, equil: int = 0, label: str = "",
                         legend: bool = True, t=None, ax=None):
    """Plot a per-frame observable and its cumulative running average vs. time.

    The observable name is shown on the y-axis. Pass an explicit ``t`` (time in
    ps per frame) to use the trajectory's real clock — e.g. a continuation run
    whose log starts partway through; otherwise the axis is ``arange(n) * dt_ps``
    starting at 0. Pass ``legend=False`` to suppress the per-axes legend (e.g.
    when a single shared legend is drawn elsewhere); the "raw"/"running avg" line
    labels are still set so the caller can build one.
    """
    plt = _import_pyplot()
    series = np.asarray(series, dtype=float)
    t = np.arange(len(series)) * dt_ps if t is None else np.asarray(t, dtype=float)
    running = np.cumsum(series) / np.arange(1, len(series) + 1)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, series, alpha=0.35, lw=0.8, label="raw")
    ax.plot(t, running, lw=2.0, label="running avg")
    if equil:
        ax.axvline(t[equil] if equil < len(t) else t[-1],
                   ls="--", color="k", alpha=0.5, label="equil")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(label or "observable")
    if legend:
        ax.legend()
    return ax


def plot_viscosity_convergence(viscosity, dt_ps: float = 1.0, *, label: str = "", ax=None):
    """Plot running viscosity η(t) (cP) vs. integration window (ps)."""
    plt = _import_pyplot()
    viscosity = np.asarray(viscosity, dtype=float)
    t = np.arange(1, len(viscosity) + 1) * dt_ps
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, 1e3 * viscosity, label=label or "η(t)")
    ax.set_xlabel("integration window (ps)")
    ax.set_ylabel("viscosity (cP)")
    ax.legend()
    return ax


def plot_property_vs_experiment(
    computed: Sequence[float],
    experimental: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    property_name: str = "property",
    ax=None,
):
    """Scatter computed vs. experimental values with a y=x parity line."""
    plt = _import_pyplot()
    computed = np.asarray(computed, dtype=float)
    experimental = np.asarray(experimental, dtype=float)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(experimental, computed, zorder=3)
    lo = float(min(computed.min(), experimental.min()))
    hi = float(max(computed.max(), experimental.max()))
    pad = 0.05 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.6, label="y = x")

    if labels is not None:
        for x, y, name in zip(experimental, computed, labels):
            ax.annotate(name, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"experimental {property_name}")
    ax.set_ylabel(f"computed {property_name}")
    ax.legend()
    return ax


def plot_property_vs_temperature(
    temperatures, values, errors=None, *, property_name: str = "property",
    experimental=None, ax=None,
):
    """Plot a property vs. temperature with optional error bars and experiment."""
    plt = _import_pyplot()
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    if errors is not None:
        ax.errorbar(temperatures, values, yerr=np.asarray(errors, dtype=float),
                    marker="o", capsize=3, label="computed")
    else:
        ax.plot(temperatures, values, marker="o", label="computed")
    if experimental is not None:
        ax.plot(temperatures, np.asarray(experimental, dtype=float),
                marker="s", ls="--", label="experiment")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(property_name)
    ax.legend()
    return ax


__all__ = [
    "plot_running_average",
    "plot_viscosity_convergence",
    "plot_property_vs_experiment",
    "plot_property_vs_temperature",
]
