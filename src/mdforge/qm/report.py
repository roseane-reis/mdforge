"""Orchestration: turn a dict of records into comparison arrays, metrics, figures.

A modernized, array-first replacement for the comparison-driver parts of
``prior internal tooling`` (the dense stringly-typed
``pair``/``pair_model`` bookkeeping and its inverted-guard bug are intentionally
not carried over). The contract is explicit: a ``dict[str, record]`` keyed by
model name, a ``reference_key``, and the field names to pull energies/gradients
from. Everything downstream is plain ``dict[str, np.ndarray]`` for
:mod:`mdforge.qm.compare`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import energy_metrics, gradient_metrics


def build_energy_dict(
    records: Mapping[str, Any],
    reference_key: str = "spice",
    *,
    reference_energy_field: str = "dft_total_energy",
    query_energy_field: str = "model_total_energy",
    relative_to_frame: int | None = None,
) -> dict[str, np.ndarray]:
    """Pull a per-model energy array off each record into ``{model: (M,)}``.

    The reference record uses ``reference_energy_field``; all others use
    ``query_energy_field``. If ``relative_to_frame`` is given, each model's
    energies are offset by that frame (``e - e[frame]``) so all share a zero —
    the relative-energy convention from the legacy report pipeline.
    """
    out: dict[str, np.ndarray] = {}
    for model, rec in records.items():
        field = reference_energy_field if model == reference_key else query_energy_field
        arr = np.asarray(getattr(rec, field), dtype=float).reshape(-1)
        if relative_to_frame is not None:
            arr = arr - arr[relative_to_frame]
        out[model] = arr
    return out


def build_gradient_dict(
    records: Mapping[str, Any],
    reference_key: str = "spice",
    *,
    reference_gradient_field: str = "dft_total_gradient",
    query_gradient_field: str = "forces_per_center",
) -> dict[str, np.ndarray]:
    """Pull a per-model gradient/force array off each record into ``{model: (M,K,3)}``."""
    out: dict[str, np.ndarray] = {}
    for model, rec in records.items():
        field = reference_gradient_field if model == reference_key else query_gradient_field
        out[model] = np.asarray(getattr(rec, field), dtype=float)
    return out


def save_metrics_csv(rows: list[dict], path: str | Path) -> Path:
    """Write a list of metric dicts to CSV (no pandas dependency)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def compare_records(
    records: Mapping[str, Any],
    reference_key: str = "spice",
    *,
    outdir: str | Path | None = None,
    energy: bool = True,
    gradient: bool = False,
    reference_energy_field: str = "dft_total_energy",
    query_energy_field: str = "model_total_energy",
    reference_gradient_field: str = "dft_total_gradient",
    query_gradient_field: str = "forces_per_center",
    relative_to_frame: int | None = None,
    quantity_name: str = "Relative energy (kcal/mol)",
) -> dict[str, Any]:
    """Build comparison dicts from records, compute metrics, and (if viz present) figures.

    Always returns metric tables (``energy_metrics`` / ``gradient_metrics`` as
    lists of dicts) — these need only numpy. Figures and a ``summary`` DataFrame
    are added when ``outdir`` is set and the viz extra is installed; CSVs are
    written to ``outdir`` regardless.

    Returns a dict with keys among: ``energy_metrics``, ``gradient_metrics``,
    ``energy_fig``, ``gradient_fig``, ``energy_csv``, ``gradient_csv``.
    """
    result: dict[str, Any] = {}

    if energy:
        edict = build_energy_dict(
            records, reference_key,
            reference_energy_field=reference_energy_field,
            query_energy_field=query_energy_field,
            relative_to_frame=relative_to_frame,
        )
        result["energy_metrics"] = energy_metrics(edict, reference_key)
        if outdir is not None:
            result["energy_csv"] = save_metrics_csv(result["energy_metrics"],
                                                    Path(outdir) / "energy_metrics.csv")
            from .compare import compare_energy_models
            summary, fig = compare_energy_models(
                edict, reference_key=reference_key, outdir=outdir,
                quantity_name=quantity_name,
            )
            result["energy_summary"] = summary
            result["energy_fig"] = fig

    if gradient:
        gdict = build_gradient_dict(
            records, reference_key,
            reference_gradient_field=reference_gradient_field,
            query_gradient_field=query_gradient_field,
        )
        result["gradient_metrics"] = gradient_metrics(gdict, reference_key)
        if outdir is not None:
            result["gradient_csv"] = save_metrics_csv(result["gradient_metrics"],
                                                      Path(outdir) / "gradient_metrics.csv")
            from .compare import compare_gradient_models
            summary, fig = compare_gradient_models(gdict, reference_key=reference_key, outdir=outdir)
            result["gradient_summary"] = summary
            result["gradient_fig"] = fig

    return result


__all__ = [
    "build_energy_dict",
    "build_gradient_dict",
    "save_metrics_csv",
    "compare_records",
]
