"""Pure-numpy regression metrics for reference-vs-model comparison (goal f).

No plotting / pandas dependency — these are the array-in metrics that
:mod:`mdforge.qm.compare` builds its figures and tables on. The comparison
contract throughout ``qm`` is ``dict[str, np.ndarray]`` keyed by model name
plus a ``reference_key`` naming the entry to compare against.
"""

from __future__ import annotations

import numpy as np


def mae(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(reference, float) - np.asarray(prediction, float))))


def rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Root-mean-square error."""
    r = np.asarray(reference, float)
    p = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((r - p) ** 2)))


def r2(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination R² (NaN if the reference has zero variance)."""
    r = np.asarray(reference, float)
    p = np.asarray(prediction, float)
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Pearson correlation coefficient (NaN for < 2 points)."""
    r = np.asarray(reference, float)
    p = np.asarray(prediction, float)
    if r.size < 2:
        return float("nan")
    return float(np.corrcoef(r, p)[0, 1])


def regression_summary(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return {MAE, RMSE, R2, Pearson r} for one reference/prediction pair."""
    return {
        "MAE": mae(reference, prediction),
        "RMSE": rmse(reference, prediction),
        "R2": r2(reference, prediction),
        "Pearson r": pearson(reference, prediction),
    }


def flatten_gradients(gradients: np.ndarray) -> np.ndarray:
    """Flatten a gradient/force array of shape (..., 3) to 1-D components."""
    return np.asarray(gradients, dtype=float).reshape(-1)


def frame_gradient_magnitudes(gradients: np.ndarray) -> np.ndarray:
    """Per-frame mean |gradient|: norm over the last axis, mean over atoms/centers.

    Input ``(M, K, 3)`` → output ``(M,)``.
    """
    g = np.asarray(gradients, dtype=float)
    return np.linalg.norm(g, axis=-1).mean(axis=-1)


def _comparison_models(data: dict[str, np.ndarray], reference_key: str,
                       models: list[str] | None) -> list[str]:
    if reference_key not in data:
        raise KeyError(f"reference_key {reference_key!r} not in data keys {list(data)}")
    names = list(data.keys()) if models is None else list(models)
    return [m for m in names if m != reference_key]


def energy_metrics(
    energies_dict: dict[str, np.ndarray],
    reference_key: str = "spice",
    models: list[str] | None = None,
) -> list[dict]:
    """Per-model energy metrics vs the reference, sorted by MAE ascending.

    Returns a list of dicts ``{"Model": name, "MAE":…, "RMSE":…, "R2":…,
    "Pearson r":…}`` — pandas-free so it works without the viz extra.
    """
    comparison = _comparison_models(energies_dict, reference_key, models)
    ref = np.asarray(energies_dict[reference_key], dtype=float).reshape(-1)
    rows = [{"Model": m, **regression_summary(ref, np.asarray(energies_dict[m], float).reshape(-1))}
            for m in comparison]
    return sorted(rows, key=lambda d: d["MAE"])


def gradient_metrics(
    gradients_dict: dict[str, np.ndarray],
    reference_key: str = "spice",
    models: list[str] | None = None,
) -> list[dict]:
    """Per-model gradient metrics vs the reference, sorted by magnitude MAE.

    Reports both flattened-component errors and per-frame |gradient| magnitude
    errors. Returns a pandas-free list of dicts.
    """
    comparison = _comparison_models(gradients_dict, reference_key, models)
    ref = np.asarray(gradients_dict[reference_key], dtype=float)
    ref_flat = flatten_gradients(ref)
    ref_mag = frame_gradient_magnitudes(ref)
    rows = []
    for m in comparison:
        pred = np.asarray(gradients_dict[m], dtype=float)
        rows.append({
            "Model": m,
            "Component MAE": mae(ref_flat, flatten_gradients(pred)),
            "Component RMSE": rmse(ref_flat, flatten_gradients(pred)),
            "Magnitude MAE": mae(ref_mag, frame_gradient_magnitudes(pred)),
            "Magnitude RMSE": rmse(ref_mag, frame_gradient_magnitudes(pred)),
            "Magnitude Pearson r": pearson(ref_mag, frame_gradient_magnitudes(pred)),
        })
    return sorted(rows, key=lambda d: d["Magnitude MAE"])


__all__ = [
    "mae", "rmse", "r2", "pearson", "regression_summary",
    "flatten_gradients", "frame_gradient_magnitudes",
    "energy_metrics", "gradient_metrics",
]
