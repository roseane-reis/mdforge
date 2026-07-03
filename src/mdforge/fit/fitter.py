"""Optimizer wrappers for force-field fitting (goal d).

Ports the two optimizers from ``analyzetool.auxfitting.Auxfit.fit_data``:
- ``least_squares`` with ``jac='3-point'``, ``loss='soft_l1'``, ``f_scale=0.5``
  (robust local refinement on a residual *vector*), and
- ``differential_evolution`` (global search on a scalarized sum-of-squares).

Both take a residual function ``x -> np.ndarray`` and return a :class:`FitResult`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FitResult:
    x: np.ndarray
    cost: float
    success: bool
    n_eval: int
    message: str
    raw: Any = None


def least_squares_fit(
    residual: Callable[[np.ndarray], np.ndarray],
    x0,
    *,
    bounds: tuple | None = None,
    diff_step: float = 0.01,
    f_scale: float = 0.5,
    loss: str = "soft_l1",
    max_nfev: int | None = None,
    verbose: int = 0,
) -> FitResult:
    """Robust least-squares refinement (scipy ``least_squares``, soft_l1 loss)."""
    from scipy.optimize import least_squares

    kwargs: dict = {"jac": "3-point", "f_scale": f_scale, "loss": loss,
                    "diff_step": diff_step, "verbose": verbose}
    if bounds is not None:
        kwargs["bounds"] = bounds
    if max_nfev is not None:
        kwargs["max_nfev"] = max_nfev
    res = least_squares(residual, np.asarray(x0, dtype=float), **kwargs)
    return FitResult(res.x, float(res.cost), bool(res.success), int(res.nfev), str(res.message), res)


def differential_evolution_fit(
    residual: Callable[[np.ndarray], np.ndarray],
    bounds,
    *,
    scalarize: bool = True,
    seed: int | None = None,
    **de_kwargs,
) -> FitResult:
    """Global search (scipy ``differential_evolution``).

    With ``scalarize`` (default), the residual vector is reduced to its
    sum-of-squares cost; pass ``scalarize=False`` if ``residual`` already returns
    a scalar.
    """
    from scipy.optimize import differential_evolution

    if scalarize:
        def objective(x):
            r = np.atleast_1d(residual(x))
            return float(np.sum(np.square(r)))
    else:
        objective = residual

    res = differential_evolution(objective, list(bounds), seed=seed, **de_kwargs)
    return FitResult(res.x, float(res.fun), bool(res.success), int(res.nfev), str(res.message), res)


__all__ = ["FitResult", "least_squares_fit", "differential_evolution_fit"]
