"""Statistical helpers for time-series analysis of MD observables.

All functions are pure: arrays in, statistics out.  Ported from the
ForceBalance-derived routines in ``analyzetool/liquid.py`` (``bzavg``,
``statisticalInefficiency``, ``mean_stderr``) with a generic bootstrap helper
added.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bzavg(obs, boltz):
    """Boltzmann-weighted average of an observable.

    Parameters
    ----------
    obs:
        1-D ``(T,)`` or 2-D ``(T, K)`` / ``(K, T)`` array of observable values.
    boltz:
        1-D ``(T,)`` array of weights.  Pass ``np.ones(T)`` for a plain mean.

    Returns
    -------
    float (1-D input) or ``np.ndarray`` (2-D input).
    """
    obs = np.asarray(obs, dtype=float)
    boltz = np.asarray(boltz, dtype=float)
    if obs.ndim == 2:
        if obs.shape[0] == len(boltz) and obs.shape[1] == len(boltz):
            raise ValueError(
                "Both dimensions of obs equal len(boltz); axis is ambiguous."
            )
        if obs.shape[0] == len(boltz):
            return np.sum(obs * boltz.reshape(-1, 1), axis=0) / np.sum(boltz)
        if obs.shape[1] == len(boltz):
            return np.sum(obs * boltz, axis=1) / np.sum(boltz)
        raise ValueError("obs dimensions do not match len(boltz).")
    if obs.ndim == 1:
        return float(np.dot(obs, boltz) / np.sum(boltz))
    raise ValueError("obs must be 1- or 2-dimensional.")


def statistical_inefficiency(A_n, B_n=None, fast: bool = False, mintime: int = 3) -> float:
    """Compute the statistical inefficiency g = 1 + 2·tau of a timeseries.

    The autocorrelation of ``A_n`` is used unless a second series ``B_n`` is
    given, in which case the cross-correlation is estimated.  ``g`` is enforced
    to be ``>= 1.0``.  This is John Chodera's routine as vendored by
    ForceBalance (Chodera et al., JCTC 3(1):26-41, 2007).
    """
    A_n = np.asarray(A_n)
    B_n = np.asarray(A_n) if B_n is None else np.asarray(B_n)
    N = A_n.shape[0]
    if A_n.shape != B_n.shape:
        raise ValueError("A_n and B_n must have the same dimensions.")

    g = 1.0
    mu_A = A_n.mean()
    mu_B = B_n.mean()
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B
    sigma2_AB = (dA_n * dB_n).mean()
    if sigma2_AB == 0:
        # Constant series: no correlation structure, g stays at its floor.
        return 1.0

    t = 1
    increment = 1
    while t < N - 1:
        C = np.sum(dA_n[0:(N - t)] * dB_n[t:N] + dB_n[0:(N - t)] * dA_n[t:N]) / (
            2.0 * float(N - t) * sigma2_AB
        )
        if (C <= 0.0) and (t > mintime):
            break
        g += 2.0 * C * (1.0 - float(t) / float(N)) * float(increment)
        t += increment
        if fast:
            increment += 1

    return max(g, 1.0)


# Legacy spelling kept so ported code / notebooks import cleanly.
statisticalInefficiency = statistical_inefficiency


def mean_stderr(ts) -> tuple[float, float, bool]:
    """Return (mean, correlation-corrected stderr, error_flag) of a timeseries.

    The standard error is inflated by ``sqrt(g / N)`` where ``g`` is the
    statistical inefficiency, matching the legacy ``liquid.py`` behaviour.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.size == 0:
        return 0.0, 0.0, True
    try:
        tsmean = float(np.mean(ts))
        ts_std = float(np.std(ts) * np.sqrt(statistical_inefficiency(ts) / len(ts)))
        return tsmean, ts_std, False
    except Exception:
        return 0.0, 0.0, True


def bootstrap_error(
    estimator: Callable[[np.ndarray], float],
    n: int,
    *,
    numboots: int = 1000,
    sample_size: int | None = None,
    seed: int | None = None,
) -> float:
    """Bootstrap standard error of a scalar estimator over ``n`` frames.

    Generalises the per-property bootstrap loops in legacy ``liquid.py``.

    Parameters
    ----------
    estimator:
        Callable taking an integer index array (into ``range(n)``) and returning
        a scalar.  Resample with replacement is handled here.
    n:
        Number of available frames.
    numboots:
        Number of bootstrap resamples (legacy default 1000).
    sample_size:
        Frames drawn per resample.  Defaults to ``n // 3`` (legacy convention).
    seed:
        Seed for reproducibility.  ``None`` matches legacy non-deterministic
        behaviour.

    Returns
    -------
    The standard deviation across bootstrap estimates.  Multiply by
    ``sqrt(statistical_inefficiency(series))`` to recover the legacy error bar.
    """
    if n <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    N = sample_size if sample_size else max(1, int(n / 3))
    vals = np.array([estimator(rng.integers(0, n, size=N)) for _ in range(numboots)])
    return float(np.std(vals))


__all__ = [
    "bzavg",
    "statistical_inefficiency",
    "statisticalInefficiency",
    "mean_stderr",
    "bootstrap_error",
]
