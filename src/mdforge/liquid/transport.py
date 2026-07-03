"""Transport-property kernels: pressure tensor and shear viscosity.

Ported from the already-decoupled ``analyzetool`` transport chain
(``get_pressure_tensor.py`` → ``compute_visc.py`` / ``compute_visc-gk.py``).
These were the cleanest "arrays in, property out" code in the legacy package
and serve as the template for the parse⟂compute split.

Two changes from the legacy scripts, both deliberate:
1. Time handling is made self-consistent — viscosity takes an explicit
   ``dt_ps`` and builds its own time axis, instead of the legacy mix of a
   ``linspace(0, total, N)`` axis with a hard-coded ``dx = 0.01 ps``.
2. The Einstein integral uses a cumulative trapezoid (O(N)) instead of the
   legacy O(N²) re-integration; the result is identical up to the usual
   trapezoid end-point convention.
The Einstein average keeps the legacy 5-independent-shear-component convention
(``Pxy, Pxz, Pyz, (Pxx−Pyy)/2, (Pyy−Pzz)/2``); Green-Kubo averages all six.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .constants import KB_J, N_A_LEGACY

ANG2_PS_TO_CM2_S = 1e-4   # 1 Å²/ps = 1e-4 cm²/s


def _box_lengths(box, n_frames: int) -> np.ndarray:
    """Normalise a box spec to per-frame ``(T, 3)`` edge lengths (orthorhombic)."""
    b = np.asarray(box, dtype=float)
    if b.ndim == 1:
        b = np.broadcast_to(b, (n_frames, b.shape[0])).copy()
    if b.shape[1] >= 6:
        tilt = b[:, 3:6]
        if np.any(np.abs(tilt) > 1e-6):
            raise ValueError(
                "unwrap supports orthorhombic cells only; the trajectory has "
                "non-zero tilt factors (xy/xz/yz)."
            )
    return b[:, :3]


def pressure_tensor(virial, velocities, masses, volume) -> np.ndarray:
    """Per-frame pressure tensor in Pa.

    P = (Σ_i m_i v_i⊗v_i · 10 − virial · 4184) / (N_A · V)

    Parameters
    ----------
    virial:
        ``(T, 3, 3)`` internal virial tensor in kcal/mol.
    velocities:
        ``(T, N, 3)`` atomic velocities in Å/ps.
    masses:
        ``(N,)`` atomic masses in amu.
    volume:
        Cell volume in Å³ — scalar (fixed-volume run) or ``(T,)`` array.

    Returns
    -------
    ``(T, 3, 3)`` pressure tensor in Pa.

    Notes
    -----
    The ``10`` and ``4184`` factors and the ``N_A·V`` normalisation reproduce
    the legacy ``get_pressure_tensor.py`` arithmetic exactly. The kinetic term
    assumes a symmetric virial (Tinker's is symmetric).
    """
    virial = np.asarray(virial, dtype=float)
    V = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    vol_m3 = 1e-30 * np.asarray(volume, dtype=float)  # Å³ → m³

    kinetic = np.einsum("i,tia,tib->tab", m, V, V)  # (T,3,3) amu·(Å/ps)²
    raw = 10.0 * kinetic - 4184.0 * virial
    div = N_A_LEGACY * vol_m3
    if np.ndim(div) == 0:
        return raw / div
    return raw / div[:, None, None]


def _shear_components(P: np.ndarray) -> np.ndarray:
    """Return the (6, T) array of shear stress components from a (T,3,3) tensor."""
    N = P.shape[0]
    shear = np.zeros((6, N), dtype=float)
    shear[0] = P[:, 0, 1]
    shear[1] = P[:, 0, 2]
    shear[2] = P[:, 1, 2]
    shear[3] = (P[:, 0, 0] - P[:, 1, 1]) / 2.0  # xx-yy
    shear[4] = (P[:, 1, 1] - P[:, 2, 2]) / 2.0  # yy-zz
    shear[5] = (P[:, 0, 0] - P[:, 2, 2]) / 2.0  # xx-zz (unused by Einstein)
    return shear


def viscosity_einstein(pressure_tensor, volume, temperature, dt_ps, skip: int = 1) -> np.ndarray:
    """Shear viscosity vs. integration window via the Einstein/Helfand method.

    η(t) = ⟨(∫₀ᵗ P_shear dt')²⟩ · V / (2·k_B·T·t),  averaged over the 5
    independent shear components.

    Parameters
    ----------
    pressure_tensor:
        ``(T, 3, 3)`` pressure tensor in Pa.
    volume:
        Cell volume in Å³ (scalar; the run is fixed-volume for viscosity).
    temperature:
        Kelvin.
    dt_ps:
        Time between pressure-tensor frames in ps.
    skip:
        Stride applied to the frames before integrating.

    Returns
    -------
    ``(M,)`` array of η(t) in Pa·s, one value per window. Multiply by 1e3 for
    cP (mPa·s); take a plateau mean for the reported viscosity.
    """
    P = np.asarray(pressure_tensor, dtype=float)
    vol_m3 = 1e-30 * float(np.mean(volume))
    shear = _shear_components(P)[:, ::skip]
    dt_s = dt_ps * 1e-12 * skip
    Ns = shear.shape[1]

    integral_sq = np.zeros(Ns, dtype=float)
    for i in range(5):  # 5 independent shear components
        cum = cumulative_trapezoid(shear[i], dx=dt_s, initial=0.0)
        integral_sq += cum ** 2 / 5.0

    kbT = KB_J * temperature
    time = np.arange(Ns) * dt_s
    return integral_sq[1:] * vol_m3 / (2.0 * kbT * time[1:])


def viscosity_green_kubo(
    pressure_tensor, volume, temperature, dt_ps, max_lag: int | None = None
) -> np.ndarray:
    """Shear viscosity vs. upper integration limit via the Green-Kubo method.

    η(t) = (V / k_B T) · ∫₀ᵗ ⟨P_shear(0)·P_shear(τ)⟩ dτ,  with the stress
    autocorrelation averaged over all six shear components.

    Parameters
    ----------
    pressure_tensor:
        ``(T, 3, 3)`` pressure tensor in Pa.
    volume:
        Cell volume in Å³ (scalar).
    temperature:
        Kelvin.
    dt_ps:
        Time between frames in ps.
    max_lag:
        Largest autocorrelation lag to compute (defaults to all frames; the
        long-lag tail is noisy and usually trimmed).

    Returns
    -------
    ``(L-1,)`` array of η(t) in Pa·s. Multiply by 1e3 for cP.
    """
    P = np.asarray(pressure_tensor, dtype=float)
    vol_m3 = 1e-30 * float(np.mean(volume))
    shear = _shear_components(P)
    N = shear.shape[1]
    size = N if max_lag is None else min(max_lag, N)

    acf = np.zeros(size, dtype=float)
    for t in range(size):
        acf[t] = np.mean(shear[:, : N - t] * shear[:, t:])

    acf *= vol_m3 / (KB_J * temperature)
    dt_s = dt_ps * 1e-12
    return cumulative_trapezoid(acf, dx=dt_s, initial=0.0)[1:]


def unwrap_com(com, box):
    """Unwrap a wrapped COM trajectory ``(T, M, 3)`` by incremental minimum image.

    Each frame-to-frame step is reduced to its minimum image under that frame's
    box and accumulated, so molecules that cross the periodic boundary trace a
    continuous path. Returns an unwrapped ``(T, M, 3)`` array (frame 0 unchanged).

    IMPORTANT — unwrapping: HOOMD GSD ``particles.image`` flags for the rigid-body
    *centres* are frozen (they do not increment when a COM crosses the cell wall),
    so ``position + image·L`` does NOT unwrap correctly. We unwrap incrementally
    with the minimum-image of each frame-to-frame step instead.
    """
    com = np.asarray(com, dtype=float)
    T = com.shape[0]
    L = _box_lengths(box, T)
    out = np.empty_like(com)
    out[0] = com[0]
    for t in range(1, T):
        d = com[t] - com[t - 1]
        d -= L[t] * np.round(d / L[t])
        out[t] = out[t - 1] + d
    return out


def msd(unwrapped, *, max_lag=None):
    """Time-origin-averaged mean-squared displacement vs lag.

    ``unwrapped`` is ``(T, M, 3)``. Returns ``msd`` ``(max_lag+1,)`` in Å²
    (averaged over molecules and all time origins). ``msd[0] = 0``.
    """
    u = np.asarray(unwrapped, dtype=float)
    T = u.shape[0]
    max_lag = (T - 1) if max_lag is None else min(max_lag, T - 1)
    out = np.zeros(max_lag + 1, dtype=float)
    for lag in range(1, max_lag + 1):
        disp = u[lag:] - u[:-lag]                    # (T-lag, M, 3)
        out[lag] = float((disp * disp).sum(-1).mean())
    return out


def self_diffusion(msd_curve, dt_ps, *, fit_lo=0.2, fit_hi=0.6):
    """Self-diffusion D (Å²/ps) from MSD via the 3-D Einstein relation.

    Fits a straight line to MSD over the lag window
    ``[fit_lo, fit_hi]·max_lag`` (default 20–60 %, avoiding the short-time
    ballistic regime and the noisy long-time tail) and returns
    ``D = slope / 6`` plus the fit for plotting.

    Returns dict: ``D_ang2_ps``, ``D_cm2_s``, ``slope``, ``intercept``,
    ``fit_slice`` (lo, hi indices), ``t`` (lag times, ps).
    """
    msd_curve = np.asarray(msd_curve, dtype=float)
    n = len(msd_curve)
    t = np.arange(n) * dt_ps
    lo, hi = int(fit_lo * (n - 1)), int(fit_hi * (n - 1))
    lo = max(lo, 1)
    slope, intercept = np.polyfit(t[lo:hi + 1], msd_curve[lo:hi + 1], 1)
    D = slope / 6.0
    return {
        "D_ang2_ps": float(D),
        "D_cm2_s": float(D * ANG2_PS_TO_CM2_S),
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_slice": (lo, hi),
        "t": t,
    }


def yeh_hummer_correction(viscosity_pa_s, box_length_ang, temperature_K):
    """Yeh–Hummer finite-size correction to D (cm²/s), given a viscosity.

    ΔD = ξ·k_B·T / (6π·η·L),  ξ = 2.837297 (cubic box). Add to the PBC D to
    estimate the infinite-system value. Needs a viscosity — we cannot compute η
    from this run, so pass a literature/reference value and label it as such.
    """
    XI = 2.837297
    KB = 1.380649e-23
    L_m = box_length_ang * 1e-10
    dD_m2_s = XI * KB * temperature_K / (6 * np.pi * viscosity_pa_s * L_m)
    return dD_m2_s * 1e4   # m²/s → cm²/s


__all__ = [
    "pressure_tensor",
    "viscosity_einstein",
    "viscosity_green_kubo",
    "unwrap_com",
    "msd",
    "self_diffusion",
    "yeh_hummer_correction",
    "ANG2_PS_TO_CM2_S",
]
