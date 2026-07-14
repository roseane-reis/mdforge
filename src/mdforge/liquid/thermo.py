"""Thermodynamic liquid-property kernels — arrays in, properties out.

Ported from the ``Liquid`` methods in ``analyzetool/liquid.py``
(``calc_alpha``, ``calc_kappa``, ``calc_cp``, ``calc_eps0``) plus density and
heat-of-vaporization, with the file-parsing entry points stripped away. Every
function operates on pre-sliced numpy arrays (equilibration trimming is the
caller's responsibility — see :func:`mdforge.liquid.equilibrate`).

Unit conventions match the legacy code so results are reproducible:
- energies / enthalpy : kcal/mol
- volume              : Angstrom³
- dipole              : e·Angstrom
- temperature         : Kelvin
"""

from __future__ import annotations

import numpy as np

from .constants import AMU_A3_TO_G_CM3, DIELECTRIC_PREFACTOR, KB_J, R_KCAL
from .stats import bzavg


def equilibrate(array: np.ndarray, equil: int) -> np.ndarray:
    """Drop the first ``equil`` frames (equilibration) from a per-frame array."""
    return np.asarray(array)[equil:]


def density(volume, total_mass) -> np.ndarray:
    """Per-frame mass density in g/cm³.

    Parameters
    ----------
    volume:
        Per-frame cell volume in Å³ (scalar or ``(T,)`` array).
    total_mass:
        Total system mass in amu.
    """
    volume = np.asarray(volume, dtype=float)
    return AMU_A3_TO_G_CM3 * total_mass / volume


def thermal_expansion(enthalpy, volume, temperature, weights=None) -> float:
    """Isobaric thermal-expansion coefficient α (legacy ``calc_alpha``).

    α = 1/(kT·T) · (⟨H·V⟩ − ⟨H⟩⟨V⟩) / ⟨V⟩,  with kT = R·T (kcal/mol).
    """
    h = np.asarray(enthalpy, dtype=float)
    v = np.asarray(volume, dtype=float)
    L = min(len(h), len(v))
    h, v = h[:L], v[:L]
    b = np.ones(L) if weights is None else np.asarray(weights, dtype=float)[:L]
    kT = R_KCAL * temperature
    return (1.0 / (kT * temperature)) * (bzavg(h * v, b) - bzavg(h, b) * bzavg(v, b)) / bzavg(v, b)


def isothermal_compressibility(volume, temperature) -> float:
    """Isothermal compressibility κ_T (legacy ``calc_kappa``).

    Returns the value in the legacy unit convention (the ``1e11`` prefactor of
    ``liquid.py``); divide/scale downstream as that code did. Volume in Å³.
    """
    V0 = 1e-30 * np.asarray(volume, dtype=float)  # Å³ → m³
    avg_volume = V0.mean()
    volume_fluct = (V0 * V0).mean() - avg_volume * avg_volume
    return 1e11 * (volume_fluct / (KB_J * temperature * avg_volume))


def heat_capacity(enthalpy, n_molecules, temperature, weights=None) -> float:
    """Isobaric heat capacity per molecule Cp (legacy ``calc_cp``), ×1000.

    Cp = 1000/(N·kT·T) · (⟨H²⟩ − ⟨H⟩²),  kT = R·T (kcal/mol).
    """
    h = np.asarray(enthalpy, dtype=float)
    L = len(h)
    b = np.ones(L) if weights is None else np.asarray(weights, dtype=float)
    kT = R_KCAL * temperature
    cp = (1.0 / (n_molecules * kT * temperature)) * (bzavg(h ** 2, b) - bzavg(h, b) ** 2)
    return cp * 1000.0


def clausius_mossotti_eps_inf(molpol: float, volume_per_molecule: float) -> float:
    """High-frequency dielectric ε_∞ from molecular polarizability.

    Matches the ``epf_inf`` expression in legacy ``calc_eps0`` setup:
    (−8π·α − 3·v) / (4π·α − 3·v),  v = volume per molecule.
    """
    v = volume_per_molecule
    return (-np.pi * 8 * molpol - 3 * v) / (np.pi * 4 * molpol - 3 * v)


def dielectric_constant(dipoles, volume, temperature, eps_inf: float = 1.0, weights=None) -> float:
    """Static dielectric constant ε₀ from cell-dipole fluctuations (``calc_eps0``).

    ε₀ = ε_∞ + prefactor · ⟨δμ²⟩ / ⟨V⟩ / T,  with the dipole variance summed
    over x, y, z components.

    Parameters
    ----------
    dipoles:
        ``(T, 3)`` array of total cell dipole in **Debye** (the prefactor is
        defined for Debye; convert an e·Å dipole with ``× 4.803``).
    volume:
        ``(T,)`` array of cell volume (Å³).
    temperature:
        Kelvin.
    eps_inf:
        High-frequency dielectric (1.0 if polarizability unknown); see
        :func:`clausius_mossotti_eps_inf`.
    """
    d = np.asarray(dipoles, dtype=float)
    v = np.asarray(volume, dtype=float)
    L = min(len(d), len(v))
    d, v = d[:L], v[:L]
    b = np.ones(L) if weights is None else np.asarray(weights, dtype=float)[:L]
    D2 = bzavg(d[:, 0] ** 2, b) - bzavg(d[:, 0], b) ** 2
    D2 += bzavg(d[:, 1] ** 2, b) - bzavg(d[:, 1], b) ** 2
    D2 += bzavg(d[:, 2] ** 2, b) - bzavg(d[:, 2], b) ** 2
    return eps_inf + DIELECTRIC_PREFACTOR * D2 / bzavg(v, b) / temperature


def heat_of_vaporization(gas_pe, liquid_pe_per_molecule, temperature) -> float:
    """Heat of vaporization ΔHvap (legacy ``HV``).

    ΔHvap = ⟨E_gas⟩ − ⟨E_liquid⟩/N + RT,  energies in kcal/mol.

    ``gas_pe`` is the per-molecule gas-phase potential (or enthalpy) average;
    ``liquid_pe_per_molecule`` is the liquid potential (or enthalpy) per molecule.
    Pass enthalpies for the legacy ``HV2`` variant.
    """
    return gas_pe - liquid_pe_per_molecule + R_KCAL * temperature


__all__ = [
    "equilibrate",
    "density",
    "thermal_expansion",
    "isothermal_compressibility",
    "heat_capacity",
    "clausius_mossotti_eps_inf",
    "dielectric_constant",
    "heat_of_vaporization",
]
