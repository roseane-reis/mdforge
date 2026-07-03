"""Physical constants and generic unit conversions used across mdforge.

All constants are derived from CODATA 2018 exact-definition values.
Ported from prior internal tooling (unchanged).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PI = math.pi
TWO_PI = 2.0 * PI
C = 299792458.0           # m/s  (exact)
MU0 = 4.0e-7 * PI        # H/m
EPSILON0 = 1.0 / (MU0 * C * C)
H = 6.62607015e-34        # J·s  (exact)
HBAR = H / TWO_PI
E = 1.602176634e-19       # C    (exact)
M_E = 9.1093837015e-31    # kg
N_A = 6.02214076e23       # mol⁻¹  (exact)
A0 = HBAR * HBAR / (M_E * E * E * (1.0 / (4.0 * PI * EPSILON0)))  # bohr [m]

HARTREE_TO_EV = (M_E * E**4) / (HBAR**2 * (4.0 * PI * EPSILON0) ** 2 * E)
HARTREE_TO_KCAL_MOL = HARTREE_TO_EV * E * N_A / 4184.0
HARTREE_TO_KJ_MOL = HARTREE_TO_EV * E * N_A / 1000.0
BOHR_TO_ANGSTROM = A0 * 1.0e10
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

_ENERGY_FACTORS_TO_HARTREE = {
    'hartree': 1.0,
    'ha': 1.0,
    'kcal/mol': 1.0 / HARTREE_TO_KCAL_MOL,
    'kcalmol': 1.0 / HARTREE_TO_KCAL_MOL,
    'kj/mol': 1.0 / HARTREE_TO_KJ_MOL,
    'kjmol': 1.0 / HARTREE_TO_KJ_MOL,
}

_LENGTH_FACTORS_TO_BOHR = {
    'bohr': 1.0,
    'a0': 1.0,
    'angstrom': ANGSTROM_TO_BOHR,
    'ang': ANGSTROM_TO_BOHR,
    'a': ANGSTROM_TO_BOHR,
    'nm': 10.0 * ANGSTROM_TO_BOHR,
}

_GRADIENT_FACTORS_TO_HARTREE_PER_BOHR = {
    'hartree/bohr': 1.0,
    'ha/bohr': 1.0,
    'kcal/mol/angstrom': BOHR_TO_ANGSTROM / HARTREE_TO_KCAL_MOL,
    'kcal/mol/a': BOHR_TO_ANGSTROM / HARTREE_TO_KCAL_MOL,
    'kcal/mol/ang': BOHR_TO_ANGSTROM / HARTREE_TO_KCAL_MOL,
    'kj/mol/angstrom': BOHR_TO_ANGSTROM / HARTREE_TO_KJ_MOL,
    'kj/mol/a': BOHR_TO_ANGSTROM / HARTREE_TO_KJ_MOL,
    'kj/mol/ang': BOHR_TO_ANGSTROM / HARTREE_TO_KJ_MOL,
}


def _norm(unit: str) -> str:
    return unit.strip().lower().replace(' ', '')


def canonical_energy_unit(unit: str) -> str:
    u = _norm(unit)
    mapping = {
        'hartree': 'Hartree', 'ha': 'Hartree',
        'kcal/mol': 'kcal/mol', 'kcalmol': 'kcal/mol',
        'kj/mol': 'kJ/mol', 'kjmol': 'kJ/mol',
    }
    if u not in mapping:
        raise ValueError(f'Unsupported energy unit: {unit!r}')
    return mapping[u]


def canonical_length_unit(unit: str) -> str:
    u = _norm(unit)
    mapping = {
        'bohr': 'bohr', 'a0': 'bohr',
        'angstrom': 'Angstrom', 'ang': 'Angstrom', 'a': 'Angstrom',
        'nm': 'nm',
    }
    if u not in mapping:
        raise ValueError(f'Unsupported length unit: {unit!r}')
    return mapping[u]


def canonical_gradient_unit(unit: str) -> str:
    u = _norm(unit)
    mapping = {
        'hartree/bohr': 'Hartree/bohr', 'ha/bohr': 'Hartree/bohr',
        'kcal/mol/angstrom': 'kcal/mol/Angstrom',
        'kcal/mol/a': 'kcal/mol/Angstrom',
        'kcal/mol/ang': 'kcal/mol/Angstrom',
        'kj/mol/angstrom': 'kJ/mol/Angstrom',
        'kj/mol/a': 'kJ/mol/Angstrom',
        'kj/mol/ang': 'kJ/mol/Angstrom',
    }
    if u not in mapping:
        raise ValueError(f'Unsupported gradient unit: {unit!r}')
    return mapping[u]


def convert_energy(values: Any, from_unit: str, to_unit: str) -> np.ndarray:
    fu = _norm(from_unit)
    tu = _norm(to_unit)
    if fu == tu:
        return np.asarray(values) if isinstance(values, (list, tuple, np.ndarray)) else values
    if fu not in _ENERGY_FACTORS_TO_HARTREE or tu not in _ENERGY_FACTORS_TO_HARTREE:
        raise ValueError(f'Unsupported energy conversion: {from_unit!r} -> {to_unit!r}')
    arr = np.asarray(values, dtype=float)
    return arr * _ENERGY_FACTORS_TO_HARTREE[fu] / _ENERGY_FACTORS_TO_HARTREE[tu]


def convert_length(values: Any, from_unit: str, to_unit: str) -> np.ndarray:
    fu = _norm(from_unit)
    tu = _norm(to_unit)
    if fu == tu:
        return np.asarray(values) if isinstance(values, (list, tuple, np.ndarray)) else values
    if fu not in _LENGTH_FACTORS_TO_BOHR or tu not in _LENGTH_FACTORS_TO_BOHR:
        raise ValueError(f'Unsupported length conversion: {from_unit!r} -> {to_unit!r}')
    arr = np.asarray(values, dtype=float)
    return arr * _LENGTH_FACTORS_TO_BOHR[fu] / _LENGTH_FACTORS_TO_BOHR[tu]


def convert_gradient(values: Any, from_unit: str, to_unit: str) -> np.ndarray:
    fu = _norm(from_unit)
    tu = _norm(to_unit)
    if fu == tu:
        return np.asarray(values) if isinstance(values, (list, tuple, np.ndarray)) else values
    if fu not in _GRADIENT_FACTORS_TO_HARTREE_PER_BOHR or tu not in _GRADIENT_FACTORS_TO_HARTREE_PER_BOHR:
        raise ValueError(f'Unsupported gradient conversion: {from_unit!r} -> {to_unit!r}')
    arr = np.asarray(values, dtype=float)
    return arr * _GRADIENT_FACTORS_TO_HARTREE_PER_BOHR[fu] / _GRADIENT_FACTORS_TO_HARTREE_PER_BOHR[tu]


# Convenience wrappers --------------------------------------------------------

def hartree_to_kcal(values: Any) -> np.ndarray:
    return convert_energy(values, 'Hartree', 'kcal/mol')


def kcal_to_hartree(values: Any) -> np.ndarray:
    return convert_energy(values, 'kcal/mol', 'Hartree')


def gradient_hartree_bohr_to_kcal_angstrom(values: Any) -> np.ndarray:
    return convert_gradient(values, 'Hartree/bohr', 'kcal/mol/Angstrom')


def gradient_kcal_angstrom_to_hartree_bohr(values: Any) -> np.ndarray:
    return convert_gradient(values, 'kcal/mol/Angstrom', 'Hartree/bohr')


__all__ = [
    'PI', 'TWO_PI', 'C', 'MU0', 'EPSILON0', 'H', 'HBAR', 'E', 'M_E', 'N_A', 'A0',
    'HARTREE_TO_EV', 'HARTREE_TO_KCAL_MOL', 'HARTREE_TO_KJ_MOL',
    'BOHR_TO_ANGSTROM', 'ANGSTROM_TO_BOHR',
    'canonical_energy_unit', 'canonical_length_unit', 'canonical_gradient_unit',
    'convert_energy', 'convert_length', 'convert_gradient',
    'hartree_to_kcal', 'kcal_to_hartree',
    'gradient_hartree_bohr_to_kcal_angstrom', 'gradient_kcal_angstrom_to_hartree_bohr',
]
