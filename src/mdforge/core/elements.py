"""Element data: symbols, atomic numbers, and atomic masses.

Single source of truth for element tables, consolidating the copies that were
scattered across the legacy code (``analyzetool/process.py`` ``mw_elements``,
``prior internal tooling`` ``ATOMIC_MASSES``, etc.).

Provides both symbol-keyed and Z-indexed mass lookups:
- ``ELEMENT_MASSES``   : {symbol -> mass in amu}
- ``ATOMIC_MASSES``    : np.ndarray indexed by atomic number Z (amu)
- ``Z_TO_SYMBOL`` / ``SYMBOL_TO_Z`` : element-symbol ↔ atomic-number maps
- ``mass_of(symbol_or_z)`` : convenience lookup
"""

from __future__ import annotations

import numpy as np

# Atomic masses in amu (IUPAC standard atomic weights, conventional values).
ELEMENT_MASSES: dict[str, float] = {
    'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811,
    'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797,
    'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386, 'Si': 28.0855,
    'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983,
    'Ca': 40.078, 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961,
    'Mn': 54.938045, 'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546,
    'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.9216, 'Se': 78.96,
    'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
    'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07,
    'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818,
    'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.90447, 'Xe': 131.293,
    'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
    'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964,
    'Gd': 157.25, 'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259,
    'Tm': 168.93421, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479,
    'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084,
    'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
    'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176,
}

# Atomic number → symbol for Z = 1..86 (H..Rn) — covers all biological /
# organic / HIPPO-relevant elements and then some.
_SYMBOLS_BY_Z: list[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

Z_TO_SYMBOL: dict[int, str] = {z: sym for z, sym in enumerate(_SYMBOLS_BY_Z, start=1)}
SYMBOL_TO_Z: dict[str, int] = {sym: z for z, sym in Z_TO_SYMBOL.items()}

# Z-indexed mass array (index 0 unused / zero), built from the tables above.
_MAX_Z = len(_SYMBOLS_BY_Z)
ATOMIC_MASSES = np.zeros(_MAX_Z + 1, dtype=float)
for _z, _sym in Z_TO_SYMBOL.items():
    if _sym in ELEMENT_MASSES:
        ATOMIC_MASSES[_z] = ELEMENT_MASSES[_sym]


def mass_of(symbol_or_z: str | int) -> float:
    """Return the atomic mass (amu) for an element symbol or atomic number."""
    if isinstance(symbol_or_z, str):
        try:
            return ELEMENT_MASSES[symbol_or_z]
        except KeyError as exc:
            raise KeyError(f"No mass for element symbol {symbol_or_z!r}") from exc
    z = int(symbol_or_z)
    if 0 < z <= _MAX_Z and ATOMIC_MASSES[z] > 0:
        return float(ATOMIC_MASSES[z])
    raise KeyError(f"No mass for atomic number {z}")


__all__ = [
    "ELEMENT_MASSES",
    "ATOMIC_MASSES",
    "Z_TO_SYMBOL",
    "SYMBOL_TO_Z",
    "mass_of",
]
