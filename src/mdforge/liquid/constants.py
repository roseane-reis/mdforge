"""Constants for liquid-phase property analysis.

These values are kept **bit-for-bit identical** to the legacy ``analyzetool``
implementation (``liquid.py`` / ``get_pressure_tensor.py``) so the refactored
array kernels reproduce historical numbers exactly. Modern CODATA equivalents
live in :mod:`mdforge.core.units`; where the two differ it is noted below.
"""

from __future__ import annotations

# Gas constant in kcal/mol/K.  kT = R_KCAL * T.
R_KCAL = 1.9872036e-3

# Boltzmann constant in J/K (CODATA 2014; core.units uses the 2018 exact value
# 1.380649e-23 — the 9th-digit difference is below MD statistical noise).
KB_J = 1.38064852e-23

# Avogadro's number used by the legacy pressure-tensor normalisation.
# (core.units uses the 2018 exact 6.02214076e23.)
N_A_LEGACY = 6.02214129e23

# amu / Å³  →  g/cm³.  Equals 1e24 / N_A.  Legacy literal preserved.
AMU_A3_TO_G_CM3 = 1.6605387831627252

# Static-dielectric prefactor:  (1/3)·debye² / (kB · eps0 · nm³).
# Active value from legacy liquid.py (line 18).
DIELECTRIC_PREFACTOR = 30.3392945e3

__all__ = [
    "R_KCAL",
    "KB_J",
    "N_A_LEGACY",
    "AMU_A3_TO_G_CM3",
    "DIELECTRIC_PREFACTOR",
]
