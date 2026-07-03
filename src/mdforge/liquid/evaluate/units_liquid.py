"""Strict unit alignment for bulk liquid properties (scoring boundary).

``core.units`` covers energy/length/gradient for the QM side; bulk-property
units (density, compressibility, diffusion, thermal expansion) are liquid-
specific and live here. The converter is deliberately **strict** — an
unrecognised unit raises rather than silently passing a wrong number. That
guards against exactly the kg/m³-vs-g/cm³ mix-up that leaked a density of
"1.178" (kg/m³) into a campaign ``results.json``.
"""

from __future__ import annotations


class UnitError(ValueError):
    """Raised when a unit is unrecognised or has no known conversion."""


def _norm(unit: str | None) -> str:
    """Canonicalise a unit string for lookup (lower-case, strip decoration)."""
    if unit is None:
        return "dimensionless"
    u = str(unit).strip().lower()
    for a, b in (("^", ""), ("·", " "), ("*", " "), ("å", "angstrom"),
                 ("ang", "angstrom"), ("angstromstrom", "angstrom")):
        u = u.replace(a, b)
    u = " ".join(u.split())  # collapse whitespace
    aliases = {
        "": "dimensionless", "-": "dimensionless", "none": "dimensionless",
        "a": "angstrom",
        "cm2/s": "cm2/s", "1e-5 cm2/s": "1e-5 cm2/s", "1e-9 m2/s": "1e-9 m2/s",
        "d": "debye",
    }
    return aliases.get(u, u)


# Conversion factors INTO each canonical reference unit:
#   value_in_ref_unit = value_in_from_unit * _FACTORS[ref][from]
_FACTORS: dict[str, dict[str, float]] = {
    "g/cm3": {"g/cm3": 1.0, "kg/m3": 1.0e-3},
    # 1e-6/bar == 1e-11/Pa numerically (1 bar = 1e5 Pa).
    "1e-6/bar": {"1e-6/bar": 1.0, "1e-11/pa": 1.0, "1/pa": 1.0e11, "1/bar": 1.0e6},
    # 1e-5 cm2/s == 1e-9 m2/s numerically.
    "1e-5 cm2/s": {"1e-5 cm2/s": 1.0, "cm2/s": 1.0e5, "1e-9 m2/s": 1.0, "m2/s": 1.0e9},
    "1e-4/k": {"1e-4/k": 1.0, "1/k": 1.0e4},
    "kcal/mol": {"kcal/mol": 1.0, "kj/mol": 1.0 / 4.184},
    "cal/mol/k": {"cal/mol/k": 1.0, "j/mol/k": 1.0 / 4.184},
    "dimensionless": {"dimensionless": 1.0},
    "angstrom": {"angstrom": 1.0, "nm": 10.0},
    "debye": {"debye": 1.0},
    "atm": {"atm": 1.0, "bar": 0.986923},
    "k": {"k": 1.0},
}


def convert(value: float, from_unit: str | None, to_unit: str | None,
            *, aliases: dict[str, dict] | None = None) -> float:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` (strict).

    ``aliases`` is an optional per-property map ``{from_unit: {"factor_to_ref": f}}``
    consulted before raising, so a property can declare unusual computed units.
    """
    f, t = _norm(from_unit), _norm(to_unit)
    if f == t:
        return float(value)
    table = _FACTORS.get(t)
    if table is not None and f in table:
        return float(value) * table[f]
    if aliases:
        norm_aliases = {_norm(k): v for k, v in aliases.items()}
        if f in norm_aliases and "factor_to_ref" in norm_aliases[f]:
            return float(value) * float(norm_aliases[f]["factor_to_ref"])
    raise UnitError(
        f"no conversion from {from_unit!r} to {to_unit!r} "
        f"(normalised {f!r} -> {t!r})"
    )


__all__ = ["convert", "UnitError"]
