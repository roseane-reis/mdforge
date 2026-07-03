"""Experimental bulk-phase property tables → :class:`BulkProperties` (goal d).

Ingests the ``reference-data/database-info`` bulk-property data — both the
consolidated pickles (``org_liq_list.pickle`` / ``org_liq_dict.pickle`` /
``molinfo_dict.pickle``) and the raw per-property CSVs (``org_liq_*.csv``).

Property vector order (after the molecule id), used throughout this project:
``[T, density, ΔHvap, dielectric, κT, αT, surface_tension]`` with ``-1`` = missing.

.. note::
   Values are stored verbatim from the source tables. Field names follow
   ``BulkProperties``, but the source ΔHvap column is reported in the data's own
   units (often kJ/mol despite the ``_kcal_mol`` suffix) and κT/αT carry the
   source's scale factors — convert at the point of use, not here.
"""

from __future__ import annotations

from pathlib import Path

from ..core.io import load_pickle
from ..core.records import BulkProperties

# index → BulkProperties field, for the 6 properties following T in a row.
_PROP_FIELDS = (
    "density_kg_m3",
    "delta_hvap_kcal_mol",
    "dielectric",
    "kappa_T",
    "alpha_T",
    "surface_tension_mN_m",
)


def bulk_properties_from_vector(
    vector, *, molecule_id: int | None = None, missing: float = -1.0
) -> BulkProperties:
    """Convert a ``[T, density, ΔHvap, diel, κT, αT, surf]`` row to ``BulkProperties``.

    Entries equal to ``missing`` (default ``-1``) or None become ``None``.
    """
    vector = list(vector)
    temperature = float(vector[0])
    kwargs = {}
    for field, value in zip(_PROP_FIELDS, vector[1:7]):
        kwargs[field] = None if (value is None or value == missing) else float(value)
    return BulkProperties(temperature_K=temperature, metadata={"molecule_id": molecule_id}, **kwargs)


def load_bulk_table(pickle_path: str | Path, *, missing: float = -1.0):
    """Load a consolidated bulk-property pickle into ``BulkProperties`` records.

    Handles the three shapes in ``database-info``:
    - ``org_liq_list.pickle`` — list of ``[id, T, …6props]`` → ``{id: BulkProperties}``
    - ``molinfo_dict.pickle`` — ``id -> [T, …6props]``     → ``{id: BulkProperties}``
    - ``org_liq_dict.pickle`` — ``id -> [[T, …6props], …]`` → ``{id: [BulkProperties, …]}``
      (one entry per measured temperature)
    """
    data = load_pickle(pickle_path)
    out: dict = {}
    if isinstance(data, list):
        for row in data:
            mid = int(row[0])
            out[mid] = bulk_properties_from_vector(row[1:8], molecule_id=mid, missing=missing)
        return out
    if isinstance(data, dict):
        for key, value in data.items():
            mid = int(key)
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                out[mid] = [bulk_properties_from_vector(r, molecule_id=mid, missing=missing)
                            for r in value]
            else:
                out[mid] = bulk_properties_from_vector(value, molecule_id=mid, missing=missing)
        return out
    raise TypeError(f"Unsupported bulk table type: {type(data).__name__}")


def _maybe_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def parse_org_liq_csv(path: str | Path) -> list[dict]:
    """Parse a raw ``org_liq_<property>.csv`` into structured rows.

    Each data row is ``id, name, T, exp, col5, [sim1, σ1, [sim2, σ2]]`` where
    ``col5`` is overloaded — a bracketed literature tag ``[N]`` *or* the
    experimental uncertainty (hence ragged 7/9-column rows). A leading ``..`` in
    the id field continues the previous molecule at a new temperature.

    Returns dicts: ``{id, name, T, exp, ref, sigma_exp, sims}`` where ``sims`` is
    a list of ``(value, sigma)`` model estimates.
    """
    rows: list[dict] = []
    last_id: int | None = None
    last_name: str | None = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if parts[0] in ("..", ""):  # continuation row (same molecule, next T)
            mid, name, rest = last_id, last_name, parts[1:]
        else:
            mid, name, rest = int(parts[0]), parts[1], parts[2:]
            last_id, last_name = mid, name
        if len(rest) < 2:
            continue
        temperature = _maybe_float(rest[0])
        exp = _maybe_float(rest[1])
        col5 = rest[2] if len(rest) > 2 else ""
        is_ref = col5.startswith("[")
        ref = col5 if is_ref else None
        sigma_exp = None if is_ref else _maybe_float(col5)
        tail = rest[3:]
        sims = [(_maybe_float(tail[i]), _maybe_float(tail[i + 1]))
                for i in range(0, len(tail) - 1, 2)]
        rows.append({"id": mid, "name": name, "T": temperature, "exp": exp,
                     "ref": ref, "sigma_exp": sigma_exp, "sims": sims})
    return rows


__all__ = ["bulk_properties_from_vector", "load_bulk_table", "parse_org_liq_csv"]
