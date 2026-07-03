"""I/O helpers for molecule records and model outputs.

Ported from prior internal tooling with these changes:
- legacy JSON loader renamed to load_center_json  (no proprietary name in public API)
- save_pickle rewritten for Python 3  (removed iteritems(), improved interface)
- longest_common_substring kept as a utility but not exported by default
"""

from __future__ import annotations

import contextlib
import json
import pickle
import sys
import types
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .units import BOHR_TO_ANGSTROM

# Legacy module paths that pickled a ``SpiceMolecule`` (or sibling records) before the
# mdforge consolidation. Old ``*.joblib`` files store the class by (module, qualname),
# so loading them requires those names to resolve to the current mdforge classes.
# NOTE: shims for a couple of private predecessor module names were dropped for the
# public release; extend via the ``extra_modules`` arg of ``legacy_record_shims`` if
# you need to load those older files (see dev notes / TODO).
_LEGACY_RECORD_MODULES: tuple[str, ...] = (
    "data_class",
    "data_class_fixed",
    "data_class_fixed_v2",
    "data_class_refactored",
    "data_class_refactored_v2",
)

# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

def longest_common_substring(str1: str, str2: str) -> str:
    """Return the longest common substring of two strings."""
    match = SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, outfn: str | Path) -> None:
    """Save an arbitrary Python object to a pickle file."""
    with open(outfn, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(filenm: str | Path) -> Any:
    """Load a pickle file and return its contents."""
    with open(filenm, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Center/atom JSON loader
# ---------------------------------------------------------------------------

def load_center_json(filename: str | Path) -> dict[str, Any]:
    """Load a center/atom mapping JSON file into sorted numpy arrays.

    Expected JSON structure::

        {
          "energy": float,
          "structure_file": str,          # optional
          "atoms":   [{"atom": int, "center": int,
                        "x": float, "y": float, "z": float}, ...],
          "centers": [{"center": int,
                        "x": float, "y": float, "z": float,
                        "fx": float, "fy": float, "fz": float,
                        "mx": float, "my": float, "mz": float}, ...]
        }

    Returns a dict with keys:
        coords, atom2center, center_coords,
        center_forces, center_torques, energy, structure_file
    """
    path = Path(filename)
    with path.open() as fh:
        data = json.load(fh)

    atoms = sorted(data["atoms"], key=lambda entry: entry["atom"])
    centers = sorted(data["centers"], key=lambda entry: entry["center"])

    return {
        "coords": np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float),
        "atom2center": np.array([a["center"] for a in atoms], dtype=int),
        "center_coords": np.array([[c["x"], c["y"], c["z"]] for c in centers], dtype=float),
        "center_forces": np.array([[c["fx"], c["fy"], c["fz"]] for c in centers], dtype=float),
        "center_torques": np.array([[c["mx"], c["my"], c["mz"]] for c in centers], dtype=float),
        "energy": float(data["energy"]),
        "structure_file": data.get("structure_file", ""),
    }


# ---------------------------------------------------------------------------
# Legacy-aware joblib loading
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def legacy_record_shims(extra_modules: tuple[str, ...] = ()):
    """Temporarily map legacy module paths → current mdforge record classes.

    Old ``.joblib`` files pickled ``SpiceMolecule`` from standalone modules
    (``data_class*``, …). Pickle records the
    class by ``(module, qualname)``; to unpickle those into the current
    :class:`mdforge.core.records.SpiceMolecule` we register shim modules that expose the
    current classes under the legacy names. Real, already-imported modules are left
    untouched, and every shim is removed on exit so there is no lasting global side effect.
    """
    from . import records as _records

    shim_attrs = {
        "SpiceMolecule": _records.SpiceMolecule,
        "Trajectory": _records.Trajectory,
        "BulkProperties": _records.BulkProperties,
        "apply_update": _records.apply_update,
    }
    installed: list[str] = []
    try:
        for name in (*_LEGACY_RECORD_MODULES, *extra_modules):
            if name in sys.modules:
                continue  # never clobber a genuinely importable module
            module = types.ModuleType(name)
            module.__dict__.update(shim_attrs)
            if "." not in name:
                module.__path__ = []  # allow `import name.sub` to resolve a sibling shim
            sys.modules[name] = module
            installed.append(name)
        yield
    finally:
        for name in installed:
            sys.modules.pop(name, None)


def load_record(path: str | Path) -> Any:
    """Load one ``.joblib`` record, transparently upgrading legacy-pickled objects.

    Tries a plain :func:`joblib.load` first; if the file references a legacy module
    (``ModuleNotFoundError``), retries under :func:`legacy_record_shims` so the object
    deserializes directly into the current mdforge classes.
    """
    path = Path(path)
    try:
        return joblib.load(path)
    except ModuleNotFoundError:
        with legacy_record_shims():
            return joblib.load(path)


def load_joblib_records(
    joblib_dir: str | Path | list[str | Path],
) -> dict[str, Any]:
    """Load all *.joblib files from one or more directories.

    Parameters
    ----------
    joblib_dir:
        A single directory path, or a list of directory paths.

    Returns
    -------
    dict mapping ``stem`` → loaded object (typically a ``SpiceMolecule``). Legacy-pickled
    records are upgraded transparently (see :func:`load_record`).
    """
    if not isinstance(joblib_dir, (list, np.ndarray)):
        joblib_dirs = [Path(joblib_dir)]
    else:
        joblib_dirs = [Path(p) for p in joblib_dir]

    files: list[Path] = []
    for directory in joblib_dirs:
        files.extend(sorted(directory.glob("*.joblib")))

    return {f.stem: load_record(f) for f in files}


# ---------------------------------------------------------------------------
# XYZ writer
# ---------------------------------------------------------------------------

def write_xyz_string(
    atomic_numbers: np.ndarray,
    conformations_bohr: np.ndarray,
    comment_template: str = "frame={frame}",
) -> str:
    """Produce a multi-frame XYZ string from atomic numbers and conformations in bohr.

    Parameters
    ----------
    atomic_numbers:
        1-D integer array of length N.
    conformations_bohr:
        Array of shape (M, N, 3) in bohr.
    comment_template:
        Format string with ``{frame}`` placeholder for the comment line.

    Returns
    -------
    A single concatenated XYZ string (all M frames).
    """
    atomic_numbers = np.asarray(atomic_numbers)
    conformations_bohr = np.asarray(conformations_bohr, dtype=float)
    symbols = [_atomic_number_to_symbol(z) for z in atomic_numbers]
    lines: list[str] = []
    for i, coords in enumerate(conformations_bohr):
        coords_ang = coords * BOHR_TO_ANGSTROM
        lines.append(str(len(atomic_numbers)))
        lines.append(comment_template.format(frame=i))
        for sym, (x, y, z) in zip(symbols, coords_ang):
            lines.append(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines) + "\n"


def _atomic_number_to_symbol(z: int) -> str:
    table = {
        1: "H",  2: "He", 3: "Li",  4: "Be",  5: "B",  6: "C",  7: "N",  8: "O",
        9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 18: "Ar", 19: "K",  20: "Ca",
        26: "Fe", 28: "Ni", 29: "Cu", 30: "Zn",
        34: "Se", 35: "Br", 53: "I",
    }
    return table.get(int(z), "X")


__all__ = [
    "load_center_json",
    "load_joblib_records",
    "load_record",
    "legacy_record_shims",
    "write_xyz_string",
    "load_pickle",
    "save_pickle",
    "longest_common_substring",
]
