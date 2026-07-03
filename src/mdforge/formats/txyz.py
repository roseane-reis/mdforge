"""Tinker XYZ / TXYZ read & write (goal c).

Handles both Tinker-format files (``index name x y z type bond1 bond2 …`` with
an optional periodic-box line) and plain raw XYZ (``element x y z``). All
parsing is pure text — no engine, no subprocess.

Consolidates the several ``ARC``/``read_xyz_file``/``update_tinker_xyz`` copies
across the legacy code (``analyzetool.process``, ``analyzetool.prmedit``,
``prior internal tooling``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _isfloat(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _isint(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


def _as_lines(source: str | Path | list[str]) -> list[str]:
    if isinstance(source, list):
        return source
    text = str(source)
    if "\n" in text:
        return text.splitlines()
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"XYZ file does not exist: {source}")
    return path.read_text().splitlines()


@dataclass
class TinkerXYZ:
    """One frame of a Tinker (or raw) XYZ structure.

    - ``names``        : element/atom-name strings, length N
    - ``coords``       : (N, 3) float
    - ``types``        : (N,) int Tinker atom types, or None for raw XYZ
    - ``connectivity`` : per-atom list of 1-based bonded atom indices ([] if none)
    - ``box``          : (6,) [a b c α β γ] or None
    - ``title``        : optional comment / title
    """

    names: list[str]
    coords: np.ndarray
    types: np.ndarray | None = None
    connectivity: list[list[int]] = field(default_factory=list)
    box: np.ndarray | None = None
    title: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.names)

    @property
    def is_tinker(self) -> bool:
        return self.types is not None


def _classify(lines: list[str]) -> tuple[str, int, bool]:
    """Return (kind, atom_start_index, has_box).

    kind is 'tinker' or 'raw'. Tinker files have no comment line; raw XYZ has a
    comment at line 1 and atoms ('element x y z') from line 2.
    """
    t1 = lines[1].split()
    # Periodic box: exactly 6 tokens, all numeric.
    if len(t1) == 6 and all(_isfloat(x) for x in t1):
        return "tinker", 2, True
    # Tinker atom line: integer index, non-numeric name, then coords.
    if len(t1) >= 5 and _isint(t1[0]) and not _isfloat(t1[1]):
        return "tinker", 1, False
    # Otherwise raw XYZ (line 1 is a free-text comment).
    return "raw", 2, False


def read_txyz(source: str | Path | list[str]) -> TinkerXYZ:
    """Read a single-frame Tinker or raw XYZ into a :class:`TinkerXYZ`."""
    # Strip only leading/trailing blank lines — an interior blank line is a
    # (legitimately empty) raw-XYZ comment and must be preserved for _classify.
    lines = list(_as_lines(source))
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    n_atoms = int(lines[0].split()[0])
    kind, start, has_box = _classify(lines)

    box = None
    title = ""
    if has_box:
        box = np.array([float(x) for x in lines[1].split()], dtype=float)
    elif kind == "raw":
        # line[1] is the comment/title
        title = lines[1] if len(lines) > 1 else ""
    else:
        # Tinker file with no box: a title may follow the count on line 0.
        rest = lines[0].split(maxsplit=1)
        title = rest[1] if len(rest) > 1 else ""

    names: list[str] = []
    coords: list[list[float]] = []
    types: list[int] = []
    connectivity: list[list[int]] = []

    for line in lines[start:start + n_atoms]:
        s = line.split()
        if len(s) < 4:
            continue
        if kind == "tinker":
            # index name x y z type [bonds...]
            names.append(s[1])
            coords.append([float(s[2]), float(s[3]), float(s[4])])
            if len(s) > 5:
                types.append(int(s[5]))
                connectivity.append([int(b) for b in s[6:]])
            else:
                connectivity.append([])
        else:
            # element x y z
            names.append(s[0])
            coords.append([float(s[1]), float(s[2]), float(s[3])])

    return TinkerXYZ(
        names=names,
        coords=np.array(coords, dtype=float),
        types=np.array(types, dtype=int) if types else None,
        connectivity=connectivity,
        box=box,
        title=title,
    )


def write_txyz(xyz: TinkerXYZ, path: str | Path | None = None) -> str:
    """Serialize a :class:`TinkerXYZ`. Writes Tinker format if types are present."""
    n = xyz.n_atoms
    if xyz.is_tinker:
        # Tinker format: title (if any) shares the count line; no comment line.
        header = f"{n:6d}" + (f"  {xyz.title}" if xyz.title else "")
        lines = [header]
        if xyz.box is not None:
            lines.append("  " + " ".join(f"{v:11.6f}" for v in xyz.box))
    else:
        # Raw XYZ requires a dedicated comment line (even if empty).
        lines = [f"{n:6d}", xyz.title]

    for i in range(n):
        x, y, z = xyz.coords[i]
        if xyz.is_tinker:
            t = int(xyz.types[i])
            row = f"{i + 1:6d}  {xyz.names[i]:<3s}{x:12.6f}{y:12.6f}{z:12.6f}{t:6d}"
            if xyz.connectivity and i < len(xyz.connectivity):
                for b in xyz.connectivity[i]:
                    row += f"{int(b):6d}"
            lines.append(row)
        else:
            lines.append(f"{xyz.names[i]:<3s}{x:12.6f}{y:12.6f}{z:12.6f}")

    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text)
    return text


def update_coords(
    source: str | Path | list[str],
    new_coords: np.ndarray | None = None,
    *,
    type_map: dict[int, int] | None = None,
    path: str | Path | None = None,
) -> str:
    """Rewrite a Tinker XYZ with new coordinates and/or remapped atom types.

    Preserves the header, box line, and connectivity. Ported from
    ``classical.update_tinker_xyz`` with arg-list output instead of file side
    effects by default. ``new_coords`` is ``(N, 3)`` aligned to atom order.
    """
    xyz = read_txyz(source)
    if new_coords is not None:
        new_coords = np.asarray(new_coords, dtype=float)
        if new_coords.shape != xyz.coords.shape:
            raise ValueError(
                f"new_coords shape {new_coords.shape} != existing {xyz.coords.shape}"
            )
        xyz.coords = new_coords
    if type_map is not None and xyz.types is not None:
        xyz.types = np.array([type_map.get(int(t), int(t)) for t in xyz.types], dtype=int)
    return write_txyz(xyz, path=path)


def raw_to_txyz(
    source: str | Path | list[str],
    types: list[int] | np.ndarray,
    connectivity: list[list[int]],
    *,
    title: str = "",
    path: str | Path | None = None,
) -> str:
    """Convert a raw XYZ to a Tinker TXYZ given atom types and connectivity."""
    raw = read_txyz(source)
    if raw.is_tinker:
        raise ValueError("Source already looks like a Tinker XYZ")
    types = np.asarray(types, dtype=int)
    if len(types) != raw.n_atoms or len(connectivity) != raw.n_atoms:
        raise ValueError("types and connectivity must have one entry per atom")
    txyz = TinkerXYZ(
        names=raw.names, coords=raw.coords, types=types,
        connectivity=[list(map(int, c)) for c in connectivity],
        title=title or raw.title,
    )
    return write_txyz(txyz, path=path)


def txyz_to_raw(source: str | Path | list[str], *, path: str | Path | None = None) -> str:
    """Convert a Tinker TXYZ to a plain raw XYZ (element x y z)."""
    txyz = read_txyz(source)
    raw = TinkerXYZ(names=txyz.names, coords=txyz.coords, title=txyz.title or "")
    return write_txyz(raw, path=path)


__all__ = [
    "TinkerXYZ",
    "read_txyz",
    "write_txyz",
    "update_coords",
    "raw_to_txyz",
    "txyz_to_raw",
]
