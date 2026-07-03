"""Tinker ARC multi-frame trajectory read & write (goal c).

An ``.arc`` file is concatenated Tinker XYZ frames (each: count line, optional
periodic-box line, then atom lines). Consolidates the four ``ARC`` reimplementations
in the legacy code into one array-based reader. Pure text — no engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .txyz import TinkerXYZ, _as_lines, _classify, write_txyz


@dataclass
class ArcTrajectory:
    """A multi-frame Tinker trajectory.

    - ``coords``       : (M, N, 3) float
    - ``names``        : element/atom-name strings, length N (from frame 0)
    - ``types``        : (N,) int Tinker atom types, or None
    - ``connectivity`` : per-atom bonded indices from frame 0 ([] if none)
    - ``box``          : (M, 6) [a b c α β γ] per frame, or None
    - ``title``        : frame-0 title
    """

    coords: np.ndarray
    names: list[str]
    types: np.ndarray | None = None
    connectivity: list[list[int]] = field(default_factory=list)
    box: np.ndarray | None = None
    title: str = ""

    @property
    def n_frames(self) -> int:
        return self.coords.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.coords.shape[1]

    def frame(self, i: int) -> TinkerXYZ:
        """Return frame ``i`` as a single :class:`~mdforge.formats.txyz.TinkerXYZ`."""
        return TinkerXYZ(
            names=self.names,
            coords=self.coords[i],
            types=self.types,
            connectivity=self.connectivity,
            box=None if self.box is None else self.box[i],
            title=self.title,
        )

    def volume(self) -> np.ndarray | None:
        """Per-frame cell volume (Å³) for orthorhombic boxes, else None.

        Returns None when no box is present or any frame is non-orthorhombic.
        """
        if self.box is None:
            return None
        a, b, c = self.box[:, 0], self.box[:, 1], self.box[:, 2]
        angles = self.box[:, 3:6]
        if not np.allclose(angles, 90.0):
            return None
        return a * b * c


def count_frames(source: str | Path | list[str]) -> int:
    """Number of frames in an ARC/XYZ file (reads only the first frame's header)."""
    lines = _as_lines(source)
    nonblank = [ln for ln in lines if ln.strip() != ""]
    n_atoms = int(nonblank[0].split()[0])
    _, start, _ = _classify(nonblank)
    per_frame = start + n_atoms
    return len(nonblank) // per_frame


def read_arc(source: str | Path | list[str]) -> ArcTrajectory:
    """Read a multi-frame Tinker ARC/XYZ into an :class:`ArcTrajectory`."""
    lines = [ln for ln in _as_lines(source) if ln.strip() != ""]
    n_atoms = int(lines[0].split()[0])
    kind, start, has_box = _classify(lines)
    per_frame = start + n_atoms
    n_frames = len(lines) // per_frame

    names: list[str] = []
    types: list[int] = []
    connectivity: list[list[int]] = []
    coords = np.empty((n_frames, n_atoms, 3), dtype=float)
    boxes = np.empty((n_frames, 6), dtype=float) if has_box else None
    title = ""

    for f in range(n_frames):
        base = f * per_frame
        if has_box:
            boxes[f] = [float(x) for x in lines[base + 1].split()]
        elif f == 0 and kind == "tinker":
            rest = lines[base].split(maxsplit=1)
            title = rest[1] if len(rest) > 1 else ""
        elif f == 0 and kind == "raw":
            title = lines[base + 1]

        atom_lines = lines[base + start: base + start + n_atoms]
        for a, line in enumerate(atom_lines):
            s = line.split()
            if kind == "tinker":
                coords[f, a] = [float(s[2]), float(s[3]), float(s[4])]
                if f == 0:
                    names.append(s[1])
                    if len(s) > 5:
                        types.append(int(s[5]))
                        connectivity.append([int(b) for b in s[6:]])
                    else:
                        connectivity.append([])
            else:
                coords[f, a] = [float(s[1]), float(s[2]), float(s[3])]
                if f == 0:
                    names.append(s[0])

    return ArcTrajectory(
        coords=coords,
        names=names,
        types=np.array(types, dtype=int) if types else None,
        connectivity=connectivity,
        box=boxes,
        title=title,
    )


def write_arc(traj: ArcTrajectory, path: str | Path | None = None) -> str:
    """Serialize an :class:`ArcTrajectory` back to concatenated Tinker frames."""
    chunks = [write_txyz(traj.frame(i)) for i in range(traj.n_frames)]
    text = "".join(chunks)
    if path is not None:
        Path(path).write_text(text)
    return text


__all__ = ["ArcTrajectory", "read_arc", "write_arc", "count_frames"]
