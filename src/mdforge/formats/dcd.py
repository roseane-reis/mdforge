"""Read (and write) CHARMM/NAMD DCD trajectories with pure numpy — no mdtraj.

DCD is a binary, Fortran-"unformatted" trajectory format written by CHARMM,
NAMD, OpenMM, HOOMD (via its DCD writer) and others. It stores *coordinates
only* — atom identities/topology come from a companion PDB/PSF/Tinker-XYZ — so
this reader pairs with :mod:`mdforge.formats.pdb` / :mod:`~mdforge.formats.txyz`
for the O/H selection the liquid kernels need.

Layout (little- or big-endian; auto-detected from the leading record marker):

- **header record** (84 bytes): ``b"CORD"`` + 20 int32 control words. Word 0 is
  NSET (frame count), word 10 is DELTA, word 11 the unit-cell flag, word 19 the
  CHARMM-version marker.
- **title record**: int32 NTITLE then ``NTITLE`` 80-char lines.
- **natom record**: a single int32, NATOM.
- **per frame**: an optional 6×float64 unit-cell record (when the cell flag is
  set), then three float32 records X, Y, Z of length NATOM.

Each "record" is framed by a leading and trailing int32 byte-count (the Fortran
unformatted convention); we validate the pair and use the leading marker (== 84
for the header) to pick the byte order.

Pure parse half (file ⟂ compute): returns numpy arrays; the kernels never open a
DCD file.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DCDTrajectory:
    """Coordinates (and, if present, per-frame unit cell) from a DCD file.

    Units: Angstrom. ``box`` is ``(T, 6)`` ``[a, b, c, alpha, beta, gamma]``
    (lengths Å, angles degrees) or ``None`` when the file carries no unit cell.
    """

    coordinates: np.ndarray            # (T, N, 3) [Angstrom]
    box: np.ndarray | None = None      # (T, 6) [a b c alpha beta gamma] or None

    @property
    def n_frames(self) -> int:
        return self.coordinates.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.coordinates.shape[1]


def _detect_endianness(raw: bytes) -> str:
    """Return the struct byte-order prefix ('<' or '>') from the first marker.

    The first Fortran record (the DCD header) is always 84 bytes, so the leading
    int32 length marker reads as 84 in the file's native byte order.
    """
    if struct.unpack("<i", raw[:4])[0] == 84:
        return "<"
    if struct.unpack(">i", raw[:4])[0] == 84:
        return ">"
    raise ValueError("not a DCD file (leading record marker != 84)")


def _angle_from_slot(x: float) -> float:
    """Interpret a DCD unit-cell angle slot as degrees.

    Different writers store the angle as either its value in degrees or as its
    cosine (CHARMM). A magnitude ``<= 1`` is treated as a cosine.
    """
    if -1.0 <= x <= 1.0:
        return float(np.degrees(np.arccos(x)))
    return float(x)


def read_dcd(path: str | Path, *, max_frames: int | None = None,
             stride: int = 1) -> DCDTrajectory:
    """Read a DCD trajectory into a :class:`DCDTrajectory`.

    Parameters
    ----------
    path:
        DCD file path.
    max_frames:
        Read at most this many *kept* frames (after striding; default: all).
    stride:
        Keep every ``stride``-th frame (default 1 = every frame). Skipped frames
        are stepped over by byte arithmetic — their coordinates are never paged
        in — so sampling e.g. every 100th frame of a multi-GB trajectory is cheap.
        Use it to span a whole long run with a few hundred frames.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    path = Path(path)
    with open(path, "rb") as fh:
        data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return _read_dcd_mmap(data, path, max_frames, stride)
        finally:
            data.close()


def _read_dcd_mmap(data, path: Path, max_frames: int | None, stride: int = 1) -> DCDTrajectory:
    bo = _detect_endianness(data[:4])
    i4 = bo + "i"
    size = len(data)

    def read_record(pos: int):
        """Return ``(payload, next_pos)``, or ``(None, pos)`` at EOF/truncation."""
        if pos + 4 > size:
            return None, pos
        (n,) = struct.unpack_from(i4, data, pos)
        if n < 0 or pos + 8 + n > size:
            return None, pos
        payload = data[pos + 4:pos + 4 + n]
        (n2,) = struct.unpack_from(i4, data, pos + 4 + n)
        if n2 != n:
            return None, pos
        return payload, pos + 8 + n

    # --- header ---------------------------------------------------------
    header, pos = read_record(0)
    if header is None or header[:4] != b"CORD":
        raise ValueError("not a coordinate DCD (missing 'CORD' magic)")
    icntrl = struct.unpack(bo + "20i", header[4:84])
    nset = icntrl[0]
    has_cell = icntrl[10] != 0

    # --- title (consumed but unused) ------------------------------------
    _, pos = read_record(pos)

    # --- natom ----------------------------------------------------------
    natom_rec, pos = read_record(pos)
    if natom_rec is None:
        raise ValueError(f"truncated DCD {path} (no NATOM record)")
    (natom,) = struct.unpack_from(i4, natom_rec, 0)

    f4 = np.dtype(bo + "f4")
    # NSET (header frame count) is unreliable for streaming writers (e.g. Tinker9
    # leaves it 0); when non-positive, read frames until EOF instead of trusting it.
    if nset > 0:
        n_want = nset if max_frames is None else min(nset, max_frames)
    else:
        n_want = max_frames  # None => read to EOF

    coords: list[np.ndarray] = []
    boxes: list[list[float]] = []
    frame_bytes: int | None = None
    while (n_want is None or len(coords) < n_want) and pos < size:
        frame_start = pos
        cell_vals = None
        if has_cell:
            rec, pos = read_record(pos)
            if rec is None or len(rec) < 48:
                break
            cell = struct.unpack(bo + "6d", rec[:48])
            a, b, c = cell[0], cell[2], cell[5]
            cell_vals = [a, b, c, _angle_from_slot(cell[4]),
                         _angle_from_slot(cell[3]), _angle_from_slot(cell[1])]
        xr, pos = read_record(pos)
        yr, pos = read_record(pos)
        zr, pos = read_record(pos)
        if xr is None or yr is None or zr is None:
            break  # incomplete trailing frame (still-writing run): stop here
        x = np.frombuffer(xr, dtype=f4, count=natom)
        y = np.frombuffer(yr, dtype=f4, count=natom)
        z = np.frombuffer(zr, dtype=f4, count=natom)
        coords.append(np.stack([x, y, z], axis=1).astype(float))
        if cell_vals is not None:
            boxes.append(cell_vals)
        # every frame has identical byte length; skip (stride-1) of them by
        # pointer arithmetic so their coordinates are never read/paged in.
        if frame_bytes is None:
            frame_bytes = pos - frame_start
        if stride > 1:
            pos = frame_start + stride * frame_bytes

    if not coords:
        raise ValueError(f"DCD trajectory {path} has no coordinate frames")

    box_arr = np.asarray(boxes, dtype=float) if (has_cell and boxes) else None
    return DCDTrajectory(coordinates=np.asarray(coords, dtype=float), box=box_arr)


def write_dcd(
    coordinates,
    *,
    box=None,
    path: str | Path,
    title: str = "mdforge",
    delta: float = 1.0,
) -> Path:
    """Write coordinates (and optional unit cell) to a CHARMM-style DCD file.

    Provided mainly for round-trip testing of :func:`read_dcd`. ``coordinates``
    is ``(T, N, 3)`` in Å; ``box`` is an optional ``(T, 6)``
    ``[a, b, c, alpha, beta, gamma]`` (angles in degrees, stored as cosines in
    the CHARMM slot order ``[a, cos γ, b, cos β, cos α, c]``).
    """
    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coordinates must have shape (T, N, 3)")
    T, N, _ = coords.shape
    has_cell = box is not None
    if has_cell:
        box = np.asarray(box, dtype=float)

    bo = "<"

    def record(payload: bytes) -> bytes:
        n = len(payload)
        return struct.pack(bo + "i", n) + payload + struct.pack(bo + "i", n)

    icntrl = [0] * 20
    icntrl[0] = T
    icntrl[10] = 1 if has_cell else 0
    icntrl[19] = 24  # non-zero => CHARMM format (DELTA read as float32)
    header = b"CORD" + struct.pack(bo + "9i", *icntrl[:9]) \
        + struct.pack(bo + "f", float(delta)) + struct.pack(bo + "10i", *icntrl[10:])

    title_payload = struct.pack(bo + "i", 1) + title.encode("latin-1")[:80].ljust(80, b" ")

    out = bytearray()
    out += record(header)
    out += record(title_payload)
    out += record(struct.pack(bo + "i", N))
    for t in range(T):
        if has_cell:
            a, b, c, alpha, beta, gamma = box[t]
            slots = [a, np.cos(np.radians(gamma)), b,
                     np.cos(np.radians(beta)), np.cos(np.radians(alpha)), c]
            out += record(struct.pack(bo + "6d", *slots))
        out += record(coords[t, :, 0].tobytes())
        out += record(coords[t, :, 1].tobytes())
        out += record(coords[t, :, 2].tobytes())

    p = Path(path)
    p.write_bytes(bytes(out))
    return p


__all__ = [
    "DCDTrajectory",
    "read_dcd",
    "write_dcd",
]
