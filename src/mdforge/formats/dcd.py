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


def read_dcd(path: str | Path, *, max_frames: int | None = None) -> DCDTrajectory:
    """Read a DCD trajectory into a :class:`DCDTrajectory`.

    Parameters
    ----------
    path:
        DCD file path.
    max_frames:
        Read at most this many frames (default: all present).
    """
    data = Path(path).read_bytes()
    bo = _detect_endianness(data)
    i4 = bo + "i"
    pos = 0

    def read_record() -> bytes:
        nonlocal pos
        (n,) = struct.unpack_from(i4, data, pos)
        pos += 4
        payload = data[pos:pos + n]
        pos += n
        (n2,) = struct.unpack_from(i4, data, pos)
        pos += 4
        if n2 != n:
            raise ValueError(f"corrupt DCD record ({n} != {n2}) at byte {pos}")
        return payload

    # --- header ---------------------------------------------------------
    header = read_record()
    if header[:4] != b"CORD":
        raise ValueError("not a coordinate DCD (missing 'CORD' magic)")
    icntrl = struct.unpack(bo + "20i", header[4:84])
    nset = icntrl[0]
    charmm = icntrl[19] != 0
    has_cell = icntrl[10] != 0
    # DELTA is a float32 for CHARMM (word 9), a float64 otherwise (words 9-10).
    if charmm:
        (delta,) = struct.unpack_from(bo + "f", header, 4 + 9 * 4)
    else:
        (delta,) = struct.unpack_from(bo + "d", header, 4 + 9 * 4)

    # --- title (consumed but unused) ------------------------------------
    read_record()

    # --- natom ----------------------------------------------------------
    natom_rec = read_record()
    (natom,) = struct.unpack_from(i4, natom_rec, 0)

    f4 = np.dtype(bo + "f4")
    n_want = nset if max_frames is None else min(nset, max_frames)

    coords: list[np.ndarray] = []
    boxes: list[list[float]] = []
    for _ in range(n_want):
        if pos >= len(data):
            break  # file truncated (still-writing run): stop at what we have
        if has_cell:
            cell = struct.unpack(bo + "6d", read_record())
            a, b, c = cell[0], cell[2], cell[5]
            gamma = _angle_from_slot(cell[1])
            beta = _angle_from_slot(cell[3])
            alpha = _angle_from_slot(cell[4])
            boxes.append([a, b, c, alpha, beta, gamma])
        x = np.frombuffer(read_record(), dtype=f4, count=natom)
        y = np.frombuffer(read_record(), dtype=f4, count=natom)
        z = np.frombuffer(read_record(), dtype=f4, count=natom)
        coords.append(np.stack([x, y, z], axis=1).astype(float))

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
