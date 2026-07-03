"""PDB read & write, and PDB ↔ Tinker XYZ conversion (goal c).

★ Net-new for mdforge. The legacy ``PDB_tools.ipynb`` only manipulated PDBs via
mdtraj (box imaging, residue renaming); there was no standalone reader/writer.
This module parses/writes fixed-column PDB ``ATOM``/``HETATM``/``CRYST1`` records
with no mdtraj dependency, keeping the core install light. Box-imaging / make-whole
operations belong to Phase 5 (simulate) and may use mdtraj there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.elements import SYMBOL_TO_Z
from .txyz import TinkerXYZ, _as_lines


@dataclass
class PDBAtom:
    serial: int
    name: str
    res_name: str
    chain: str
    res_seq: int
    element: str
    x: float
    y: float
    z: float
    record: str = "ATOM"


@dataclass
class PDBStructure:
    atoms: list[PDBAtom] = field(default_factory=list)
    box: np.ndarray | None = None  # (6,) [a b c α β γ] from CRYST1
    title: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def coords(self) -> np.ndarray:
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=float)

    @property
    def elements(self) -> list[str]:
        return [a.element for a in self.atoms]

    @classmethod
    def from_arrays(
        cls,
        coords: np.ndarray,
        elements: list[str],
        *,
        res_name: str | Sequence[str] = "MOL",
        chain: str | Sequence[str] = "A",
        res_seq: int | Sequence[int] = 1,
        box: np.ndarray | None = None,
        title: str = "",
        names: list[str] | None = None,
    ) -> PDBStructure:
        """Build a :class:`PDBStructure` from a coords array + element symbols.

        ``coords`` is ``(N, 3)``; ``elements`` is length ``N`` (e.g. ``["O", "H", "H"]``).
        Atom names default to ``element + 1-based index`` when ``names`` is omitted.
        Removes the per-atom :class:`PDBAtom` boilerplate when feeding a structure
        from raw arrays.

        ``res_name``, ``chain`` and ``res_seq`` each take either a single value
        (applied to every atom) or a per-atom sequence of length ``N``. Pass a
        per-atom ``res_seq`` to keep distinct molecules in separate residues —
        center-based engines may infer molecular connectivity from residue grouping,
        so a single shared residue can fuse separate molecules.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3); got {coords.shape}")
        n = len(coords)
        if len(elements) != n:
            raise ValueError(f"elements has {len(elements)} entries; expected {n}")
        if names is not None and len(names) != n:
            raise ValueError(f"names has {len(names)} entries; expected {n}")
        res_names = _broadcast_per_atom(res_name, n, "res_name")
        chains = _broadcast_per_atom(chain, n, "chain")
        res_seqs = _broadcast_per_atom(res_seq, n, "res_seq")
        atoms: list[PDBAtom] = []
        for i, (xyz, raw_elem) in enumerate(zip(coords, elements)):
            s = str(raw_elem)
            elem = s.capitalize() if len(s) == 2 else s.upper()
            atoms.append(PDBAtom(
                serial=i + 1,
                name=names[i] if names is not None else f"{elem}{i + 1}",
                res_name=str(res_names[i]),
                chain=str(chains[i]),
                res_seq=int(res_seqs[i]),
                element=elem,
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
            ))
        return cls(atoms=atoms, box=box, title=title)


def _broadcast_per_atom(value, n: int, label: str) -> list:
    """Return a length-``n`` list: a scalar ``value`` repeated, or a sequence as-is.

    Strings count as scalars (broadcast), not as per-atom sequences.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        return [value] * n
    seq = list(value)
    if len(seq) != n:
        raise ValueError(f"{label} has {len(seq)} entries; expected {n}")
    return seq


def _element_from_name(name: str) -> str:
    """Best-effort element from a PDB atom name (e.g. 'C01' → 'C', 'CL2' → 'Cl')."""
    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return "X"
    # Two-letter element (e.g. Cl, Br) only if it matches a known symbol.
    two = letters[:2].capitalize()
    if len(letters) >= 2 and two in SYMBOL_TO_Z:
        return two
    return letters[0].upper()


def read_pdb(source: str | Path | list[str]) -> PDBStructure:
    """Parse a PDB file/text into a :class:`PDBStructure` (fixed-column)."""
    lines = _as_lines(source)
    struct = PDBStructure()
    for line in lines:
        rec = line[:6].strip()
        if rec == "CRYST1":
            try:
                struct.box = np.array([
                    float(line[6:15]), float(line[15:24]), float(line[24:33]),
                    float(line[33:40]), float(line[40:47]), float(line[47:54]),
                ], dtype=float)
            except ValueError:
                pass
        elif rec in ("ATOM", "HETATM"):
            name = line[12:16].strip()
            elem = line[76:78].strip()
            if not elem:
                elem = _element_from_name(name)
            else:
                elem = elem.capitalize() if len(elem) == 2 else elem.upper()
            struct.atoms.append(PDBAtom(
                serial=int(line[6:11]) if line[6:11].strip() else len(struct.atoms) + 1,
                name=name,
                res_name=line[17:20].strip() or "MOL",
                chain=line[21:22].strip() or "A",
                res_seq=int(line[22:26]) if line[22:26].strip() else 1,
                element=elem,
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                record=rec,
            ))
        elif rec == "TITLE":
            struct.title = line[10:].strip()
    return struct


def _format_atom(a: PDBAtom) -> str:
    # PDB convention: 1-letter elements get a leading space in the name field.
    name4 = a.name[:4] if len(a.name) >= 4 else f" {a.name:<3s}"
    return (
        f"{a.record:<6s}{a.serial:>5d} {name4}{'':1s}{a.res_name:>3s} "
        f"{a.chain:1s}{a.res_seq:>4d}{'':1s}   "
        f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}{1.0:6.2f}{0.0:6.2f}          {a.element:>2s}"
    )


def write_pdb(struct: PDBStructure, path: str | Path | None = None) -> str:
    """Serialize a :class:`PDBStructure` to PDB text."""
    lines: list[str] = []
    if struct.title:
        lines.append(f"TITLE     {struct.title}")
    if struct.box is not None:
        b = struct.box
        lines.append(
            f"CRYST1{b[0]:9.3f}{b[1]:9.3f}{b[2]:9.3f}"
            f"{b[3]:7.2f}{b[4]:7.2f}{b[5]:7.2f} P 1           1"
        )
    for a in struct.atoms:
        lines.append(_format_atom(a))
    lines.append("END")
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text)
    return text


def to_pdb_string(
    coords: np.ndarray,
    elements: list[str],
    *,
    res_name: str | Sequence[str] = "MOL",
    chain: str | Sequence[str] = "A",
    res_seq: int | Sequence[int] = 1,
    box: np.ndarray | None = None,
    title: str = "",
    names: list[str] | None = None,
) -> str:
    """Serialize a coords array + element symbols straight to PDB text.

    Convenience over ``write_pdb(PDBStructure.from_arrays(...))`` for callers that
    work from raw ``(N, 3)`` coordinates instead of an assembled structure. See
    :meth:`PDBStructure.from_arrays` for the per-atom ``res_name``/``chain``/``res_seq``
    broadcasting rules.
    """
    return write_pdb(
        PDBStructure.from_arrays(
            coords, elements, res_name=res_name, chain=chain, res_seq=res_seq,
            box=box, title=title, names=names,
        )
    )


def txyz_to_pdb(xyz: TinkerXYZ, *, res_name: str = "MOL", chain: str = "A") -> PDBStructure:
    """Convert a :class:`~mdforge.formats.txyz.TinkerXYZ` to a single-residue PDB.

    Element is taken from each atom name (digits stripped); atom names are made
    unique per element (C1, C2, H1, …).
    """
    counts: dict[str, int] = {}
    atoms: list[PDBAtom] = []
    for i, nm in enumerate(xyz.names):
        elem = _element_from_name(nm)
        counts[elem] = counts.get(elem, 0) + 1
        x, y, z = xyz.coords[i]
        atoms.append(PDBAtom(
            serial=i + 1, name=f"{elem}{counts[elem]}", res_name=res_name,
            chain=chain, res_seq=1, element=elem, x=x, y=y, z=z, record="HETATM",
        ))
    box = None
    if xyz.box is not None:
        box = np.asarray(xyz.box, dtype=float)
    return PDBStructure(atoms=atoms, box=box, title=xyz.title)


def pdb_to_txyz(struct: PDBStructure) -> TinkerXYZ:
    """Convert a :class:`PDBStructure` to a raw (typeless) Tinker XYZ.

    Atom types and connectivity are NOT inferred (use Tinker's ``xyzedit`` or a
    parameter assignment step for that); names are set to element symbols.
    """
    return TinkerXYZ(
        names=list(struct.elements),
        coords=struct.coords,
        box=None if struct.box is None else np.asarray(struct.box, dtype=float),
        title=struct.title,
    )


__all__ = [
    "PDBAtom",
    "PDBStructure",
    "read_pdb",
    "write_pdb",
    "to_pdb_string",
    "txyz_to_pdb",
    "pdb_to_txyz",
]
