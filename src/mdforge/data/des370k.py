"""DES370K (Donchev) database access helpers (goal d).

The published DES370K dimer database is keyed on integer PubChem **CIDs** and a per-pair
**group id** (``gid``), and ships its geometry scans as CSV rows. This module centralizes
the lookups needed to (a) **sort** SPICE DES370K records back into the original published
order and (b) **map** externally-modeled molecules/dimers onto DES370K records:

- :class:`Des370kIndex` — small in-memory index built from the published lookup pickles
  (``dimers_map``, ``gID_smiles``, ``gID_CID_info``): canonical-SMILES↔CID, and the
  symmetric ``(cid1, cid2) -> gid`` dimer lookup.
- :class:`MonomerResolver` — identify a molecule *from geometry alone* (element symbols +
  coordinates) against DES370K's monomers, using a bond-order-agnostic Weisfeiler–Lehman
  graph hash. Geometry sources (PDB/xyz) routinely carry wrong/zero-order bonds, so naive
  SMILES perception fails; matching element-labeled **connectivity** is robust and
  correctly separates same-formula isomers (e.g. pyrazole vs imidazole, furan vs THF).
- :func:`parse_des370k_row` — decode one ``data_per_gid`` conformation CSV row into
  coordinates (Å), element symbols, and the DESRES group/subgroup labels.

rdkit is a **lazy import** (only the SMILES→graph path needs it); the WL hash itself is
pure-Python so geometry resolution works without any optional dependency.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Covalent radii (Å, Cordero 2008, common subset) for distance-based bond perception.
COVALENT_RADII: dict[str, float] = {
    'H': 0.31, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'Na': 1.66, 'Mg': 1.41, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    'K': 2.03, 'Ca': 1.76, 'Se': 1.20, 'Br': 1.20, 'I': 1.39, 'Li': 1.28,
}
_DEFAULT_RADIUS = 0.77

ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 34: 'Se',
    35: 'Br', 53: 'I', 3: 'Li',
}

_RDKIT_MISSING_MSG = (
    "rdkit is required for SMILES handling here. "
    "Install it with: pip install 'mdforge[chem]'  (or  pip install rdkit)"
)


def _rdkit_chem():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_RDKIT_MISSING_MSG) from exc
    return Chem


def canonical_smiles(smiles: str) -> str | None:
    """Canonical SMILES via rdkit, or ``None`` if it cannot be parsed."""
    Chem = _rdkit_chem()
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


# ---------------------------------------------------------------------------
# Weisfeiler–Lehman graph hash (element-labeled, bond-order agnostic)
# ---------------------------------------------------------------------------

def _h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def wl_graph_hash(elements, edges, iterations: int = 5) -> str:
    """Deterministic Weisfeiler–Lehman hash of an element-labeled undirected graph.

    ``elements`` maps node index → element symbol; ``edges`` is an iterable of
    ``(i, j)`` index pairs. Bond orders are ignored — only connectivity + element
    labels matter — so the same molecule hashes identically whether built from a
    correct SMILES or from a geometry with imperfect bond perception.
    """
    n = len(elements)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        if i == j:
            continue
        adj[i].append(j)
        adj[j].append(i)
    labels = [_h(str(e)) for e in elements]
    for _ in range(iterations):
        labels = [_h(labels[i] + "|" + ",".join(sorted(labels[k] for k in adj[i]))) for i in range(n)]
    return _h("|".join(sorted(labels)))


def bonds_from_geometry(symbols, coords, tol: float = 0.45) -> list[tuple[int, int]]:
    """Perceive connectivity from 3D coordinates (bond if dist ≤ r_i + r_j + tol)."""
    coords = np.asarray(coords, dtype=float)
    n = len(symbols)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        ri = COVALENT_RADII.get(symbols[i], _DEFAULT_RADIUS)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(symbols[j], _DEFAULT_RADIUS)
            if float(np.linalg.norm(coords[i] - coords[j])) <= ri + rj + tol:
                edges.append((i, j))
    return edges


def geometry_graph_hash(symbols, coords, *, tol: float = 0.45, iterations: int = 5) -> str:
    """WL hash of a molecule from element symbols + coordinates (geometry-perceived bonds)."""
    return wl_graph_hash(list(symbols), bonds_from_geometry(symbols, coords, tol=tol), iterations=iterations)


def smiles_graph_hash(smiles: str, *, iterations: int = 5) -> str | None:
    """WL hash of a molecule from a SMILES string (rdkit topology, all H explicit)."""
    Chem = _rdkit_chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return wl_graph_hash(elements, edges, iterations=iterations)


# ---------------------------------------------------------------------------
# Monomer identity resolution (geometry → DES370K CID/SMILES)
# ---------------------------------------------------------------------------

@dataclass
class MonomerMatch:
    smiles: str | None
    cid: int | None
    source: str           # "gid_smiles" | "sdfs" | ...
    n_heavy: int


@dataclass
class MonomerResolver:
    """Resolve a molecule (element symbols + coordinates) to a DES370K monomer.

    Build with :meth:`from_des370k`; resolve with :meth:`resolve`. Matching is by
    WL graph hash, so it is bond-order-agnostic and isomer-safe.
    """

    by_hash: dict[str, MonomerMatch] = field(default_factory=dict)
    iterations: int = 5
    tol: float = 0.45

    def add_smiles(self, smiles: str, cid: int | None, source: str) -> None:
        key = smiles_graph_hash(smiles, iterations=self.iterations)
        if key is None:
            return
        cs = canonical_smiles(smiles)
        Chem = _rdkit_chem()
        m = Chem.MolFromSmiles(smiles)
        n_heavy = m.GetNumHeavyAtoms() if m is not None else 0
        existing = self.by_hash.get(key)
        # Prefer an entry that carries a CID over one that does not.
        if existing is None or (existing.cid is None and cid is not None):
            self.by_hash[key] = MonomerMatch(smiles=cs or smiles, cid=cid, source=source, n_heavy=n_heavy)

    @classmethod
    def from_des370k(
        cls,
        gid_smiles: dict,
        gid_cid_info: dict,
        *,
        extra_smiles: list[str] | None = None,
        iterations: int = 5,
        tol: float = 0.45,
    ) -> MonomerResolver:
        """Build from the published ``gID_smiles`` + ``gID_CID_info`` lookups.

        Each gid contributes its two monomers (SMILES + CID). ``extra_smiles`` (e.g.
        the SMILES-named SDF files in ``SDFS/``) supplement monomers that may not
        appear in the loaded dimer set; those get a CID only if a CID-bearing entry
        with the same graph already exists.
        """
        self = cls(iterations=iterations, tol=tol)
        for gid, frags in gid_smiles.items():
            info = gid_cid_info.get(gid)
            if not info:
                continue
            for frag, cid in ((frags[0], info[0]), (frags[1], info[1])):
                self.add_smiles(frag[0], int(cid), source="gid_smiles")
        for smi in (extra_smiles or []):
            self.add_smiles(smi, None, source="sdfs")
        return self

    def resolve(self, symbols, coords) -> MonomerMatch | None:
        """Return the DES370K monomer matching this geometry, or ``None``."""
        key = geometry_graph_hash(symbols, coords, tol=self.tol, iterations=self.iterations)
        return self.by_hash.get(key)


# ---------------------------------------------------------------------------
# Dimer index
# ---------------------------------------------------------------------------

@dataclass
class Des370kIndex:
    """In-memory index over the published DES370K lookup tables.

    - ``dimers_map``   : ``{cid1: {cid2: gid}}`` (symmetric in the source)
    - ``gid_smiles``   : ``{gid: [[smiles1, n1, ...], [smiles2, n2, ...]]}``
    - ``gid_cid_info`` : ``{gid: [cid1, cid2, ...]}``
    """

    dimers_map: dict
    gid_smiles: dict
    gid_cid_info: dict

    def dimer_gid(self, cid1: int, cid2: int) -> int | None:
        """Group id for the (cid1, cid2) dimer, trying both orderings; ``None`` if absent."""
        g = self.dimers_map.get(cid1, {}).get(cid2)
        if g is not None:
            return g
        return self.dimers_map.get(cid2, {}).get(cid1)

    def gid_cids(self, gid: int) -> tuple[int, int] | None:
        info = self.gid_cid_info.get(gid)
        return (int(info[0]), int(info[1])) if info else None

    def gid_monomer_smiles(self, gid: int) -> tuple[str, str] | None:
        frags = self.gid_smiles.get(gid)
        return (frags[0][0], frags[1][0]) if frags else None

    def gid_monomer_natoms(self, gid: int) -> tuple[int, int] | None:
        frags = self.gid_smiles.get(gid)
        return (int(frags[0][1]), int(frags[1][1])) if frags else None

    def monomer_resolver(self, *, extra_smiles: list[str] | None = None, iterations: int = 5, tol: float = 0.45) -> MonomerResolver:
        return MonomerResolver.from_des370k(
            self.gid_smiles, self.gid_cid_info,
            extra_smiles=extra_smiles, iterations=iterations, tol=tol,
        )

    @classmethod
    def from_pickles(
        cls,
        *,
        dimers_map_path,
        gid_smiles_path,
        gid_cid_info_path,
    ) -> Des370kIndex:
        """Load the three small lookup pickles from disk."""
        from ..core.io import load_pickle
        return cls(
            dimers_map=load_pickle(dimers_map_path),
            gid_smiles=load_pickle(gid_smiles_path),
            gid_cid_info=load_pickle(gid_cid_info_path),
        )


# ---------------------------------------------------------------------------
# data_per_gid CSV row parsing
# ---------------------------------------------------------------------------

# Column layout of a DES370K ``data_per_gid`` conformation row (comma-separated):
#   0,1   monomer-1 / monomer-2 short SMILES
#   4,5   monomer-1 / monomer-2 atom counts
#   6     group id (gid)
#   7     DESRES group label (md_dimer / qm_opt_dimer / md_solvation / md_nmer / ...)
#   8     DESRES subgroup id (sgid)
#   -2    flattened xyz coordinates (Å), whitespace-separated, row-major (N,3)
#   -1    element symbols, whitespace-separated
_COL_NATOMS1, _COL_NATOMS2 = 4, 5
_COL_GID, _COL_GROUP, _COL_SGID = 6, 7, 8


@dataclass
class Des370kConformation:
    coords_angstrom: np.ndarray   # (N, 3)
    symbols: list[str]            # length N
    group: str
    subgroup_id: int
    gid: int
    n_atoms_mol1: int
    n_atoms_mol2: int


def parse_des370k_row(row: str) -> Des370kConformation:
    """Decode one ``data_per_gid[gid][i]`` CSV row into a :class:`Des370kConformation`."""
    parts = row.split(",")
    symbols = parts[-1].split()
    coords = np.array([float(x) for x in parts[-2].split()], dtype=float).reshape(-1, 3)
    if len(symbols) != coords.shape[0]:
        raise ValueError(f"row atom/coord mismatch: {len(symbols)} symbols vs {coords.shape[0]} coords")
    return Des370kConformation(
        coords_angstrom=coords,
        symbols=symbols,
        group=parts[_COL_GROUP],
        subgroup_id=int(parts[_COL_SGID]),
        gid=int(parts[_COL_GID]),
        n_atoms_mol1=int(parts[_COL_NATOMS1]),
        n_atoms_mol2=int(parts[_COL_NATOMS2]),
    )


def conformation_groups_from_desres(desres_entry: dict) -> dict[str, list[int]]:
    """Map DESRES group label → sorted list of conformation indices.

    ``desres_entry`` is ``desres_dimer_info[cid][cid2] = {sgid: [group, count]}``.
    Conformations are laid out contiguously per subgroup, in the dict's insertion
    order — matching how ``data_per_gid[gid]`` rows are ordered.
    """
    groups: dict[str, list[int]] = {}
    start = 0
    for _sgid, (group, count) in desres_entry.items():
        groups.setdefault(group, []).extend(range(start, start + count))
        start += count
    return groups


def heavy_formula(symbols) -> str:
    """Hill-ish heavy-atom formula string (e.g. ``C3N2``) for quick auditing."""
    c = Counter(s for s in symbols if s != "H")
    return "".join(f"{el}{n}" for el, n in sorted(c.items()))


__all__ = [
    "COVALENT_RADII",
    "ATOMIC_NUMBER_TO_SYMBOL",
    "canonical_smiles",
    "wl_graph_hash",
    "bonds_from_geometry",
    "geometry_graph_hash",
    "smiles_graph_hash",
    "MonomerMatch",
    "MonomerResolver",
    "Des370kIndex",
    "Des370kConformation",
    "parse_des370k_row",
    "conformation_groups_from_desres",
    "heavy_formula",
]
