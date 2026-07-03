"""Molecule identity registry: integer ID ↔ name ↔ PubChem CID ↔ formula ↔ SMILES.

The parametrization database keys everything on an integer molecule ID
(``1 = chloroform``, …). This registry bridges that ID to the other identifiers,
consolidating the ``reference-data/database-info/*.pickle`` lookup tables
(``full_database.pickle`` id→[name, formula, CID, CID2], ``mw_mol.pickle`` id→MW,
``ncia_names_cid.pickle`` name→CID).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MoleculeIdentity:
    """Identity record for one molecule (any field may be None)."""

    molecule_id: int | None = None
    name: str | None = None
    formula: str | None = None
    cid: int | None = None
    cid2: int | None = None
    smiles: str | None = None
    mw: float | None = None


class IdentityRegistry:
    """Bidirectional molecule-identity lookup (by ID, name, or CID)."""

    def __init__(self) -> None:
        self._by_id: dict[int, MoleculeIdentity] = {}
        self._by_name: dict[str, MoleculeIdentity] = {}
        self._by_cid: dict[int, MoleculeIdentity] = {}

    def __len__(self) -> int:
        return len(self._by_id) or len(self._by_name)

    def __iter__(self) -> Iterator[MoleculeIdentity]:
        seen = set()
        for d in (self._by_id, self._by_name):
            for rec in d.values():
                key = id(rec)
                if key not in seen:
                    seen.add(key)
                    yield rec

    def __repr__(self) -> str:
        return f"IdentityRegistry({len(self)} molecules)"

    def add(self, ident: MoleculeIdentity) -> MoleculeIdentity:
        """Index an identity record by whichever of id/name/cid it carries."""
        if ident.molecule_id is not None:
            self._by_id[int(ident.molecule_id)] = ident
        if ident.name:
            self._by_name[ident.name] = ident
        if ident.cid is not None:
            self._by_cid[int(ident.cid)] = ident
        return ident

    # -- lookups --
    def get(self, molecule_id: int) -> MoleculeIdentity | None:
        return self._by_id.get(int(molecule_id))

    def get_by_name(self, name: str) -> MoleculeIdentity | None:
        return self._by_name.get(name)

    def get_by_cid(self, cid: int) -> MoleculeIdentity | None:
        return self._by_cid.get(int(cid))

    def get_smiles(self, name: str) -> str | None:
        rec = self._by_name.get(name)
        return rec.smiles if rec else None

    def get_cid(self, name: str) -> int | None:
        rec = self._by_name.get(name)
        return rec.cid if rec else None

    def ids(self) -> list[int]:
        return sorted(self._by_id)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    # -- constructors --
    @classmethod
    def from_full_database(
        cls,
        full_database: dict,
        *,
        mw: dict | None = None,
        name_to_cid: dict | None = None,
    ) -> IdentityRegistry:
        """Build from a ``full_database`` dict (``id -> [name, formula, CID, CID2]``).

        Optional ``mw`` (``id -> g/mol``) and ``name_to_cid`` supplement the records.
        """
        reg = cls()
        for mid, row in full_database.items():
            row = list(row)
            name = row[0] if len(row) > 0 else None
            formula = row[1] if len(row) > 1 else None
            cid = row[2] if len(row) > 2 else None
            cid2 = row[3] if len(row) > 3 else None
            reg.add(MoleculeIdentity(
                molecule_id=int(mid), name=name, formula=formula,
                cid=int(cid) if cid not in (None, -1) else None,
                cid2=int(cid2) if cid2 not in (None, -1) else None,
                mw=(float(mw[mid]) if mw and mid in mw else None),
            ))
        if name_to_cid:
            for name, cid in name_to_cid.items():
                if name not in reg._by_name and cid is not None:
                    reg.add(MoleculeIdentity(name=name, cid=int(cid)))
        return reg

    @classmethod
    def from_pickle_dir(cls, directory: str | Path) -> IdentityRegistry:
        """Load from a ``database-info`` directory of identity pickles.

        Reads ``full_database.pickle`` (required), plus ``mw_mol.pickle`` and
        ``ncia_names_cid.pickle`` if present.
        """
        from .io import load_pickle

        directory = Path(directory)
        full_db_path = directory / "full_database.pickle"
        if not full_db_path.is_file():
            # tolerate the legacy misspelled filename
            alt = directory / "org_liq_databse.pickle"
            full_db_path = alt if alt.is_file() else full_db_path
        if not full_db_path.is_file():
            raise FileNotFoundError(f"No full_database.pickle in {directory}")

        full_database = load_pickle(full_db_path)
        mw = _try_load(directory / "mw_mol.pickle")
        name_to_cid = _try_load(directory / "ncia_names_cid.pickle")
        return cls.from_full_database(full_database, mw=mw, name_to_cid=name_to_cid)


def _try_load(path: Path):
    if not path.is_file():
        return None
    from .io import load_pickle
    try:
        return load_pickle(path)
    except Exception:
        return None


__all__ = ["MoleculeIdentity", "IdentityRegistry"]
