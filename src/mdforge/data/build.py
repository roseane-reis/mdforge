"""Assemble the parametrization database (goal d).

Combines the identity layer (`core.identity`), the experimental bulk-phase
targets (`data.bulk`), and the QM dimer references (`data.dimers`) into one
queryable :class:`Database`. This is the regenerated ``full_database`` the fit
(Phase 6) consumes: per-molecule identity + bulk targets, and discoverable dimer
interaction-energy sets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..core.identity import IdentityRegistry, MoleculeIdentity
from .bulk import load_bulk_table
from .dimers import DimerSet, load_sapt_dimer


@dataclass
class Database:
    """Assembled parametrization database: identity + bulk targets."""

    identity: IdentityRegistry
    bulk: dict = field(default_factory=dict)   # molecule_id -> BulkProperties | list
    metadata: dict = field(default_factory=dict)

    def molecule(self, molecule_id: int) -> MoleculeIdentity | None:
        return self.identity.get(molecule_id)

    def bulk_properties(self, molecule_id: int):
        return self.bulk.get(int(molecule_id))

    def molecule_ids(self) -> list[int]:
        return sorted(self.bulk)

    def __repr__(self) -> str:
        return (f"Database(identity={len(self.identity)} molecules, "
                f"bulk={len(self.bulk)} entries)")


def load_reference_database(
    reference_data_dir: str | Path,
    *,
    bulk_pickle: str = "molinfo_dict.pickle",
) -> Database:
    """Load identity + bulk targets from a ``reference-data`` tree.

    Expects ``<reference_data_dir>/database-info/`` with ``full_database.pickle``
    and the bulk-property pickle (default ``molinfo_dict.pickle``).
    """
    dbinfo = Path(reference_data_dir) / "database-info"
    if not dbinfo.is_dir():
        raise FileNotFoundError(f"No database-info/ under {reference_data_dir}")
    identity = IdentityRegistry.from_pickle_dir(dbinfo)
    bulk = {}
    bulk_path = dbinfo / bulk_pickle
    if bulk_path.is_file():
        bulk = load_bulk_table(bulk_path)
    return Database(identity=identity, bulk=bulk,
                    metadata={"reference_data_dir": str(reference_data_dir),
                              "bulk_pickle": bulk_pickle})


def iter_s101x7(databases_dir: str | Path) -> Iterator[DimerSet]:
    """Yield every S101x7 dimer pair as a :class:`DimerSet`.

    Walks ``<databases_dir>/S101x7/*/*.npy`` (skipping the monomer-split files).
    """
    root = Path(databases_dir) / "S101x7"
    for npy in sorted(root.glob("*/*.npy")):
        if npy.stem.endswith("-mol1") or npy.stem.endswith("-mol2"):
            continue
        yield load_sapt_dimer(npy.with_suffix(""), source="S101x7")


__all__ = ["Database", "load_reference_database", "iter_s101x7"]
