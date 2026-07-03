"""mdforge.data — parametrization database access (goal d).

Ingests the reference data trees into records the fitter (Phase 6) consumes:
- **bulk** — experimental liquid properties → :class:`~mdforge.core.records.BulkProperties`
- **dimers** — QM interaction-energy sets (S101x7, qm-calc SAPT npy, DES370K,
  NCIA) → :class:`~mdforge.data.dimers.DimerSet` (compact ``[elst,exch,ind,disp,total]``)
- **build** — assemble identity + bulk into a queryable :class:`Database`

Identity (ID ↔ name ↔ CID ↔ formula) lives in :mod:`mdforge.core.identity`.
Engine-free, like the rest of the analysis surface — builds on ``core`` + ``formats``.
"""

from __future__ import annotations

from ..core.identity import IdentityRegistry, MoleculeIdentity
from . import build, bulk, des370k, dimers, matching
from .build import Database, iter_s101x7, load_reference_database
from .bulk import bulk_properties_from_vector, load_bulk_table, parse_org_liq_csv
from .des370k import (
    Des370kConformation,
    Des370kIndex,
    MonomerMatch,
    MonomerResolver,
    canonical_smiles,
    conformation_groups_from_desres,
    geometry_graph_hash,
    heavy_formula,
    parse_des370k_row,
    smiles_graph_hash,
    wl_graph_hash,
)
from .dimers import (
    SAPT_COMPONENTS,
    DimerSet,
    load_des370k_gid,
    load_s101x7_pair,
    load_sapt_dimer,
    parse_ncia_benchmark,
    project_des370k_components,
)
from .matching import (
    ConformationMatchResult,
    apply_atom_permutation_to_spicemolecule,
    apply_conformation_permutation_to_spicemolecule,
    build_mapping_diagnostics,
    match_conformations_dimer_no_query_smiles,
    monomer_rmsd_between_conformations,
    reorder_reference_spicemolecule_to_query,
    reorder_reference_spicemolecule_to_query_atom_order,
    run_alignment,
)

__all__ = [
    # submodules
    "bulk", "dimers", "build", "matching", "des370k",
    # identity (re-exported for convenience)
    "IdentityRegistry", "MoleculeIdentity",
    # bulk
    "bulk_properties_from_vector", "load_bulk_table", "parse_org_liq_csv",
    # dimers
    "SAPT_COMPONENTS", "DimerSet", "load_sapt_dimer", "load_s101x7_pair",
    "project_des370k_components", "load_des370k_gid", "parse_ncia_benchmark",
    # build
    "Database", "load_reference_database", "iter_s101x7",
    # matching (dimer atom + conformation matcher)
    "ConformationMatchResult", "match_conformations_dimer_no_query_smiles",
    "apply_atom_permutation_to_spicemolecule", "apply_conformation_permutation_to_spicemolecule",
    "reorder_reference_spicemolecule_to_query", "reorder_reference_spicemolecule_to_query_atom_order",
    "build_mapping_diagnostics", "monomer_rmsd_between_conformations", "run_alignment",
    # des370k (identity + dimer index + row parsing)
    "Des370kIndex", "Des370kConformation", "MonomerResolver", "MonomerMatch",
    "canonical_smiles", "wl_graph_hash", "geometry_graph_hash", "smiles_graph_hash",
    "parse_des370k_row", "conformation_groups_from_desres", "heavy_formula",
]
