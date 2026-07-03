"""Tests for mdforge.data.matching (dimer matcher) and mdforge.data.des370k helpers.

All fixtures are synthetic and self-contained — no external data tree needed. Tests
that need SMILES handling skip when rdkit is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdforge.core.records import SpiceMolecule
from mdforge.data import (
    Des370kIndex,
    apply_conformation_permutation_to_spicemolecule,
    conformation_groups_from_desres,
    geometry_graph_hash,
    heavy_formula,
    match_conformations_dimer_no_query_smiles,
    parse_des370k_row,
    reorder_reference_spicemolecule_to_query_atom_order,
    smiles_graph_hash,
    wl_graph_hash,
)

try:
    import rdkit  # noqa: F401
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

needs_rdkit = pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")


# ---------------------------------------------------------------------------
# Synthetic water-dimer fixture (rigid monomers, distinct rigid-body placements)
# ---------------------------------------------------------------------------

def _rot(axis, theta):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


# A rigid water monomer (O, H, H) in bohr-ish units; internal geometry fixed.
_WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def _make_water_dimer_reference(n_conf=5):
    rng = np.random.default_rng(0)
    confs = np.zeros((n_conf, 6, 3), dtype=np.float64)
    for i in range(n_conf):
        r1 = _rot(rng.normal(size=3), rng.uniform(0, np.pi))
        r2 = _rot(rng.normal(size=3), rng.uniform(0, np.pi))
        w1 = _WATER @ r1.T
        w2 = _WATER @ r2.T + np.array([3.0 + 0.5 * i, 0.4 * i, 0.0])  # separated, distinct
        confs[i, :3] = w1
        confs[i, 3:] = w2
    return SpiceMolecule(
        name="water-water", subset="DES370K",
        smiles="[H:2][O:1][H:3].[H:5][O:4][H:6]",
        atomic_numbers=np.array([8, 1, 1, 8, 1, 1]),
        conformations=confs,
        dft_total_energy=np.arange(n_conf, dtype=float),
        dft_total_gradient=np.zeros((n_conf, 6, 3)),
        formation_energy=np.zeros(n_conf),
    )


@needs_rdkit
def test_matcher_recovers_atom_and_conformation_permutation():
    ref = _make_water_dimer_reference(n_conf=5)
    n = ref.n_conformations

    # Build a query: permute atoms (swap the two H of each water) and shuffle confs.
    atom_perm_q_from_ref = np.array([0, 2, 1, 3, 5, 4])   # query atom k <- ref atom atom_perm_q_from_ref[k]
    conf_order = np.array([2, 0, 4, 1, 3])                 # query conf i <- ref conf conf_order[i]
    query = np.zeros_like(ref.conformations)
    for i in range(n):
        query[i] = ref.conformations[conf_order[i]][atom_perm_q_from_ref]
    query_atoms = ref.atomic_numbers[atom_perm_q_from_ref]

    result = match_conformations_dimer_no_query_smiles(
        query_coords=query, query_atoms=query_atoms, n_atoms_per_mol=[3, 3],
        reference_set=ref, conformation_match_mode="aligned", heavy_key_atoms=1,
    )
    # query_to_reference_conf_perm[i] = ref index that query i matches -> recovers conf_order
    np.testing.assert_array_equal(result.query_to_reference_conf_perm, conf_order)

    # Reorder reference into query atom + conf order; coords must align to query (near 0 RMSD).
    reordered = reorder_reference_spicemolecule_to_query_atom_order(ref, result)
    reordered = apply_conformation_permutation_to_spicemolecule(reordered, result.query_to_reference_conf_perm)
    max_dev = np.max(np.abs(reordered.conformations - query))
    assert max_dev < 1e-9, f"reordered reference does not reproduce query (max dev {max_dev:.2e})"


@needs_rdkit
def test_matcher_exact_mode_identity():
    ref = _make_water_dimer_reference(n_conf=4)
    # Query == reference (same atom order, same conf order) -> identity perms, exact match.
    result = match_conformations_dimer_no_query_smiles(
        query_coords=ref.conformations.copy(), query_atoms=ref.atomic_numbers,
        n_atoms_per_mol=[3, 3], reference_set=ref,
        conformation_match_mode="exact", key_decimals=6, exact_tol=1e-6,
    )
    np.testing.assert_array_equal(result.query_to_reference_conf_perm, np.arange(4))


# ---------------------------------------------------------------------------
# WL graph hash
# ---------------------------------------------------------------------------

def test_wl_hash_deterministic_and_isomorphism_invariant():
    # Two labelings of the same path graph C-C-O must hash identically.
    h1 = wl_graph_hash(["C", "C", "O"], [(0, 1), (1, 2)])
    h2 = wl_graph_hash(["O", "C", "C"], [(0, 1), (1, 2)])  # reversed order, same graph
    assert h1 == h2
    # A different connectivity (C-O-C) must hash differently.
    h3 = wl_graph_hash(["C", "O", "C"], [(0, 1), (1, 2)])
    assert h3 != h1


def test_geometry_hash_separates_isomers():
    # Square vs path of 4 carbons -> different graphs, different hashes.
    square = np.array([[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [0, 1.5, 0]], float)
    h_sq = geometry_graph_hash(["C", "C", "C", "C"], square, tol=0.45)
    chain = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]], float)
    h_ch = geometry_graph_hash(["C", "C", "C", "C"], chain, tol=0.45)
    assert h_sq != h_ch


@needs_rdkit
def test_geometry_and_smiles_hash_agree_for_water():
    # Water from geometry must hash the same as water from SMILES.
    h_geom = geometry_graph_hash(["O", "H", "H"], _WATER, tol=0.45)
    h_smi = smiles_graph_hash("O")
    assert h_geom == h_smi


# ---------------------------------------------------------------------------
# Des370kIndex
# ---------------------------------------------------------------------------

def test_des370k_index_dimer_lookup_is_symmetric():
    idx = Des370kIndex(
        dimers_map={241: {962: 12399}},   # benzene-water only stored one way
        gid_smiles={12399: [["c1ccccc1", 12, 0.0], ["O", 3, 0.0]]},
        gid_cid_info={12399: [241, 962, 1, 29]},
    )
    assert idx.dimer_gid(241, 962) == 12399
    assert idx.dimer_gid(962, 241) == 12399      # reverse order resolves
    assert idx.dimer_gid(241, 999) is None
    assert idx.gid_cids(12399) == (241, 962)
    assert idx.gid_monomer_natoms(12399) == (12, 3)
    assert idx.gid_monomer_smiles(12399) == ("c1ccccc1", "O")


# ---------------------------------------------------------------------------
# data_per_gid row parsing + DESRES grouping
# ---------------------------------------------------------------------------

def test_parse_des370k_row():
    # Minimal water-water-like row matching the documented column layout.
    cols = ["O", "O", "0", "0", "3", "3", "11056", "md_dimer", "1968145", "100",
            "x", "avqz"] + ["0.0"] * 40
    cols[-2] = "0 0 0 0.96 0 0 -0.24 0.93 0 3 0 0 3.96 0 0 2.76 0.93 0"
    cols[-1] = "O H H O H H\n"
    row = ",".join(cols)
    conf = parse_des370k_row(row)
    assert conf.gid == 11056
    assert conf.group == "md_dimer"
    assert conf.subgroup_id == 1968145
    assert conf.n_atoms_mol1 == 3 and conf.n_atoms_mol2 == 3
    assert conf.symbols == ["O", "H", "H", "O", "H", "H"]
    assert conf.coords_angstrom.shape == (6, 3)


def test_conformation_groups_from_desres():
    entry = {  # {sgid: [group, count]} laid out contiguously in order
        100: ["md_dimer", 3],
        200: ["md_dimer", 2],
        300: ["md_solvation", 4],
    }
    groups = conformation_groups_from_desres(entry)
    assert groups["md_dimer"] == [0, 1, 2, 3, 4]
    assert groups["md_solvation"] == [5, 6, 7, 8]


def test_heavy_formula():
    assert heavy_formula(["C", "C", "N", "H", "H", "H", "H"]) == "C2N1"
