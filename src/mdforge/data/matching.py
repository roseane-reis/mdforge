"""Dimer atom + conformation matching (goal d).

Ported from ``prior internal tooling`` (the rdkit Kabsch/Hungarian
matcher) — the piece needed for **monomer↔dimer re-matching**: given a dimer geometry
scan in some arbitrary atom/conformation order (e.g. the original published DES370K
records) and a reference :class:`~mdforge.core.records.SpiceMolecule` for the same
dimer, recover the atom permutation and the conformation permutation that map one onto
the other, so per-conformation reference data (energies, gradients, model outputs) can
be aligned 1-to-1.

How it works
------------
1. The reference's two monomers are identified by splitting its (mapped or plain)
   SMILES into exactly two disconnected fragments (rdkit ``GetMolFrags``).
2. Per fragment, atoms are matched element-wise by an iterated Kabsch + Hungarian
   assignment on coordinates (``_match_atoms_one_fragment``), giving a full
   reference→query atom permutation. Both fragment orderings are tried when the
   monomers are size-compatible; the lower-RMSD assignment wins.
3. Conformations are matched either by global aligned-RMSD Hungarian assignment
   (``mode="aligned"``) or by exact coordinate equality with a heavy-atom hash key
   (``mode="exact"``).

Design notes
------------
- **rdkit is a lazy import.** Only the SMILES-fragment helpers need it; importing
  ``mdforge.data.matching`` (and ``mdforge.data``) works without rdkit installed.
  Install it via the ``[chem]`` extra.
- ``scipy.optimize.linear_sum_assignment`` (a core dependency) does the Hungarian step.
- The matcher compares **coordinates + element symbols only** — it never needs correct
  bond orders, so geometry-only sources (PDB/xyz scans) match fine against the
  chemically-correct reference SMILES.
- Fixes vs the legacy source: ``run_alignment`` called ``reorder_query_arrays_to_reference``
  with non-existent ``query_coords``/``extra_arrays`` kwargs — corrected here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

try:  # progress bar is optional
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Uppercase element symbols: the matcher compares fragment compositions as strings,
# so a self-consistent uppercase table is all it needs (case is normalized in _to_symbols).
ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: 'H',  2: 'HE', 3: 'LI', 4: 'BE', 5: 'B',  6: 'C',  7: 'N',  8: 'O',
    9: 'F',  10: 'NE', 11: 'NA', 12: 'MG', 13: 'AL', 14: 'SI', 15: 'P',
    16: 'S', 17: 'CL', 18: 'AR', 19: 'K',  20: 'CA', 26: 'FE', 28: 'NI',
    29: 'CU', 30: 'ZN', 34: 'SE', 35: 'BR', 53: 'I',
}

_RDKIT_MISSING_MSG = (
    "rdkit is required for SMILES-based dimer fragment splitting. "
    "Install it with: pip install 'mdforge[chem]'  (or  pip install rdkit)"
)


def _rdkit():
    """Lazy rdkit import with a friendly error (keeps mdforge.data import-light)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdmolfiles
    except ImportError as exc:  # pragma: no cover - exercised only without rdkit
        raise ImportError(_RDKIT_MISSING_MSG) from exc
    return Chem, rdmolfiles


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FragmentInfo:
    frag1_idx: np.ndarray
    frag2_idx: np.ndarray
    source: str


@dataclass
class FragmentedAtomMapResult:
    atom_perm_reference_to_query: np.ndarray
    fragment_assignment: tuple[int, int]
    rmsd_anchor: float
    reference_fragment_source: str


@dataclass
class ConformationMatchResult:
    query_to_reference_conf_perm: np.ndarray
    pair_rmsd: np.ndarray
    cost_matrix: np.ndarray | None
    atom_perm: np.ndarray
    fragment_assignment: tuple[int, int]
    anchor_rmsd: float
    reference_fragment_source: str


@dataclass
class MappingDiagnostics:
    query_fragment_sizes: tuple[int, int]
    reference_fragment_sizes: tuple[int, int]
    query_fragment_compositions: tuple[dict[str, int], dict[str, int]]
    reference_fragment_compositions: tuple[dict[str, int], dict[str, int]]
    chosen_fragment_assignment: tuple[int, int]
    anchor_rmsd: float
    reference_fragment_source: str


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def _to_symbols(atoms) -> np.ndarray:
    """Coerce atomic numbers or symbol strings to an uppercase symbol array."""
    arr = np.asarray(atoms)
    if np.issubdtype(arr.dtype, np.integer):
        return np.array([ATOMIC_NUMBER_TO_SYMBOL[int(z)] for z in arr], dtype=object)
    return np.char.upper(arr.astype(str))


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - Pc @ R
    return R, t


def _apply_transform(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return P @ R + t


def _rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def _squared_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    AB = A @ B.T
    return AA + BB - 2.0 * AB


# ---------------------------------------------------------------------------
# SMILES fragment splitting (rdkit — lazy)
# ---------------------------------------------------------------------------

def _mol_from_smiles_keep_hs(smiles: str):
    Chem, rdmolfiles = _rdkit()
    params = rdmolfiles.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def _mapped_smiles_matches_reference(reference_smiles: str, reference_atomic_numbers) -> bool:
    try:
        mol = _mol_from_smiles_keep_hs(reference_smiles)
    except Exception:
        return False
    ref_z = np.asarray(reference_atomic_numbers, dtype=int)
    mapped_z = np.full(len(ref_z), -1, dtype=int)
    for atom in mol.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap <= 0 or amap > len(ref_z):
            return False
        mapped_z[amap - 1] = atom.GetAtomicNum()
    return np.all(mapped_z > 0) and np.array_equal(mapped_z, ref_z)


def _split_reference_fragments_from_mapped_smiles(reference_smiles: str) -> tuple[np.ndarray, np.ndarray]:
    Chem, _ = _rdkit()
    mol = _mol_from_smiles_keep_hs(reference_smiles)
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    if len(frags) != 2:
        raise ValueError(f"Expected exactly 2 disconnected fragments, got {len(frags)}")
    frag_arrays = []
    for frag in frags:
        frag_ref_idx = []
        for atom_idx in frag:
            atom = mol.GetAtomWithIdx(atom_idx)
            amap = atom.GetAtomMapNum()
            if amap <= 0:
                raise ValueError("All atoms must have positive atom-map numbers")
            frag_ref_idx.append(amap - 1)
        frag_arrays.append(np.array(frag_ref_idx, dtype=int))
    return frag_arrays[0], frag_arrays[1]


def _split_reference_fragments_from_smiles(reference_smiles: str, reference_atomic_numbers) -> tuple[np.ndarray, np.ndarray]:
    Chem, _ = _rdkit()
    mol = _mol_from_smiles_keep_hs(reference_smiles)
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    if len(frags) != 2:
        raise ValueError(f"Expected exactly 2 disconnected fragments, got {len(frags)}")
    smiles_atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    ref_atomic_numbers = np.asarray(reference_atomic_numbers, dtype=int)
    if not np.array_equal(smiles_atomic_numbers, ref_atomic_numbers):
        raise ValueError(
            "Regular SMILES fallback is unsafe because RDKit atom order does not match "
            "reference_atomic_numbers. Use mapped SMILES or a reference-order-consistent SMILES."
        )
    return np.array(frags[0], dtype=int), np.array(frags[1], dtype=int)


def _get_reference_fragment_indices(reference_smiles: str, reference_atomic_numbers, prefer_mapped_smiles: bool = True) -> FragmentInfo:
    if prefer_mapped_smiles and _mapped_smiles_matches_reference(reference_smiles, reference_atomic_numbers):
        f1, f2 = _split_reference_fragments_from_mapped_smiles(reference_smiles)
        return FragmentInfo(f1, f2, "mapped_smiles")
    f1, f2 = _split_reference_fragments_from_smiles(reference_smiles, reference_atomic_numbers)
    return FragmentInfo(f1, f2, "regular_smiles")


# ---------------------------------------------------------------------------
# Atom matching (one fragment / full dimer)
# ---------------------------------------------------------------------------

def _match_atoms_one_fragment(
    query_coords: np.ndarray,
    query_atoms,
    ref_coords: np.ndarray,
    ref_atoms,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    q = np.asarray(query_coords, dtype=float)
    r = np.asarray(ref_coords, dtype=float)
    qsym = _to_symbols(query_atoms)
    rsym = _to_symbols(ref_atoms)
    if q.shape != r.shape:
        raise ValueError(f"Fragment coordinate shapes do not match: {q.shape} vs {r.shape}")
    uq, cq = np.unique(qsym, return_counts=True)
    ur, cr = np.unique(rsym, return_counts=True)
    if not (np.array_equal(uq, ur) and np.array_equal(cq, cr)):
        raise ValueError("Fragment element compositions do not match")
    q0 = q - q.mean(axis=0)
    r0 = r - r.mean(axis=0)
    perm = np.empty(len(qsym), dtype=int)
    for elem in uq:
        q_idx = np.where(qsym == elem)[0]
        r_idx = np.where(rsym == elem)[0]
        cost = _squared_distance_matrix(q0[q_idx], r0[r_idx])
        row_ind, col_ind = linear_sum_assignment(cost)
        perm[r_idx[col_ind]] = q_idx[row_ind]
    prev_rmsd = np.inf
    for _ in range(max_iter):
        q_reordered = q[perm]
        R, t = _kabsch(q_reordered, r)
        q_aligned_all = _apply_transform(q, R, t)
        new_perm = np.empty_like(perm)
        for elem in uq:
            q_idx = np.where(qsym == elem)[0]
            r_idx = np.where(rsym == elem)[0]
            cost = _squared_distance_matrix(q_aligned_all[q_idx], r[r_idx])
            row_ind, col_ind = linear_sum_assignment(cost)
            new_perm[r_idx[col_ind]] = q_idx[row_ind]
        q_reordered = q[new_perm]
        R, t = _kabsch(q_reordered, r)
        curr_rmsd = _rmsd(_apply_transform(q_reordered, R, t), r)
        if np.array_equal(new_perm, perm) or abs(prev_rmsd - curr_rmsd) < tol:
            perm = new_perm
            break
        perm = new_perm
        prev_rmsd = curr_rmsd
    q_reordered = q[perm]
    R, t = _kabsch(q_reordered, r)
    rmsd = _rmsd(_apply_transform(q_reordered, R, t), r)
    return perm, rmsd


def get_fixed_dimer_atom_map_without_query_smiles(
    query_coords_anchor: np.ndarray,
    query_atoms,
    n_atoms_per_mol,
    reference_coords_anchor: np.ndarray,
    reference_atomic_numbers,
    reference_smiles: str,
    prefer_mapped_smiles: bool = True,
) -> FragmentedAtomMapResult:
    qcoords = np.asarray(query_coords_anchor, dtype=float)
    rcoords = np.asarray(reference_coords_anchor, dtype=float)
    qatoms = _to_symbols(query_atoms)
    ratoms = _to_symbols(reference_atomic_numbers)
    n1, n2 = map(int, n_atoms_per_mol)
    if n1 + n2 != len(qatoms):
        raise ValueError("n_atoms_per_mol does not sum to query atom count")
    q_frag1 = slice(0, n1)
    q_frag2 = slice(n1, n1 + n2)
    ref_frag_info = _get_reference_fragment_indices(reference_smiles, reference_atomic_numbers, prefer_mapped_smiles)
    r_frag1_idx, r_frag2_idx = ref_frag_info.frag1_idx, ref_frag_info.frag2_idx
    candidates = []

    retry = True
    if len(r_frag1_idx) == n1 and len(r_frag2_idx) == n2:
        retry = False
        try:
            perm1, _ = _match_atoms_one_fragment(qcoords[q_frag1], qatoms[q_frag1], rcoords[r_frag1_idx], ratoms[r_frag1_idx])
            perm2, _ = _match_atoms_one_fragment(qcoords[q_frag2], qatoms[q_frag2], rcoords[r_frag2_idx], ratoms[r_frag2_idx])
            full_perm = np.empty(len(ratoms), dtype=int)
            full_perm[r_frag1_idx] = q_frag1.start + perm1
            full_perm[r_frag2_idx] = q_frag2.start + perm2
            q_reordered = qcoords[full_perm]
            R, t = _kabsch(q_reordered, rcoords)
            rmsd = _rmsd(_apply_transform(q_reordered, R, t), rcoords)
            candidates.append((full_perm, (0, 1), rmsd))
        except Exception:
            if n1 == n2:
                retry = True

    if len(r_frag2_idx) == n1 and len(r_frag1_idx) == n2 and retry:
        perm1, _ = _match_atoms_one_fragment(qcoords[q_frag1], qatoms[q_frag1], rcoords[r_frag2_idx], ratoms[r_frag2_idx])
        perm2, _ = _match_atoms_one_fragment(qcoords[q_frag2], qatoms[q_frag2], rcoords[r_frag1_idx], ratoms[r_frag1_idx])
        full_perm = np.empty(len(ratoms), dtype=int)
        full_perm[r_frag2_idx] = q_frag1.start + perm1
        full_perm[r_frag1_idx] = q_frag2.start + perm2
        q_reordered = qcoords[full_perm]
        R, t = _kabsch(q_reordered, rcoords)
        rmsd = _rmsd(_apply_transform(q_reordered, R, t), rcoords)
        candidates.append((full_perm, (1, 0), rmsd))
    if not candidates:
        raise ValueError("Could not match query monomer sizes/compositions to reference fragments")
    best_perm, best_assignment, best_rmsd = min(candidates, key=lambda x: x[2])
    return FragmentedAtomMapResult(best_perm, best_assignment, best_rmsd, ref_frag_info.source)


def reorder_query_atoms_to_reference(query_conformations: np.ndarray, atom_perm_reference_to_query: np.ndarray) -> np.ndarray:
    coords = np.asarray(query_conformations)
    if coords.ndim == 2:
        return coords[atom_perm_reference_to_query]
    if coords.ndim == 3:
        return coords[:, atom_perm_reference_to_query, :]
    raise ValueError("query_conformations must have shape (N,3) or (M,N,3)")


# ---------------------------------------------------------------------------
# Conformation matching
# ---------------------------------------------------------------------------

def _make_progress_iter(n: int, show_progress: bool, desc: str):
    if not show_progress:
        return range(n)
    if tqdm is not None:
        return tqdm(range(n), desc=desc)
    print(f"{desc}: starting {n} items")
    return range(n)


def _coords_key(coords: np.ndarray, atom_indices: np.ndarray, decimals: int) -> tuple[float, ...]:
    return tuple(np.round(coords[atom_indices], decimals=decimals).ravel().tolist())


def match_conformations_with_fixed_atom_order(
    query_conformations: np.ndarray,
    reference_conformations: np.ndarray,
    *,
    mode: str = "aligned",
    heavy_atom_indices: np.ndarray | None = None,
    heavy_key_atoms: int = 2,
    exact_tol: float = 1e-8,
    key_decimals: int = 8,
    show_progress: bool = False,
    progress_desc: str = "Matching conformations",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    query_conformations = np.asarray(query_conformations, dtype=float)
    reference_conformations = np.asarray(reference_conformations, dtype=float)
    Mq, Nq, _ = query_conformations.shape
    Mr, Nr, _ = reference_conformations.shape
    if Mq != Mr:
        raise ValueError(f"1-to-1 matching requires equal numbers of conformations; got {Mq} and {Mr}")
    if Nq != Nr:
        raise ValueError(f"Different number of atoms: {Nq} vs {Nr}")

    if mode == "aligned":
        cost = np.zeros((Mq, Mr), dtype=float)
        for i in _make_progress_iter(Mq, show_progress, progress_desc):
            for j in range(Mr):
                R, t = _kabsch(query_conformations[i], reference_conformations[j])
                q_aligned = _apply_transform(query_conformations[i], R, t)
                cost[i, j] = _rmsd(q_aligned, reference_conformations[j])
            if show_progress and tqdm is None:
                print(f"{progress_desc}: finished row {i + 1}/{Mq}")
        row_ind, col_ind = linear_sum_assignment(cost)
        conf_perm = np.empty(Mq, dtype=int)
        pair_rmsd = np.empty(Mq, dtype=float)
        for i, j in zip(row_ind, col_ind):
            conf_perm[i] = j
            pair_rmsd[i] = cost[i, j]
        return conf_perm, pair_rmsd, cost

    if mode != "exact":
        raise ValueError("mode must be 'aligned' or 'exact'")

    if heavy_atom_indices is None:
        heavy_atom_indices = np.arange(Nq, dtype=int)
    else:
        heavy_atom_indices = np.asarray(heavy_atom_indices, dtype=int)
    if len(heavy_atom_indices) == 0:
        heavy_atom_indices = np.arange(Nq, dtype=int)
    small_idx = heavy_atom_indices[: max(1, min(heavy_key_atoms, len(heavy_atom_indices)))]

    small_buckets: dict[tuple[float, ...], list[int]] = {}
    full_heavy_buckets: dict[tuple[float, ...], list[int]] = {}
    for j in range(Mr):
        small_buckets.setdefault(_coords_key(reference_conformations[j], small_idx, key_decimals), []).append(j)
        full_heavy_buckets.setdefault(_coords_key(reference_conformations[j], heavy_atom_indices, key_decimals), []).append(j)

    used_refs = set()
    conf_perm = np.full(Mq, -1, dtype=int)
    pair_rmsd = np.full(Mq, np.nan, dtype=float)

    for i in _make_progress_iter(Mq, show_progress, progress_desc):
        q_small = _coords_key(query_conformations[i], small_idx, key_decimals)
        candidates = small_buckets.get(q_small, [])
        if not candidates:
            raise ValueError(f"No reference candidates found for query conformation {i} using the small heavy-atom key")
        q_heavy = _coords_key(query_conformations[i], heavy_atom_indices, key_decimals)
        candidates = [j for j in full_heavy_buckets.get(q_heavy, []) if j in candidates]
        if not candidates:
            candidates = small_buckets[q_small]
        chosen = None
        for j in candidates:
            if j in used_refs:
                continue
            if np.allclose(query_conformations[i], reference_conformations[j], atol=exact_tol, rtol=0.0):
                chosen = j
                break
        if chosen is None:
            for j in range(Mr):
                if j in used_refs:
                    continue
                if np.allclose(query_conformations[i], reference_conformations[j], atol=exact_tol, rtol=0.0):
                    chosen = j
                    break
        if chosen is None:
            raise ValueError(f"Could not find an exact 1-to-1 match for query conformation {i}")
        used_refs.add(chosen)
        conf_perm[i] = chosen
        pair_rmsd[i] = 0.0
        if show_progress and tqdm is None:
            print(f"{progress_desc}: matched {i + 1}/{Mq}")

    if len(used_refs) != Mr:
        raise ValueError("Exact matching did not produce a full 1-to-1 cover of the reference conformations")
    return conf_perm, pair_rmsd, None


def match_conformations_dimer_no_query_smiles(
    query_coords: np.ndarray,
    query_atoms,
    n_atoms_per_mol,
    reference_set: Any,
    anchor_query_idx: int = 0,
    anchor_ref_idx: int = 0,
    prefer_mapped_smiles: bool = True,
    show_progress: bool = False,
    conformation_match_mode: str = "aligned",
    heavy_key_atoms: int = 2,
    exact_tol: float = 1e-8,
    key_decimals: int = 8,
) -> ConformationMatchResult:
    """Match a dimer geometry scan to a reference :class:`SpiceMolecule`.

    ``reference_set`` must expose ``.conformations`` (M,N,3), ``.atomic_numbers`` (N,)
    and ``.smiles`` (a 2-fragment dimer SMILES). Returns the atom permutation
    (reference→query) and the query→reference conformation permutation.
    """
    atom_map = get_fixed_dimer_atom_map_without_query_smiles(
        query_coords_anchor=query_coords[anchor_query_idx],
        query_atoms=query_atoms,
        n_atoms_per_mol=n_atoms_per_mol,
        reference_coords_anchor=reference_set.conformations[anchor_ref_idx],
        reference_atomic_numbers=reference_set.atomic_numbers,
        reference_smiles=reference_set.smiles,
        prefer_mapped_smiles=prefer_mapped_smiles,
    )
    query_coords_reordered = reorder_query_atoms_to_reference(query_coords, atom_map.atom_perm_reference_to_query)
    heavy_atom_indices = np.where(np.asarray(reference_set.atomic_numbers) != 1)[0]
    conf_perm, pair_rmsd, cost_matrix = match_conformations_with_fixed_atom_order(
        query_coords_reordered,
        reference_set.conformations,
        mode=conformation_match_mode,
        heavy_atom_indices=heavy_atom_indices,
        heavy_key_atoms=heavy_key_atoms,
        exact_tol=exact_tol,
        key_decimals=key_decimals,
        show_progress=show_progress,
    )
    return ConformationMatchResult(
        query_to_reference_conf_perm=conf_perm,
        pair_rmsd=pair_rmsd,
        cost_matrix=cost_matrix,
        atom_perm=atom_map.atom_perm_reference_to_query,
        fragment_assignment=atom_map.fragment_assignment,
        anchor_rmsd=atom_map.rmsd_anchor,
        reference_fragment_source=atom_map.reference_fragment_source,
    )


# ---------------------------------------------------------------------------
# Applying permutations to records / arrays
# ---------------------------------------------------------------------------

def _permute_array_axes(arr: np.ndarray, atom_perm: np.ndarray | None, conf_perm: np.ndarray | None) -> np.ndarray:
    out = arr
    if conf_perm is not None and out.ndim >= 1 and out.shape[0] == len(conf_perm):
        out = out[conf_perm]
    if atom_perm is None:
        return out
    if out.ndim >= 2 and out.shape[1] == len(atom_perm):
        out = out[:, atom_perm, ...]
    if out.ndim >= 3 and out.shape[2] == len(atom_perm):
        out = out[:, :, atom_perm, ...]
    return out


def apply_atom_permutation_to_spicemolecule(mol: Any, atom_perm: np.ndarray) -> Any:
    if not is_dataclass(mol):
        raise TypeError("mol must be a dataclass instance")
    updates = {}
    for f in fields(mol):
        value = getattr(mol, f.name)
        if value is None:
            updates[f.name] = None
        elif isinstance(value, np.ndarray):
            if f.name == "atomic_numbers" and value.ndim == 1:
                updates[f.name] = value[atom_perm]
            else:
                updates[f.name] = _permute_array_axes(value, atom_perm=atom_perm, conf_perm=None)
        else:
            updates[f.name] = value
    return replace(mol, **updates)


def apply_conformation_permutation_to_spicemolecule(mol: Any, conf_perm: np.ndarray) -> Any:
    if not is_dataclass(mol):
        raise TypeError("mol must be a dataclass instance")
    updates = {}
    for f in fields(mol):
        value = getattr(mol, f.name)
        if value is None:
            updates[f.name] = None
        elif isinstance(value, np.ndarray):
            updates[f.name] = _permute_array_axes(value, atom_perm=None, conf_perm=conf_perm)
        else:
            updates[f.name] = value
    return replace(mol, **updates)


def reorder_reference_spicemolecule_to_query(reference_set, match_result):
    """Reorder the reference's conformations into the query conformation order."""
    perm = match_result.query_to_reference_conf_perm
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return apply_conformation_permutation_to_spicemolecule(reference_set, inv_perm)


def reorder_reference_spicemolecule_to_query_atom_order(reference_set, match_result):
    """Reorder reference atoms to match the query atom order.

    Assumes ``match_result.atom_perm[ref_idx] = query_idx``.
    """
    perm = np.asarray(match_result.atom_perm, dtype=int)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return apply_atom_permutation_to_spicemolecule(reference_set, inv_perm)


def reorder_query_arrays_to_reference(query_arrays: dict[str, np.ndarray], atom_perm: np.ndarray, conf_perm: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in query_arrays.items():
        arr = np.asarray(value)
        out[key] = _permute_array_axes(arr, atom_perm=atom_perm, conf_perm=conf_perm)
    return out


# ---------------------------------------------------------------------------
# Diagnostics / convenience
# ---------------------------------------------------------------------------

def build_mapping_diagnostics(query_atoms, n_atoms_per_mol, reference_set: Any, result: ConformationMatchResult) -> MappingDiagnostics:
    qatoms = _to_symbols(query_atoms)
    ratoms = _to_symbols(reference_set.atomic_numbers)
    n1, n2 = map(int, n_atoms_per_mol)
    q1 = qatoms[:n1]
    q2 = qatoms[n1:n1 + n2]
    ref_frag_info = _get_reference_fragment_indices(reference_set.smiles, reference_set.atomic_numbers, prefer_mapped_smiles=True)
    r1 = ratoms[ref_frag_info.frag1_idx]
    r2 = ratoms[ref_frag_info.frag2_idx]

    def comp(arr: np.ndarray) -> dict[str, int]:
        u, c = np.unique(arr, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    return MappingDiagnostics(
        query_fragment_sizes=(len(q1), len(q2)),
        reference_fragment_sizes=(len(r1), len(r2)),
        query_fragment_compositions=(comp(q1), comp(q2)),
        reference_fragment_compositions=(comp(r1), comp(r2)),
        chosen_fragment_assignment=result.fragment_assignment,
        anchor_rmsd=result.anchor_rmsd,
        reference_fragment_source=result.reference_fragment_source,
    )


def monomer_rmsd_between_conformations(
    reference_coords: np.ndarray,
    query_coords: np.ndarray,
    n_atoms_per_mol,
    max_rmsd: float = 0.0,
) -> dict[str, Any]:
    """Per-monomer + whole-dimer RMSD between two same-ordered dimer conformations.

    Monomer 1 = first ``n_atoms_per_mol[0]`` atoms, monomer 2 = the next
    ``n_atoms_per_mol[1]``. Each RMSD is after independent Kabsch alignment.
    """
    ref = np.asarray(reference_coords, dtype=float)
    qry = np.asarray(query_coords, dtype=float)
    if ref.ndim != 2 or qry.ndim != 2 or ref.shape != qry.shape or ref.shape[1] != 3:
        raise ValueError("reference_coords and query_coords must both have shape (N, 3) and match")
    n1, n2 = map(int, n_atoms_per_mol)
    if n1 + n2 != ref.shape[0]:
        raise ValueError("n_atoms_per_mol does not sum to the number of atoms")
    ref1, ref2 = ref[:n1], ref[n1:n1 + n2]
    qry1, qry2 = qry[:n1], qry[n1:n1 + n2]
    R1, t1 = _kabsch(qry1, ref1)
    R2, t2 = _kabsch(qry2, ref2)
    Rd, td = _kabsch(qry, ref)
    rmsd_mon1 = _rmsd(_apply_transform(qry1, R1, t1), ref1)
    rmsd_mon2 = _rmsd(_apply_transform(qry2, R2, t2), ref2)
    rmsd_dimer = _rmsd(_apply_transform(qry, Rd, td), ref)
    return {
        "rmsd_mon1": float(rmsd_mon1),
        "rmsd_mon2": float(rmsd_mon2),
        "rmsd_dimer": float(rmsd_dimer),
        "mon_distance_ref": float(np.linalg.norm(ref1.mean(axis=0) - ref2.mean(axis=0))),
        "mon_distance_query": float(np.linalg.norm(qry1.mean(axis=0) - qry2.mean(axis=0))),
        "pass_": bool((rmsd_mon1 <= max_rmsd) and (rmsd_mon2 <= max_rmsd)),
    }


def run_alignment(query_coords, query_atoms, reference_set, n_atoms_per_mol, query_extra=None):
    """Convenience: match a dimer scan and return (match, diagnostics, reordered_query_arrays).

    ``query_extra`` is an optional dict of extra query-side arrays to reorder alongside
    the coordinates (e.g. per-conformation energies/gradients).
    """
    match = match_conformations_dimer_no_query_smiles(
        query_coords=query_coords,
        query_atoms=query_atoms,
        n_atoms_per_mol=n_atoms_per_mol,
        reference_set=reference_set,
        anchor_query_idx=0,
        anchor_ref_idx=0,
    )
    diagnostics = build_mapping_diagnostics(
        query_atoms=query_atoms,
        n_atoms_per_mol=n_atoms_per_mol,
        reference_set=reference_set,
        result=match,
    )
    query_arrays = {"coords": np.asarray(query_coords)}
    if query_extra:
        query_arrays.update(query_extra)
    reordered_query = reorder_query_arrays_to_reference(
        query_arrays,
        atom_perm=match.atom_perm,
        conf_perm=match.query_to_reference_conf_perm,
    )
    return match, diagnostics, reordered_query


__all__ = [
    "FragmentInfo",
    "FragmentedAtomMapResult",
    "ConformationMatchResult",
    "MappingDiagnostics",
    "ATOMIC_NUMBER_TO_SYMBOL",
    "get_fixed_dimer_atom_map_without_query_smiles",
    "match_conformations_with_fixed_atom_order",
    "match_conformations_dimer_no_query_smiles",
    "reorder_query_atoms_to_reference",
    "apply_atom_permutation_to_spicemolecule",
    "apply_conformation_permutation_to_spicemolecule",
    "reorder_reference_spicemolecule_to_query",
    "reorder_reference_spicemolecule_to_query_atom_order",
    "reorder_query_arrays_to_reference",
    "build_mapping_diagnostics",
    "monomer_rmsd_between_conformations",
    "run_alignment",
]
