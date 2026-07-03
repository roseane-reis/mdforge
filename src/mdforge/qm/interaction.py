"""Interaction energies/gradients and monomer↔dimer matching (goal f).

Lifted from ``prior internal tooling``. Interaction
quantities are ``dimer − monomer1 − monomer2`` (block-wise for gradients).

Changes from the source:
- rdkit is now **optional** — only the SMILES-fragment path imports it, lazily,
  with a clear error if missing. The common paths (explicit ``n_atoms_per_mol``
  or matching by atomic-number blocks) need no rdkit.
- Fixed the ``mon_order`` UnboundLocalError in ``infer_n_atoms_per_mol`` (it is
  now initialised to the input order before the matching loop).
- Atomic masses come from :mod:`mdforge.core.elements`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.elements import ATOMIC_MASSES

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MonomerMatchResult:
    query_to_reference_index: np.ndarray
    matched_mask: np.ndarray
    unmatched_query_indices: np.ndarray
    match_rmsd: np.ndarray


@dataclass
class PairInteractionEnergyResult:
    pair: str
    dimer_record_key: str
    monomer_record_keys: tuple[str, str]
    n_atoms_per_mol: tuple[int, int]
    dimer_energy: np.ndarray
    monomer1_energy: np.ndarray
    monomer2_energy: np.ndarray
    interaction_energy: np.ndarray
    monomer1_match_index: np.ndarray
    monomer2_match_index: np.ndarray
    match_mask: np.ndarray
    unmatched_query_indices: np.ndarray
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _as_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Coordinates must have shape (N,3) or (M,N,3)")
    return arr


def compute_center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Center of mass per frame: ``(M,N,3)`` + ``(N,)`` → ``(M,3)`` (or ``(3,)``)."""
    xyz = _as_coords(coords)
    m = np.asarray(masses, dtype=float).reshape(-1)
    if xyz.shape[1] != m.size:
        raise ValueError("Mass array length does not match atom count")
    com = np.sum(xyz * m[None, :, None], axis=1) / np.sum(m)
    return com[0] if np.asarray(coords).ndim == 2 else com


def center_coordinates(coords: np.ndarray, *, center_method: str = "centroid",
                       masses: np.ndarray | None = None) -> np.ndarray:
    """Translate coordinates so the centroid (or COM) is at the origin."""
    xyz = _as_coords(coords)
    if center_method not in {"centroid", "com"}:
        raise ValueError("center_method must be 'centroid' or 'com'")
    if center_method == "centroid":
        center = xyz.mean(axis=1)
    else:
        if masses is None:
            raise ValueError("masses are required when center_method='com'")
        center = compute_center_of_mass(xyz, masses)
        if center.ndim == 1:
            center = center[None, :]
    out = xyz - center[:, None, :]
    return out[0] if np.asarray(coords).ndim == 2 else out


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R, Qc - Pc @ R


def _rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def _record_masses(record: Any) -> np.ndarray:
    z = np.asarray(record.atomic_numbers, dtype=int)
    if np.max(z) >= len(ATOMIC_MASSES):
        raise ValueError("core.elements.ATOMIC_MASSES does not cover one or more atomic numbers")
    return np.asarray(ATOMIC_MASSES[z], dtype=float)


def monomer_com_distance(conformations: np.ndarray, n_atoms_per_mol: Sequence[int],
                         masses: np.ndarray | None = None) -> np.ndarray:
    """Per-frame distance between the two monomers' centers (COM or centroid).

    ``conformations`` is ``(M, N, 3)``; ``n_atoms_per_mol`` is ``(n1, n2)``.
    Uses centroids when ``masses`` is None. Returns ``(M,)``.
    """
    xyz = _as_coords(conformations)
    n1, n2 = int(n_atoms_per_mol[0]), int(n_atoms_per_mol[1])
    if n1 + n2 != xyz.shape[1]:
        raise ValueError("n_atoms_per_mol does not sum to the atom count")
    m1 = xyz[:, :n1, :]
    m2 = xyz[:, n1:n1 + n2, :]
    if masses is None:
        c1, c2 = m1.mean(axis=1), m2.mean(axis=1)
    else:
        masses = np.asarray(masses, dtype=float)
        c1 = compute_center_of_mass(m1, masses[:n1])
        c2 = compute_center_of_mass(m2, masses[n1:n1 + n2])
    return np.linalg.norm(c1 - c2, axis=-1)


# ---------------------------------------------------------------------------
# Fragment / monomer-count inference  (rdkit optional)
# ---------------------------------------------------------------------------

def _atoms_per_mol_from_smiles(smiles: str) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Return (counts, per-fragment atomic numbers) from a 2-fragment SMILES.

    Lazy-imports rdkit; raises a clear error if it is not installed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdmolfiles
    except ImportError as exc:  # pragma: no cover - only without rdkit
        raise ImportError(
            "Inferring monomer counts from SMILES requires rdkit. Either install rdkit "
            "or pass n_atoms_per_mol explicitly."
        ) from exc
    params = rdmolfiles.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    frag_z = tuple(tuple(int(mol.GetAtomWithIdx(int(i)).GetAtomicNum()) for i in frag) for frag in frags)
    return tuple(len(z) for z in frag_z), frag_z


def infer_n_atoms_per_mol(
    n_atoms_per_mol: Sequence[int] | None = None,
    *,
    smiles: str | None = None,
    record: Any | None = None,
    check_atom_order: bool = True,
) -> tuple[int, int]:
    """Resolve the two monomer atom counts.

    Preference order: explicit ``n_atoms_per_mol`` → SMILES fragments (rdkit).
    When matching against a record's atomic numbers, the monomer order is
    chosen to match the record's atom-block ordering.
    """
    if n_atoms_per_mol is not None:
        if len(n_atoms_per_mol) != 2:
            raise ValueError("n_atoms_per_mol must contain exactly two monomer sizes")
        return int(n_atoms_per_mol[0]), int(n_atoms_per_mol[1])

    if smiles is None and record is not None:
        smiles = getattr(record, "smiles", None)
    if not smiles:
        raise ValueError(
            "Could not infer n_atoms_per_mol; pass n_atoms_per_mol or a record/smiles "
            "with two fragments"
        )

    counts, frag_z = _atoms_per_mol_from_smiles(smiles)
    n1, n2 = counts

    if check_atom_order and record is not None:
        number_of_atoms = {"mon1": frag_z[0], "mon2": frag_z[1]}
        natoms = record.n_atoms
        mon_order = ["mon1", "mon2"]  # default — fixes legacy UnboundLocalError
        for a, b in (["mon1", "mon2"], ["mon2", "mon1"]):
            zA, zB = number_of_atoms[a], number_of_atoms[b]
            qa, qb = len(zA), len(zB)
            if qa + qb != natoms:
                continue
            block1 = np.sort([int(x) for x in record.atomic_numbers[:qa]])[::-1]
            block2 = np.sort([int(x) for x in record.atomic_numbers[qa:natoms]])[::-1]
            if (np.array_equal(np.sort(zA)[::-1], block1)
                    and np.array_equal(np.sort(zB)[::-1], block2)):
                mon_order = [a, b]
                break
        n1 = len(number_of_atoms[mon_order[0]])
        n2 = len(number_of_atoms[mon_order[1]])
    return n1, n2


def split_dimer_conformations(conformations: np.ndarray,
                              n_atoms_per_mol: Sequence[int] | None = None,
                              *, record: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split dimer coordinates into the two monomer blocks."""
    xyz = _as_coords(conformations)
    n1, n2 = infer_n_atoms_per_mol(n_atoms_per_mol, record=record)
    if n1 + n2 != xyz.shape[1]:
        raise ValueError("n_atoms_per_mol does not sum to the atom count")
    return xyz[:, :n1, :], xyz[:, n1:n1 + n2, :]


def infer_pair_monomer_keys(pair: str,
                            monomer_key_map: Mapping[str, Sequence[str]] | None = None) -> tuple[str, str]:
    """Return the two monomer record keys for ``pair`` (``left_right`` convention)."""
    if monomer_key_map is not None and pair in monomer_key_map:
        keys = tuple(monomer_key_map[pair])
        if len(keys) != 2:
            raise ValueError("monomer_key_map values must have length 2")
        return str(keys[0]), str(keys[1])
    left, right = pair.split("_", 1)
    return left, right


def _atomic_multiset_matches(a: Sequence[int], b: Sequence[int]) -> bool:
    return np.array_equal(np.sort(np.asarray(a, dtype=int)), np.sort(np.asarray(b, dtype=int)))


def infer_monomer_order_in_dimer(dimer_record, mono1_record, mono2_record,
                                 monomer_keys: tuple[str, str]) -> tuple[tuple[str, str], tuple[int, int]]:
    """Decide which monomer occupies the first atom block of the dimer."""
    dz = np.asarray(dimer_record.atomic_numbers, dtype=int)
    z1 = np.asarray(mono1_record.atomic_numbers, dtype=int)
    z2 = np.asarray(mono2_record.atomic_numbers, dtype=int)
    n1, n2 = z1.size, z2.size
    if n1 + n2 != dz.size:
        raise ValueError("Monomer atom counts do not sum to dimer atom count")
    k1, k2 = monomer_keys
    if _atomic_multiset_matches(dz[:n1], z1) and _atomic_multiset_matches(dz[n1:], z2):
        return (k1, k2), (n1, n2)
    if _atomic_multiset_matches(dz[:n2], z2) and _atomic_multiset_matches(dz[n2:], z1):
        return (k2, k1), (n2, n1)
    raise ValueError("Could not infer monomer order in dimer from atomic-number blocks")


def match_monomer_conformations(query_conformations, reference_conformations, *,
                                center_method: str = "centroid", masses=None,
                                rmsd_tol: float = 1e-8, allow_reuse: bool = True) -> MonomerMatchResult:
    """Match each query monomer frame to a reference frame by Kabsch RMSD."""
    query = _as_coords(query_conformations)
    reference = _as_coords(reference_conformations)
    if query.shape[1:] != reference.shape[1:]:
        raise ValueError(f"query/reference monomer shapes differ: {query.shape[1:]} vs {reference.shape[1:]}")
    if reference.shape[0] == 0:
        raise ValueError("reference_conformations is empty")
    q = center_coordinates(query, center_method=center_method, masses=masses)
    r = center_coordinates(reference, center_method=center_method, masses=masses)
    n_q, n_r = q.shape[0], r.shape[0]
    cost = np.empty((n_q, n_r), dtype=float)
    for i in range(n_q):
        for j in range(n_r):
            R, t = _kabsch(q[i], r[j])
            cost[i, j] = _rmsd(q[i] @ R + t, r[j])
    match_index = np.full(n_q, -1, dtype=int)
    matched = np.zeros(n_q, dtype=bool)
    match_rmsd = np.full(n_q, np.nan, dtype=float)
    used = np.zeros(n_r, dtype=bool)
    for i in np.argsort(np.min(cost, axis=1)):
        for j in np.argsort(cost[i]):
            if not allow_reuse and used[j]:
                continue
            if cost[i, j] <= rmsd_tol:
                match_index[i], matched[i], match_rmsd[i], used[j] = int(j), True, cost[i, j], True
                break
    return MonomerMatchResult(match_index, matched, np.where(~matched)[0], match_rmsd)


# ---------------------------------------------------------------------------
# Interaction energy / gradient
# ---------------------------------------------------------------------------

def _get_energy_array(record: Any, field: str | None) -> np.ndarray:
    if field is None:
        field = "model_total_energy" if getattr(record, "model_total_energy", None) is not None else "dft_total_energy"
    arr = np.asarray(getattr(record, field), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Energy arrays must be 1D")
    return arr


def _get_gradient_array(record: Any, field: str | None) -> np.ndarray | None:
    if field is None:
        field = "forces_per_center" if getattr(record, "forces_per_center", None) is not None else "dft_total_gradient"
    value = getattr(record, field, None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"Gradient-like array {field!r} must have trailing shape (*, 3)")
    return arr


def _blockwise_interaction_gradient(dimer_grad, mon1_grad, mon2_grad) -> np.ndarray:
    dimer_grad = np.asarray(dimer_grad, dtype=float)
    mon1_grad = np.asarray(mon1_grad, dtype=float)
    mon2_grad = np.asarray(mon2_grad, dtype=float)
    if dimer_grad.shape[0] != mon1_grad.shape[0] or dimer_grad.shape[0] != mon2_grad.shape[0]:
        raise ValueError("Frame counts do not match for interaction gradient subtraction")
    n1, n2 = mon1_grad.shape[1], mon2_grad.shape[1]
    if dimer_grad.shape[1] != n1 + n2:
        raise ValueError("Block sizes do not sum to the dimer gradient size")
    out = np.array(dimer_grad, copy=True)
    out[:, :n1, ...] -= mon1_grad
    out[:, n1:n1 + n2, ...] -= mon2_grad
    return out


def _broadcast_energy(arr, target_n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Energy arrays must be 1D")
    if arr.shape[0] == target_n:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_n)
    raise ValueError(f"Energy length {arr.shape[0]} != target {target_n}")


def _broadcast_gradient(arr, target_n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == target_n:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_n, axis=0)
    raise ValueError(f"Gradient first dim {arr.shape[0]} != target {target_n}")


def compute_model_interactions_no_match(
    record: Any,
    *,
    records_monomer: Mapping[str, Any] | None = None,
    monomer_record_keys: Sequence[str] | None = None,
    monomer_record_keys_field: str = "monomer_record_keys",
    model_energy_field: str = "model_total_energy",
    model_gradient_field: str | None = "forces_per_center",
    monomer_energy_field: str = "model_total_energy",
    monomer_gradient_field: str | None = None,
    monomer1_energy_field: str = "monomer1_total_energy",
    monomer2_energy_field: str = "monomer2_total_energy",
    monomer1_gradient_field: str = "monomer1_total_gradient",
    monomer2_gradient_field: str = "monomer2_total_gradient",
    interaction_energy_field: str = "interaction_total_energy",
    interaction_gradient_field: str = "interaction_total_gradient",
) -> Any:
    """Set interaction energy/gradient on ``record`` from already-aligned monomer totals.

    Monomer totals are read from attached fields on ``record`` (default) or from
    ``records_monomer`` via ``monomer_record_keys``. No conformation matching is
    done — frame counts must already correspond.
    """
    dimer_energy = _get_energy_array(record, model_energy_field)
    dimer_grad = _get_gradient_array(record, model_gradient_field) if model_gradient_field else None
    target_n = dimer_energy.shape[0]

    if records_monomer is None:
        mon1_e = getattr(record, monomer1_energy_field, None)
        mon2_e = getattr(record, monomer2_energy_field, None)
        mon1_g = getattr(record, monomer1_gradient_field, None) if model_gradient_field else None
        mon2_g = getattr(record, monomer2_gradient_field, None) if model_gradient_field else None
    else:
        keys = tuple(monomer_record_keys or getattr(record, monomer_record_keys_field, ()) or ())
        if len(keys) != 2:
            raise ValueError("Two monomer record keys are required when records_monomer is provided")
        rec1, rec2 = records_monomer[keys[0]], records_monomer[keys[1]]
        mon1_e, mon2_e = getattr(rec1, monomer_energy_field), getattr(rec2, monomer_energy_field)
        grad_field = monomer_gradient_field or model_gradient_field
        mon1_g = getattr(rec1, grad_field, None) if grad_field else None
        mon2_g = getattr(rec2, grad_field, None) if grad_field else None
        record.monomer_record_keys = keys

    if mon1_e is None or mon2_e is None:
        raise ValueError("Monomer energy arrays are missing")
    mon1_e = _broadcast_energy(mon1_e, target_n)
    mon2_e = _broadcast_energy(mon2_e, target_n)
    setattr(record, monomer1_energy_field, mon1_e)
    setattr(record, monomer2_energy_field, mon2_e)
    setattr(record, interaction_energy_field, dimer_energy - mon1_e - mon2_e)

    if dimer_grad is not None:
        if mon1_g is None or mon2_g is None:
            raise ValueError("Monomer gradient arrays are missing")
        mon1_g = _broadcast_gradient(mon1_g, dimer_grad.shape[0])
        mon2_g = _broadcast_gradient(mon2_g, dimer_grad.shape[0])
        setattr(record, monomer1_gradient_field, mon1_g)
        setattr(record, monomer2_gradient_field, mon2_g)
        setattr(record, interaction_gradient_field,
                _blockwise_interaction_gradient(dimer_grad, mon1_g, mon2_g))
    return record


def _resolve_record_key(base_key: str, records: Mapping[str, Any], model: str | None) -> str:
    if model is None:
        if base_key not in records:
            raise KeyError(f"Missing record key: {base_key}")
        return base_key
    candidate = f"{base_key}_{model}"
    if candidate in records:
        return candidate
    if base_key in records:
        return base_key
    raise KeyError(f"Could not resolve record key for {base_key!r} and model {model!r}")


def compute_pair_interaction_energies(
    pair: str,
    records_dimer: Mapping[str, Any],
    records_monomer: Mapping[str, Any],
    *,
    model: str | None = None,
    dimer_record_key: str | None = None,
    monomer_record_keys: Sequence[str] | None = None,
    monomer_key_map: Mapping[str, Sequence[str]] | None = None,
    n_atoms_per_mol: Sequence[int] | None = None,
    dimer_energy_field: str | None = None,
    monomer_energy_field: str | None = None,
    center_method: str = "centroid",
    rmsd_tol: float = 1e-6,
    allow_reuse: bool = True,
) -> PairInteractionEnergyResult:
    """Interaction energy for one dimer pair, matching dimer frames to monomers."""
    dimer_key = _resolve_record_key(dimer_record_key or pair, records_dimer, model)
    dimer_record = records_dimer[dimer_key]
    requested = (infer_pair_monomer_keys(pair, monomer_key_map=monomer_key_map)
                 if monomer_record_keys is None else tuple(monomer_record_keys))
    if len(requested) != 2:
        raise ValueError("monomer_record_keys must have length 2")
    k1 = _resolve_record_key(requested[0], records_monomer, model)
    k2 = _resolve_record_key(requested[1], records_monomer, model)
    resolved_keys, n_resolved = infer_monomer_order_in_dimer(
        dimer_record, records_monomer[k1], records_monomer[k2], (k1, k2))
    mono1_key, mono2_key = resolved_keys
    mono1, mono2 = records_monomer[mono1_key], records_monomer[mono2_key]
    n1, n2 = n_resolved if n_atoms_per_mol is None else infer_n_atoms_per_mol(n_atoms_per_mol, record=dimer_record)

    dimer_energy = _get_energy_array(dimer_record, dimer_energy_field)
    dimer_m1, dimer_m2 = split_dimer_conformations(dimer_record.conformations, (n1, n2))
    masses1 = _record_masses(mono1) if center_method == "com" else None
    masses2 = _record_masses(mono2) if center_method == "com" else None
    match1 = match_monomer_conformations(dimer_m1, mono1.conformations, center_method=center_method,
                                         masses=masses1, rmsd_tol=rmsd_tol, allow_reuse=allow_reuse)
    match2 = match_monomer_conformations(dimer_m2, mono2.conformations, center_method=center_method,
                                         masses=masses2, rmsd_tol=rmsd_tol, allow_reuse=allow_reuse)
    match_mask = match1.matched_mask & match2.matched_mask
    e1_ref = _get_energy_array(mono1, monomer_energy_field)
    e2_ref = _get_energy_array(mono2, monomer_energy_field)
    mon1_energy = np.full_like(dimer_energy, np.nan)
    mon2_energy = np.full_like(dimer_energy, np.nan)
    mon1_energy[match1.matched_mask] = e1_ref[match1.query_to_reference_index[match1.matched_mask]]
    mon2_energy[match2.matched_mask] = e2_ref[match2.query_to_reference_index[match2.matched_mask]]
    interaction = np.full_like(dimer_energy, np.nan)
    interaction[match_mask] = dimer_energy[match_mask] - mon1_energy[match_mask] - mon2_energy[match_mask]

    return PairInteractionEnergyResult(
        pair=pair, dimer_record_key=dimer_key, monomer_record_keys=(mono1_key, mono2_key),
        n_atoms_per_mol=(n1, n2), dimer_energy=dimer_energy,
        monomer1_energy=mon1_energy, monomer2_energy=mon2_energy, interaction_energy=interaction,
        monomer1_match_index=match1.query_to_reference_index,
        monomer2_match_index=match2.query_to_reference_index,
        match_mask=match_mask, unmatched_query_indices=np.where(~match_mask)[0],
        metadata={"center_method": center_method, "rmsd_tol": rmsd_tol,
                  "n_total": int(dimer_energy.shape[0]), "n_matched": int(match_mask.sum())},
    )


def build_interaction_energy_dict(pairs: Sequence[str], records_dimer, records_monomer,
                                  **kwargs) -> dict[str, PairInteractionEnergyResult]:
    """Run :func:`compute_pair_interaction_energies` over many pairs."""
    return {pair: compute_pair_interaction_energies(pair, records_dimer, records_monomer, **kwargs)
            for pair in pairs}


__all__ = [
    "MonomerMatchResult", "PairInteractionEnergyResult",
    "compute_center_of_mass", "center_coordinates", "monomer_com_distance",
    "infer_n_atoms_per_mol", "split_dimer_conformations",
    "infer_pair_monomer_keys", "infer_monomer_order_in_dimer",
    "match_monomer_conformations",
    "compute_model_interactions_no_match", "compute_pair_interaction_energies",
    "build_interaction_energy_dict",
]
