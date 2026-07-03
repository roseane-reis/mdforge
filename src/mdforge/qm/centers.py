"""Reduce per-atom forces into per-center net force + torque.

Lifted from ``prior internal tooling`` (which already supersets the
loose ``center_update_helpers.py``). The core kernel
:func:`batch_atom_forces_to_center` is vectorized here (the legacy version was a
python double-loop over frames×atoms, flagged as a perf issue) and takes plain
arrays — the **generic, engine-agnostic API** that private companion plugins
producing center-based forces are expected to call.

Units: coordinates must share one length unit; force units are preserved;
torque carries the corresponding force·length unit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..core.records import get_temporary_field, set_temporary_field


def batch_atom_forces_to_center(
    center_coords_batch: np.ndarray,
    atom2center_batch: np.ndarray,
    atomic_coords_batch: np.ndarray,
    per_atom_forces_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-atom forces to per-center net force + torque.

    Parameters
    ----------
    center_coords_batch:
        ``(C, 3)`` or ``(F, C, 3)`` center coordinates.
    atom2center_batch:
        ``(N,)`` or ``(F, N)`` map from atom index → center id (0-based).
    atomic_coords_batch:
        ``(F, N, 3)`` atom coordinates.
    per_atom_forces_batch:
        ``(F, N, 3)`` per-atom forces (NOT gradients — negate first if needed).

    Returns
    -------
    ``(center_forces, center_torques)``, each ``(F, C, 3)``.
    """
    center_coords_batch = np.asarray(center_coords_batch, dtype=float)
    atom2center_batch = np.asarray(atom2center_batch, dtype=int)
    atomic_coords_batch = np.asarray(atomic_coords_batch, dtype=float)
    per_atom_forces_batch = np.asarray(per_atom_forces_batch, dtype=float)

    if atomic_coords_batch.shape != per_atom_forces_batch.shape:
        raise ValueError(
            f"atomic_coords_batch shape {atomic_coords_batch.shape} != "
            f"per_atom_forces_batch shape {per_atom_forces_batch.shape}"
        )
    if atomic_coords_batch.ndim != 3 or atomic_coords_batch.shape[-1] != 3:
        raise ValueError("atomic_coords_batch must have shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = atomic_coords_batch.shape

    if center_coords_batch.ndim == 2:
        center_coords_batch = np.repeat(center_coords_batch[None], n_frames, axis=0)
    if atom2center_batch.ndim == 1:
        atom2center_batch = np.repeat(atom2center_batch[None], n_frames, axis=0)

    if center_coords_batch.shape[0] != n_frames:
        raise ValueError("center_coords_batch must be (n_centers, 3) or (n_frames, n_centers, 3)")
    if atom2center_batch.shape != (n_frames, n_atoms):
        raise ValueError("atom2center_batch must be (n_atoms,) or (n_frames, n_atoms)")

    n_centers = center_coords_batch.shape[1]
    if atom2center_batch.min() < 0 or atom2center_batch.max() >= n_centers:
        raise ValueError(f"atom2center index out of range for n_centers={n_centers}")

    # Center coordinate for each atom: (F, N, 3)
    idx = np.broadcast_to(atom2center_batch[:, :, None], (n_frames, n_atoms, 3))
    center_of_atom = np.take_along_axis(center_coords_batch, idx, axis=1)
    lever = atomic_coords_batch - center_of_atom
    torque_per_atom = np.cross(lever, per_atom_forces_batch)

    # Scatter-add atoms into their centers (handles repeated indices).
    frame_idx = np.broadcast_to(np.arange(n_frames)[:, None], (n_frames, n_atoms))
    center_forces = np.zeros((n_frames, n_centers, 3), dtype=float)
    center_torques = np.zeros((n_frames, n_centers, 3), dtype=float)
    np.add.at(center_forces, (frame_idx, atom2center_batch), per_atom_forces_batch)
    np.add.at(center_torques, (frame_idx, atom2center_batch), torque_per_atom)

    return center_forces, center_torques


# ---------------------------------------------------------------------------
# Record-aware helpers (operate on SpiceMolecule-like objects)
# ---------------------------------------------------------------------------

def _attr_name(obj: Any, *candidates: str) -> str:
    for name in candidates:
        if hasattr(obj, name):
            return name
    raise AttributeError(f"None of {candidates} exist on {type(obj).__name__}")


def _broadcast_reference_to_target(reference_record: Any, target_record: Any) -> tuple[np.ndarray, np.ndarray]:
    center_coords = np.asarray(getattr(reference_record, "center_coords"))
    atom2center = np.asarray(getattr(reference_record, _attr_name(reference_record, "atom_to_center", "atom2center")))
    if center_coords is None or atom2center is None:
        raise ValueError("reference_record must have center_coords and atom_to_center populated")

    n_frames = np.asarray(target_record.conformations).shape[0]
    n_atoms = np.asarray(target_record.conformations).shape[1]

    if center_coords.ndim == 2:
        center_coords = np.repeat(center_coords[None], n_frames, axis=0)
    elif center_coords.ndim == 3 and center_coords.shape[0] == 1 and n_frames != 1:
        center_coords = np.repeat(center_coords, n_frames, axis=0)
    elif center_coords.ndim != 3 or center_coords.shape[0] != n_frames:
        raise ValueError("reference center_coords frame count does not match target")

    if atom2center.ndim == 1:
        atom2center = np.repeat(atom2center[None], n_frames, axis=0)
    elif atom2center.ndim == 2 and atom2center.shape[0] == 1 and n_frames != 1:
        atom2center = np.repeat(atom2center, n_frames, axis=0)
    elif atom2center.ndim != 2 or atom2center.shape[0] != n_frames:
        raise ValueError("reference atom2center frame count does not match target")

    if atom2center.shape[1] != n_atoms:
        raise ValueError(f"reference atom2center has {atom2center.shape[1]} atoms but target has {n_atoms}")
    return center_coords, atom2center


def update_record_center_fields_from_reference(
    record: Any,
    reference_record: Any,
    per_atom_values_batch: np.ndarray,
    *,
    values_are_gradients: bool = True,
    magnitude_as_attribute: bool = True,
    in_place: bool = True,
) -> Any:
    """Copy a reference center map onto ``record`` and compute center force/torque.

    When ``values_are_gradients`` is True the forces are ``-per_atom_values_batch``.
    Caches ``center_force_magnitude`` / ``center_torque_magnitude``.
    """
    if not in_place:
        record = joblib.loads(joblib.dumps(record))

    center_coords_batch, atom2center_batch = _broadcast_reference_to_target(reference_record, record)
    per_atom_values_batch = np.asarray(per_atom_values_batch, dtype=float)
    atomic_coords_batch = np.asarray(record.conformations, dtype=float)
    if per_atom_values_batch.shape != atomic_coords_batch.shape:
        raise ValueError(
            f"per_atom_values_batch shape {per_atom_values_batch.shape} != "
            f"record.conformations shape {atomic_coords_batch.shape}"
        )

    forces = -per_atom_values_batch if values_are_gradients else per_atom_values_batch
    center_forces, center_torques = batch_atom_forces_to_center(
        center_coords_batch, atom2center_batch, atomic_coords_batch, forces
    )

    setattr(record, "center_coords", center_coords_batch)
    setattr(record, _attr_name(record, "atom_to_center", "atom2center"),
            np.asarray(atom2center_batch, dtype=int))
    setattr(record, "forces_per_center", center_forces)
    setattr(record, "torques_per_center", center_torques)

    record.set_cache("center_force_magnitude", np.linalg.norm(center_forces, axis=-1),
                     as_attribute=magnitude_as_attribute)
    record.set_cache("center_torque_magnitude", np.linalg.norm(center_torques, axis=-1),
                     as_attribute=magnitude_as_attribute)
    return record


def compute_center_magnitude(
    record: Any,
    *,
    field_name: str = "forces_per_center",
    output_field: str = "center_force_magnitude",
    overwrite: bool = False,
) -> np.ndarray:
    """Compute and cache |F_center| from ``record.<field_name>`` → ``(M, C)``."""
    existing = get_temporary_field(record, output_field)
    if existing is not None and not overwrite:
        return existing
    forces = getattr(record, field_name, None)
    if forces is None:
        raise ValueError(f"record has no field {field_name!r}")
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise ValueError(f"{field_name} must be (n_confs, n_centers, 3); got {forces.shape}")
    magnitude = np.linalg.norm(forces, axis=-1)
    return set_temporary_field(record, output_field, magnitude, as_attribute=True, overwrite=True)


def update_record_dict_center_fields(
    record_dict: MutableMapping[str, Any],
    reference_record: Any,
    per_atom_values_by_key: Mapping[str, np.ndarray],
    *,
    gradient_keys: Iterable[str] | None = None,
) -> MutableMapping[str, Any]:
    """Apply one reference center map to a dict of model records."""
    grad_keys = set(per_atom_values_by_key.keys() if gradient_keys is None else gradient_keys)
    for key, values in per_atom_values_by_key.items():
        if key not in record_dict:
            raise KeyError(f"{key!r} not found in record_dict")
        update_record_center_fields_from_reference(
            record=record_dict[key], reference_record=reference_record,
            per_atom_values_batch=values, values_are_gradients=(key in grad_keys), in_place=True,
        )
    return record_dict


def update_joblib_center_fields_from_reference(
    record_joblib: str | Path,
    reference_joblib: str | Path,
    per_atom_values_batch: np.ndarray,
    *,
    output_joblib: str | Path | None = None,
    values_are_gradients: bool = True,
) -> Any:
    """Load target + reference joblibs, update target's center fields, optionally save."""
    record = joblib.load(record_joblib)
    reference_record = joblib.load(reference_joblib)
    updated = update_record_center_fields_from_reference(
        record, reference_record, per_atom_values_batch,
        values_are_gradients=values_are_gradients, in_place=True,
    )
    if output_joblib is not None:
        joblib.dump(updated, output_joblib, compress=3)
    return updated


__all__ = [
    "batch_atom_forces_to_center",
    "update_record_center_fields_from_reference",
    "update_record_dict_center_fields",
    "update_joblib_center_fields_from_reference",
    "compute_center_magnitude",
]
