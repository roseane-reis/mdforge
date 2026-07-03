"""Read HOOMD-blue GSD rigid-body trajectories into numpy arrays.

A rigid-body GSD stores, per molecule, a *center* particle carrying the
centre-of-mass position and an orientation quaternion, plus massless
*constituent* particles that ride along rigidly. The center particles are the
ones whose ``body`` index equals their own particle index; constituents point
back at their center and are skipped here (they are reconstructed from the
center body frame when real atom coordinates are needed).

This module is deliberately engine-agnostic: it reads the generic HOOMD
rigid-body layout and reconstructs whole-molecule atom positions **from a
caller-supplied reference geometry** (``{species: (N_atoms, 3)}`` in the body
frame). It never imports a force-field / model package — the body-frame
geometry is the caller's responsibility, so any center-based rigid-body engine
can reuse this reader.

``gsd`` is an optional dependency (``pip install gsd``); it is imported lazily
so the rest of :mod:`mdforge.formats` works without it.

Parse ⟂ compute: this is the *parse* half. Structural kernels
(:mod:`mdforge.liquid.structure`) take the arrays produced here and never open
a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _import_gsd():
    try:
        import gsd.hoomd  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without gsd
        raise ImportError(
            "Reading GSD trajectories requires the 'gsd' package. "
            "Install it with: pip install gsd"
        ) from exc
    import gsd.hoomd

    return gsd.hoomd


def quaternion_to_rotation_matrix(quats: np.ndarray) -> np.ndarray:
    """Convert unit quaternions ``(..., 4)`` ``[w, x, y, z]`` to ``(..., 3, 3)``.

    Uses the same column convention as HOOMD's rigid-body rotation: a body-frame
    point ``p`` maps to the lab frame as ``R @ p``.
    """
    q = np.asarray(quats, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class RigidTrajectory:
    """Per-frame rigid-body data read from a GSD trajectory.

    Units: Angstrom (positions/box). The box is the HOOMD 6-tuple
    ``[Lx, Ly, Lz, xy, xz, yz]`` per frame (variable under NPT).
    """

    com: np.ndarray            # (T, M, 3)   centre-of-mass positions [Å]
    orientation: np.ndarray    # (T, M, 4)   body quaternions [w, x, y, z]
    box: np.ndarray            # (T, 6)      [Lx, Ly, Lz, xy, xz, yz] [Å]
    species: list[str]         # length M    per-molecule species name
    types: list[str] = field(default_factory=list)  # the GSD type table

    @property
    def n_frames(self) -> int:
        return self.com.shape[0]

    @property
    def n_molecules(self) -> int:
        return self.com.shape[1]

    @property
    def volume(self) -> np.ndarray:
        """Per-frame cell volume in Å³ (orthorhombic + tilt-aware)."""
        Lx, Ly, Lz = self.box[:, 0], self.box[:, 1], self.box[:, 2]
        return Lx * Ly * Lz  # tilt factors do not change the cell volume


def read_rigid_bodies(path: str | Path, *, max_frames: int | None = None) -> RigidTrajectory:
    """Read a HOOMD rigid-body GSD into a :class:`RigidTrajectory`.

    Center particles are those with ``body == own index``; constituent
    particles are skipped. The per-molecule species is read from each center's
    ``typeid`` so mixed-species systems work. The molecule layout (which
    particles are centers, and their species) is taken from the first frame and
    assumed constant across the trajectory.

    Parameters
    ----------
    path:
        GSD file path.
    max_frames:
        Read at most this many frames (default: all).
    """
    gsd_hoomd = _import_gsd()
    with gsd_hoomd.open(str(path), "r") as traj:
        n_frames = len(traj) if max_frames is None else min(len(traj), max_frames)
        if n_frames == 0:
            raise ValueError(f"GSD trajectory {path} has no frames")

        first = traj[0]
        body = np.asarray(first.particles.body)
        idx = np.arange(len(body))
        centrals = idx[body == idx]
        if len(centrals) == 0:
            # No rigid constraint recorded: treat every particle as a molecule.
            centrals = idx
        types = list(first.particles.types)
        tid = np.asarray(first.particles.typeid)
        species = [types[tid[c]] for c in centrals]

        M = len(centrals)
        com = np.empty((n_frames, M, 3), dtype=float)
        orientation = np.empty((n_frames, M, 4), dtype=float)
        box = np.empty((n_frames, 6), dtype=float)
        for i in range(n_frames):
            snap = traj[i]
            com[i] = snap.particles.position[centrals]
            orientation[i] = snap.particles.orientation[centrals]
            box[i] = snap.configuration.box

    return RigidTrajectory(com=com, orientation=orientation, box=box,
                           species=species, types=types)


def reconstruct_atoms(
    rbt: RigidTrajectory,
    reference_geometry: dict[str, np.ndarray],
    *,
    wrap_molecules: bool = True,
) -> np.ndarray:
    """Reconstruct whole-molecule atom positions for every frame.

    ``reference_geometry`` maps each species name present in ``rbt.species`` to
    its body-frame atom coordinates ``(N_atoms, 3)`` in Angstrom. The caller
    supplies this (e.g. extracted once from the engine's reference shapes) — the
    reader stays model-free.

    For each molecule m at frame t: ``atoms = R(quat) @ body_local.T + COM``.
    With ``wrap_molecules`` the COM is wrapped into the primary box ``[-L/2,
    L/2]`` and the whole molecule follows (atoms are kept intact, never split
    across the boundary).

    Returns an array ``(T, N_total_atoms, 3)`` with molecules concatenated in
    center order; the per-atom species/order follows ``reference_geometry``.
    """
    missing = sorted(set(rbt.species) - set(reference_geometry))
    if missing:
        raise KeyError(f"reference_geometry missing species: {missing}")

    local = {s: np.asarray(reference_geometry[s], dtype=float) for s in set(rbt.species)}
    n_atoms = sum(len(local[s]) for s in rbt.species)
    T = rbt.n_frames
    out = np.empty((T, n_atoms, 3), dtype=float)

    for t in range(T):
        com = rbt.com[t].copy()
        if wrap_molecules:
            L = rbt.box[t, :3]
            com -= L * np.round(com / L)
        R = quaternion_to_rotation_matrix(rbt.orientation[t])  # (M, 3, 3)
        a = 0
        for m, s in enumerate(rbt.species):
            pts = local[s]
            out[t, a:a + len(pts)] = (R[m] @ pts.T).T + com[m]
            a += len(pts)
    return out


def reference_geometry_from_gsd(
    path: str | Path,
    *,
    exclude_central: bool | None = None,
) -> dict[str, np.ndarray]:
    """Recover body-frame atom geometry per species from GSD frame 0.

    Returns ``{species: (N_atoms, 3)}`` suitable as the ``reference_geometry``
    argument of :func:`reconstruct_atoms` — recovered directly from the
    trajectory's own rigid-body definition, so no external/force-field geometry
    package is needed (fully engine-agnostic).

    For each species (identified by the central particle's type) the first
    molecule's constituent particles are taken, shifted to the central-particle
    frame (minimum image), and rotated back by the inverse of the central's
    orientation quaternion — giving the fixed body-frame template. The
    constituent order in the file is preserved.

    Parameters
    ----------
    exclude_central:
        Whether to drop the central particle itself from the template. By
        default (``None``) the central is dropped when it has constituents
        (a separate centre-of-mass site) and kept otherwise (the central *is*
        an atom). The constituent order in the file is otherwise preserved.
    """
    gsd_hoomd = _import_gsd()
    with gsd_hoomd.open(str(path), "r") as traj:
        if len(traj) == 0:
            raise ValueError(f"GSD trajectory {path} has no frames")
        f0 = traj[0]
        body = np.asarray(f0.particles.body)
        idx = np.arange(len(body))
        pos = np.asarray(f0.particles.position, dtype=float)
        orient = np.asarray(f0.particles.orientation, dtype=float)
        types = list(f0.particles.types)
        tid = np.asarray(f0.particles.typeid)
        Lbox = np.asarray(f0.configuration.box, dtype=float)[:3]

    centrals = idx[body == idx]
    geom: dict[str, np.ndarray] = {}
    for c in centrals:
        species = types[tid[c]]
        if species in geom:
            continue
        members = idx[body == c]
        constituents = members[members != c]
        drop_central = exclude_central
        if drop_central is None:
            drop_central = len(constituents) > 0
        atom_idx = constituents if drop_central else members
        if len(atom_idx) == 0:
            atom_idx = np.asarray([c])
        rel = pos[atom_idx] - pos[c]
        rel -= Lbox * np.round(rel / Lbox)                       # minimum image
        R0 = quaternion_to_rotation_matrix(orient[c])            # (3, 3)
        geom[species] = (R0.T @ rel.T).T                         # body frame
    return geom


def species_atom_index(rbt: RigidTrajectory, reference_geometry: dict[str, np.ndarray],
                       species: str, local_atom: int | slice | list[int]) -> np.ndarray:
    """Indices (into the reconstructed atom axis) of one local atom across all molecules.

    Example: the carbon atoms of benzene are local indices 0..5, so
    ``species_atom_index(rbt, geom, "benzene", slice(0, 6))`` returns the global
    atom indices of every carbon — ready to slice the array from
    :func:`reconstruct_atoms` for an element-specific RDF.
    """
    counts = {s: len(np.asarray(reference_geometry[s])) for s in set(rbt.species)}
    if isinstance(local_atom, slice):
        wanted = list(range(*local_atom.indices(counts[species])))
    elif isinstance(local_atom, int):
        wanted = [local_atom]
    else:
        wanted = list(local_atom)
    out: list[int] = []
    a = 0
    for s in rbt.species:
        if s == species:
            out.extend(a + w for w in wanted)
        a += counts[s]
    return np.asarray(out, dtype=int)


__all__ = [
    "RigidTrajectory",
    "read_rigid_bodies",
    "reconstruct_atoms",
    "reference_geometry_from_gsd",
    "species_atom_index",
    "quaternion_to_rotation_matrix",
]
