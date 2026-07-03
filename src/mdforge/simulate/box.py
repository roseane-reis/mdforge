"""Box / density math and a simple cubic-lattice box builder (goal a).

The density↔box-size↔molecule-count conversions are ported from
``analyzetool.process`` (``num_molecules``/``calc_density``/``calc_box_s``).
``replicate_cubic`` places copies of a molecule on a cubic grid to seed a liquid
box — a crude starting configuration meant to be relaxed by a subsequent
minimize/equilibration (engine-agnostic; geometry only).
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.elements import mass_of
from ..formats.txyz import TinkerXYZ

# Avogadro's number (CODATA 2018).
_N_A = 6.02214076e23


def molar_mass(molecule: TinkerXYZ | list[str]) -> float:
    """Total molar mass (g/mol) of a molecule, summed from element masses.

    Atom names are reduced to their element by stripping trailing digits.
    """
    names = molecule.names if isinstance(molecule, TinkerXYZ) else list(molecule)
    total = 0.0
    for nm in names:
        elem = "".join(c for c in nm if not c.isdigit())
        total += mass_of(elem)
    return total


def box_edge_for_density(n_molecules: int, molar_mass_g: float, density_g_cm3: float) -> float:
    """Cubic box edge (Å) holding ``n_molecules`` at the target density."""
    mass_g = n_molecules * molar_mass_g / _N_A
    volume_cm3 = mass_g / density_g_cm3
    return float((volume_cm3 * 1e24) ** (1.0 / 3.0))


def n_molecules_for_box(box_edge: float, molar_mass_g: float, density_g_cm3: float) -> int:
    """Number of molecules in a cubic box of edge ``box_edge`` (Å) at a density."""
    volume_cm3 = (box_edge ** 3) * 1e-24
    mass_g = volume_cm3 * density_g_cm3
    return int(round(mass_g / (molar_mass_g / _N_A)))


def density_of_box(n_molecules: int, molar_mass_g: float, box_edge: float) -> float:
    """Density (g/cm³) of ``n_molecules`` in a cubic box of edge ``box_edge`` (Å)."""
    mass_g = n_molecules * molar_mass_g / _N_A
    volume_cm3 = (box_edge ** 3) * 1e-24
    return float(mass_g / volume_cm3)


def replicate_cubic(
    molecule: TinkerXYZ,
    n_copies: int,
    box_edge: float,
    *,
    title: str | None = None,
) -> TinkerXYZ:
    """Replicate ``molecule`` onto a cubic grid inside a box of edge ``box_edge`` (Å).

    Returns a single :class:`TinkerXYZ` of ``n_copies`` molecules with the box
    vectors set and per-copy connectivity offset. Copies are centered in grid
    cells; overlaps are expected and should be relaxed by minimization.
    """
    if not molecule.is_tinker:
        raise ValueError("replicate_cubic needs a Tinker XYZ with atom types")
    n_atoms = molecule.n_atoms
    per_side = int(np.ceil(round(n_copies ** (1.0 / 3.0), 6)))
    while per_side ** 3 < n_copies:
        per_side += 1
    spacing = box_edge / per_side

    base = molecule.coords - molecule.coords.mean(axis=0)

    coords_blocks: list[np.ndarray] = []
    names: list[str] = []
    types: list[int] = []
    connectivity: list[list[int]] = []
    cells = itertools.product(range(per_side), repeat=3)
    for placed, (ix, iy, iz) in zip(range(n_copies), cells):
        offset = (np.array([ix, iy, iz]) + 0.5) * spacing
        coords_blocks.append(base + offset)
        names.extend(molecule.names)
        types.extend(int(t) for t in molecule.types)
        atom_offset = placed * n_atoms
        for bonds in molecule.connectivity:
            connectivity.append([b + atom_offset for b in bonds])

    return TinkerXYZ(
        names=names,
        coords=np.concatenate(coords_blocks, axis=0),
        types=np.array(types, dtype=int),
        connectivity=connectivity,
        box=np.array([box_edge, box_edge, box_edge, 90.0, 90.0, 90.0]),
        title=title or f"{n_copies} x {molecule.title or 'molecule'} cubic box",
    )


__all__ = [
    "molar_mass",
    "box_edge_for_density",
    "n_molecules_for_box",
    "density_of_box",
    "replicate_cubic",
]
