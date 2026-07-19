"""Water species profile — the only species-specific knowledge in the pipeline.

Supports both 3-site water (``O, H, H``) and 4-site (or higher) models that add
one or more massless *ghost* charge sites — the ``M``-site of TIP4P /
TIP4P-2005-style models. A ghost site carries partial charge but no mass: it is
excluded from the centre of mass and from the O/H radial distribution functions,
but included in the cell dipole that sets the dielectric constant. Physical
partial charges default to the common 3-point values (``O = -0.68 e``,
``H = +0.34 e``; net-neutral) but are normally overridden from the user config so
the module stays model-agnostic. Every site (including each ghost site) needs a
charge in ``charges_e``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....core.elements import ELEMENT_MASSES

# Common 3-point water partial charges (net-neutral); overridable from config.
_DEFAULT_CHARGES = {"O": -0.68, "H": 0.34}
_WATER_MOLAR_MASS = 18.01528  # g/mol


@dataclass
class WaterProfile:
    """Static, engine-agnostic description of a water molecule (3- or 4-site).

    A 4-site model records the ghost (``M``-site) local index in
    ``virtual_local_indices``; those sites are massless — dropped from the mass /
    centre-of-mass and RDF selections but kept for the cell dipole.
    """

    atoms_per_molecule: int = 3
    oxygen_local_index: int = 0
    hydrogen_local_indices: tuple[int, ...] = (1, 2)
    element_order: tuple[str, ...] = ("O", "H", "H")
    virtual_local_indices: tuple[int, ...] = ()
    charges_e: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_CHARGES))
    molar_mass_g_mol: float = _WATER_MOLAR_MASS
    species_name: str = "water"

    def per_molecule_charges(self) -> np.ndarray:
        """Per-atom charges for one molecule, in ``element_order`` (e)."""
        return np.array([self.charges_e[e] for e in self.element_order], dtype=float)

    def per_molecule_masses(self) -> np.ndarray:
        """Per-atom masses for one molecule, in ``element_order`` (amu).

        Ghost (``virtual_local_indices``) sites are massless, so they drop out of
        the mass-weighted centre of mass and never require an entry in the element
        mass table (an ``M``-site has no atomic mass).
        """
        virt = set(self.virtual_local_indices)
        return np.array(
            [0.0 if i in virt else ELEMENT_MASSES[e]
             for i, e in enumerate(self.element_order)],
            dtype=float,
        )

    def per_atom_charges(self, n_molecules: int) -> np.ndarray:
        """Per-atom charges for the whole system (``n_molecules × atoms``), e.

        Used for the atomistic cell-dipole of the DCD path (no rigid-body
        orientations available).
        """
        return np.tile(self.per_molecule_charges(), n_molecules)

    def net_charge(self) -> float:
        """Net molecular charge (should be ~0 for a physical water model)."""
        return float(self.per_molecule_charges().sum())

    def body_dipole(self, body_geometry: np.ndarray) -> np.ndarray:
        """Body-frame molecular dipole ``μ = Σ q_i r_i`` (e·Å).

        ``body_geometry`` is the ``(atoms_per_molecule, 3)`` body-frame template
        (any origin — the dipole is origin-independent for a net-neutral
        molecule). Charges follow ``element_order``; a ghost site contributes
        through its charge and position like any other site.
        """
        g = np.asarray(body_geometry, dtype=float)
        if g.shape[0] != self.atoms_per_molecule:
            raise ValueError(
                f"body_geometry has {g.shape[0]} atoms; expected "
                f"{self.atoms_per_molecule} for {self.species_name}"
            )
        return (self.per_molecule_charges()[:, None] * g).sum(axis=0)

    @property
    def real_element_order(self) -> tuple[str, ...]:
        """Element order of the real (massive) sites only — ghost sites dropped."""
        virt = set(self.virtual_local_indices)
        return tuple(e for i, e in enumerate(self.element_order) if i not in virt)

    def topology_atoms_per_molecule(self, elements: list[str]) -> int:
        """Validate a topology's per-molecule sites; return its atoms/molecule.

        A topology may list *every* site — ``O, H, H`` plus each ghost site, e.g.
        ``O, H, H, M`` (a Tinker txyz) — or *only the real atoms* ``O, H, H`` (a
        HOOMD PDB, where the ghost site lives only in the trajectory). Returns the
        matched per-molecule atom count (``atoms_per_molecule`` for the full
        layout, or the real-atom count); raises if neither layout matches.
        """
        def _cap(seq):
            return tuple(e.capitalize() for e in seq)

        full, real = self.element_order, self.real_element_order
        if len(elements) >= len(full) and _cap(elements[:len(full)]) == full:
            return len(full)
        if len(elements) >= len(real) and _cap(elements[:len(real)]) == real:
            return len(real)
        want = full if len(elements) >= len(full) else real
        raise ValueError(
            f"topology molecule sites {_cap(elements[:len(want)])} do not match "
            f"the water profile {want}"
        )


def water_profile(
    *,
    charges_e: dict[str, float] | None = None,
    molar_mass_g_mol: float | None = None,
    atoms_per_molecule: int | None = None,
    virtual_sites: list[str] | None = None,
) -> WaterProfile:
    """Build a :class:`WaterProfile`, optionally for a 4-site (M-site) model.

    ``virtual_sites`` names the massless charge (ghost) sites, e.g. ``["M"]`` for
    a TIP4P-style model; the per-molecule layout is then ``O, H, H`` followed by
    those sites, so ``atoms_per_molecule`` (when given) must equal ``3 +
    len(virtual_sites)``. Every site — including each ghost site — needs a charge
    in ``charges_e`` (the 3-point default only covers ``O``/``H``).

    Ghost-site names are matched as element symbols against the topology (the
    common single-symbol ``M``-site of 4-site water). A name that does not match
    the topology's per-molecule site order raises a clear error rather than
    silently mis-selecting atoms.
    """
    virtual = [str(s).strip().capitalize() for s in (virtual_sites or [])]
    order = ("O", "H", "H") + tuple(virtual)
    apm = atoms_per_molecule if atoms_per_molecule is not None else len(order)
    if apm != len(order):
        raise ValueError(
            f"atoms_per_molecule={apm} disagrees with the layout O, H, H + "
            f"virtual {virtual} ({len(order)} sites); set atoms_per_molecule="
            f"{len(order)} or adjust virtual_sites"
        )

    prof = WaterProfile()
    prof.atoms_per_molecule = apm
    prof.element_order = order
    prof.virtual_local_indices = tuple(range(3, len(order)))
    prof.oxygen_local_index = 0
    prof.hydrogen_local_indices = (1, 2)

    if charges_e:
        prof.charges_e = {str(k).capitalize(): float(v) for k, v in charges_e.items()}
    missing = [e for e in dict.fromkeys(order) if e not in prof.charges_e]
    if missing:
        raise ValueError(
            f"no charge for site(s) {missing}; set system.charges_e for every "
            f"site {list(dict.fromkeys(order))} (a 4-site model must give the "
            f"ghost-site charge, e.g. M)"
        )
    if molar_mass_g_mol:
        prof.molar_mass_g_mol = float(molar_mass_g_mol)
    return prof


# Registry hook for future liquids (keyed by species name).
_PROFILES = {"water": water_profile}


def get_profile(species: str, **kwargs) -> WaterProfile:
    """Return the profile builder for ``species`` (only ``water`` today)."""
    try:
        builder = _PROFILES[species.lower()]
    except KeyError as exc:
        raise KeyError(
            f"no evaluation profile for species {species!r}; available: "
            f"{sorted(_PROFILES)}"
        ) from exc
    return builder(**kwargs)


__all__ = ["WaterProfile", "water_profile", "get_profile"]
