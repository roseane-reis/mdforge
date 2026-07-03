"""Water species profile — the only species-specific knowledge in the pipeline.

3-site water: one oxygen (the ordering/RDF site) and two hydrogens per molecule,
element order ``O, H, H``. Physical partial charges default to the common
3-point values (``O = -0.68 e``, ``H = +0.34 e``; net-neutral) but are normally
overridden from the user config so the module stays model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Common 3-point water partial charges (net-neutral); overridable from config.
_DEFAULT_CHARGES = {"O": -0.68, "H": 0.34}
_WATER_MOLAR_MASS = 18.01528  # g/mol


@dataclass
class WaterProfile:
    """Static, engine-agnostic description of a rigid 3-site water molecule."""

    atoms_per_molecule: int = 3
    oxygen_local_index: int = 0
    hydrogen_local_indices: tuple[int, ...] = (1, 2)
    element_order: tuple[str, ...] = ("O", "H", "H")
    charges_e: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_CHARGES))
    molar_mass_g_mol: float = _WATER_MOLAR_MASS
    species_name: str = "water"

    def per_molecule_charges(self) -> np.ndarray:
        """Per-atom charges for one molecule, in ``element_order`` (e)."""
        return np.array([self.charges_e[e] for e in self.element_order], dtype=float)

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
        molecule). Charges follow ``element_order``.
        """
        g = np.asarray(body_geometry, dtype=float)
        if g.shape[0] != self.atoms_per_molecule:
            raise ValueError(
                f"body_geometry has {g.shape[0]} atoms; expected "
                f"{self.atoms_per_molecule} for {self.species_name}"
            )
        return (self.per_molecule_charges()[:, None] * g).sum(axis=0)

    def validate_elements(self, elements: list[str]) -> None:
        """Check that a per-molecule element list matches ``O, H, H``."""
        got = tuple(e.capitalize() for e in elements[: self.atoms_per_molecule])
        if got != self.element_order:
            raise ValueError(
                f"topology molecule elements {got} do not match the water "
                f"profile {self.element_order}"
            )


def water_profile(
    *,
    charges_e: dict[str, float] | None = None,
    molar_mass_g_mol: float | None = None,
) -> WaterProfile:
    """Build a :class:`WaterProfile`, overriding charges / molar mass from config."""
    prof = WaterProfile()
    if charges_e:
        prof.charges_e = {str(k).capitalize(): float(v) for k, v in charges_e.items()}
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
