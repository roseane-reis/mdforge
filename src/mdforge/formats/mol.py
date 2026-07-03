"""Coordinate → connectivity/SMILES perception via RDKit (goal c, optional).

The legacy ``analyzetool.xyz2mol`` vendored ~840 lines of the Jensen-group
``xyz2mol`` bond-perception code. Modern RDKit (≥2022.09) provides the same
capability natively via ``rdkit.Chem.rdDetermineBonds``, so this module is a
thin, lazy wrapper instead of a vendored fork (install ``mdforge[chem]``).

If a future need outgrows RDKit's perception, the Jensen ``xyz2mol`` can be
re-vendored here behind the same API.
"""

from __future__ import annotations

import numpy as np

from ..core.elements import Z_TO_SYMBOL


def _import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds  # noqa: F401  (registers the function)
    except ImportError as exc:  # pragma: no cover - only without rdkit
        raise ImportError(
            "Coordinate→molecule perception requires rdkit (>=2022.09). "
            "Install it with: pip install 'mdforge[chem]'"
        ) from exc
    return Chem


def _xyz_block(atomic_numbers, coords: np.ndarray, comment: str = "") -> str:
    coords = np.asarray(coords, dtype=float)
    symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in atomic_numbers]
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:<3s} {x:18.10f} {y:18.10f} {z:18.10f}")
    return "\n".join(lines) + "\n"


def xyz_to_rdkit(atomic_numbers, coords: np.ndarray, charge: int = 0):
    """Build an RDKit Mol with perceived bonds from atomic numbers + coordinates.

    Parameters
    ----------
    atomic_numbers:
        Length-N atomic numbers.
    coords:
        ``(N, 3)`` coordinates in Angstrom.
    charge:
        Total molecular charge (passed to bond perception).
    """
    Chem = _import_rdkit()
    from rdkit.Chem import rdDetermineBonds

    mol = Chem.MolFromXYZBlock(_xyz_block(atomic_numbers, coords))
    if mol is None:
        raise ValueError("RDKit could not parse the generated XYZ block")
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def xyz_to_smiles(atomic_numbers, coords: np.ndarray, charge: int = 0) -> str:
    """Return a canonical SMILES perceived from coordinates."""
    from rdkit import Chem

    return Chem.MolToSmiles(xyz_to_rdkit(atomic_numbers, coords, charge=charge))


__all__ = ["xyz_to_rdkit", "xyz_to_smiles"]
