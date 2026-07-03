"""Core data records for mdforge.

Contains:
- ``SpiceMolecule`` — canonical QM/model record; ported from
  prior internal tooling with the following fixes:
    * ``all_energy`` field-group no longer includes ``torques_per_center``
      (torques carry energy-like units but are NOT energies; converting them
      with update_energy_units() was silently wrong)
    * BOHR_TO_ANGSTROM / ANGSTROM_TO_BOHR imported from units.py instead of
      being duplicated as module-level literals
    * h5py import guarded so the module loads without h5py for non-SPICE workflows
- ``Trajectory``     — stub for MD trajectory data (Phase 4/5)
- ``BulkProperties`` — stub for computed liquid-phase properties (Phase 4)

Migration helpers from legacy joblib files are preserved: upgrade_legacy_to_v2,
ensure_cache, set_temporary_field, get_temporary_field.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from joblib import dump, load

from .units import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
)
from .units import (
    convert_energy as _ce,
)
from .units import (
    convert_gradient as _cg,
)
from .units import (
    convert_length as _cl,
)

# h5py is only needed for load_subset (SPICE HDF5 ingest); guard the import
# so mdforge.core loads cleanly even without h5py in the environment.
try:
    import h5py as _h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# h5py is a listed core dependency so this should never fire in normal usage;
# if it does the error message points at the fix.
_H5PY_MISSING_MSG = (
    "h5py is required for SPICE HDF5 ingest. "
    "Install it with: pip install h5py>=3.9"
)

# Approximate conversion for display/export; authoritative value lives in units.py
_HARTREE2KCALMOL = 627.5095

ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: 'H',  2: 'He', 3: 'Li',  4: 'Be',  5: 'B',  6: 'C',  7: 'N',  8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
    26: 'Fe', 28: 'Ni', 29: 'Cu', 30: 'Zn', 34: 'Se', 35: 'Br', 53: 'I',
}

SUBSET_FILTERS: dict[str, str] = {
    'dipeptides':         'Dipeptides',
    'solvated':           'Solvated Amino Acids',
    'amino-acid-ligand':  'Amino Acid Ligand',
    'des-monomers':       'DES Monomers',
    'des370k':            'DES370K',
    'ion-pairs':          'Ion Pairs',
    'pubchem':            'PubChem Set',
    'pubchem-boron':      'PubChem Boron Silicon',
    'solvated-pubchem':   'Solvated PubChem',
    'water':              'Water Clusters',
}

ModelOutputFormat = Literal["single_dict", "list", "dict_by_frame"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_model_output_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and "atoms" in obj and "centers" in obj


def _numeric_key_or_none(key: Any) -> float | None:
    if isinstance(key, (int, float, np.integer, np.floating)):
        return float(key)
    if isinstance(key, str):
        try:
            return float(key.strip())
        except ValueError:
            return None
    return None


def _ordered_mapping_items(mapping: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    items = list(mapping.items())
    numeric_keys = [_numeric_key_or_none(k) for k, _ in items]
    if all(v is not None for v in numeric_keys):
        return [item for _, item in sorted(zip(numeric_keys, items), key=lambda pair: pair[0])]
    return items


def normalize_model_outputs_payload(
    obj: Any, data_format: ModelOutputFormat
) -> list[dict[str, Any]]:
    """Normalize model outputs using an explicitly provided format.

    Supported formats
    -----------------
    ``"single_dict"``
        One conformation dict with keys ``{"atoms", "centers"}``.
    ``"list"``
        List/tuple of per-conformation dicts.
    ``"dict_by_frame"``
        Mapping like ``{0: {...}, 1: {...}, ...}``.
    """
    if data_format == "single_dict":
        if not _is_model_output_dict(obj):
            raise ValueError("data_format='single_dict' requires a dict with keys 'atoms' and 'centers'")
        return [dict(obj)]

    if data_format == "list":
        if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
            raise ValueError("data_format='list' requires a list/tuple of per-conformation dicts")
        values = list(obj)
        if not all(_is_model_output_dict(item) for item in values):
            raise ValueError("Every item in model_outputs must be a dict with keys 'atoms' and 'centers'")
        return [dict(item) for item in values]

    if data_format == "dict_by_frame":
        if not isinstance(obj, Mapping):
            raise ValueError("data_format='dict_by_frame' requires a mapping like {0: {...}, 1: {...}}")
        items = _ordered_mapping_items(obj)
        values = [value for _, value in items]
        if not all(_is_model_output_dict(value) for value in values):
            raise ValueError(
                "Every value in a dict_by_frame payload must be a dict with keys 'atoms' and 'centers'"
            )
        return [dict(value) for value in values]

    raise ValueError(f"Unsupported data_format: {data_format!r}")


def _rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# SpiceMolecule — canonical QM/model record
# ---------------------------------------------------------------------------

# Organic-subset SMILES atoms that may appear unbracketed (two-letter first).
_SMILES_ORGANIC_TWO = ('Cl', 'Br')
_SMILES_ORGANIC_ONE = set('BCNOPSFIbcnops')  # incl. aromatic lowercase b c n o p s


def _count_smiles_atoms(frag: str) -> int:
    """Count atoms in a SMILES fragment without rdkit.

    Counts each bracketed atom (``[...]``, including explicit/mapped H) once and
    each bare organic-subset atom once. Implicit hydrogens are NOT added, so the
    count is exact only for fully explicit (atom-mapped) SMILES — the caller guards
    on that.
    """
    n = i = 0
    while i < len(frag):
        ch = frag[i]
        if ch == '[':
            n += 1
            j = frag.find(']', i)
            i = len(frag) if j < 0 else j + 1
        elif frag[i:i + 2] in _SMILES_ORGANIC_TWO:
            n += 1
            i += 2
        elif ch in _SMILES_ORGANIC_ONE:
            n += 1
            i += 1
        else:
            i += 1
    return n


def _residue_ids_from_smiles(smiles: str | None, n_atoms: int) -> list[int]:
    """Per-atom residue numbers, split by SMILES fragment (monomer).

    A dimer/complex SMILES separates monomers with ``.``; DES370K / SPICE
    atom-mapped SMILES list every atom explicitly and in fragment order, so
    counting atoms per fragment recovers the monomer boundaries — fragment *k*
    becomes residue *k+1*. So a dimer gets monomer 1 → resSeq 1, monomer 2 →
    resSeq 2 (engines that infer connectivity from residue grouping then keep the
    monomers separate). Falls back to a single residue (all ones) when the SMILES
    is absent, has no ``.``, or its atom count does not match ``n_atoms`` (e.g.
    implicit-hydrogen SMILES, where the split can't be trusted).
    """
    if not smiles or '.' not in smiles:
        return [1] * n_atoms
    counts = [_count_smiles_atoms(frag) for frag in smiles.split('.')]
    if sum(counts) != n_atoms:
        return [1] * n_atoms
    res_ids: list[int] = []
    for res_seq, count in enumerate(counts, start=1):
        res_ids.extend([res_seq] * count)
    return res_ids


@dataclass
class SpiceMolecule:
    """Canonical record for one molecule from the SPICE dataset (or a model).

    Native units (as stored in the joblib file):
    - positions / conformations: **bohr**
    - energies: **Hartree**
    - gradients: **Hartree/bohr**

    Use ``update_energy_units``, ``update_gradient_units``,
    ``update_position_units`` to convert fields in-place.
    """

    name: str
    subset: str
    smiles: str

    atomic_numbers: np.ndarray          # (N,)     int
    conformations: np.ndarray           # (M,N,3)  float32 [bohr]
    dft_total_energy: np.ndarray        # (M,)     float64 [Hartree]
    dft_total_gradient: np.ndarray      # (M,N,3)  float32 [Hartree/bohr]
    formation_energy: np.ndarray        # (M,)     float64 [Hartree]

    # Optional QM property arrays
    mbis_charges: np.ndarray | None = None
    mbis_dipoles: np.ndarray | None = None
    mbis_quadrupoles: np.ndarray | None = None
    mbis_octupoles: np.ndarray | None = None
    scf_dipole: np.ndarray | None = None
    scf_quadrupole: np.ndarray | None = None
    mayer_indices: np.ndarray | None = None
    wiberg_lowdin_indices: np.ndarray | None = None

    # Per-center (rigid-body / multipole site) fields
    atom_to_center: np.ndarray | None = None       # (N,)     int
    rmsd_per_conf: np.ndarray | None = None        # (M,)     float
    flagged_confs: np.ndarray | None = None        # (M,)     bool
    center_ids: np.ndarray | None = None           # (C,)     int
    center_coords: np.ndarray | None = None        # (M,C,3)  [bohr]
    forces_per_center: np.ndarray | None = None    # (M,C,3)
    torques_per_center: np.ndarray | None = None   # (M,C,3)

    # Model output fields
    model_total_energy: np.ndarray | None = None
    center_force_unit: str | None = None
    center_torque_unit: str | None = None
    source_label: str | None = None

    # Dimer / interaction energy fields
    monomer_record_keys: tuple[str, str] | None = None
    monomer1_match_index: np.ndarray | None = None
    monomer2_match_index: np.ndarray | None = None
    monomer_match_mask: np.ndarray | None = None
    monomer1_total_energy: np.ndarray | None = None
    monomer2_total_energy: np.ndarray | None = None
    monomer1_total_gradient: np.ndarray | None = None
    monomer2_total_gradient: np.ndarray | None = None
    interaction_total_energy: np.ndarray | None = None
    interaction_total_gradient: np.ndarray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Initialisation / migration
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._ensure_optional_fields()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support unpickling of older serialized objects."""
        self.__dict__.update(state)
        self._ensure_optional_fields()

    def _ensure_optional_fields(self) -> None:
        """Add any fields that may be missing from an older serialized object."""
        defaults: dict[str, Any] = {
            'mbis_charges': None, 'mbis_dipoles': None,
            'mbis_quadrupoles': None, 'mbis_octupoles': None,
            'scf_dipole': None, 'scf_quadrupole': None,
            'mayer_indices': None, 'wiberg_lowdin_indices': None,
            'atom_to_center': None, 'center_ids': None,
            'center_coords': None, 'forces_per_center': None,
            'torques_per_center': None, 'rmsd_per_conf': None,
            'flagged_confs': None, 'model_total_energy': None,
            'center_force_unit': None, 'center_torque_unit': None,
            'source_label': None, 'monomer_record_keys': None,
            'monomer1_match_index': None, 'monomer2_match_index': None,
            'monomer_match_mask': None,
            'monomer1_total_energy': None, 'monomer2_total_energy': None,
            'monomer1_total_gradient': None, 'monomer2_total_gradient': None,
            'interaction_total_energy': None, 'interaction_total_gradient': None,
        }
        for key, default in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, default)

        # Migrate old field names from legacy serialized objects
        legacy_aliases = {
            'monomer_record_keys_qm': 'monomer_record_keys',
            'monomer1_qm_match_index': 'monomer1_match_index',
            'monomer2_qm_match_index': 'monomer2_match_index',
            'monomer_qm_match_mask': 'monomer_match_mask',
            'monomer1_qm_dft_total_energy': 'monomer1_total_energy',
            'monomer2_qm_dft_total_energy': 'monomer2_total_energy',
            'monomer1_qm_dft_total_gradient': 'monomer1_total_gradient',
            'monomer2_qm_dft_total_gradient': 'monomer2_total_gradient',
            'interaction_qm_dft_total_energy': 'interaction_total_energy',
            'interaction_qm_dft_total_gradient': 'interaction_total_gradient',
        }
        for old_name, new_name in legacy_aliases.items():
            if hasattr(self, old_name) and getattr(self, new_name) is None:
                setattr(self, new_name, getattr(self, old_name))

        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        # ``cache`` is a dataclass field (repr=False); legacy-pickled objects predate it,
        # so restore it here or dataclasses.fields() iteration (e.g. in the dimer matcher)
        # would AttributeError on a missing attribute.
        if not hasattr(self, 'cache') or self.cache is None:
            self.cache = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def n_conformations(self) -> int:
        return len(self.conformations)

    @property
    def n_centers(self) -> int:
        return 0 if self.center_ids is None else len(self.center_ids)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, filename: str | Path, compress: int = 3) -> None:
        """Save this record to a joblib file."""
        dump(self, filename, compress=compress)
        print(f"  Saved {self.name} -> {filename}")

    @classmethod
    def load(cls, filename: str | Path) -> SpiceMolecule:
        """Load a SpiceMolecule (or legacy equivalent) from a joblib file.

        Objects pickled from pre-consolidation standalone modules are
        deserialized transparently via the legacy module shims in
        :mod:`mdforge.core.io`.
        """
        try:
            obj = load(filename)
        except ModuleNotFoundError:
            from .io import legacy_record_shims
            with legacy_record_shims():
                obj = load(filename)
        print(f"  Loaded {filename}")
        if not isinstance(obj, cls):
            try:
                obj.atomic_numbers  # noqa: B018  — probe for duck-type compatibility
            except Exception as exc:
                raise TypeError(
                    f"{filename} does not contain a SpiceMolecule-like object"
                ) from exc
            obj = upgrade_legacy_to_v2(obj)
        obj._ensure_optional_fields()
        return obj

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def set_cache(self, name: str, value: Any, *, as_attribute: bool = False) -> Any:
        self.cache[name] = value
        if as_attribute:
            setattr(self, name, value)
        return value

    def get_cache(self, name: str, default: Any = None) -> Any:
        return self.cache.get(name, default)

    def has_cache(self, name: str) -> bool:
        return name in self.cache

    def clear_cache(self, name: str | None = None) -> None:
        if name is None:
            self.cache.clear()
        else:
            self.cache.pop(name, None)

    # ------------------------------------------------------------------
    # Constructors from model outputs
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_atoms_and_centers(
        model_output: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        atoms = sorted(model_output['atoms'], key=lambda item: int(item['atom']))
        centers = sorted(model_output['centers'], key=lambda item: int(item['center']))
        return atoms, centers

    @classmethod
    def from_model_outputs(
        cls,
        *,
        name: str,
        atomic_numbers: Sequence[int] | np.ndarray,
        model_outputs: Any,
        data_format: ModelOutputFormat,
        subset: str = 'model-output',
        smiles: str = '',
        reference_conformations: np.ndarray | None = None,
        gradients: np.ndarray | None = None,
        formation_energy: np.ndarray | None = None,
        source_label: str = 'model',
        center_force_unit: str = 'unknown',
        center_torque_unit: str = 'unknown',
        energy_unit: str = 'hartree',
    ) -> SpiceMolecule:
        outputs = normalize_model_outputs_payload(model_outputs, data_format=data_format)
        if len(outputs) == 0:
            raise ValueError('model_outputs must contain at least one conformation')

        do_rmsd = reference_conformations is not None
        if do_rmsd:
            reference_conformations = np.asarray(reference_conformations, dtype=np.float32)

        atomic_numbers = np.asarray(atomic_numbers, dtype=int).flatten()
        n_atoms = len(atomic_numbers)
        n_conf = len(outputs)

        _, first_centers = cls._sorted_atoms_and_centers(outputs[0])
        n_centers = len(first_centers)

        conformations = np.full((n_conf, n_atoms, 3), np.nan, dtype=np.float32)
        center_coords = np.full((n_conf, n_centers, 3), np.nan, dtype=np.float32)
        forces_per_center = np.full((n_conf, n_centers, 3), np.nan, dtype=np.float64)
        torques_per_center = np.full((n_conf, n_centers, 3), np.nan, dtype=np.float64)
        model_total_energy = np.full((n_conf,), np.nan, dtype=np.float64)
        rmsd_per_conf = np.full((n_conf,), np.nan, dtype=np.float64)
        flagged_confs = np.zeros((n_conf,), dtype=bool)
        atoms2center = np.full((n_conf, n_atoms), -1, dtype=np.int32)
        centerids = np.full((n_conf, n_centers), -1, dtype=np.int32)

        atom_to_center = None
        center_ids = None

        for i, output in enumerate(outputs):
            atoms, centers = cls._sorted_atoms_and_centers(output)

            bad_shape = (len(atoms) != n_atoms) or (len(centers) != n_centers)
            if bad_shape:
                flagged_confs[i] = True
                model_total_energy[i] = float(output.get('energy', np.nan))
                model_e = float(output.get('energy', np.nan))
                test_e = -100
                try:
                    test_e = float(output['pairwise'][0]['energy'])
                except Exception:
                    pass
                print(test_e, model_e)
                if test_e != -100 and test_e != model_e:
                    model_total_energy[i] = test_e
                if do_rmsd and reference_conformations is not None:
                    conformations[i] = reference_conformations[i]
                elif len(atoms) == n_atoms:
                    atom_coords_ang = np.array(
                        [[a['x'], a['y'], a['z']] for a in atoms], dtype=np.float64
                    )
                    conformations[i] = atom_coords_ang * ANGSTROM_TO_BOHR
                else:
                    raise ValueError(
                        f'Conformation {i} has {len(atoms)} atoms and {len(centers)} centers; '
                        f'expected {n_atoms} atoms and {n_centers} centers'
                    )
                continue

            atom_ids = np.array([int(a['atom']) for a in atoms], dtype=int)
            expected_atom_ids = np.arange(1, n_atoms + 1, dtype=int)
            if not np.array_equal(atom_ids, expected_atom_ids):
                raise ValueError(
                    f'Conformation {i} atom ids must run 1..N; got {atom_ids}'
                )

            current_atom_to_center = np.array([int(a['center']) for a in atoms], dtype=int)
            current_center_ids = np.array([int(c['center']) for c in centers], dtype=int)
            atoms2center[i] = current_atom_to_center
            centerids[i] = current_center_ids

            if atom_to_center is None:
                atom_to_center = current_atom_to_center.copy()
            elif not np.array_equal(atom_to_center, current_atom_to_center):
                flagged_confs[i] = True

            if center_ids is None:
                center_ids = current_center_ids.copy()
            elif not np.array_equal(center_ids, current_center_ids):
                flagged_confs[i] = True

            atom_coords_ang = np.array(
                [[a['x'], a['y'], a['z']] for a in atoms], dtype=np.float64
            )
            center_coords_ang = np.array(
                [[c['x'], c['y'], c['z']] for c in centers], dtype=np.float64
            )
            conformations[i] = atom_coords_ang * ANGSTROM_TO_BOHR
            center_coords[i] = center_coords_ang * ANGSTROM_TO_BOHR
            forces_per_center[i] = np.array(
                [[c['fx'], c['fy'], c['fz']] for c in centers], dtype=np.float64
            )
            torques_per_center[i] = np.array(
                [[c['mx'], c['my'], c['mz']] for c in centers], dtype=np.float64
            )
            model_total_energy[i] = float(output.get('energy', np.nan))
            if do_rmsd and reference_conformations is not None:
                rmsd_per_conf[i] = _rmsd(reference_conformations[i], conformations[i])

        if gradients is None:
            gradients = np.full((n_conf, n_atoms, 3), np.nan, dtype=np.float32)
        else:
            gradients = np.asarray(gradients, dtype=np.float32)
            if gradients.shape != (n_conf, n_atoms, 3):
                raise ValueError(f'gradients shape {gradients.shape} != {(n_conf, n_atoms, 3)}')

        if formation_energy is None:
            formation_energy = np.full((n_conf,), np.nan, dtype=np.float64)
        else:
            formation_energy = np.asarray(formation_energy, dtype=np.float64)
            if formation_energy.shape != (n_conf,):
                raise ValueError(f'formation_energy shape {formation_energy.shape} != {(n_conf,)}')

        if flagged_confs.any():
            if atom_to_center is not None:
                print(f"Default Atom2Center: {' '.join(map(str, atom_to_center))}")
            if center_ids is not None:
                print(f"Default Center IDs: {' '.join(map(str, center_ids))}")
            for i in np.where(flagged_confs)[0]:
                print(f"Conformation {i} was flagged.")

        return cls(
            name=name, subset=subset, smiles=smiles,
            atomic_numbers=atomic_numbers,
            conformations=conformations,
            dft_total_energy=model_total_energy.copy(),
            dft_total_gradient=gradients,
            formation_energy=formation_energy,
            atom_to_center=atom_to_center,
            rmsd_per_conf=rmsd_per_conf,
            flagged_confs=flagged_confs,
            center_ids=center_ids,
            center_coords=center_coords,
            forces_per_center=forces_per_center,
            torques_per_center=torques_per_center,
            model_total_energy=model_total_energy,
            center_force_unit=center_force_unit,
            center_torque_unit=center_torque_unit,
            source_label=source_label,
            metadata={
                'energy_unit': energy_unit,
                'coordinates_input_unit': 'angstrom',
                'model_output_format': data_format,
            },
        )

    def attach_model_outputs(
        self,
        model_outputs: Any,
        *,
        data_format: ModelOutputFormat,
        source_label: str = 'model',
        center_force_unit: str = 'unknown',
        center_torque_unit: str = 'unknown',
        energy_unit: str = 'hartree',
        verify_coordinates: bool = True,
        atol_bohr: float = 1e-4,
    ) -> None:
        parsed = SpiceMolecule.from_model_outputs(
            name=self.name, subset=self.subset, smiles=self.smiles,
            atomic_numbers=self.atomic_numbers,
            reference_conformations=self.conformations,
            model_outputs=model_outputs,
            data_format=data_format,
            source_label=source_label,
            center_force_unit=center_force_unit,
            center_torque_unit=center_torque_unit,
            energy_unit=energy_unit,
        )
        if parsed.conformations.shape != self.conformations.shape and verify_coordinates:
            raise ValueError(
                f'Model conformations shape {parsed.conformations.shape} '
                f'!= reference shape {self.conformations.shape}'
            )
        if not np.allclose(parsed.conformations, self.conformations, atol=atol_bohr, rtol=0.0):
            diff_val = np.abs(parsed.conformations - self.conformations)
            print(
                f'Model coordinates do not match reference within {atol_bohr} bohr '
                f'(max abs diff = {np.max(diff_val):.4e} bohr)'
            )
            if verify_coordinates:
                raise ValueError('Failed coordinate verification')

        self.atom_to_center = parsed.atom_to_center
        self.rmsd_per_conf = parsed.rmsd_per_conf
        self.flagged_confs = parsed.flagged_confs
        self.center_ids = parsed.center_ids
        self.center_coords = parsed.center_coords
        self.forces_per_center = parsed.forces_per_center
        self.torques_per_center = parsed.torques_per_center
        self.model_total_energy = parsed.model_total_energy
        self.center_force_unit = parsed.center_force_unit
        self.center_torque_unit = parsed.center_torque_unit
        self.source_label = source_label
        self.metadata.update(parsed.metadata)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_pdb(self, output_dir: Path, conf_indices=None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if conf_indices is None:
            conf_indices = range(self.conformations.shape[0])
        symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(int(z), 'X') for z in self.atomic_numbers]
        element_counts: dict[str, int] = {}
        atom_names = []
        for sym in symbols:
            element_counts[sym] = element_counts.get(sym, 0) + 1
            atom_names.append(f"{sym}{element_counts[sym]}")
        res_ids = _residue_ids_from_smiles(self.smiles, len(symbols))
        safe_name = self.name.replace('/', '_').replace(' ', '_')
        for i in conf_indices:
            pdb_path = output_dir / f"{safe_name}_conf{i:04d}.pdb"
            coords_ang = self.conformations[i] * BOHR_TO_ANGSTROM
            with open(pdb_path, 'w') as fh:
                fh.write(f"REMARK  Molecule         : {self.name}\n")
                fh.write(f"REMARK  Subset           : {self.subset}\n")
                fh.write(f"REMARK  SMILES           : {self.smiles}\n")
                fh.write(f"REMARK  Conformation     : {i}\n")
                try:
                    fh.write(f"REMARK  DFT Total Energy : {self.dft_total_energy[i]:.10f} Ha\n")
                    fh.write(f"REMARK  Formation Energy : {self.formation_energy[i]:.10f} Ha\n")
                except Exception:
                    pass
                for j, (aname, sym, pos) in enumerate(zip(atom_names, symbols, coords_ang)):
                    fh.write(
                        f"HETATM{j + 1:5d} {aname:<4s} MOL A{res_ids[j]:>4d}    "
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                        f"  1.00  0.00          {sym:>2s}\n"
                    )
                fh.write("END\n")

    def to_xyz(
        self, output_dir: Path, conf_indices=None, comment: str = "", suffix: str = ""
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if conf_indices is None:
            conf_indices = range(self.n_conformations)
        symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(int(z), 'X') for z in self.atomic_numbers]
        safe_name = self.name.replace('/', '_').replace(' ', '_')
        xyz_path = (
            output_dir / f"{safe_name}_{suffix}.xyz"
            if suffix else output_dir / f"{safe_name}.xyz"
        )
        with open(xyz_path, 'w') as fh:
            for i in conf_indices:
                fh.write(f"{len(symbols)}\n")
                e_f = _HARTREE2KCALMOL * self.dft_total_energy[i]
                cur_comment = (
                    comment if comment
                    else f"{safe_name} | Conf {i} | DFT E: {e_f:.6f} kcal/mol"
                )
                fh.write(f"{cur_comment}\n")
                coords_ang = self.conformations[i] * BOHR_TO_ANGSTROM
                for symbol, (x, y, z) in zip(symbols, coords_ang):
                    fh.write(f"{symbol:<3} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    def to_xyz_template(self, conf_indices=None, comment: str = "") -> str:
        if conf_indices is None:
            conf_indices = range(self.n_conformations)
        symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(int(z), 'X') for z in self.atomic_numbers]
        xyz_data = ""
        for i in conf_indices:
            xyz_data += f"{len(symbols)}\n{comment}\n"
            coords_ang = self.conformations[i] * BOHR_TO_ANGSTROM
            for symbol, (x, y, z) in zip(symbols, coords_ang):
                xyz_data += f"{symbol:<3} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        return xyz_data

    # ------------------------------------------------------------------
    # Legacy property aliases (backward compat with old notebooks)
    # ------------------------------------------------------------------

    @property
    def monomer_record_keys_qm(self):
        return self.monomer_record_keys

    @monomer_record_keys_qm.setter
    def monomer_record_keys_qm(self, value):
        self.monomer_record_keys = value

    @property
    def monomer1_qm_match_index(self):
        return self.monomer1_match_index

    @monomer1_qm_match_index.setter
    def monomer1_qm_match_index(self, value):
        self.monomer1_match_index = value

    @property
    def monomer2_qm_match_index(self):
        return self.monomer2_match_index

    @monomer2_qm_match_index.setter
    def monomer2_qm_match_index(self, value):
        self.monomer2_match_index = value

    @property
    def monomer_qm_match_mask(self):
        return self.monomer_match_mask

    @monomer_qm_match_mask.setter
    def monomer_qm_match_mask(self, value):
        self.monomer_match_mask = value

    @property
    def monomer1_qm_dft_total_energy(self):
        return self.monomer1_total_energy

    @monomer1_qm_dft_total_energy.setter
    def monomer1_qm_dft_total_energy(self, value):
        self.monomer1_total_energy = value

    @property
    def monomer2_qm_dft_total_energy(self):
        return self.monomer2_total_energy

    @monomer2_qm_dft_total_energy.setter
    def monomer2_qm_dft_total_energy(self, value):
        self.monomer2_total_energy = value

    @property
    def monomer1_qm_dft_total_gradient(self):
        return self.monomer1_total_gradient

    @monomer1_qm_dft_total_gradient.setter
    def monomer1_qm_dft_total_gradient(self, value):
        self.monomer1_total_gradient = value

    @property
    def monomer2_qm_dft_total_gradient(self):
        return self.monomer2_total_gradient

    @monomer2_qm_dft_total_gradient.setter
    def monomer2_qm_dft_total_gradient(self, value):
        self.monomer2_total_gradient = value

    @property
    def interaction_qm_dft_total_energy(self):
        return self.interaction_total_energy

    @interaction_qm_dft_total_energy.setter
    def interaction_qm_dft_total_energy(self, value):
        self.interaction_total_energy = value

    @property
    def interaction_qm_dft_total_gradient(self):
        return self.interaction_total_gradient

    @interaction_qm_dft_total_gradient.setter
    def interaction_qm_dft_total_gradient(self, value):
        self.interaction_total_gradient = value


# ---------------------------------------------------------------------------
# Trajectory  (Phase 4 / 5)
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Time-ordered MD trajectory / per-frame observables from a simulation.

    This is the array-based hand-off between *parsing* (engine output → arrays)
    and *computation* (arrays → properties).  The kernels in ``mdforge.liquid``
    consume these arrays directly and never touch a log file.

    All fields are optional: a trajectory parsed from a Tinker ``analyze`` log
    may carry only ``potential_energy``, ``dipole``, and ``volume`` (no
    coordinates), while one built for viscosity carries ``velocities`` and
    ``virial`` but no ``dipole``.

    Units (convention)
    ------------------
    - positions:  Angstrom            (T, N, 3)
    - velocities: Angstrom/ps         (T, N, 3)
    - energies:   kcal/mol            (T,)
    - volume:     Angstrom³           (T,)
    - virial:     kcal/mol            (T, 3, 3)
    - dipole:     e·Angstrom          (T, 3)   (Tinker "Dipole X,Y,Z-Components")
    - box:        Angstrom            (T, 3, 3) or (T, 6) [a b c α β γ]
    - masses:     amu                 (N,)
    """

    positions: np.ndarray | None = None          # (T, N, 3)  [Angstrom]
    velocities: np.ndarray | None = None          # (T, N, 3)  [Angstrom/ps]
    potential_energy: np.ndarray | None = None    # (T,)        [kcal/mol]
    kinetic_energy: np.ndarray | None = None      # (T,)        [kcal/mol]
    total_energy: np.ndarray | None = None        # (T,)        [kcal/mol]
    volume: np.ndarray | None = None              # (T,)        [Angstrom³]
    dipole: np.ndarray | None = None              # (T, 3)      [e·Angstrom]
    virial: np.ndarray | None = None              # (T, 3, 3)   [kcal/mol]
    box_vectors: np.ndarray | None = None         # (T, 3, 3) or (T, 6)
    masses: np.ndarray | None = None              # (N,)        [amu]
    n_molecules: int | None = None
    n_atoms_per_molecule: int | None = None
    temperature_K: float = 298.15
    dt_ps: float = 0.001
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of frames, inferred from whichever per-frame array exists."""
        for arr in (self.potential_energy, self.kinetic_energy, self.total_energy,
                    self.volume, self.dipole, self.virial, self.positions, self.velocities):
            if arr is not None:
                return len(arr)
        return 0

    @property
    def n_atoms(self) -> int:
        """Number of atoms, inferred from coordinates, velocities, or masses."""
        for arr in (self.positions, self.velocities):
            if arr is not None:
                return arr.shape[1]
        if self.masses is not None:
            return len(self.masses)
        return 0

    @property
    def total_mass(self) -> float | None:
        """Total system mass in amu (sum of per-atom masses), or None."""
        return None if self.masses is None else float(np.sum(self.masses))

    @property
    def enthalpy(self) -> np.ndarray | None:
        """H = PE + KE per frame, when both are available (else PE, else None)."""
        if self.potential_energy is not None and self.kinetic_energy is not None:
            n = min(len(self.potential_energy), len(self.kinetic_energy))
            return self.potential_energy[:n] + self.kinetic_energy[:n]
        return self.potential_energy


# ---------------------------------------------------------------------------
# BulkProperties stub  (Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class BulkProperties:
    """Computed bulk-phase properties for a liquid simulation.

    .. note::
        Stub — full implementation in Phase 4 (liquid/).  The ``mdforge.liquid``
        module will provide functions that accept a ``Trajectory`` and return a
        ``BulkProperties`` record.

    All experimental reference values carry the same field names so they can
    be compared directly with computed values.
    """

    temperature_K: float
    density_kg_m3: float | None = None          # kg/m³
    delta_hvap_kcal_mol: float | None = None    # kcal/mol
    dielectric: float | None = None             # dimensionless
    kappa_T: float | None = None                # 10⁻⁶ bar⁻¹  (isothermal compressibility)
    alpha_T: float | None = None                # 10⁻⁴ K⁻¹    (thermal expansivity)
    surface_tension_mN_m: float | None = None   # mN/m
    diffusion_cm2_s: float | None = None        # cm²/s
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HDF5 ingest
# ---------------------------------------------------------------------------

def _read_str(dataset) -> str:
    val = dataset[()]
    if isinstance(val, np.ndarray):
        val = val.flat[0]
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val)


def _optional(group, key) -> np.ndarray | None:
    return group[key][:] if key in group else None


def load_subset(
    hdf5_path: Path, subset_keyword: str | None
) -> dict[str, SpiceMolecule]:
    """Load a subset of the SPICE HDF5 file into a dict of SpiceMolecule objects.

    Parameters
    ----------
    hdf5_path:
        Path to the SPICE ``*.hdf5`` file.
    subset_keyword:
        String to match against the ``subset`` field (e.g. ``"Dipeptides"``).
        Pass ``None`` to load all groups.
    """
    if not _HAS_H5PY:
        raise ImportError(_H5PY_MISSING_MSG)

    molecules: dict[str, SpiceMolecule] = {}
    with _h5py.File(hdf5_path, 'r') as f:
        all_keys = list(f.keys())
        label = f"containing '{subset_keyword}'" if subset_keyword else "(all groups)"
        print(f"Scanning {len(all_keys)} groups {label} …")
        for name in all_keys:
            grp = f[name]
            if 'subset' not in grp:
                continue
            subset_str = _read_str(grp['subset'])
            if subset_keyword is not None and subset_keyword not in subset_str:
                continue
            molecules[name] = SpiceMolecule(
                name=name,
                subset=subset_str,
                smiles=_read_str(grp['smiles']),
                atomic_numbers=grp['atomic_numbers'][:].astype(int).flatten(),
                conformations=grp['conformations'][:],
                dft_total_energy=grp['dft_total_energy'][:],
                dft_total_gradient=grp['dft_total_gradient'][:],
                formation_energy=grp['formation_energy'][:],
                mbis_charges=_optional(grp, 'mbis_charges'),
                mbis_dipoles=_optional(grp, 'mbis_dipoles'),
                mbis_quadrupoles=_optional(grp, 'mbis_quadrupoles'),
                mbis_octupoles=_optional(grp, 'mbis_octupoles'),
                scf_dipole=_optional(grp, 'scf_dipole'),
                scf_quadrupole=_optional(grp, 'scf_quadrupole'),
                mayer_indices=_optional(grp, 'mayer_indices'),
                wiberg_lowdin_indices=_optional(grp, 'wiberg_lowdin_indices'),
            )
    print(f"  → Found {len(molecules)} molecules.\n")
    return molecules


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def apply_update(mol: Any, new_data: Sequence[Any], field_names: Sequence[str]) -> Any:
    """Return a copy of *mol* with specified fields replaced by *new_data*."""
    if not is_dataclass(mol) or isinstance(mol, type):
        raise TypeError("mol must be a dataclass instance")
    field_names = list(field_names)
    new_data = list(new_data)
    if len(new_data) != len(field_names):
        raise ValueError(
            f"new_data and field_names must have the same length; "
            f"got {len(new_data)} and {len(field_names)}"
        )
    valid_fields = {f.name for f in fields(mol)}
    updates: dict[str, Any] = {}
    for fname, data in zip(field_names, new_data):
        if fname not in valid_fields:
            raise AttributeError(f"{type(mol).__name__} has no field {fname!r}")
        current = getattr(mol, fname)
        if isinstance(current, np.ndarray):
            updates[fname] = np.asarray(data, dtype=current.dtype)
        elif current is None and isinstance(data, (list, tuple)):
            updates[fname] = np.asarray(data)
        else:
            updates[fname] = data
    return replace(mol, **updates)


def ensure_cache(record: Any, cache_name: str = "cache") -> dict:
    """Ensure *record* has a mutable cache dict and return it."""
    cache = getattr(record, cache_name, None)
    if cache is None:
        cache = {}
        setattr(record, cache_name, cache)
    elif not isinstance(cache, dict):
        raise TypeError(f"{cache_name!r} exists but is not a dict.")
    return cache


def set_temporary_field(
    record: Any,
    name: str,
    value: Any,
    *,
    cache_name: str = "cache",
    as_attribute: bool = True,
    overwrite: bool = True,
) -> Any:
    """Store a temporary computed field in *record.cache* (and optionally as an attribute)."""
    cache = ensure_cache(record, cache_name=cache_name)
    if not overwrite and name in cache:
        return cache[name]
    cache[name] = value
    if as_attribute:
        setattr(record, name, value)
    return value


def get_temporary_field(
    record: Any,
    name: str,
    *,
    cache_name: str = "cache",
    default: Any = None,
) -> Any:
    """Retrieve a temporary field from cache first, then from attribute."""
    cache = getattr(record, cache_name, None)
    if isinstance(cache, dict) and name in cache:
        return cache[name]
    return getattr(record, name, default)


def upgrade_legacy_to_v2(legacy_obj: Any) -> SpiceMolecule:
    """Coerce a legacy (older-module) SpiceMolecule-like object to the current class."""
    def get(name: str, default=None):
        return getattr(legacy_obj, name, default)

    return SpiceMolecule(
        name=get('name'),
        subset=get('subset', ''),
        smiles=get('smiles', ''),
        atomic_numbers=np.asarray(get('atomic_numbers'), dtype=int),
        conformations=np.asarray(get('conformations'), dtype=np.float32),
        dft_total_energy=np.asarray(get('dft_total_energy'), dtype=np.float64),
        dft_total_gradient=np.asarray(get('dft_total_gradient'), dtype=np.float32),
        formation_energy=np.asarray(get('formation_energy'), dtype=np.float64),
        mbis_charges=get('mbis_charges'),
        mbis_dipoles=get('mbis_dipoles'),
        mbis_quadrupoles=get('mbis_quadrupoles'),
        mbis_octupoles=get('mbis_octupoles'),
        scf_dipole=get('scf_dipole'),
        scf_quadrupole=get('scf_quadrupole'),
        mayer_indices=get('mayer_indices'),
        wiberg_lowdin_indices=get('wiberg_lowdin_indices'),
        atom_to_center=get('atom_to_center'),
        center_ids=get('center_ids'),
        center_coords=get('center_coords'),
        rmsd_per_conf=get('rmsd_per_conf'),
        flagged_confs=get('flagged_confs'),
        forces_per_center=get('forces_per_center'),
        torques_per_center=get('torques_per_center'),
        model_total_energy=get('model_total_energy'),
        center_force_unit=get('center_force_unit'),
        center_torque_unit=get('center_torque_unit'),
        source_label=get('source_label'),
        monomer_record_keys=get('monomer_record_keys', get('monomer_record_keys_qm')),
        monomer1_match_index=get('monomer1_match_index', get('monomer1_qm_match_index')),
        monomer2_match_index=get('monomer2_match_index', get('monomer2_qm_match_index')),
        monomer_match_mask=get('monomer_match_mask', get('monomer_qm_match_mask')),
        monomer1_total_energy=get('monomer1_total_energy', get('monomer1_qm_dft_total_energy')),
        monomer2_total_energy=get('monomer2_total_energy', get('monomer2_qm_dft_total_energy')),
        monomer1_total_gradient=get('monomer1_total_gradient', get('monomer1_qm_dft_total_gradient')),
        monomer2_total_gradient=get('monomer2_total_gradient', get('monomer2_qm_dft_total_gradient')),
        interaction_total_energy=get('interaction_total_energy', get('interaction_qm_dft_total_energy')),
        interaction_total_gradient=get('interaction_total_gradient', get('interaction_qm_dft_total_gradient')),
        metadata=dict(get('metadata', {}) or {}),
        cache=dict(get('cache', {}) or {}),
    )


# ---------------------------------------------------------------------------
# Unit-conversion method patching (keeps SpiceMolecule definition clean above)
# _ce / _cg / _cl (convert_energy/gradient/length) are imported at module top.
# ---------------------------------------------------------------------------

def _field_groups() -> dict[str, list[str]]:
    return {
        'monomer_total_energy': ['monomer1_total_energy', 'monomer2_total_energy'],
        'monomer_total_gradient': ['monomer1_total_gradient', 'monomer2_total_gradient'],
        # NOTE: torques_per_center is intentionally NOT in all_energy.
        #       Torques carry energy-like units but are NOT energies; converting
        #       them via update_energy_units() was a latent bug in the upstream code.
        'all_energy': [
            'dft_total_energy', 'formation_energy', 'model_total_energy',
            'monomer1_total_energy', 'monomer2_total_energy', 'interaction_total_energy',
        ],
        'all_gradient': [
            'dft_total_gradient', 'forces_per_center',
            'monomer1_total_gradient', 'monomer2_total_gradient', 'interaction_total_gradient',
        ],
        'all_position': ['conformations', 'center_coords'],
        'positions': ['conformations', 'center_coords'],
    }


def _expand_fields(field_list):
    if isinstance(field_list, str):
        field_list = [field_list]
    groups = _field_groups()
    out, seen = [], set()
    for f in field_list:
        for expanded in groups.get(f, [f]):
            if expanded not in seen:
                seen.add(expanded)
                out.append(expanded)
    return out


def _default_field_units(self):
    metadata = getattr(self, 'metadata', None) or {}
    model_energy_unit = metadata.get('energy_unit', 'Hartree')
    return {
        'conformations': 'bohr',
        'center_coords': 'bohr',
        'dft_total_energy': 'Hartree',
        'formation_energy': 'Hartree',
        'dft_total_gradient': 'Hartree/bohr',
        'model_total_energy': model_energy_unit,
        'forces_per_center': getattr(self, 'center_force_unit', None) or 'Hartree/bohr',
        'torques_per_center': getattr(self, 'center_torque_unit', None) or model_energy_unit,
        'monomer1_total_energy': model_energy_unit,
        'monomer2_total_energy': model_energy_unit,
        'interaction_total_energy': model_energy_unit,
        'monomer1_total_gradient': getattr(self, 'center_force_unit', None) or 'Hartree/bohr',
        'monomer2_total_gradient': getattr(self, 'center_force_unit', None) or 'Hartree/bohr',
        'interaction_total_gradient': getattr(self, 'center_force_unit', None) or 'Hartree/bohr',
    }


def _ensure_field_units(self):
    if not hasattr(self, 'metadata') or self.metadata is None:
        self.metadata = {}
    self.metadata.setdefault('field_units', {})
    for f, u in _default_field_units(self).items():
        self.metadata['field_units'].setdefault(f, u)
    return self.metadata['field_units']


def _update_units(self, *, converter, from_unit, to_unit, fields):
    resolved = _expand_fields(fields)
    field_units = _ensure_field_units(self)
    for f in resolved:
        if not hasattr(self, f):
            raise AttributeError(f"{type(self).__name__} has no field {f!r}")
        value = getattr(self, f)
        if value is None:
            continue
        setattr(self, f, converter(value, from_unit, to_unit))
        field_units[f] = to_unit
        if f == 'model_total_energy':
            self.metadata['energy_unit'] = to_unit
        if f == 'forces_per_center':
            self.center_force_unit = to_unit
        if f == 'torques_per_center':
            self.center_torque_unit = to_unit
    return self


def _update_energy_units(self, from_unit='Hartree', to_unit='kcal/mol', fields=('model_total_energy',)):
    return _update_units(self, converter=_ce, from_unit=from_unit, to_unit=to_unit, fields=fields)


def _update_gradient_units(self, from_unit='Hartree/bohr', to_unit='kcal/mol/Angstrom', fields=('forces_per_center',)):
    return _update_units(self, converter=_cg, from_unit=from_unit, to_unit=to_unit, fields=fields)


def _update_position_units(self, from_unit='bohr', to_unit='Angstrom', fields=('conformations', 'center_coords')):
    return _update_units(self, converter=_cl, from_unit=from_unit, to_unit=to_unit, fields=fields)


def _sync_total_fields_from_legacy(self, overwrite=False, compute_interaction=True):
    def _extract_two(name):
        value = getattr(self, name, None)
        if value is None:
            return None, None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return value[0], value[1]
        if isinstance(value, dict):
            for ka, kb in [(0, 1), (1, 2), ('monomer1', 'monomer2')]:
                if ka in value and kb in value:
                    return value[ka], value[kb]
        return None, None

    if overwrite or getattr(self, 'monomer1_total_energy', None) is None or \
            getattr(self, 'monomer2_total_energy', None) is None:
        a = getattr(self, 'monomer1_qm_dft_total_energy', None)
        b = getattr(self, 'monomer2_qm_dft_total_energy', None)
        if a is None or b is None:
            a, b = _extract_two('monomer_dft_total_energy')
        if a is not None and b is not None:
            self.monomer1_total_energy = np.asarray(a, dtype=float)
            self.monomer2_total_energy = np.asarray(b, dtype=float)

    if overwrite or getattr(self, 'monomer1_total_gradient', None) is None or \
            getattr(self, 'monomer2_total_gradient', None) is None:
        a = getattr(self, 'monomer1_qm_dft_total_gradient', None)
        b = getattr(self, 'monomer2_qm_dft_total_gradient', None)
        if a is None or b is None:
            a, b = _extract_two('monomer_dft_total_gradient')
        if a is not None and b is not None:
            self.monomer1_total_gradient = np.asarray(a, dtype=float)
            self.monomer2_total_gradient = np.asarray(b, dtype=float)

    if overwrite or getattr(self, 'interaction_total_energy', None) is None:
        a = getattr(self, 'interaction_qm_dft_total_energy', None)
        if a is not None:
            self.interaction_total_energy = np.asarray(a, dtype=float)

    if overwrite or getattr(self, 'interaction_total_gradient', None) is None:
        a = getattr(self, 'interaction_qm_dft_total_gradient', None)
        if a is not None:
            self.interaction_total_gradient = np.asarray(a, dtype=float)

    if compute_interaction and getattr(self, 'interaction_total_energy', None) is None:
        dimer_e = getattr(self, 'dft_total_energy', None)
        mon1_e = getattr(self, 'monomer1_total_energy', None)
        mon2_e = getattr(self, 'monomer2_total_energy', None)
        if all(x is not None for x in (dimer_e, mon1_e, mon2_e)):
            d, m1, m2 = (np.asarray(x, dtype=float) for x in (dimer_e, mon1_e, mon2_e))
            if d.shape == m1.shape == m2.shape:
                self.interaction_total_energy = d - m1 - m2

    return self


# Attach methods to the dataclass
SpiceMolecule.ensure_field_units = _ensure_field_units
SpiceMolecule.update_energy_units = _update_energy_units
SpiceMolecule.update_gradient_units = _update_gradient_units
SpiceMolecule.update_position_units = _update_position_units
SpiceMolecule.sync_total_fields_from_legacy = _sync_total_fields_from_legacy


# ---------------------------------------------------------------------------
# CLI entry point  (spice-model-tools-compatible)
# ---------------------------------------------------------------------------

def _load_jsonish(path: Path) -> Any:
    text = path.read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def main() -> None:  # pragma: no cover
    known_aliases = list(SUBSET_FILTERS.keys()) + ['both', 'all']
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--hdf5', default='SPICE-2.0.1.hdf5')
    parser.add_argument('--out-dir', default='extracted')
    parser.add_argument('--subset', default='both', metavar='ALIAS')
    parser.add_argument('--keyword', default=None, metavar='STRING')
    parser.add_argument('--no-pdb', action='store_true')
    args = parser.parse_args()
    if args.keyword and args.subset != 'both':
        parser.error('--keyword and --subset are mutually exclusive')
    hdf5_path = Path(args.hdf5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.keyword:
        subsets_to_run = [(args.keyword.replace(' ', '_'), args.keyword)]
    elif args.subset == 'both':
        subsets_to_run = [
            ('dipeptides', SUBSET_FILTERS['dipeptides']),
            ('solvated', SUBSET_FILTERS['solvated']),
        ]
    elif args.subset == 'all':
        subsets_to_run = [('all', None)]
    elif args.subset in SUBSET_FILTERS:
        subsets_to_run = [(args.subset, SUBSET_FILTERS[args.subset])]
    else:
        parser.error(f"Unknown subset alias {args.subset!r}. Use one of: {', '.join(known_aliases)}")

    from joblib import dump as _dump  # noqa: F401
    for subset_key, subset_keyword in subsets_to_run:
        label = subset_keyword if subset_keyword else 'all groups'
        print(f"=== Extracting: {label} ===")
        molecules = load_subset(hdf5_path, subset_keyword)
        joblib_dir = out_dir / subset_key / 'joblib'
        pdb_dir = out_dir / subset_key / 'pdb'
        joblib_dir.mkdir(parents=True, exist_ok=True)
        for mol_name, mol in molecules.items():
            safe = mol_name.replace('/', '_').replace(' ', '_')
            mol.save(joblib_dir / f"{safe}.joblib")
            if not args.no_pdb:
                mol.to_pdb(pdb_dir / safe)
        print(f"  Joblib files : {joblib_dir}")
        if not args.no_pdb and molecules:
            last_mol = next(reversed(molecules.values()))
            print(f"  PDB files    : {pdb_dir}  ({last_mol.n_conformations} conformations)")
        print()


if __name__ == '__main__':
    main()


__all__ = [
    'SpiceMolecule',
    'Trajectory',
    'BulkProperties',
    'ATOMIC_NUMBER_TO_SYMBOL',
    'SUBSET_FILTERS',
    'ModelOutputFormat',
    'normalize_model_outputs_payload',
    'apply_update',
    'ensure_cache',
    'set_temporary_field',
    'get_temporary_field',
    'upgrade_legacy_to_v2',
    'load_subset',
]
