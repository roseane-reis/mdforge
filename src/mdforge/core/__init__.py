"""mdforge.core — units, records, and I/O primitives.

Ported from prior internal tooling with the following changes:
- legacy JSON loader → load_center_json  (removes proprietary name)
- all_energy field-group no longer includes torques_per_center  (bug fix)
- save_pickle updated for Python 3  (removed iteritems())
- Trajectory and BulkProperties stubs added for Phase 4/5
- rdkit import guarded as an optional extra
"""

from .identity import IdentityRegistry, MoleculeIdentity
from .io import (
    load_center_json,
    load_joblib_records,
    load_pickle,
    save_pickle,
    write_xyz_string,
)
from .records import (
    BulkProperties,
    SpiceMolecule,
    Trajectory,
    apply_update,
    get_temporary_field,
    load_subset,
    set_temporary_field,
    upgrade_legacy_to_v2,
)
from .units import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_KCAL_MOL,
    HARTREE_TO_KJ_MOL,
    convert_energy,
    convert_gradient,
    convert_length,
    gradient_hartree_bohr_to_kcal_angstrom,
    gradient_kcal_angstrom_to_hartree_bohr,
    hartree_to_kcal,
    kcal_to_hartree,
)

__all__ = [
    # units
    "BOHR_TO_ANGSTROM", "ANGSTROM_TO_BOHR",
    "HARTREE_TO_KCAL_MOL", "HARTREE_TO_KJ_MOL",
    "convert_energy", "convert_gradient", "convert_length",
    "hartree_to_kcal", "kcal_to_hartree",
    "gradient_hartree_bohr_to_kcal_angstrom", "gradient_kcal_angstrom_to_hartree_bohr",
    # records
    "SpiceMolecule", "Trajectory", "BulkProperties",
    "apply_update", "set_temporary_field", "get_temporary_field",
    "upgrade_legacy_to_v2", "load_subset",
    # io
    "load_center_json", "load_joblib_records",
    "write_xyz_string", "load_pickle", "save_pickle",
    # identity
    "IdentityRegistry", "MoleculeIdentity",
]
