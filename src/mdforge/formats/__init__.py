"""mdforge.formats — Tinker/PDB file parsing, conversion, and editing (goal c).

Pure file ⟂ engine: every parser takes text or a path and returns plain Python /
numpy data; nothing here spawns Tinker. The engine (Phase 2) runs the binaries
and feeds their stdout to :mod:`~mdforge.formats.analyze_out`.

Modules
-------
- :mod:`~mdforge.formats.txyz`        Tinker/raw XYZ read/write, coord/type edit
- :mod:`~mdforge.formats.arc`         multi-frame ARC trajectory read/write
- :mod:`~mdforge.formats.prm`         Tinker .prm parse/write + .key writer (HIPPO terms)
- :mod:`~mdforge.formats.analyze_out` parse ``analyze``/``testgrad`` stdout → arrays
- :mod:`~mdforge.formats.pdb`         PDB read/write + PDB↔TXYZ (net-new)
- :mod:`~mdforge.formats.mol`         coordinate→SMILES via RDKit (optional, [chem])
- :mod:`~mdforge.formats.gsd`         HOOMD rigid-body GSD reader + atom reconstruction (optional, [gsd])
- :mod:`~mdforge.formats.dcd`         CHARMM/NAMD DCD trajectory reader/writer (pure numpy)
- :mod:`~mdforge.formats.epsr`        EPSR experimental g(r) / angular-g(r,theta) readers

Engine-free, like ``liquid`` and ``qm`` — builds on ``core`` only.
"""

from __future__ import annotations

from . import analyze_out, arc, dcd, epsr, gsd, mol, pdb, prm, txyz
from .analyze_out import (
    ENERGY_TERMS,
    energy_components,
    parse_energy_breakdown,
    parse_testgrad,
    sapt_components,
)
from .arc import ArcTrajectory, count_frames, read_arc, write_arc
from .dcd import DCDTrajectory, read_dcd, write_dcd
from .epsr import read_epsr_angular_rdf, read_epsr_rdf
from .gsd import (
    RigidTrajectory,
    read_rigid_bodies,
    reconstruct_atoms,
    reference_geometry_from_gsd,
    species_atom_index,
)
from .pdb import PDBStructure, pdb_to_txyz, read_pdb, to_pdb_string, txyz_to_pdb, write_pdb
from .prm import multipole_factors, process_prm, write_key, write_prm
from .txyz import (
    TinkerXYZ,
    raw_to_txyz,
    read_txyz,
    txyz_to_raw,
    update_coords,
    write_txyz,
)

__all__ = [
    # submodules
    "txyz", "arc", "prm", "analyze_out", "pdb", "mol", "gsd", "dcd", "epsr",
    # gsd (HOOMD rigid-body trajectories)
    "RigidTrajectory", "read_rigid_bodies", "reconstruct_atoms",
    "reference_geometry_from_gsd", "species_atom_index",
    # dcd (CHARMM/NAMD trajectories)
    "DCDTrajectory", "read_dcd", "write_dcd",
    # epsr (experimental RDF readers)
    "read_epsr_rdf", "read_epsr_angular_rdf",
    # txyz
    "TinkerXYZ", "read_txyz", "write_txyz", "update_coords", "raw_to_txyz", "txyz_to_raw",
    # arc
    "ArcTrajectory", "read_arc", "write_arc", "count_frames",
    # prm
    "process_prm", "write_prm", "write_key", "multipole_factors",
    # analyze_out
    "ENERGY_TERMS", "parse_energy_breakdown", "energy_components",
    "sapt_components", "parse_testgrad",
    # pdb
    "PDBStructure", "read_pdb", "write_pdb", "to_pdb_string", "txyz_to_pdb", "pdb_to_txyz",
]
