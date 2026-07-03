"""Ingest topology + trajectory + logs into numpy arrays (parse ⟂ compute).

Produces a :class:`LegData` bundle per simulation leg. No compute kernels are
called here — this is purely the parse half. Three trajectory sources are
unified:

- **GSD** (HOOMD rigid bodies): body-frame geometry is recovered from the file
  itself (:func:`mdforge.formats.gsd.reference_geometry_from_gsd`), so no
  force-field/model geometry package is needed.
- **DCD** (CHARMM/NAMD, coordinates only): atom identities and per-molecule
  membership come from the topology; the centre of mass is mass-weighted.
- **logs** (``.npy`` / ``.csv`` per-frame thermo) via
  :func:`mdforge.liquid.parse.trajectory_from_hoomd_npy`.

Charges for the dielectric come from the config/topology (physical values), so
the HOOMD ``√332.06371`` GSD-charge scaling is sidestepped entirely.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.elements import ELEMENT_MASSES
from ...core.records import Trajectory
from ...formats.dcd import read_dcd
from ...formats.gsd import (
    read_rigid_bodies,
    reconstruct_atoms,
    reference_geometry_from_gsd,
)
from ...formats.pdb import read_pdb
from ...formats.txyz import read_txyz
from .. import parse
from .config import EvalConfig, LegSpec
from .profiles import WaterProfile
from .profiles.water import get_profile


@dataclass
class LegData:
    """Numpy arrays for one simulation leg (thermo + structure + transport inputs)."""

    name: str
    ensemble: str
    equil_frac: float
    source: str                              # "gsd" | "dcd" | "log-only"
    traj: Trajectory | None = None           # per-frame thermo
    raw_columns: dict = field(default_factory=dict)  # pressure_atm, density_gcc, pe_* ...
    atoms: np.ndarray | None = None          # (T, N, 3) reconstructed / DCD coords
    com: np.ndarray | None = None            # (T, M, 3) per-molecule COM (for diffusion)
    box: np.ndarray | None = None            # (T, 3) orthorhombic edge lengths (Å)
    volume: np.ndarray | None = None         # (T,) cell volume (Å³)
    o_idx: np.ndarray | None = None
    h_idx: np.ndarray | None = None
    mol_o: np.ndarray | None = None
    mol_h: np.ndarray | None = None
    charges: np.ndarray | None = None        # (N,) per-atom (e)
    n_molecules: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_traj_frames(self) -> int:
        if self.com is not None:
            return self.com.shape[0]
        if self.atoms is not None:
            return self.atoms.shape[0]
        return 0

    def equil_frames(self, n: int) -> int:
        return int(round(self.equil_frac * n))


# ---------------------------------------------------------------------------
# Topology → profile
# ---------------------------------------------------------------------------

def _element_from_name(name: str) -> str:
    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return "X"
    return letters[0].upper()


def _topology_elements(config: EvalConfig) -> tuple[list[str], int, np.ndarray | None]:
    """Return (per-atom element symbols, n_atoms, box) from the topology file."""
    if config.topology.pdb:
        struct = read_pdb(str(config.resolve(config.topology.pdb)))
        return [a.element for a in struct.atoms], struct.n_atoms, struct.box
    if config.topology.txyz:
        xyz = read_txyz(str(config.resolve(config.topology.txyz)))
        elements = [_element_from_name(n) for n in xyz.names]
        return elements, xyz.n_atoms, xyz.box
    raise ValueError("config topology has neither pdb nor txyz")


def water_profile_from_topology(config: EvalConfig) -> tuple[WaterProfile, int]:
    """Build the species profile and molecule count from config + topology.

    Charges come from ``config.system.charges_e`` when given (physical values),
    otherwise the profile defaults. The first molecule's elements are validated
    against the profile (``O, H, H`` for water).
    """
    profile = get_profile(
        config.species,
        charges_e=config.system.charges_e,
        molar_mass_g_mol=config.system.molar_mass_g_mol,
    )
    apm = config.system.atoms_per_molecule
    elements, n_atoms, _ = _topology_elements(config)
    if len(elements) >= apm:
        profile.validate_elements(elements[:apm])
    n_molecules = config.system.n_molecules
    if n_molecules is None:
        if n_atoms % apm != 0:
            raise ValueError(
                f"topology has {n_atoms} atoms, not a multiple of "
                f"atoms_per_molecule={apm}; set system.n_molecules explicitly"
            )
        n_molecules = n_atoms // apm
    return profile, int(n_molecules)


def _atom_selection(profile: WaterProfile, n_molecules: int):
    """Return (o_idx, h_idx, mol_o, mol_h, per_atom_charges) for the atom axis."""
    apm = profile.atoms_per_molecule
    base = np.arange(n_molecules) * apm
    o_idx = base + profile.oxygen_local_index
    h_local = np.asarray(profile.hydrogen_local_indices)
    h_idx = (base[:, None] + h_local[None, :]).ravel()          # [H0a,H0b,H1a,...]
    mol_o = np.arange(n_molecules)
    mol_h = np.repeat(np.arange(n_molecules), len(h_local))
    charges = profile.per_atom_charges(n_molecules)
    return o_idx, h_idx, mol_o, mol_h, charges


# ---------------------------------------------------------------------------
# Per-frame thermo logs (.npy / .csv)
# ---------------------------------------------------------------------------

def _csv_to_structured(path: Path) -> np.ndarray:
    """Read a HOOMD-style per-frame CSV into a numpy structured array."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader if row]
    cols = list(zip(*rows)) if rows else [[] for _ in header]
    dtype = [(name, "i8" if name == "step" else "f8") for name in header]
    arr = np.zeros(len(rows), dtype=dtype)
    for name, col in zip(header, cols):
        arr[name] = np.asarray(col, dtype=(np.int64 if name == "step" else float))
    return arr


def _load_log(path: Path, profile: WaterProfile, n_molecules: int,
              temperature_K: float) -> tuple[Trajectory, dict]:
    """Load a .npy/.csv thermo log into a Trajectory + a raw-columns dict."""
    if path.suffix == ".csv":
        arr = _csv_to_structured(path)
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        tmp.close()
        try:
            np.save(tmp.name, arr)
            traj = parse.trajectory_from_hoomd_npy(
                tmp.name, n_molecules=n_molecules,
                molar_mass_g_mol=profile.molar_mass_g_mol, temperature_K=temperature_K)
        finally:
            os.unlink(tmp.name)
    else:
        arr = np.load(path)
        traj = parse.trajectory_from_hoomd_npy(
            path, n_molecules=n_molecules,
            molar_mass_g_mol=profile.molar_mass_g_mol, temperature_K=temperature_K)
    raw = {name: np.asarray(arr[name]) for name in (arr.dtype.names or ())}
    return traj, raw


# ---------------------------------------------------------------------------
# Trajectory sources (GSD / DCD)
# ---------------------------------------------------------------------------

def _ingest_gsd(path: Path, profile: WaterProfile, n_molecules: int | None,
                max_frames: int | None) -> dict:
    geom = reference_geometry_from_gsd(path)
    rbt = read_rigid_bodies(path, max_frames=max_frames)
    if n_molecules is None:
        n_molecules = rbt.n_molecules
    if rbt.n_molecules != n_molecules:
        raise ValueError(
            f"GSD has {rbt.n_molecules} molecules; config/topology says {n_molecules}"
        )
    atoms = reconstruct_atoms(rbt, geom, wrap_molecules=True)
    o_idx, h_idx, mol_o, mol_h, charges = _atom_selection(profile, n_molecules)
    return {
        "atoms": atoms, "com": rbt.com, "box": rbt.box[:, :3].copy(),
        "volume": rbt.volume, "o_idx": o_idx, "h_idx": h_idx,
        "mol_o": mol_o, "mol_h": mol_h, "charges": charges, "n_molecules": n_molecules,
    }


def _mass_weighted_com(atoms: np.ndarray, n_molecules: int, apm: int,
                       masses_local: np.ndarray) -> np.ndarray:
    """Per-molecule mass-weighted COM ``(T, M, 3)`` from atoms ``(T, N, 3)``."""
    T = atoms.shape[0]
    a = atoms.reshape(T, n_molecules, apm, 3)
    w = masses_local / masses_local.sum()
    return np.einsum("tmar,a->tmr", a, w)


def _ingest_dcd(path: Path, profile: WaterProfile, n_molecules: int | None,
                max_frames: int | None, warnings: list[str]) -> dict:
    dcd = read_dcd(path, max_frames=max_frames)
    apm = profile.atoms_per_molecule
    if n_molecules is None:
        n_molecules = dcd.n_atoms // apm
    if dcd.n_atoms != n_molecules * apm:
        raise ValueError(
            f"DCD has {dcd.n_atoms} atoms; expected {n_molecules * apm} "
            f"({n_molecules} molecules × {apm})"
        )
    atoms = dcd.coordinates
    o_idx, h_idx, mol_o, mol_h, charges = _atom_selection(profile, n_molecules)
    masses_local = np.array([ELEMENT_MASSES[e] for e in profile.element_order], dtype=float)
    com = _mass_weighted_com(atoms, n_molecules, apm, masses_local)
    box = None
    volume = None
    if dcd.box is not None:
        box = dcd.box[:, :3].copy()               # a, b, c edge lengths
        volume = box[:, 0] * box[:, 1] * box[:, 2]
    else:
        warnings.append(f"{path.name}: DCD has no unit cell; structure/diffusion skipped")
    return {
        "atoms": atoms, "com": com, "box": box, "volume": volume,
        "o_idx": o_idx, "h_idx": h_idx, "mol_o": mol_o, "mol_h": mol_h,
        "charges": charges, "n_molecules": n_molecules,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_leg(leg: LegSpec, config: EvalConfig, profile: WaterProfile,
               n_molecules: int, *, max_frames: int | None = None) -> LegData:
    """Ingest one leg's topology/trajectory/log into a :class:`LegData`."""
    warnings: list[str] = []
    data = LegData(
        name=leg.name, ensemble=leg.ensemble.upper(),
        equil_frac=leg.resolved_equil_frac(), source="log-only",
        n_molecules=n_molecules, warnings=warnings,
    )

    traj_path = config.resolve(leg.trajectory)
    if traj_path is not None:
        suffix = traj_path.suffix.lower()
        if suffix == ".gsd":
            try:
                arrays = _ingest_gsd(traj_path, profile, n_molecules, max_frames)
                data.source = "gsd"
            except ImportError as exc:
                warnings.append(f"{leg.name}: GSD unavailable ({exc}); "
                                "structure/diffusion/dielectric skipped")
                arrays = None
        elif suffix == ".dcd":
            arrays = _ingest_dcd(traj_path, profile, n_molecules, max_frames, warnings)
            data.source = "dcd"
        else:
            raise ValueError(f"leg {leg.name!r}: unsupported trajectory {traj_path.name}")
        if arrays is not None:
            for k, v in arrays.items():
                setattr(data, k, v)

    log_path = config.resolve(leg.log)
    if log_path is not None and log_path.is_file():
        traj, raw = _load_log(log_path, profile, data.n_molecules or n_molecules,
                              config.state.temperature_K)
        data.traj = traj
        data.raw_columns = raw

    return data


# ---------------------------------------------------------------------------
# Campaign auto-discovery
# ---------------------------------------------------------------------------

_CAMPAIGN_ENSEMBLE = {"npt": "NPT", "nvt": "NVT", "nvt2": "NVT", "nve": "NVE"}


def legs_from_campaign(run_dir: str | Path) -> list[LegSpec]:
    """Discover production legs from a campaign run directory + ``meta.json``.

    Scans ``{npt,nvt,nvt2,nve}/`` (``heat/`` is equilibration and skipped),
    using the actual filenames present (note ``nvt2/`` holds files named
    ``nvt.*``). A leg is emitted when it has a trajectory and/or a log.
    """
    run = Path(run_dir)
    meta_path = run / "meta.json"
    _ = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    legs: list[LegSpec] = []
    for sub in ("npt", "nvt", "nvt2", "nve"):
        d = run / sub
        if not d.is_dir():
            continue
        # nvt2/ stores files named nvt.*; otherwise files match the dir name.
        stem = "nvt" if sub == "nvt2" else sub
        gsd = d / f"{stem}.gsd"
        npy = d / f"{stem}.npy"
        csvf = d / f"{stem}.csv"
        traj = str(gsd) if gsd.is_file() else None
        log = str(npy) if npy.is_file() else (str(csvf) if csvf.is_file() else None)
        if traj is None and log is None:
            continue
        legs.append(LegSpec(name=sub, ensemble=_CAMPAIGN_ENSEMBLE[sub],
                            trajectory=traj, log=log))
    return legs


__all__ = [
    "LegData", "ingest_leg", "water_profile_from_topology", "legs_from_campaign",
]
