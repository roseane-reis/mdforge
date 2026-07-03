"""mdforge.simulate — run MD over the engine (goal a).

Orchestrates :meth:`Engine.dynamics` into liquid-box NPT and gas-phase runs,
producing :class:`~mdforge.core.records.Trajectory` records that flow directly
into ``mdforge.liquid`` (Phase 4). Box construction and batch dispatch are
engine-agnostic; *where* a run executes (local CPU, remote GPU) is the engine's
runner concern, not hardcoded here.

    from mdforge.simulate import run_npt, build_openmm_water_box
    eng = build_openmm_water_box(n_waters=300)
    traj = run_npt(eng, None, nsteps=5000, dt_fs=2.0, temperature=298.15, pressure=1.0)
    from mdforge.liquid import compute_bulk_properties
    props = compute_bulk_properties(traj, equil=1000, n_molecules=300)
"""

from __future__ import annotations

from . import box, gas, jobs, liquid_box
from .box import (
    box_edge_for_density,
    density_of_box,
    molar_mass,
    n_molecules_for_box,
    replicate_cubic,
)
from .gas import gas_average_pe, run_gas
from .jobs import JobResult, run_jobs
from .liquid_box import build_openmm_water_box, run_npt

__all__ = [
    # submodules
    "box", "liquid_box", "gas", "jobs",
    # box math + builder
    "molar_mass", "box_edge_for_density", "n_molecules_for_box", "density_of_box",
    "replicate_cubic",
    # orchestration
    "run_npt", "build_openmm_water_box", "run_gas", "gas_average_pe",
    # dispatch
    "run_jobs", "JobResult",
]
