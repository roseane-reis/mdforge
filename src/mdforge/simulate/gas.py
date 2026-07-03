"""Gas-phase single-molecule runs (goal a).

A gas-phase NVT (or stochastic) trajectory of an isolated molecule, used mainly
to get the gas-phase potential average that feeds the heat-of-vaporization
(``liquid.thermo.heat_of_vaporization``). Engine-agnostic — drives
``engine.dynamics(ensemble='nvt', ...)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.records import Trajectory
from ..engine.base import Engine, require
from .liquid_box import _next_structure


def run_gas(
    engine: Engine,
    structure: Any,
    *,
    nsteps: int,
    dt_fs: float = 1.0,
    temperature: float = 298.15,
    ensemble: str = "nvt",
    minimize: bool = False,
    minimize_tol: float = 1.0,
    **dyn_opt,
) -> Trajectory:
    """Run a gas-phase single-molecule trajectory and return it."""
    require(engine, "dynamics")
    current = structure
    if minimize and engine.capabilities.minimize:
        current = _next_structure(engine.minimize(current, tol=minimize_tol), current)
    return engine.dynamics(
        current, nsteps=nsteps, dt_fs=dt_fs, ensemble=ensemble,
        temperature=temperature, **dyn_opt,
    )


def gas_average_pe(traj: Trajectory, *, equil: int = 0, per_molecule: bool = False) -> float:
    """Mean gas-phase potential energy (optionally per molecule) for ΔHvap."""
    if traj.potential_energy is None:
        raise ValueError("Trajectory has no potential_energy")
    pe = np.asarray(traj.potential_energy)[equil:]
    mean = float(pe.mean())
    if per_molecule and traj.n_molecules:
        mean /= traj.n_molecules
    return mean


__all__ = ["run_gas", "gas_average_pe"]
