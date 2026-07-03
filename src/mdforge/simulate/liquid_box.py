"""Liquid-box NPT orchestration over the engine (goal a).

``run_npt`` drives any :class:`~mdforge.engine.base.Engine` (Tinker or OpenMM)
through an optional minimize then a production NPT ``dynamics`` call, returning a
:class:`~mdforge.core.records.Trajectory` that flows straight into Phase 4
(``mdforge.liquid``). Replaces the hardcoded ``run_sim.py`` / ``Auxfit.run_npt``
dispatch with an engine-agnostic flow; equilibration is handled downstream via
the ``equil`` argument to ``liquid.compute_bulk_properties``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.records import Trajectory
from ..engine.base import Engine, require
from ..formats.txyz import TinkerXYZ


def _next_structure(min_result, structure: Any) -> Any:
    """Build the structure to continue from after a minimize (engine-agnostic)."""
    extra = getattr(min_result, "extra", None) or {}
    if extra.get("minimized") is not None:          # Tinker → a TinkerXYZ
        return extra["minimized"]
    if extra.get("minimized_positions") is not None:  # OpenMM → (N,3) Angstrom
        pos = extra["minimized_positions"]
        if isinstance(structure, TinkerXYZ):
            return TinkerXYZ(names=structure.names, coords=np.asarray(pos),
                             types=structure.types, connectivity=structure.connectivity,
                             box=structure.box, title=structure.title)
        return pos
    return structure


def run_npt(
    engine: Engine,
    structure: Any,
    *,
    nsteps: int,
    dt_fs: float = 1.0,
    temperature: float = 298.15,
    pressure: float = 1.0,
    minimize: bool = True,
    minimize_tol: float = 1.0,
    **dyn_opt,
) -> Trajectory:
    """Run an NPT simulation and return the production :class:`Trajectory`.

    Optionally minimizes first (if the engine supports it), then runs
    ``engine.dynamics(ensemble='npt', ...)``. Extra keyword args are forwarded to
    ``dynamics`` (e.g. ``save_ps`` for Tinker, ``report_interval`` for OpenMM).
    """
    require(engine, "dynamics", "npt")
    current = structure
    if minimize and engine.capabilities.minimize:
        current = _next_structure(engine.minimize(current, tol=minimize_tol), current)
    return engine.dynamics(
        current, nsteps=nsteps, dt_fs=dt_fs, ensemble="npt",
        temperature=temperature, pressure=pressure, **dyn_opt,
    )


def build_openmm_water_box(
    n_waters: int,
    *,
    model: str = "tip3p",
    forcefield: list[str] | None = None,
    nonbonded_method: str = "PME",
    nonbonded_cutoff_nm: float = 0.9,
    constraints: str = "HBonds",
    platform_name: str | None = None,
):
    """Build a cubic TIP3P-family water box as a ready :class:`OpenMMEngine`.

    Uses OpenMM's ``Modeller.addSolvent`` to pack ``n_waters`` molecules; returns
    an :class:`~mdforge.engine.openmm.OpenMMEngine` with the system, topology,
    and starting positions populated (NPT-ready: PME + periodic box).
    """
    from ..engine.openmm import OpenMMEngine, _require_openmm
    openmm, app, unit = _require_openmm()

    ff_files = forcefield or ["amber14-all.xml", "amber14/tip3p.xml"]
    ff = app.ForceField(*ff_files)
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model=model, numAdded=n_waters, boxShape="cube")
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=getattr(app, nonbonded_method),
        nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
        constraints=getattr(app, constraints), rigidWater=True,
    )
    positions = np.array(modeller.positions.value_in_unit(unit.angstrom), dtype=float)
    return OpenMMEngine(system=system, topology=modeller.topology,
                        default_positions=positions, platform_name=platform_name)


__all__ = ["run_npt", "build_openmm_water_box"]
