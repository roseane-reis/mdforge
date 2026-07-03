"""OpenMM engine (in-process, Python API).

Extracted from the ``OpenMM_compute_energies`` / ``test-md-openmm`` notebooks and
shaped to the :class:`Engine` interface. Key improvements over the notebooks:
``batch_single_point`` builds the ``Context`` **once** and only ``setPositions``
per frame (the notebooks rebuilt it per frame), and ``dynamics`` adds a
``MonteCarloBarostat`` for NPT.

Native OpenMM units are kJ/mol and nm; results are reported as kJ/mol with
gradients and coordinates in Angstrom (so :mod:`mdforge.core.units` can convert
to Tinker's kcal/mol·Å for cross-engine comparison).

OpenMM is imported lazily, so ``import mdforge.engine.openmm`` works without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.records import Trajectory
from .base import Capabilities, EngineResult

_NM_PER_ANGSTROM = 0.1
_ANGSTROM_PER_NM = 10.0


def _require_openmm():
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:  # pragma: no cover - only without openmm
        raise ImportError(
            "The OpenMM engine requires openmm. Install it with: pip install 'mdforge[openmm]' "
            "(or conda install -c conda-forge openmm)"
        ) from exc
    return openmm, app, unit


@dataclass
class OpenMMEngine:
    """Evaluate energies/forces and run MD via OpenMM.

    Construct directly from a prebuilt ``system`` + ``topology``, or via
    :meth:`from_pdb` (FF-XML) / :meth:`from_ml` (MACE-OFF23 etc. via openmm-ml).
    ``default_positions`` (Angstrom) is used when ``single_point`` is called
    without an explicit structure.
    """

    system: Any
    topology: Any = None
    platform_name: str | None = None
    default_positions: np.ndarray | None = None  # (N,3) Angstrom
    capabilities: Capabilities = field(default_factory=lambda: Capabilities(
        single_point=True, gradient=True, minimize=True,
        dynamics=True, npt=True, batched=True,
    ))

    def __post_init__(self):
        self._context = None  # cached for batched single points

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pdb(cls, pdb_path: str | Path, forcefield: list[str], *,
                 nonbonded_method: str = "NoCutoff", platform_name: str | None = None,
                 **create_kwargs) -> OpenMMEngine:
        """Build a system from a PDB and a list of OpenMM force-field XML files."""
        openmm, app, unit = _require_openmm()
        pdb = app.PDBFile(str(pdb_path))
        ff = app.ForceField(*forcefield)
        method = getattr(app, nonbonded_method)
        system = ff.createSystem(pdb.topology, nonbondedMethod=method, **create_kwargs)
        pos = np.array(pdb.positions.value_in_unit(unit.angstrom), dtype=float)
        return cls(system=system, topology=pdb.topology, platform_name=platform_name,
                   default_positions=pos)

    @classmethod
    def from_ml(cls, pdb_path: str | Path, model_name: str = "mace-off23-small",
                platform_name: str | None = None) -> OpenMMEngine:
        """Build an ML-potential system (MACE-OFF23 / UMA) via openmm-ml."""
        openmm, app, unit = _require_openmm()
        try:
            from openmmml import MLPotential
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "from_ml requires openmm-ml. Install it with: pip install 'mdforge[openmm,ml]'"
            ) from exc
        pdb = app.PDBFile(str(pdb_path))
        system = MLPotential(model_name).createSystem(pdb.topology)
        pos = np.array(pdb.positions.value_in_unit(unit.angstrom), dtype=float)
        return cls(system=system, topology=pdb.topology, platform_name=platform_name,
                   default_positions=pos)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _platform(self):
        openmm, _, _ = _require_openmm()
        if self.platform_name:
            return openmm.Platform.getPlatformByName(self.platform_name)
        return None

    def _make_context(self):
        openmm, _, unit = _require_openmm()
        integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
        platform = self._platform()
        if platform is not None:
            return openmm.Context(self.system, integrator, platform)
        return openmm.Context(self.system, integrator)

    def _coerce_positions(self, structure: Any) -> np.ndarray:
        """Return an (M, N, 3) Angstrom array from various structure inputs."""
        if structure is None:
            if self.default_positions is None:
                raise ValueError("No positions: pass a structure or set default_positions")
            arr = np.asarray(self.default_positions, dtype=float)
        elif hasattr(structure, "coords") and not isinstance(structure, np.ndarray):
            arr = np.asarray(structure.coords, dtype=float)
        else:
            arr = np.asarray(structure, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"positions must be (N,3) or (M,N,3); got {arr.shape}")
        return arr

    def _eval(self, context, frame_angstrom: np.ndarray, want_forces: bool):
        openmm, app, unit = _require_openmm()
        context.setPositions((frame_angstrom * _NM_PER_ANGSTROM).tolist() * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=want_forces)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        grad = None
        if want_forces:
            forces = np.array(
                state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
                dtype=float,
            )
            grad = -forces * _NM_PER_ANGSTROM  # force→gradient, /nm → /Angstrom
        return e, grad

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def single_point(self, structure: Any = None, *, breakdown: bool = False,
                     gradient: bool = False, **opt) -> EngineResult:
        positions = self._coerce_positions(structure)
        if self._context is None:
            self._context = self._make_context()
        energies, grads = [], []
        for frame in positions:
            e, g = self._eval(self._context, frame, want_forces=gradient)
            energies.append(e)
            if gradient:
                grads.append(g)
        return EngineResult(
            energy=np.array(energies), energy_unit="kJ/mol",
            gradient=np.array(grads) if gradient else None,
            force_unit="kJ/mol/Angstrom" if gradient else None,
            length_unit="angstrom",
        )

    def gradient(self, structure: Any = None, **opt) -> EngineResult:
        return self.single_point(structure, gradient=True, **opt)

    def batch_single_point(self, structures: Any, *, gradient: bool = False, **opt) -> EngineResult:
        """Single point over many frames, reusing one Context (only setPositions per frame)."""
        return self.single_point(structures, gradient=gradient, **opt)

    def minimize(self, structure: Any = None, *, tol: float = 10.0, max_iterations: int = 0,
                 **opt) -> EngineResult:
        """Minimize with ``LocalEnergyMinimizer``; ``tol`` is in kJ/mol/nm."""
        openmm, app, unit = _require_openmm()
        positions = self._coerce_positions(structure)[0]
        context = self._make_context()
        context.setPositions((positions * _NM_PER_ANGSTROM).tolist() * unit.nanometer)
        openmm.LocalEnergyMinimizer.minimize(
            context, tol * unit.kilojoule_per_mole / unit.nanometer, max_iterations
        )
        state = context.getState(getEnergy=True, getPositions=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        minimized = np.array(
            state.getPositions().value_in_unit(unit.angstrom), dtype=float
        )
        return EngineResult(
            energy=np.array([e]), energy_unit="kJ/mol", length_unit="angstrom",
            extra={"minimized_positions": minimized},
        )

    def dynamics(self, structure: Any = None, *, nsteps: int, dt_fs: float,
                 ensemble: str = "nvt", temperature: float = 298.15,
                 pressure: float | None = None, friction_ps: float = 1.0,
                 report_interval: int = 100, **opt) -> Trajectory:
        """Run MD and return a :class:`Trajectory` (positions Å, energies kJ/mol).

        ``ensemble='npt'`` adds a ``MonteCarloBarostat`` at ``pressure`` bar
        (default 1.0). ``nve`` uses a Verlet integrator; otherwise LangevinMiddle.
        """
        openmm, app, unit = _require_openmm()
        ensemble = ensemble.lower()
        positions = self._coerce_positions(structure)[0]

        # The barostat mutates the system; copy via serialization to stay reentrant.
        system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(self.system))
        if ensemble == "npt":
            if pressure is None:
                pressure = 1.0
            system.addForce(openmm.MonteCarloBarostat(pressure * unit.bar,
                                                      temperature * unit.kelvin))

        if ensemble == "nve":
            integrator = openmm.VerletIntegrator(dt_fs * unit.femtosecond)
        else:
            integrator = openmm.LangevinMiddleIntegrator(
                temperature * unit.kelvin, friction_ps / unit.picosecond, dt_fs * unit.femtosecond
            )
        platform = self._platform()
        context = (openmm.Context(system, integrator, platform) if platform
                   else openmm.Context(system, integrator))
        context.setPositions((positions * _NM_PER_ANGSTROM).tolist() * unit.nanometer)
        context.setVelocitiesToTemperature(temperature * unit.kelvin)

        pos_frames, pe, ke, vol, boxes = [], [], [], [], []
        n_reports = max(1, nsteps // max(1, report_interval))
        for _ in range(n_reports):
            integrator.step(report_interval)
            state = context.getState(getEnergy=True, getPositions=True)
            pos_frames.append(np.array(state.getPositions().value_in_unit(unit.angstrom), dtype=float))
            pe.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            ke.append(state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole))
            box = state.getPeriodicBoxVectors().value_in_unit(unit.angstrom)
            box = np.array(box, dtype=float)
            boxes.append(box)
            vol.append(float(np.linalg.det(box)))

        masses = np.array([
            self.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(self.system.getNumParticles())
        ], dtype=float)
        return Trajectory(
            positions=np.array(pos_frames),
            potential_energy=np.array(pe),
            kinetic_energy=np.array(ke),
            volume=np.array(vol),
            box_vectors=np.array(boxes),
            masses=masses,
            temperature_K=temperature, dt_ps=dt_fs / 1000.0,
            metadata={"engine": "openmm", "ensemble": ensemble,
                      "energy_unit": "kJ/mol", "report_interval": report_interval},
        )


__all__ = ["OpenMMEngine"]
