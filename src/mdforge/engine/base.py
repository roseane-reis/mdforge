"""Engine interface: the contract every MD backend implements (Phase 2 keystone).

One ``Engine`` Protocol absorbs the differences between Tinker (file +
subprocess, kcal/mol·Å) and OpenMM (in-process, kJ/mol·nm). Every
:class:`EngineResult` self-describes its units; :mod:`mdforge.core.units`
normalizes before any cross-engine comparison. ``Capabilities`` gate usage so a
plugin that only does single points raises cleanly on ``dynamics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


class UnsupportedOperation(NotImplementedError):
    """Raised when an engine is asked for a capability it does not provide."""


@dataclass
class EngineResult:
    """Result of an engine energy/force evaluation (possibly multi-frame).

    Native units are reported, never assumed — use :meth:`energy_in` /
    :meth:`gradient_in` (or :mod:`mdforge.core.units`) to normalize.
    """

    energy: np.ndarray                                  # (M,) in energy_unit
    energy_unit: str
    components: dict[str, np.ndarray] | None = None      # Tinker analyze breakdown
    intermolecular: np.ndarray | None = None             # (M,)
    gradient: np.ndarray | None = None                   # (M,N,3) per-atom; force = -gradient
    forces_per_center: np.ndarray | None = None          # (M,C,3) center-based plugins
    torques_per_center: np.ndarray | None = None         # (M,C,3)
    center_coords: np.ndarray | None = None              # (M,C,3) in length_unit
    atom_to_center: np.ndarray | None = None             # (N,)
    force_unit: str | None = None
    length_unit: str = "angstrom"
    extra: dict | None = None

    def energy_in(self, unit: str) -> np.ndarray:
        """Return the energy converted to ``unit`` (e.g. 'kcal/mol', 'kJ/mol')."""
        from ..core.units import convert_energy
        return convert_energy(self.energy, self.energy_unit, unit)

    def gradient_in(self, unit: str) -> np.ndarray:
        """Return the per-atom gradient converted to ``unit`` (e.g. 'kcal/mol/Angstrom')."""
        if self.gradient is None:
            raise ValueError("EngineResult has no gradient")
        if self.force_unit is None:
            raise ValueError("EngineResult.force_unit is unset; cannot convert")
        from ..core.units import convert_gradient
        return convert_gradient(self.gradient, self.force_unit, unit)


@dataclass
class Capabilities:
    """What an engine can do. Consumers gate on these via :func:`require`."""

    single_point: bool = False
    gradient: bool = False                 # per-atom gradient
    forces_per_center: bool = False
    components: bool = False               # energy-component breakdown
    minimize: bool = False
    dynamics: bool = False
    npt: bool = False
    batched: bool = False                  # efficient multi-structure single points
    esp: bool = False
    polarizability: bool = False


@runtime_checkable
class Engine(Protocol):
    """The unified MD-engine interface (structural; duck-typed at runtime)."""

    capabilities: Capabilities

    def single_point(self, structure: Any, *, breakdown: bool = False, **opt) -> EngineResult: ...
    def gradient(self, structure: Any, **opt) -> EngineResult: ...
    def minimize(self, structure: Any, *, tol: float = 0.1, **opt) -> EngineResult: ...
    def dynamics(
        self, structure: Any, *, nsteps: int, dt_fs: float, ensemble: str = "nvt",
        temperature: float = 298.15, pressure: float | None = None, **opt,
    ) -> Any: ...
    def batch_single_point(self, structures: Any, **opt) -> EngineResult: ...


def require(engine: Engine, *caps: str) -> None:
    """Raise :class:`UnsupportedOperation` unless ``engine`` has all ``caps``."""
    missing = [c for c in caps if not getattr(engine.capabilities, c, False)]
    if missing:
        raise UnsupportedOperation(
            f"{type(engine).__name__} lacks required capabilities: {', '.join(missing)}"
        )


__all__ = ["EngineResult", "Capabilities", "Engine", "UnsupportedOperation", "require"]
