"""Fitting targets — each produces a residual from an engine evaluation (goal d).

Ports the residual terms of ``Auxfit.optimize_prms``, but **every per-evaluation
energy/property comes from an** :class:`~mdforge.engine.base.Engine` (default
Tinker) rather than hardcoded Tinker calls:

- :class:`DimerInteractionTarget` — model SAPT-5 components (``dimer − mon1 − mon2``,
  projected via the same mapping as :mod:`mdforge.formats.analyze_out`) vs. QM.
- :class:`PolarizabilityTarget` — molecular-polarizability eigenvalues vs. reference.
- :class:`BulkPropertyTarget` — NPT bulk property (e.g. density) %-error vs. experiment.

Each target exposes ``residual(engine) -> np.ndarray``; the workflow concatenates
them into the vector the optimizer drives. (The ESP/``potential`` target is
deferred — it needs a Tinker ``potential`` grid-fitting wrapper not yet on the engine.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..formats.arc import ArcTrajectory
from ..formats.txyz import TinkerXYZ


def split_dimer(arc: ArcTrajectory, n_atoms_mol1: int) -> tuple[ArcTrajectory, ArcTrajectory]:
    """Split a dimer trajectory into its two monomer trajectories.

    Coordinates/types are sliced at ``n_atoms_mol1``; connectivity is restricted
    to intra-monomer bonds and renumbered so each monomer is a standalone system.
    """
    n1 = n_atoms_mol1

    def _conn(conn, lo, hi):
        out = []
        for bonds in conn[lo:hi]:
            out.append([b - lo for b in bonds if lo < b <= hi])
        return out

    types = arc.types
    mon1 = ArcTrajectory(
        coords=arc.coords[:, :n1, :], names=arc.names[:n1],
        types=None if types is None else types[:n1],
        connectivity=_conn(arc.connectivity, 0, n1) if arc.connectivity else [],
        box=None, title=f"{arc.title} mol1",
    )
    mon2 = ArcTrajectory(
        coords=arc.coords[:, n1:, :], names=arc.names[n1:],
        types=None if types is None else types[n1:],
        connectivity=_conn(arc.connectivity, n1, arc.n_atoms) if arc.connectivity else [],
        box=None, title=f"{arc.title} mol2",
    )
    return mon1, mon2


def project_components_to_sapt(diff: dict[str, np.ndarray], n_geom: int) -> np.ndarray:
    """Project a per-term interaction-energy dict to ``(n_geom, 5)`` SAPT components.

    Same grouping as :func:`mdforge.formats.analyze_out.sapt_components`:
    elst = Atomic Multipoles + Charge-Charge; exch = Repulsion;
    ind = Polarization + Charge Transfer; disp = Dispersion (or Van der Waals if
    no Dispersion term, e.g. AMOEBA); total = sum of all terms.
    """
    def grab(*names) -> np.ndarray:
        out = np.zeros(n_geom)
        for nm in names:
            v = diff.get(nm)
            if v is not None:
                out = out + np.asarray(v, dtype=float)
        return out

    es = grab("Atomic Multipoles", "Charge-Charge")
    exch = grab("Repulsion")
    ind = grab("Polarization", "Charge Transfer")
    disp = grab("Dispersion")
    if np.abs(disp).sum() < 1e-9:
        disp = grab("Van der Waals")
    arrays = [np.asarray(v, dtype=float) for v in diff.values()]
    total = np.sum(arrays, axis=0) if arrays else np.zeros(n_geom)
    return np.column_stack([es, exch, ind, disp, total])


@dataclass
class DimerInteractionTarget:
    """Residual on SAPT-5 interaction components for one dimer geometry scan."""

    name: str
    dimer: ArcTrajectory
    n_atoms_mol1: int
    qm_components: np.ndarray          # (n_geom, 5) reference
    weight: float = 1.0
    components: tuple[int, ...] = (0, 1, 2, 3, 4)

    def model_components(self, engine) -> np.ndarray:
        """Compute model SAPT-5 components via ``dimer − mon1 − mon2`` from the engine."""
        n = self.dimer.n_frames
        dimer_c = engine.batch_single_point(self.dimer, breakdown=True).components or {}
        mon1, mon2 = split_dimer(self.dimer, self.n_atoms_mol1)
        c1 = engine.batch_single_point(mon1, breakdown=True).components or {}
        c2 = engine.batch_single_point(mon2, breakdown=True).components or {}
        terms = set(dimer_c) | set(c1) | set(c2)
        zero = np.zeros(n)
        diff = {t: np.asarray(dimer_c.get(t, zero)) - np.asarray(c1.get(t, zero))
                - np.asarray(c2.get(t, zero)) for t in terms}
        return project_components_to_sapt(diff, n)

    def residual(self, engine) -> np.ndarray:
        model = self.model_components(engine)
        cols = list(self.components)
        return (self.weight * (model[:, cols] - self.qm_components[:, cols])).ravel()


@dataclass
class PolarizabilityTarget:
    """Residual on molecular-polarizability eigenvalues vs. a reference."""

    structure: TinkerXYZ
    reference_eigenvalues: np.ndarray  # (3,)
    weight: float = 1.0

    def residual(self, engine) -> np.ndarray:
        result = engine.polarizability(self.structure)
        eig = (result.extra or {}).get("polarizability_eigenvalues")
        if eig is None:
            raise ValueError("engine.polarizability returned no eigenvalues")
        return self.weight * (np.sort(np.asarray(eig)) - np.sort(np.asarray(self.reference_eigenvalues)))


@dataclass
class BulkPropertyTarget:
    """Residual on NPT bulk-property %-errors vs. experiment (e.g. density)."""

    box: Any
    experimental: Any                  # BulkProperties (experimental reference)
    n_molecules: int
    properties: tuple[str, ...] = ("density_kg_m3",)
    nsteps: int = 5000
    dt_fs: float = 2.0
    temperature: float = 298.15
    pressure: float = 1.0
    equil: int = 0
    weight: float = 1.0
    run_kwargs: dict = field(default_factory=dict)

    def residual(self, engine) -> np.ndarray:
        from ..liquid import compute_bulk_properties
        from ..simulate import run_npt

        traj = run_npt(engine, self.box, nsteps=self.nsteps, dt_fs=self.dt_fs,
                       temperature=self.temperature, pressure=self.pressure, **self.run_kwargs)
        props = compute_bulk_properties(traj, equil=self.equil, n_molecules=self.n_molecules)
        res = []
        for name in self.properties:
            model = getattr(props, name)
            ref = getattr(self.experimental, name)
            if model is None or ref in (None, 0):
                res.append(0.0)
            else:
                res.append(self.weight * (model - ref) / ref)  # fractional error
        return np.asarray(res, dtype=float)


__all__ = [
    "split_dimer",
    "project_components_to_sapt",
    "DimerInteractionTarget",
    "PolarizabilityTarget",
    "BulkPropertyTarget",
]
