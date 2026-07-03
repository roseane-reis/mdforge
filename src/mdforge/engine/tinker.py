"""Tinker / Tinker9 engine (file + subprocess), local or remote.

Merges the analyze/testgrad/minimize/dynamic scrapers from
``analyzetool.auxtinker``, ``prior internal tooling``, and
``analyzetool.run_sim``/``Auxfit.calltinker`` behind the :class:`Engine`
interface. Output parsing is delegated to :mod:`mdforge.formats.analyze_out`;
geometry I/O to :mod:`mdforge.formats.txyz`/``arc``.

Cleanups folded in (per ARCHITECTURE §4): no ``shell=True``, no ``os.chdir``
(commands run with ``cwd=`` via a :class:`~mdforge.engine.runner.Runner`); the
``cuda_device``/SSH location is the runner's concern, not baked into the command;
parameter-file paths are absolutized when staging so a temp workdir resolves them.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.records import Trajectory
from ..formats import analyze_out
from ..formats import arc as arcfmt
from ..formats import txyz as txyzfmt
from .base import Capabilities, EngineResult
from .runner import LocalRunner, Runner

_ENSEMBLE_CODE = {"nve": 1, "nvt": 2, "nph": 3, "npt": 4}


@dataclass
class TinkerEngine:
    """Drive Tinker (CPU) or Tinker9 (GPU) via a pluggable runner.

    Parameters
    ----------
    bin_dir:
        Directory containing the executables (CPU) or the ``tinker9`` binary.
    key_file:
        A ``.key`` bound at construction (the parameter set). Its ``parameters``
        line is absolutized when staged into a temp workdir.
    tinker9:
        If True, commands are ``{bin_dir}/tinker9 <verb>`` instead of ``{bin_dir}/<verb>``.
    runner:
        Executes commands; :class:`LocalRunner` (default) or
        :class:`~mdforge.engine.runner.SSHRunner` for remote GPU.
    workdir, keep_files:
        Base directory for temp run dirs and whether to keep them (debugging).
    timeout:
        Per-command timeout in seconds.
    """

    bin_dir: str | Path
    key_file: str | Path | None = None
    tinker9: bool = False
    runner: Runner = field(default_factory=LocalRunner)
    workdir: str | Path | None = None
    keep_files: bool = False
    timeout: float | None = 600.0
    capabilities: Capabilities = field(default_factory=lambda: Capabilities(
        single_point=True, gradient=True, components=True, minimize=True,
        dynamics=True, npt=True, batched=True, polarizability=True,
    ))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _exe(self, verb: str) -> list[str]:
        b = str(self.bin_dir).rstrip("/")
        return [f"{b}/tinker9", verb] if self.tinker9 else [f"{b}/{verb}"]

    @contextmanager
    def _rundir(self):
        base = str(self.workdir) if self.workdir else None
        d = Path(tempfile.mkdtemp(prefix="mdforge_tk_", dir=base))
        try:
            yield d
        finally:
            if not self.keep_files:
                shutil.rmtree(d, ignore_errors=True)

    def _write_structure(self, wd: Path, structure: Any, base: str = "mol") -> tuple[str, str]:
        """Write structure to the workdir; return (basename, extension)."""
        if isinstance(structure, arcfmt.ArcTrajectory):
            arcfmt.write_arc(structure, wd / f"{base}.arc")
            return base, "arc"
        if isinstance(structure, txyzfmt.TinkerXYZ):
            if not structure.is_tinker:
                raise ValueError("TinkerEngine needs a Tinker XYZ with atom types")
            txyzfmt.write_txyz(structure, wd / f"{base}.xyz")
            return base, "xyz"
        raise TypeError(
            f"TinkerEngine cannot serialize {type(structure).__name__}; "
            "pass a formats.TinkerXYZ or formats.ArcTrajectory"
        )

    def _write_key(self, wd: Path, base: str) -> None:
        if self.key_file is None:
            return
        key_path = Path(self.key_file)
        keydir = key_path.parent.resolve()
        out: list[str] = []
        for line in key_path.read_text().splitlines():
            toks = line.split()
            if toks and toks[0].lower() == "parameters" and len(toks) > 1:
                out.append(f"parameters {self._stage_parameters(wd, keydir, toks[1])}")
            else:
                out.append(line)
        (wd / f"{base}.key").write_text("\n".join(out) + "\n")

    @staticmethod
    def _stage_parameters(wd: Path, keydir: Path, token: str) -> str:
        """Copy the referenced ``.prm`` into the workdir and return its basename.

        A self-contained workdir resolves the parameter file in ``cwd`` both
        locally and after an :class:`SSHRunner` stages it to a remote host
        (Tinker appends ``.prm`` if the reference lacks it). If the file is not
        found locally, fall back to an absolute path (assume it exists on the
        run host).
        """
        p = Path(token)
        bases = [p] if p.is_absolute() else [keydir / token]
        candidates = []
        for b in bases:
            candidates += [b, b.with_suffix(".prm"), Path(f"{b}.prm")]
        for c in candidates:
            if c.is_file():
                shutil.copyfile(c, wd / c.name)
                return c.name
        return str((keydir / token).resolve()) if not p.is_absolute() else token

    def _run(self, wd: Path, argv: list[str], stdin: str | None = None) -> str:
        res = self.runner.run_in(wd, argv, stdin=stdin, timeout=self.timeout)
        if res.returncode != 0:
            raise RuntimeError(
                f"Tinker command failed ({argv[0].split('/')[-1]}, rc={res.returncode}):\n"
                f"{res.stderr or res.stdout[-800:]}"
            )
        return res.stdout

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def single_point(self, structure: Any, *, breakdown: bool = False, **opt) -> EngineResult:
        with self._rundir() as wd:
            base, ext = self._write_structure(wd, structure)
            self._write_key(wd, base)
            stdout = self._run(wd, self._exe("analyze") + [f"{base}.{ext}", "E"])
            frames = analyze_out.parse_energy_breakdown(stdout)
            if not frames:
                raise RuntimeError(f"Could not parse analyze output:\n{stdout[-800:]}")
            energy = np.array([f.get("Total", np.nan) for f in frames])
            inter = np.array([f.get("Intermolecular", np.nan) for f in frames])
            components = None
            if breakdown:
                # Every component the force field emits (not just the canonical
                # ENERGY_TERMS), keyed by its full Tinker label.
                keys = sorted({k for f in frames for k in f
                               if k not in ("Total", "Intermolecular")})
                components = {k: np.array([f.get(k, np.nan) for f in frames]) for k in keys}
            return EngineResult(
                energy=energy, energy_unit="kcal/mol", components=components,
                intermolecular=inter if np.isfinite(inter).any() else None,
                length_unit="angstrom",
            )

    def gradient(self, structure: Any, **opt) -> EngineResult:
        with self._rundir() as wd:
            base, ext = self._write_structure(wd, structure)
            self._write_key(wd, base)
            # testgrad prompts: analytic? y, numerical? n.
            stdout = self._run(wd, self._exe("testgrad") + [f"{base}.{ext}"], stdin="y\nn\n")
            energies, grads = analyze_out.parse_testgrad(stdout)
            if grads.size == 0:
                raise RuntimeError(f"Could not parse testgrad output:\n{stdout[-800:]}")
            return EngineResult(
                energy=energies, energy_unit="kcal/mol", gradient=grads,
                force_unit="kcal/mol/Angstrom", length_unit="angstrom",
            )

    def minimize(self, structure: Any, *, tol: float = 0.1, **opt) -> EngineResult:
        with self._rundir() as wd:
            base, ext = self._write_structure(wd, structure)
            self._write_key(wd, base)
            stdout = self._run(wd, self._exe("minimize") + [f"{base}.{ext}", str(tol)])
            final = np.nan
            for line in stdout.splitlines():
                if "Final Function Value" in line:
                    final = float(line.split()[-1])
            minimized = None
            produced = sorted(wd.glob(f"{base}.xyz_*"))
            if produced:
                minimized = txyzfmt.read_txyz(produced[-1])
            return EngineResult(
                energy=np.array([final]), energy_unit="kcal/mol",
                length_unit="angstrom", extra={"minimized": minimized},
            )

    def batch_single_point(self, structures: Any, *, breakdown: bool = False, **opt) -> EngineResult:
        """Single point over many frames in one ``analyze`` call (writes an .arc)."""
        traj = self._coerce_to_arc(structures)
        return self.single_point(traj, breakdown=breakdown, **opt)

    def dynamics(
        self, structure: Any, *, nsteps: int, dt_fs: float, ensemble: str = "nvt",
        temperature: float = 298.15, pressure: float | None = None,
        save_ps: float = 0.1, **opt,
    ) -> Trajectory:
        """Run Tinker ``dynamic`` (blocking) and return a :class:`Trajectory`.

        .. note::
            Basic blocking run — the GPU watchdog / multi-job dispatch
            (``run_sim``) is Phase 5 (``simulate.jobs``).
        """
        code = _ENSEMBLE_CODE.get(ensemble.lower())
        if code is None:
            raise ValueError(f"Unknown ensemble {ensemble!r}; use nve/nvt/nph/npt")
        if code == 4 and pressure is None:
            pressure = 1.0
        with self._rundir() as wd:
            base, ext = self._write_structure(wd, structure)
            self._write_key(wd, base)
            args = [f"{base}.{ext}", str(int(nsteps)), str(float(dt_fs)), str(float(save_ps)),
                    str(code), str(float(temperature))]
            if code in (3, 4):
                args.append(str(float(pressure)))
            stdout = self._run(wd, self._exe("dynamic") + args)

            arc_path = wd / f"{base}.arc"
            positions = box = masses = None
            volume_from_arc = None
            if arc_path.is_file():
                traj_arc = arcfmt.read_arc(arc_path)
                positions = traj_arc.coords
                box = traj_arc.box
                volume_from_arc = traj_arc.volume()
                from ..core.elements import mass_of
                masses = np.array([
                    mass_of("".join(c for c in nm if not c.isdigit()))
                    for nm in traj_arc.names
                ], dtype=float)
            # Per-save energies from the dynamic stdout/log.
            from ..liquid.parse import parse_dynamics_log
            (wd / "dyn.log").write_text(stdout)
            dyn = parse_dynamics_log(wd / "dyn.log")
            return Trajectory(
                positions=positions,
                potential_energy=dyn["potential_energy"] if dyn["potential_energy"].size else None,
                kinetic_energy=dyn["kinetic_energy"] if dyn["kinetic_energy"].size else None,
                volume=dyn["volume"] if dyn["volume"].size else volume_from_arc,
                box_vectors=box,
                masses=masses,
                temperature_K=temperature, dt_ps=dt_fs / 1000.0,
                metadata={"engine": "tinker", "ensemble": ensemble},
            )

    def polarizability(self, structure: Any, **opt) -> EngineResult:
        """Run ``polarize`` and return the molecular polarizability eigenvalues in ``extra``."""
        with self._rundir() as wd:
            base, ext = self._write_structure(wd, structure)
            self._write_key(wd, base)
            stdout = self._run(wd, self._exe("polarize") + [f"{base}.{ext}"])
            eig = []
            for line in stdout.splitlines():
                if "Polarizability Tensor Eigenvalues" in line:
                    nxt = stdout.splitlines()
                    idx = nxt.index(line)
                    for probe in nxt[idx:idx + 4]:
                        vals = [float(t) for t in probe.split() if _isfloat(t)]
                        if len(vals) >= 3:
                            eig = vals[-3:]
                            break
            return EngineResult(
                energy=np.array([np.nan]), energy_unit="kcal/mol",
                extra={"polarizability_eigenvalues": np.array(eig) if eig else None,
                       "stdout": stdout},
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_to_arc(structures: Any) -> arcfmt.ArcTrajectory:
        if isinstance(structures, arcfmt.ArcTrajectory):
            return structures
        if isinstance(structures, (list, tuple)) and structures:
            first = structures[0]
            if not isinstance(first, txyzfmt.TinkerXYZ):
                raise TypeError("batch expects a list of formats.TinkerXYZ or an ArcTrajectory")
            coords = np.stack([s.coords for s in structures], axis=0)
            return arcfmt.ArcTrajectory(
                coords=coords, names=first.names, types=first.types,
                connectivity=first.connectivity,
                box=None if first.box is None else np.stack([
                    (s.box if s.box is not None else first.box) for s in structures]),
                title=first.title,
            )
        raise TypeError("batch_single_point expects an ArcTrajectory or list of TinkerXYZ")


def _isfloat(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


__all__ = ["TinkerEngine"]
