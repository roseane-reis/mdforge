"""Fit orchestration: assemble parameters + targets + engine into a fit (goal d).

Ports the driver shape of ``Auxfit.optimize_prms`` (build a weighted residual
from the targets at the current parameters) and ``runfit.py`` (the sequential
per-term recipe ``chgpen → dispersion → repulsion → polarize → chgtrn``), but the
per-evaluation work goes through any :class:`~mdforge.engine.base.Engine`.

``FitProblem.residual(x)`` is the function the optimizer drives: vector → prmdict
→ engine (built by ``engine_factory``) → concatenated target residuals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fitter import FitResult, differential_evolution_fit, least_squares_fit
from .parameters import ParameterSpace


@dataclass
class FitProblem:
    """A fit: parameter space + targets + a prmdict→engine factory."""

    param_space: ParameterSpace
    targets: list
    engine_factory: Callable[[dict], Any]

    def residual(self, x) -> np.ndarray:
        """vector → prmdict → engine → concatenated weighted target residuals."""
        prmdict = self.param_space.from_vector(x)
        engine = self.engine_factory(prmdict)
        parts = [np.atleast_1d(t.residual(engine)) for t in self.targets]
        return np.concatenate(parts) if parts else np.zeros(0)

    def fit(self, *, method: str = "least_squares", bounds=None, **kwargs) -> FitResult:
        """Run the optimizer. ``method`` ∈ {'least_squares', 'genetic'}."""
        x0 = self.param_space.to_vector()
        if bounds is None:
            bounds = self.param_space.bounds()
        if method == "least_squares":
            return least_squares_fit(self.residual, x0, bounds=bounds, **kwargs)
        if method in ("genetic", "differential_evolution"):
            de_bounds = list(zip(np.asarray(bounds[0]), np.asarray(bounds[1])))
            return differential_evolution_fit(self.residual, de_bounds, **kwargs)
        raise ValueError(f"Unknown method {method!r}; use 'least_squares' or 'genetic'")


def tinker_engine_factory(
    bin_dir: str | Path,
    workdir: str | Path,
    *,
    key_lines: list[str] | None = None,
    tinker9: bool = False,
    **engine_kwargs,
) -> Callable[[dict], Any]:
    """Return a ``prmdict -> TinkerEngine`` factory.

    Each call writes the prmdict to ``<workdir>/ff.prm`` and a key referencing it
    (minimal gas-style key by default; pass ``key_lines`` to append PBC/integrator
    settings for NPT bulk targets), then returns a :class:`TinkerEngine` bound to
    that key. The engine stages a self-contained workdir, so this works locally
    and over SSH.
    """
    from ..engine.tinker import TinkerEngine
    from ..formats.prm import write_prm

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def factory(prmdict: dict):
        prm_path = workdir / "ff.prm"
        write_prm(prmdict, prm_path)
        key_path = workdir / "ff.key"
        text = f"parameters {prm_path}\n"
        if key_lines:
            text += "\n".join(key_lines) + "\n"
        key_path.write_text(text)
        return TinkerEngine(bin_dir=bin_dir, key_file=key_path, tinker9=tinker9, **engine_kwargs)

    return factory


def run_sequential_fit(
    prmdict: dict,
    recipe: list[list[str]],
    target_factory: Callable[[dict, list[str]], list],
    engine_factory: Callable[[dict], Any],
    *,
    method: str = "least_squares",
    **fit_kwargs,
) -> tuple[dict, list[tuple[list[str], FitResult]]]:
    """Fit terms stage-by-stage, threading the updated prmdict through each stage.

    ``recipe`` is a list of term groups (e.g. ``[['chgpen'], ['dispersion'], …]``);
    ``target_factory(current_prmdict, termfit)`` returns the targets for that stage.
    Returns ``(final_prmdict, [(termfit, FitResult), …])``.
    """
    current = prmdict
    results: list[tuple[list[str], FitResult]] = []
    for termfit in recipe:
        space = ParameterSpace(current, termfit)
        targets = target_factory(current, termfit)
        problem = FitProblem(space, targets, engine_factory)
        result = problem.fit(method=method, **fit_kwargs)
        current = space.from_vector(result.x)
        results.append((termfit, result))
    return current, results


__all__ = ["FitProblem", "tinker_engine_factory", "run_sequential_fit"]
