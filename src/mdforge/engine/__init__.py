"""mdforge.engine — the unified MD-engine layer (Phase 2 keystone).

One :class:`Engine` Protocol, two public implementations (Tinker primary, OpenMM
secondary), and a :func:`register` plugin seam so a private companion package can
add engines without touching this codebase. Goals (a) run sims and (b) run
analysis route through here; the fitter (Phase 6) pulls every per-evaluation
energy/gradient/property from an ``Engine``.

    from mdforge.engine import get_engine
    eng = get_engine("tinker", bin_dir="/path/to/tinker/bin", key_file="mol.key")
    result = eng.single_point(structure, breakdown=True)
    result.energy_in("kcal/mol")   # units self-described + normalizable

Both engine modules import cleanly without their backend present (Tinker binaries
/ the openmm package are only needed when a method actually runs).
"""

from __future__ import annotations

from .base import Capabilities, Engine, EngineResult, UnsupportedOperation, require
from .openmm import OpenMMEngine
from .registry import available, get_engine, register, unregister
from .runner import LocalRunner, Runner, RunResult, SSHRunner
from .tinker import TinkerEngine

__all__ = [
    # interface
    "Engine", "EngineResult", "Capabilities", "UnsupportedOperation", "require",
    # registry / plugin seam
    "get_engine", "register", "unregister", "available",
    # runners
    "Runner", "RunResult", "LocalRunner", "SSHRunner",
    # implementations
    "TinkerEngine", "OpenMMEngine",
]
