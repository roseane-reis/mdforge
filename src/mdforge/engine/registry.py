"""Engine registry + plugin seam.

``get_engine(name, **config)`` constructs a registered engine; ``register(name,
factory)`` lets a private companion package inject additional engines at runtime
without touching this codebase. Built-in factories import their engine module
lazily, so importing the registry never requires Tinker or OpenMM to be present.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Engine

_REGISTRY: dict[str, Callable[..., Engine]] = {}


def register(name: str, factory: Callable[..., Engine], *, overwrite: bool = False) -> None:
    """Register an engine factory under ``name``.

    The factory is any callable returning an object satisfying the
    :class:`~mdforge.engine.base.Engine` protocol.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Engine {name!r} already registered (pass overwrite=True to replace)")
    _REGISTRY[name] = factory


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def available() -> list[str]:
    """Return the sorted names of all registered engines."""
    return sorted(_REGISTRY)


def get_engine(name: str, **config) -> Engine:
    """Construct the engine registered under ``name`` with ``config`` kwargs."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown engine {name!r}; available: {available()}")
    return _REGISTRY[name](**config)


# --- built-in engines (lazy imports) ----------------------------------------

def _tinker_factory(**config) -> Engine:
    from .tinker import TinkerEngine
    return TinkerEngine(**config)


def _openmm_factory(**config) -> Engine:
    from .openmm import OpenMMEngine
    return OpenMMEngine(**config)


register("tinker", _tinker_factory)
register("openmm", _openmm_factory)


__all__ = ["register", "unregister", "available", "get_engine"]
