"""Batch simulation dispatch (goal a).

Generalizes the ``run_sim.py`` batch/watchdog logic into an engine- and
host-agnostic dispatcher: run many independent jobs (each a no-arg callable —
typically an ``engine.dynamics`` call) with a concurrency cap, capturing
results/exceptions and timing. *Where* each job runs (local CPU, remote GPU via
``SSHRunner``, etc.) is the engine/runner's concern, not hardcoded here — so the
old ``elf*``-specific SSH/SLURM coupling is gone.

A hard per-job timeout that kills a runaway subprocess belongs on the engine
(``TinkerEngine.timeout``); ``run_jobs(timeout=...)`` is a soft batch deadline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass
class JobResult:
    name: str
    ok: bool
    value: Any = None
    error: str | None = None
    elapsed_s: float = 0.0


def _run_one(name: str, fn: Callable[[], Any]) -> JobResult:
    t0 = time.perf_counter()
    try:
        value = fn()
        return JobResult(name, True, value, None, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - jobs isolate failures
        return JobResult(name, False, None, f"{type(exc).__name__}: {exc}",
                         time.perf_counter() - t0)


def run_jobs(
    jobs: dict[str, Callable[[], Any]] | list[Callable[[], Any]],
    *,
    max_parallel: int = 4,
    timeout: float | None = None,
) -> list[JobResult]:
    """Run jobs concurrently (capped at ``max_parallel``); return per-job results.

    Parameters
    ----------
    jobs:
        A ``{name: callable}`` mapping or a list of callables (named by index).
        Each callable takes no args and returns the job's result.
    max_parallel:
        Maximum concurrent jobs.
    timeout:
        Optional soft batch deadline (seconds). Jobs not finished by then are
        marked failed with ``error='timeout (batch)'``.

    Failures are isolated: one job raising does not abort the batch. Results are
    returned in the input order.
    """
    items = list(jobs.items()) if isinstance(jobs, dict) else [
        (str(i), fn) for i, fn in enumerate(jobs)
    ]
    out: dict[str, JobResult] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fut_to_name = {ex.submit(_run_one, name, fn): name for name, fn in items}
        try:
            for fut in as_completed(fut_to_name, timeout=timeout):
                res = fut.result()
                out[res.name] = res
        except TimeoutError:
            for fut, name in fut_to_name.items():
                if name not in out:
                    fut.cancel()
                    out[name] = JobResult(name, False, None, "timeout (batch)", 0.0)
    return [out[name] for name, _ in items]


__all__ = ["JobResult", "run_jobs"]
