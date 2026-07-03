"""Parse Tinker ``analyze`` / ``testgrad`` stdout into numpy arrays (goal c).

Pure text in, arrays out — the engine (Phase 2) runs Tinker and hands the
captured stdout here; this module never spawns a process. Consolidates and
reconciles the scrapers in ``analyzetool.auxtinker`` (12-term vocabulary) and
``prior internal tooling`` (13-term, adds ``Charge``).

Terms are matched **by name** rather than by position, so both the AMOEBA-style
(with ``Charge``) and HIPPO-style (no ``Charge``) breakdowns parse correctly and
``sapt_components`` is robust to the term-list difference that made the legacy
positional indexing fragile.
"""

from __future__ import annotations

import numpy as np

# Canonical superset of Tinker "Energy Component Breakdown" labels, matched by
# their FULL name (verified against real Tinker output). Matching the last word
# alone is ambiguous — e.g. "Torsional Angle" and "Angle Bending" both end in a
# word that collides — which is a latent bug in the legacy scrapers. Absent
# terms simply stay zero for a given force field.
ENERGY_TERMS: tuple[str, ...] = (
    "Bond Stretching",
    "Angle Bending",
    "Stretch-Bend",
    "Urey-Bradley",
    "Out-of-Plane Bend",
    "Improper Dihedral",
    "Torsional Angle",
    "Van der Waals",
    "Charge-Charge",       # fixed-charge electrostatics (AMBER/CHARMM/OPLS)
    "Atomic Multipoles",   # AMOEBA/HIPPO electrostatics
    "Repulsion",           # HIPPO
    "Dispersion",          # HIPPO
    "Polarization",
    "Charge Transfer",     # HIPPO
)

# SAPT-like decomposition mapping (HIPPO → SAPT components), by full name.
_SAPT_MAP = {
    "electrostatics": ("Atomic Multipoles", "Charge-Charge"),
    "exchange": ("Repulsion",),
    "induction": ("Polarization", "Charge Transfer"),
    "dispersion": ("Dispersion",),
}


def _isfloat(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _isint(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


def parse_energy_breakdown(text: str) -> list[dict[str, float]]:
    """Parse one dict of energy components per frame from ``analyze e`` output.

    Each dict maps a component keyword (e.g. ``"Multipoles"``) to its energy,
    plus ``"Total"`` and (for multi-molecule systems) ``"Intermolecular"``.
    Works for single- and multi-frame (``.arc``) output.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    inter_idx = [i for i, ln in enumerate(lines) if "Intermolecular Energy" in ln]
    total_idx = [i for i, ln in enumerate(lines) if "Total Potential" in ln]
    use_inter = len(inter_idx) > 0
    markers = sorted(inter_idx if use_inter else total_idx)
    if not markers:
        return []

    bounds = markers + [len(lines)]
    frames: list[dict[str, float]] = []
    for fi in range(len(markers)):
        block = lines[bounds[fi]:bounds[fi + 1]]
        frame: dict[str, float] = {}
        for ln in block:
            s = ln.split()
            if "Total Potential" in ln and len(s) >= 2 and _isfloat(s[-2]):
                frame["Total"] = float(s[-2])
            elif "Intermolecular Energy" in ln and len(s) >= 2 and _isfloat(s[-2]):
                frame["Intermolecular"] = float(s[-2])
            elif len(s) >= 3 and _isint(s[-1]) and _isfloat(s[-2]) and not _isfloat(s[0]):
                # "<component name…> <energy> <interaction count>" — capture the
                # FULL multi-word component name, not just its last token.
                frame[" ".join(s[:-2])] = float(s[-2])
        frames.append(frame)
    return frames


def energy_components(
    text: str, terms: tuple[str, ...] = ENERGY_TERMS
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(components, intermolecular)`` arrays from ``analyze e`` output.

    ``components`` is ``(n_frames, len(terms))`` in the given term order;
    ``intermolecular`` is ``(n_frames,)`` (NaN where absent).
    """
    frames = parse_energy_breakdown(text)
    comps = np.zeros((len(frames), len(terms)), dtype=float)
    inter = np.full(len(frames), np.nan, dtype=float)
    for i, fr in enumerate(frames):
        for j, t in enumerate(terms):
            if t in fr:
                comps[i, j] = fr[t]
        if "Intermolecular" in fr:
            inter[i] = fr["Intermolecular"]
    return comps, inter


def sapt_components(source, terms: tuple[str, ...] = ENERGY_TERMS) -> np.ndarray:
    """Map a HIPPO energy breakdown to ``[elst, exch, ind, disp, total]``.

    Accepts a single parsed frame dict, or a ``(len(terms),)`` / ``(F, len(terms))``
    component array (as returned by :func:`energy_components`).
    """
    if isinstance(source, dict):
        def g(*names):
            return float(sum(source.get(n, 0.0) for n in names))
        elst = g(*_SAPT_MAP["electrostatics"])
        exch = g(*_SAPT_MAP["exchange"])
        ind = g(*_SAPT_MAP["induction"])
        disp = g(*_SAPT_MAP["dispersion"])
        total = float(source.get("Total", elst + exch + ind + disp))
        return np.array([elst, exch, ind, disp, total], dtype=float)

    arr = np.asarray(source, dtype=float)
    idx = {t: j for j, t in enumerate(terms)}

    def col(names, a):
        return sum(a[..., idx[n]] for n in names if n in idx)

    if arr.ndim == 1:
        elst = col(_SAPT_MAP["electrostatics"], arr)
        exch = col(_SAPT_MAP["exchange"], arr)
        ind = col(_SAPT_MAP["induction"], arr)
        disp = col(_SAPT_MAP["dispersion"], arr)
        return np.array([elst, exch, ind, disp, arr.sum()], dtype=float)

    elst = col(_SAPT_MAP["electrostatics"], arr)
    exch = col(_SAPT_MAP["exchange"], arr)
    ind = col(_SAPT_MAP["induction"], arr)
    disp = col(_SAPT_MAP["dispersion"], arr)
    return np.array([elst, exch, ind, disp, arr.sum(axis=1)])


def parse_testgrad(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``testgrad`` output into ``(energies, gradients)``.

    ``energies`` is ``(n_frames,)`` and ``gradients`` is ``(n_frames, n_atoms, 3)``,
    read from the "Cartesian Gradient Breakdown over Individual Atoms" blocks.
    Ported from ``TinkerRunner.analyze_gradients`` (text-only).
    """
    energies: list[float] = []
    gradients: list[list[list[float]]] = []
    current: list[list[float]] = []
    reading = False

    for line in text.splitlines():
        if "Total Potential Energy :" in line:
            if current:
                gradients.append(current)
            current = []
            energies.append(float(line.split()[-2]))
            reading = False
            continue
        if "Cartesian Gradient Breakdown over Individual Atoms" in line:
            reading = True
            continue
        if reading and "Anlyt" in line:
            parts = line.split()
            # "Anlyt <atom> dE/dX dE/dY dE/dZ [Norm]" — take the 3 gradient
            # components by position (the legacy parts[-3:] wrongly grabbed
            # [dY, dZ, Norm] because real Tinker appends a Norm column).
            if len(parts) >= 5 and _isint(parts[1]):
                current.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if current:
        gradients.append(current)

    return np.asarray(energies, dtype=float), np.asarray(gradients, dtype=float)


__all__ = [
    "ENERGY_TERMS",
    "parse_energy_breakdown",
    "energy_components",
    "sapt_components",
    "parse_testgrad",
]
