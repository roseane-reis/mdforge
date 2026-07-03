"""Parse Tinker simulation outputs into array-based :class:`Trajectory` records.

This is the *parsing* half of the parse⟂compute split — it reads Tinker log /
``.arc`` / ``.vel`` files and emits numpy arrays. The compute kernels
(:mod:`~mdforge.liquid.thermo`, :mod:`~mdforge.liquid.transport`) never call
into here; they take the arrays.

.. note::
   **Interim implementation.** These readers are lifted from
   ``analyzetool`` (``liquid.Liquid``, ``gas.GasLog``, ``convert_to_numpy``).
   In Phase 1 the low-level format readers move to ``mdforge.formats``
   (``arc.py``, ``analyze_out.py``) and this module becomes a thin adapter that
   assembles a :class:`Trajectory` from them. The public functions here
   (:func:`trajectory_from_tinker`, the ``parse_*`` helpers) are the stable
   surface; their internals will be swapped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.elements import ELEMENT_MASSES  # noqa: F401  (re-exported for back-compat)
from ..core.records import Trajectory

# Tinker atom-name aliases occasionally seen in box files.
_TYPE_ALIASES = {"CA": "C", "HA": "H"}


def _mass_of(symbol: str) -> float:
    if symbol in ELEMENT_MASSES:
        return ELEMENT_MASSES[symbol]
    if symbol in _TYPE_ALIASES:
        return ELEMENT_MASSES[_TYPE_ALIASES[symbol]]
    raise KeyError(f"No mass for atom symbol {symbol!r}")


def _read_lines(path: str | Path) -> list[str]:
    with open(path) as fh:
        return fh.readlines()


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Box / topology
# ---------------------------------------------------------------------------

def parse_box_xyz(path: str | Path) -> dict:
    """Read a Tinker box ``.xyz`` for atom count, masses, and (if PBC) volume.

    Returns a dict with keys ``masses`` (N,), ``n_atoms``, ``symbols``, and
    ``volume`` (Å³ or None for a non-orthorhombic / non-PBC box).
    """
    lines = _read_lines(path)
    n_atoms = int(lines[0].split()[0])

    # A PBC box file has a lattice line (6 floats) as line 2.
    second = lines[1].split()
    has_lattice = len(second) >= 6 and all(_to_float(t) is not None for t in second[:6])
    start = 2 if has_lattice else 1

    volume = None
    if has_lattice:
        a, b, c, alpha, beta, gamma = (float(x) for x in second[:6])
        if round(alpha) == 90 and round(beta) == 90 and round(gamma) == 90:
            volume = a * b * c

    symbols: list[str] = []
    masses: list[float] = []
    for line in lines[start:start + n_atoms]:
        s = line.split()
        if len(s) < 2:
            continue
        sym = s[1]
        symbols.append(sym)
        masses.append(_mass_of(sym))

    return {
        "n_atoms": n_atoms,
        "symbols": symbols,
        "masses": np.array(masses, dtype=float),
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# Dynamics log  (Tinker `dynamic` stdout)
# ---------------------------------------------------------------------------

def parse_dynamics_log(path: str | Path) -> dict:
    """Parse a Tinker dynamics log for per-frame PE, KE, density, and volume.

    Scrapes the ``Current Potential``, ``Current Kinetic``, ``Density``, and
    ``Lattice Lengths`` report lines. Returns a dict of numpy arrays (any key
    may be empty if absent from the log).
    """
    pe, ke, dens, vol = [], [], [], []
    for line in _read_lines(path):
        s = line.split()
        if "Current Potential" in line:
            v = _to_float(s[2]) if len(s) > 2 else None
            if v is not None:
                pe.append(v)
        elif "Current Kinetic" in line:
            v = _to_float(s[2]) if len(s) > 2 else None
            if v is not None:
                ke.append(v)
        elif "Density" in line:
            v = _to_float(s[1]) if len(s) > 1 else None
            if v is not None:
                dens.append(v)
        elif "Lattice Lengths" in line and len(s) >= 5:
            a, b, c = _to_float(s[2]), _to_float(s[3]), _to_float(s[4])
            if None not in (a, b, c):
                vol.append(a * b * c)
    return {
        "potential_energy": np.array(pe, dtype=float),
        "kinetic_energy": np.array(ke, dtype=float),
        "density": np.array(dens, dtype=float),
        "volume": np.array(vol, dtype=float),
    }


# ---------------------------------------------------------------------------
# Analyze log  (Tinker `analyze` stdout)
# ---------------------------------------------------------------------------

def parse_analyze_log(path: str | Path) -> dict:
    """Parse a Tinker ``analyze`` log for PE, cell dipole, volume, and mass.

    Scrapes ``Total Potential Energy``, ``Dipole X,Y,Z-Components``,
    ``Cell Volume``, and ``Total System Mass``. Returns a dict of arrays plus a
    scalar ``mass`` (amu, or None).
    """
    pe, dip, vol = [], [], []
    mass: float | None = None
    for line in _read_lines(path):
        s = line.split()
        if "Total System Mass" in line:
            mass = _to_float(s[-1])
        elif "Total Potential Energy : " in line and len(s) > 4:
            v = _to_float(s[4])
            if v is not None:
                pe.append(v)
        elif "Dipole X,Y,Z-Components :" in line and len(s) >= 3:
            comps = [_to_float(s[i]) for i in range(-3, 0)]
            if None not in comps:
                dip.append(comps)
        elif "Cell Volume" in line:
            v = _to_float(s[-1])
            if v is not None:
                vol.append(v)
    return {
        "potential_energy": np.array(pe, dtype=float),
        "dipole": np.array(dip, dtype=float) if dip else np.empty((0, 3)),
        "volume": np.array(vol, dtype=float),
        "mass": mass,
    }


# ---------------------------------------------------------------------------
# Gas-phase log
# ---------------------------------------------------------------------------

def parse_gas_log(path: str | Path) -> dict:
    """Parse a gas-phase Tinker log; return per-frame PE and a reduced average.

    Mirrors ``analyzetool.gas.GasLog``: drops the first half (≤500 frames) as
    equilibration; if dynamics-style PE is found, the reported ``avg_pe`` is the
    mean of a Maxwell fit (matching the legacy heat-of-vaporization input),
    otherwise the plain mean of analyze-style energies.
    """
    edyn, eanl = [], []
    for line in _read_lines(path):
        s = line.split()
        if len(s) < 2:
            continue
        if "Current Potential" in line or "Potential Energy" in line:
            v = _to_float(s[2]) if len(s) > 2 else None
            if v is not None:
                edyn.append(v)
        if "Total Potential Energy : " in line:
            v = _to_float(s[4]) if len(s) > 4 else None
            if v is not None:
                eanl.append(v)

    if eanl:
        pe = np.array(eanl, dtype=float)
        half = min(int(pe.shape[0] / 2), 500)
        pe = pe[half:]
        return {"potential_energy": pe, "avg_pe": float(pe.mean()), "std_pe": float(pe.std())}

    if edyn:
        pe = np.array(edyn, dtype=float)
        half = min(int(pe.shape[0] / 2), 500)
        pe = pe[half:]
        avg, std = float(pe.mean()), float(pe.std())
        try:  # Maxwell fit, as in the legacy GasLog
            from scipy.stats import maxwell

            loc1, scale1 = maxwell.fit(pe)
            m2, v2 = maxwell.stats(loc=loc1, scale=(scale1 - 0.1))
            avg, std = float(m2), float(np.sqrt(v2))
        except Exception:
            pass
        return {"potential_energy": pe, "avg_pe": avg, "std_pe": std}

    return {"potential_energy": np.empty(0), "avg_pe": 0.0, "std_pe": 0.0}


# ---------------------------------------------------------------------------
# Velocity dump  (Tinker `.vel`)
# ---------------------------------------------------------------------------

def parse_velocity_dump(path: str | Path, n_atoms: int, max_frames: int | None = None) -> np.ndarray:
    """Read a Tinker ``.vel`` dump into a ``(T, N, 3)`` array (Å/ps).

    Each frame is a count line followed by ``n_atoms`` lines whose last three
    columns are the velocity components (Fortran ``D`` exponents accepted).
    """
    lines = _read_lines(path)
    frame_len = n_atoms + 1
    n_frames = len(lines) // frame_len
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    vels = np.zeros((n_frames, n_atoms, 3), dtype=float)
    for f in range(n_frames):
        base = f * frame_len + 1  # skip the per-frame count line
        for a in range(n_atoms):
            toks = lines[base + a].split()[-3:]
            vels[f, a] = [float(t.replace("D", "e")) for t in toks]
    return vels


# ---------------------------------------------------------------------------
# Virial tensor  (from an `analyze` log)
# ---------------------------------------------------------------------------

def parse_virial(path: str | Path) -> np.ndarray:
    """Extract the per-frame internal virial tensor ``(T, 3, 3)`` from a log.

    Ported from ``analyzetool.convert_to_numpy.get_virial``: locates blocks
    beginning with ``Int`` (``Internal Virial Tensor``) and reads the 3×3 block.
    """
    data = _read_lines(path)
    prefixes = np.array([ln[0:4] for ln in data])
    data_arr = np.array(data, dtype=object)
    starts = np.where(prefixes == " Int")[0]

    tensors = []
    for ind in starts:
        rows = [[float(x) for x in data_arr[ind].split()[-3:]]]
        r1 = data_arr[ind + 1].split()[-3:]
        r2 = data_arr[ind + 2].split()[-3:]
        if len(r1) == 3 and len(r2) == 3:
            rows.append([float(x) for x in r1])
            rows.append([float(x) for x in r2])
        if len(rows) == 3:
            tensors.append(rows)
    return np.array(tensors, dtype=float)


# ---------------------------------------------------------------------------
# High-level assembly
# ---------------------------------------------------------------------------

def trajectory_from_tinker(
    sim_path: str | Path,
    *,
    xyzfile: str = "liquid.xyz",
    dynamics_log: str | None = "liquid.log",
    analyze_log: str | None = "analysis.log",
    velocity_file: str | None = None,
    virial_from: str | None = None,
    temperature: float = 298.15,
    dt_ps: float = 0.001,
    n_atoms_per_molecule: int | None = None,
) -> Trajectory:
    """Assemble a :class:`Trajectory` from the outputs of a Tinker liquid run.

    Prefers the ``analyze`` log for PE / dipole / volume / mass; falls back to
    the dynamics log. Velocities and virial are read only if requested
    (needed for viscosity). Missing files are skipped silently — inspect the
    returned record to see which arrays were populated.

    Parameters
    ----------
    sim_path:
        Directory containing the run's files.
    xyzfile:
        Box ``.xyz`` filename (for masses, atom count, and a fallback volume).
    dynamics_log, analyze_log:
        Log filenames (relative to ``sim_path``); pass ``None`` to skip.
    velocity_file, virial_from:
        Optional ``.vel`` and analyze-log filenames for transport analysis.
    temperature, dt_ps, n_atoms_per_molecule:
        Recorded on the trajectory for downstream kernels.
    """
    sim = Path(sim_path)

    box = None
    box_path = sim / xyzfile
    if box_path.is_file():
        box = parse_box_xyz(box_path)

    kwargs: dict = {
        "temperature_K": temperature,
        "dt_ps": dt_ps,
        "n_atoms_per_molecule": n_atoms_per_molecule,
        "metadata": {"source": "tinker", "sim_path": str(sim)},
    }

    if box is not None:
        kwargs["masses"] = box["masses"]
        if n_atoms_per_molecule:
            kwargs["n_molecules"] = box["n_atoms"] // n_atoms_per_molecule

    # Analyze log (preferred for thermodynamics)
    analyze = None
    if analyze_log and (sim / analyze_log).is_file():
        analyze = parse_analyze_log(sim / analyze_log)
        if analyze["potential_energy"].size:
            kwargs["potential_energy"] = analyze["potential_energy"]
        if analyze["dipole"].size:
            kwargs["dipole"] = analyze["dipole"]
        if analyze["volume"].size:
            kwargs["volume"] = analyze["volume"]

    # Dynamics log (fills any gaps; supplies KE)
    if dynamics_log and (sim / dynamics_log).is_file():
        dyn = parse_dynamics_log(sim / dynamics_log)
        if "potential_energy" not in kwargs and dyn["potential_energy"].size:
            kwargs["potential_energy"] = dyn["potential_energy"]
        if dyn["kinetic_energy"].size:
            kwargs["kinetic_energy"] = dyn["kinetic_energy"]
        if "volume" not in kwargs and dyn["volume"].size:
            kwargs["volume"] = dyn["volume"]

    # Fallback to the static box volume if no per-frame volume was found.
    if "volume" not in kwargs and box is not None and box["volume"] is not None:
        n = len(kwargs.get("potential_energy", [])) or 1
        kwargs["volume"] = np.full(n, box["volume"], dtype=float)

    # Transport inputs (optional)
    if virial_from and (sim / virial_from).is_file():
        virial = parse_virial(sim / virial_from)
        if virial.size:
            kwargs["virial"] = virial
    if velocity_file and (sim / velocity_file).is_file() and box is not None:
        kwargs["velocities"] = parse_velocity_dump(sim / velocity_file, box["n_atoms"])

    return Trajectory(**kwargs)


# Column name aliases: HOOMD per-frame log → Trajectory field.
_HOOMD_NPY_ALIASES = {
    "potential_energy": ("pe", "potential_energy"),
    "kinetic_energy": ("ke", "kinetic_energy"),
    "total_energy": ("e_total", "total_energy"),
    "volume": ("volume_ang3", "volume"),
    "temperature": ("temp_K", "temperature_K"),
    "time_ps": ("time_ps",),
}


def trajectory_from_hoomd_npy(
    npy_path: str | Path,
    *,
    n_molecules: int,
    molar_mass_g_mol: float | None = None,
    total_mass_amu: float | None = None,
    temperature_K: float | None = None,
) -> Trajectory:
    """Assemble a :class:`Trajectory` from a HOOMD run's per-frame ``.npy`` log.

    The ``.npy`` is a numpy structured array with self-describing columns
    (``step, time_ps, temp_K, ke, pe, e_total, volume_ang3, ...``). Energies are
    mapped so that ``Trajectory.enthalpy`` resolves to ``PE + KE`` — i.e. the
    enthalpy uses the simulation's total energy, **not** the per-frame
    instantaneous virial pressure. At ordinary external pressure the ``P·V``
    term is negligible (≈1 kcal/mol at 1 atm), and feeding the noisy
    instantaneous ``P·V`` instead would swamp the real energy fluctuations and
    corrupt Cp / α. Supply density mass via ``molar_mass_g_mol`` (per molecule)
    or ``total_mass_amu``.

    Parameters
    ----------
    npy_path:
        Path to the run's ``<ens>.npy`` structured-array log.
    n_molecules:
        Molecule count (recorded for per-molecule Cp/ΔHvap).
    molar_mass_g_mol, total_mass_amu:
        System mass for the density kernel. Give one; ``total_mass`` wins if
        both are passed. Numerically amu/molecule and g/mol coincide.
    temperature_K:
        Scalar temperature for the kernels. Defaults to the mean of the
        ``temp_K`` column.
    """
    arr = np.load(Path(npy_path))
    names = set(arr.dtype.names or ())

    def col(field_key: str) -> np.ndarray | None:
        for cand in _HOOMD_NPY_ALIASES[field_key]:
            if cand in names:
                return np.asarray(arr[cand], dtype=float)
        return None

    kwargs: dict = {"n_molecules": n_molecules}
    for fld in ("potential_energy", "kinetic_energy", "total_energy", "volume"):
        vals = col(fld)
        if vals is not None:
            kwargs[fld] = vals

    temp = col("temperature")
    if temperature_K is None and temp is not None:
        temperature_K = float(np.mean(temp))
    kwargs["temperature_K"] = 298.15 if temperature_K is None else float(temperature_K)

    if total_mass_amu is None and molar_mass_g_mol is not None:
        total_mass_amu = n_molecules * molar_mass_g_mol
    if total_mass_amu is not None:
        # Encode as per-molecule masses so Trajectory.total_mass sums correctly.
        kwargs["masses"] = np.full(n_molecules, total_mass_amu / n_molecules, dtype=float)

    tp = col("time_ps")
    if tp is not None and len(tp) > 1:
        kwargs["dt_ps"] = float(np.median(np.diff(tp)))

    kwargs["metadata"] = {"source": "hoomd_npy", "npy_path": str(npy_path),
                          "columns": sorted(names)}
    return Trajectory(**kwargs)


__all__ = [
    "ELEMENT_MASSES",
    "parse_box_xyz",
    "parse_dynamics_log",
    "parse_analyze_log",
    "parse_gas_log",
    "parse_velocity_dump",
    "parse_virial",
    "trajectory_from_tinker",
    "trajectory_from_hoomd_npy",
]
