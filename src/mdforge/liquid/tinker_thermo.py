"""Pre-converter: Tinker liquid-run logs → a HOOMD-style per-frame thermo array.

The evaluation ingest layer reads per-frame thermodynamics from a HOOMD-style
structured array (``.npy`` / ``.csv``; see
:func:`mdforge.liquid.parse.trajectory_from_hoomd_npy`). Tinker instead writes
human-readable text logs — a dynamics log (per-frame ``Current Potential`` /
``Current Kinetic`` / ``Lattice Lengths`` / ``Current Time``) and an ``analyze``
log (per-frame potential energy + cell dipole, and the system mass / atom count).

This module bridges the two so a Tinker run can be scored by ``mdforge-eval``:
it scrapes the logs (reusing :func:`~mdforge.liquid.parse.parse_dynamics_log`
and :func:`~mdforge.liquid.parse.parse_analyze_log`) and writes a structured
array with the columns the ingest layer understands::

    step, time_ps, temp_K, pe, ke, e_total, volume_ang3, density_gcc

and, when the analyze log carries a per-frame cell dipole, ``dipole_{x,y,z}``
(Debye) so the evaluation can compute a proper ε₀ (including induced dipoles).

Point a leg's ``log:`` at the emitted file in ``eval.yaml``. The DCD trajectory
still supplies structure/diffusion; this file supplies the thermodynamics
(density, ΔH_vap, Cp, α_T, κ_T) and the dielectric.

CLI::

    python -m mdforge.liquid.tinker_thermo <run_dir>            # writes <run_dir>/liquid_thermo.npy
    python -m mdforge.liquid.tinker_thermo liquid.log -o t.csv  # single log -> CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .parse import parse_analyze_log, parse_dynamics_log

_KB_KCAL_MOL_K = 0.0019872041      # Boltzmann constant (kcal/mol/K)
# density [g/cm^3] = mass_amu / (N_A * 1e-24 * V[A^3]) = mass_amu / (0.602214076 * V)
_AMU_A3_TO_GCC = 0.602214076


def build_thermo_array(
    dynamics_log: str | Path,
    *,
    analyze_log: str | Path | None = None,
    n_atoms: int | None = None,
    total_mass_amu: float | None = None,
    save_interval_ps: float | None = None,
) -> np.ndarray:
    """Assemble a HOOMD-style per-frame thermo array from Tinker logs.

    ``dynamics_log`` supplies per-frame PE / KE / volume / time. When an
    ``analyze_log`` is given it fills in the system mass and atom count (for
    density and kinetic temperature) if those are not passed explicitly. The
    kinetic temperature uses ``T = 2·KE / ((3·n_atoms − 3)·k_B)``; density uses
    the system mass. Columns that cannot be derived (no mass → no ``density_gcc``;
    no atom count → no ``temp_K``) are simply omitted.
    """
    dyn = parse_dynamics_log(dynamics_log)
    pe, ke, vol, tps = (dyn["potential_energy"], dyn["kinetic_energy"],
                        dyn["volume"], dyn["time_ps"])
    n = min(len(pe), len(ke))
    if n == 0:
        raise ValueError(
            f"{dynamics_log}: no per-frame 'Current Potential'/'Current Kinetic' "
            "lines found (is this a Tinker dynamics log?)"
        )

    dipole = None
    if analyze_log is not None:
        anl = parse_analyze_log(analyze_log)
        if total_mass_amu is None:
            total_mass_amu = anl.get("mass")
        if n_atoms is None:
            n_atoms = anl.get("n_atoms")
        dip = anl.get("dipole")
        if dip is not None and len(dip) >= n:
            dipole = np.asarray(dip, dtype=float)[:n]   # total cell dipole (Debye)

    cols: dict[str, np.ndarray] = {"step": np.arange(1, n + 1, dtype=np.int64)}
    if len(tps) >= n:
        cols["time_ps"] = tps[:n]
    elif save_interval_ps:
        cols["time_ps"] = np.arange(1, n + 1, dtype=float) * float(save_interval_ps)

    if n_atoms:
        ndf = 3 * int(n_atoms) - 3
        cols["temp_K"] = 2.0 * ke[:n] / (ndf * _KB_KCAL_MOL_K)

    cols["pe"] = pe[:n]
    cols["ke"] = ke[:n]
    cols["e_total"] = pe[:n] + ke[:n]

    if len(vol) >= n:
        cols["volume_ang3"] = vol[:n]
        if total_mass_amu:
            cols["density_gcc"] = float(total_mass_amu) / (_AMU_A3_TO_GCC * vol[:n])

    # per-frame total cell dipole (Debye) → enables a proper ε (incl. induced)
    if dipole is not None:
        cols["dipole_x"], cols["dipole_y"], cols["dipole_z"] = (
            dipole[:, 0], dipole[:, 1], dipole[:, 2])

    dtype = [(k, "i8" if k == "step" else "f8") for k in cols]
    arr = np.zeros(n, dtype=dtype)
    for k, v in cols.items():
        arr[k] = v
    return arr


def write_thermo(out_path: str | Path, arr: np.ndarray) -> Path:
    """Write ``arr`` as ``.npy`` (structured array) or ``.csv`` (with header)."""
    out = Path(out_path)
    names = list(arr.dtype.names)
    if out.suffix.lower() == ".csv":
        cols = np.column_stack([arr[n] for n in names])
        fmt = ["%d" if n == "step" else "%.8g" for n in names]
        np.savetxt(out, cols, delimiter=",", header=",".join(names),
                   comments="", fmt=fmt)
    else:
        np.save(out, arr)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mdforge-tinker-thermo",
        description="Convert Tinker dynamics/analyze logs into a HOOMD-style "
                    "per-frame thermo array (.npy/.csv) for mdforge-eval.",
    )
    p.add_argument("path", type=Path,
                   help="Run directory (uses --dynamics-log/--analyze-log within it) "
                        "or a single dynamics-log file.")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output file (.npy or .csv). Default: <dir>/liquid_thermo.npy.")
    p.add_argument("--dynamics-log", default="liquid.log",
                   help="Dynamics-log name within a run directory (default: liquid.log).")
    p.add_argument("--analyze-log", default="analysis.log",
                   help="Analyze-log name within a run directory (default: analysis.log).")
    p.add_argument("--n-atoms", type=int, default=None,
                   help="Atom count for kinetic temperature (else read from the analyze log).")
    p.add_argument("--save-interval-ps", type=float, default=None,
                   help="Fallback frame spacing (ps) if the log has no 'Current Time' lines.")
    args = p.parse_args(argv)

    if args.path.is_dir():
        dyn = args.path / args.dynamics_log
        anl = args.path / args.analyze_log
        anl = anl if anl.is_file() else None
        out = args.out or (args.path / "liquid_thermo.npy")
    else:
        dyn = args.path
        anl = None
        out = args.out or args.path.with_suffix(".npy")

    if not Path(dyn).is_file():
        raise SystemExit(f"error: dynamics log not found: {dyn}")

    arr = build_thermo_array(dyn, analyze_log=anl, n_atoms=args.n_atoms,
                             save_interval_ps=args.save_interval_ps)
    write_thermo(out, arr)
    print(f"wrote {out}  ({len(arr)} frames; columns: {', '.join(arr.dtype.names)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
