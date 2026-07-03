#!/usr/bin/env python3
"""Compute the number of water molecules to add to a monomer to reach a target
total atom count, plus the cubic box edge for a desired density.

System = 1 monomer (solute) + N waters. Total atoms = n_mono + 3*N.
"""
from __future__ import annotations

import argparse

from mdforge.formats.txyz import read_txyz
from mdforge.simulate.box import box_edge_for_density, molar_mass

WATER = ["O", "H", "H"]  # 3 atoms / molecule


def waters_for_total(monomer_txyz: str, total_atoms: int, density_g_cm3: float):
    mono = read_txyz(monomer_txyz)
    n_mono = mono.n_atoms

    rem = (total_atoms - n_mono) % 3
    if total_atoms < n_mono or rem:
        raise ValueError(
            f"total_atoms={total_atoms} not reachable: monomer has {n_mono} atoms; "
            f"(total - {n_mono}) must be a non-negative multiple of 3 (off by {rem})."
        )
    n_water = (total_atoms - n_mono) // 3

    # Total molar mass of the whole system, then box edge for the target density.
    total_mm = molar_mass(mono) + n_water * molar_mass(WATER)
    edge = box_edge_for_density(n_molecules=1, molar_mass_g=total_mm, density_g_cm3=density_g_cm3)
    return n_mono, n_water, total_mm, edge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("monomer_txyz", help="Tinker .xyz of the monomer/solute")
    ap.add_argument("total_atoms", type=int, help="desired total atom count")
    ap.add_argument("density", type=float, help="target density (g/cm^3)")
    args = ap.parse_args()

    n_mono, n_water, total_mm, edge = waters_for_total(
        args.monomer_txyz, args.total_atoms, args.density
    )
    print(f"monomer atoms : {n_mono}")
    print(f"water molecules: {n_water}  ({3 * n_water} atoms)")
    print(f"total atoms    : {n_mono + 3 * n_water}")
    print(f"system molar mass: {total_mm:.3f} g/mol")
    print(f"cubic box edge @ {args.density} g/cm^3: {edge:.3f} Å")


if __name__ == "__main__":
    main()
