"""QM interaction-energy dimer databases → records (goal d).

Loads this project's compact SAPT format (S101x7, qm-calc ``sapt-res``/``DESRES``
npy: ``(n_geom, 5) = [elst, exch, ind, disp, CCSD(T)/total]`` kcal/mol) plus the
two big external benchmarks (Donchev DES370K wide-component dict → projected to
the compact 5; NCIA total-E_int benchmark files). These are the fitting targets
Phase 6 (`fit`) consumes: geometries to run the engine on + reference components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Compact SAPT-component column order (kcal/mol).
SAPT_COMPONENTS = ("electrostatics", "exchange", "induction", "dispersion", "total")


@dataclass
class DimerSet:
    """A dimer's geometry scan with QM interaction-energy reference.

    - ``qm_components``      : ``(n_geom, 5)`` ``[elst, exch, ind, disp, total]`` kcal/mol
    - ``interaction_energy`` : ``(n_geom,)`` total/CCSD(T) reference
    - ``geometries``         : ``(n_geom, N, 3)`` Angstrom (optional)
    - ``n_atoms_per_mol``    : ``(n1, n2)`` monomer atom-count split (optional)
    """

    name: str
    qm_components: np.ndarray | None = None
    interaction_energy: np.ndarray | None = None
    geometries: np.ndarray | None = None
    elements: list[str] | None = None
    n_atoms_per_mol: tuple[int, int] | None = None
    smiles: tuple[str, str] | None = None
    cids: tuple[int, int] | None = None
    source: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def n_geometries(self) -> int:
        for arr in (self.qm_components, self.interaction_energy, self.geometries):
            if arr is not None:
                return len(arr)
        return 0


def load_sapt_dimer(base_path: str | Path, *, source: str = "sapt", name: str | None = None) -> DimerSet:
    """Load a compact-SAPT dimer from a ``<base>`` stem.

    Reads ``<base>.npy`` ``(n,5)`` (required) and, if present, ``<base>.arc``
    (geometries) and ``<base>-mol1.xyz``/``-mol2.xyz`` (monomer split). Covers the
    S101x7 and qm-calc ``DESRES_*`` conventions.
    """
    base = Path(base_path)
    npy = base.with_suffix(".npy")
    comps = np.asarray(np.load(npy), dtype=float)
    if comps.ndim != 2 or comps.shape[1] != 5:
        raise ValueError(f"{npy} must be (n_geom, 5); got {comps.shape}")

    geometries = elements = None
    arc = base.with_suffix(".arc")
    if arc.is_file():
        from ..formats.arc import read_arc
        traj = read_arc(arc)
        geometries, elements = traj.coords, traj.names

    n_atoms_per_mol = None
    mol1 = base.parent / f"{base.name}-mol1.xyz"
    mol2 = base.parent / f"{base.name}-mol2.xyz"
    if mol1.is_file() and mol2.is_file():
        from ..formats.txyz import read_txyz
        n_atoms_per_mol = (read_txyz(mol1).n_atoms, read_txyz(mol2).n_atoms)

    return DimerSet(
        name=name or base.name, qm_components=comps, interaction_energy=comps[:, 4],
        geometries=geometries, elements=elements, n_atoms_per_mol=n_atoms_per_mol,
        source=source, metadata={"npy": str(npy)},
    )


def load_s101x7_pair(npy_path: str | Path) -> DimerSet:
    """Load one S101x7 dimer pair (``S101x7/<a>/<a>-<b>.npy`` + siblings)."""
    return load_sapt_dimer(Path(npy_path).with_suffix(""), source="S101x7")


# ---------------------------------------------------------------------------
# DES370K (Donchev) — wide named components → compact 5
# ---------------------------------------------------------------------------

def project_des370k_components(gid_value: dict, *, delta_hf_in_induction: bool = True) -> np.ndarray:
    """Project a DES370K ``data_per_gid_proc`` entry to ``(n_geom, 5)`` compact SAPT.

    Grouping (standard SAPT physical components):
    - electrostatics = ``sapt_es``
    - exchange       = ``sapt_ex + sapt_exs2``
    - induction      = ``sapt_ind + sapt_exind`` (+ ``sapt_delta_HF`` by default)
    - dispersion     = ``sapt_disp + sapt_exdisp_os + sapt_exdisp_ss``
    - total          = ``cc_CCSD(T)_all`` (fallback ``sapt_all``)
    """
    def col(key: str) -> np.ndarray:
        v = gid_value.get(key)
        return np.zeros(_n_geom(gid_value)) if v is None else np.asarray(v, dtype=float)

    es = col("sapt_es")
    exch = col("sapt_ex") + col("sapt_exs2")
    ind = col("sapt_ind") + col("sapt_exind")
    if delta_hf_in_induction:
        ind = ind + col("sapt_delta_HF")
    disp = col("sapt_disp") + col("sapt_exdisp_os") + col("sapt_exdisp_ss")
    total = gid_value.get("cc_CCSD(T)_all")
    total = col("sapt_all") if total is None else np.asarray(total, dtype=float)
    return np.column_stack([es, exch, ind, disp, total])


def _n_geom(gid_value: dict) -> int:
    for v in gid_value.values():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            return len(v)
    return 0


def load_des370k_gid(proc_dict: dict, gid, *, delta_hf_in_induction: bool = True) -> DimerSet:
    """Build a :class:`DimerSet` (components only) for one DES370K group id."""
    value = proc_dict[gid]
    comps = project_des370k_components(value, delta_hf_in_induction=delta_hf_in_induction)
    return DimerSet(
        name=f"DES370K_gid{gid}", qm_components=comps, interaction_energy=comps[:, 4],
        source="DES370K", metadata={"gid": gid, "cc_basis": value.get("cc_basis")},
    )


# ---------------------------------------------------------------------------
# NCIA — total CCSD(T)/CBS interaction-energy benchmark
# ---------------------------------------------------------------------------

def parse_ncia_benchmark(path: str | Path) -> dict[str, float]:
    """Parse an ``NCIA_*_benchmark.txt`` into ``{systemID: E_int}`` (kcal/mol).

    systemID encodes the molecule pair + distance scaling (e.g. ``1.001_080``);
    comment lines start with ``#``.
    """
    out: dict[str, float] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


__all__ = [
    "SAPT_COMPONENTS",
    "DimerSet",
    "load_sapt_dimer",
    "load_s101x7_pair",
    "project_des370k_components",
    "load_des370k_gid",
    "parse_ncia_benchmark",
]
