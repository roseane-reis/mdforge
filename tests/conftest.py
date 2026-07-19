"""Shared synthetic fixtures for the evaluation tests.

Builders (returned as fixtures so they take ``tmp_path``) for a tiny rigid-body
water GSD, a matching HOOMD per-frame ``.npy`` log, and a matching PDB topology —
enough to exercise the ingest → pipeline → report chain without real trajectories.
"""

from __future__ import annotations

import numpy as np
import pytest

# Body-frame template for a 3-site water (O, H, H), Å.
WATER_TEMPLATE = np.array([
    [0.0, 0.0, 0.0],            # O
    [0.7570, 0.5860, 0.0],      # H
    [-0.7570, 0.5860, 0.0],     # H
], dtype=float)
WATER_ELEMENTS = ["O", "H", "H"]

# 4-site (TIP4P-style) template: O, H, H, then a massless M-site ~0.15 Å from O
# along the H–H bisector. Element/site order matches Tinker's tip4pbox.xyz.
WATER4_TEMPLATE = np.array([
    [0.0, 0.0, 0.0],            # O
    [0.7570, 0.5860, 0.0],      # H
    [-0.7570, 0.5860, 0.0],     # H
    [0.0, 0.1500, 0.0],         # M (ghost charge site)
], dtype=float)
WATER4_ELEMENTS = ["O", "H", "H", "M"]


@pytest.fixture
def make_water_4site():
    """Write a Tinker txyz topology + a matching multi-frame DCD for 4-site water.

    Returns ``(txyz_path, dcd_path)``. Per molecule the sites are ``O, H, H, M``
    (M is the massless ghost charge site), matching Tinker's tip4pbox.xyz layout.
    """
    from mdforge.formats.dcd import write_dcd
    from mdforge.formats.txyz import TinkerXYZ, write_txyz

    def _make(dir_path, *, n_mol=27, box_L=14.0, n_frames=8, seed=0):
        rng = np.random.default_rng(seed)
        side = int(round(n_mol ** (1 / 3))) + 1
        grid = [(i, j, k) for i in range(side) for j in range(side)
                for k in range(side)][:n_mol]
        centers = (np.array(grid, dtype=float) + 0.5) * (box_L / side)
        base = (centers[:, None, :] + WATER4_TEMPLATE[None, :, :]).reshape(-1, 3)

        names = WATER4_ELEMENTS * n_mol
        types = np.tile([1, 2, 2, 3], n_mol)
        conn = []
        for m in range(n_mol):
            o = m * 4 + 1                       # 1-based O index
            conn.append([o + 1, o + 2, o + 3])  # O bonded to H, H, M
            conn.append([o]); conn.append([o]); conn.append([o])
        box6 = np.array([box_L, box_L, box_L, 90.0, 90.0, 90.0])
        txyz = TinkerXYZ(names=names, coords=base, types=types,
                         connectivity=conn, box=box6, title="4-site water")
        txyz_path = dir_path / "liquid.xyz"
        write_txyz(txyz, path=txyz_path)

        frames = base[None] + rng.normal(scale=0.05, size=(n_frames, len(names), 3))
        box = np.tile(box6, (n_frames, 1))
        dcd_path = dir_path / "liquid.dcd"
        write_dcd(frames, box=box, path=dcd_path)
        return txyz_path, dcd_path
    return _make


@pytest.fixture
def make_water_4site_gsd():
    """Build a HOOMD rigid-body GSD for 4-site water (central + O, H, H, M).

    Mirrors a TIP4P-style HOOMD run: each rigid body is a central COM particle
    plus four constituents in ``O, H, H, M`` order (M is the massless ghost
    charge site, a rigid-body constituent). Identity orientations keep the body
    template recoverable by :func:`reference_geometry_from_gsd`.
    """
    def _make(path, *, n_mol=27, box_L=14.0, n_frames=6, seed=0):
        gsd_hoomd = pytest.importorskip("gsd.hoomd")
        rng = np.random.default_rng(seed)
        side = int(round(n_mol ** (1 / 3))) + 1
        grid = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        centers0 = (np.array(grid[:n_mol], dtype=float) + 0.5) * (box_L / side) - box_L / 2

        per_mol = 1 + len(WATER4_TEMPLATE)          # central + O,H,H,M = 5
        N = n_mol * per_mol
        types = ["water", "O", "H", "M"]
        typeid = np.tile([0, 1, 2, 2, 3], n_mol)     # central, O, H, H, M
        body = np.repeat(np.arange(n_mol) * per_mol, per_mol)
        q = np.sqrt(332.06371)
        charge = np.tile([0.0, 0.0, 0.55975 * q, 0.55975 * q, -1.1195 * q], n_mol)

        with gsd_hoomd.open(str(path), "w") as traj:
            centers = centers0.copy()
            for _ in range(n_frames):
                centers = centers + rng.normal(0, 0.15, size=(n_mol, 3))
                centers -= box_L * np.round(centers / box_L)
                pos = np.zeros((N, 3))
                orient = np.zeros((N, 4))
                for m in range(n_mol):
                    base = m * per_mol
                    pos[base] = centers[m]
                    pos[base + 1:base + per_mol] = centers[m] + WATER4_TEMPLATE
                    orient[base:base + per_mol] = [1.0, 0.0, 0.0, 0.0]
                frame = gsd_hoomd.Frame()
                frame.particles.N = N
                frame.particles.types = types
                frame.particles.typeid = typeid
                frame.particles.position = pos
                frame.particles.orientation = orient
                frame.particles.body = body
                frame.particles.charge = charge
                frame.configuration.box = [box_L, box_L, box_L, 0, 0, 0]
                traj.append(frame)
        return path
    return _make


@pytest.fixture
def make_water_gsd():
    def _make(path, *, n_mol=27, box_L=12.0, n_frames=6, seed=0):
        gsd_hoomd = pytest.importorskip("gsd.hoomd")
        rng = np.random.default_rng(seed)
        side = int(round(n_mol ** (1 / 3))) + 1
        grid = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        centers0 = (np.array(grid[:n_mol], dtype=float) + 0.5) * (box_L / side) - box_L / 2

        # per-molecule constant orientation (identity) keeps the template recoverable
        quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_mol, 1))
        # particle layout per molecule: [COM-center, O, H, H]
        per_mol = 1 + len(WATER_TEMPLATE)
        N = n_mol * per_mol
        types = ["water", "O", "H"]
        typeid = np.tile([0, 1, 2, 2], n_mol)
        body = np.repeat(np.arange(n_mol) * per_mol, per_mol)
        charge = np.tile([0.0, -0.68, 0.34, 0.34], n_mol)

        with gsd_hoomd.open(str(path), "w") as traj:
            centers = centers0.copy()
            for f in range(n_frames):
                centers = centers + rng.normal(0, 0.15, size=(n_mol, 3))
                centers -= box_L * np.round(centers / box_L)   # wrap into box
                pos = np.zeros((N, 3))
                orient = np.zeros((N, 4))
                for m in range(n_mol):
                    base = m * per_mol
                    pos[base] = centers[m]
                    pos[base + 1:base + 4] = centers[m] + WATER_TEMPLATE
                    orient[base] = quat[m]
                    orient[base + 1:base + 4] = quat[m]
                frame = gsd_hoomd.Frame()
                frame.particles.N = N
                frame.particles.types = types
                frame.particles.typeid = typeid
                frame.particles.position = pos
                frame.particles.orientation = orient
                frame.particles.body = body
                frame.particles.charge = charge
                frame.configuration.box = [box_L, box_L, box_L, 0, 0, 0]
                traj.append(frame)
        return path
    return _make


@pytest.fixture
def make_hoomd_npy():
    def _make(path, *, n_frames=60, ensemble="npt", n_molecules=27, box_L=12.0, seed=1):
        rng = np.random.default_rng(seed)
        cols = ["step", "time_ps", "temp_K", "ke", "pe", "e_total",
                "volume_ang3", "density_gcc", "pe_rigg", "pe_ewald", "pe_pppm"]
        if ensemble == "npt":
            cols.append("pressure_atm")
        dt = np.dtype([(c, "i8" if c == "step" else "f8") for c in cols])
        a = np.zeros(n_frames, dtype=dt)
        a["step"] = np.arange(n_frames) * 2500
        a["time_ps"] = np.arange(n_frames) * 5.0
        a["temp_K"] = 298.0 + rng.normal(0, 0.3, n_frames)
        a["ke"] = 900.0 + rng.normal(0, 5, n_frames)
        a["pe"] = -3500.0 + rng.normal(0, 20, n_frames)
        a["e_total"] = a["ke"] + a["pe"]
        a["volume_ang3"] = box_L ** 3 + rng.normal(0, 30, n_frames)
        a["density_gcc"] = 1.10 + rng.normal(0, 0.005, n_frames)
        a["pe_rigg"] = rng.normal(2.0, 0.1, n_frames)
        a["pe_ewald"] = rng.normal(-25.0, 0.5, n_frames)
        a["pe_pppm"] = rng.normal(-1.0, 0.1, n_frames)
        if ensemble == "npt":
            a["pressure_atm"] = 1.0 + rng.normal(0, 40, n_frames)
        np.save(path, a)
        return path
    return _make


@pytest.fixture
def make_water_pdb():
    def _make(path, *, n_mol=27, box_L=12.0):
        from mdforge.formats.pdb import to_pdb_string
        coords = []
        elements = []
        res_seq = []
        for m in range(n_mol):
            coords.extend((np.array([m % 3, (m // 3) % 3, 0.0]) + WATER_TEMPLATE).tolist())
            elements.extend(WATER_ELEMENTS)
            res_seq.extend([m + 1] * 3)
        box = np.array([box_L, box_L, box_L, 90.0, 90.0, 90.0])
        text = to_pdb_string(np.array(coords), elements, res_name="HOH",
                             res_seq=res_seq, box=box)
        path.write_text(text)
        return path
    return _make
