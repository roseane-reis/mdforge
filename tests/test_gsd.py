"""Tests for HOOMD rigid-body GSD reading, atom reconstruction, and the
structural (RDF) + npy-adapter kernels."""

from __future__ import annotations

import numpy as np
import pytest

from mdforge.formats.gsd import (
    RigidTrajectory,
    quaternion_to_rotation_matrix,
    read_rigid_bodies,
    reconstruct_atoms,
    species_atom_index,
)
from mdforge.liquid import (
    angular_rdf,
    coordination_number,
    msd,
    normals_from_orientations,
    plane_normal_from_points,
    rdf,
    self_diffusion,
    unwrap_com,
)
from mdforge.liquid.parse import trajectory_from_hoomd_npy


# --------------------------------------------------------------------------
# quaternion → rotation matrix
# --------------------------------------------------------------------------

def test_quaternion_identity():
    R = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3))


def test_quaternion_90deg_about_z():
    # 90° about z: [w,x,y,z] = [cos45, 0, 0, sin45]; x->y, y->-x
    s = np.sqrt(0.5)
    R = quaternion_to_rotation_matrix(np.array([s, 0.0, 0.0, s]))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    assert np.allclose(R @ np.array([0, 1.0, 0]), [-1, 0, 0], atol=1e-12)


def test_quaternion_batched_shape():
    q = np.tile([1.0, 0, 0, 0], (5, 1))
    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (5, 3, 3)


# --------------------------------------------------------------------------
# atom reconstruction
# --------------------------------------------------------------------------

def _one_molecule_traj(com, quat, box=(100.0, 100.0, 100.0, 0, 0, 0)):
    return RigidTrajectory(
        com=np.array([[com]], dtype=float),
        orientation=np.array([[quat]], dtype=float),
        box=np.array([box], dtype=float),
        species=["X"],
        types=["X"],
    )


def test_reconstruct_identity_at_origin():
    geom = {"X": np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])}
    rbt = _one_molecule_traj([0.0, 0, 0], [1.0, 0, 0, 0])
    atoms = reconstruct_atoms(rbt, geom, wrap_molecules=False)
    assert atoms.shape == (1, 3, 3)
    assert np.allclose(atoms[0], geom["X"])


def test_reconstruct_translation_and_rotation():
    geom = {"X": np.array([[1.0, 0, 0]])}
    s = np.sqrt(0.5)
    rbt = _one_molecule_traj([10.0, 5.0, 0.0], [s, 0, 0, s])  # 90° about z
    atoms = reconstruct_atoms(rbt, geom, wrap_molecules=False)
    # (1,0,0) rotated 90° about z -> (0,1,0), then + COM
    assert np.allclose(atoms[0, 0], [10.0, 6.0, 0.0], atol=1e-12)


def test_reconstruct_wraps_whole_molecule():
    # COM outside the primary box; the molecule must stay intact (atoms follow COM).
    geom = {"X": np.array([[0.5, 0, 0], [-0.5, 0, 0]])}
    rbt = _one_molecule_traj([60.0, 0, 0], [1.0, 0, 0, 0], box=(100, 100, 100, 0, 0, 0))
    atoms = reconstruct_atoms(rbt, geom, wrap_molecules=True)
    # COM 60 wraps to -40; atoms at -39.5 and -40.5; bond length preserved.
    bond = np.linalg.norm(atoms[0, 0] - atoms[0, 1])
    assert np.isclose(bond, 1.0)
    assert np.isclose(atoms[0].mean(axis=0)[0], -40.0)


def test_reconstruct_missing_species_raises():
    rbt = _one_molecule_traj([0.0, 0, 0], [1.0, 0, 0, 0])
    with pytest.raises(KeyError):
        reconstruct_atoms(rbt, {"Y": np.zeros((1, 3))})


def test_species_atom_index_selects_carbons():
    rbt = RigidTrajectory(
        com=np.zeros((1, 2, 3)), orientation=np.zeros((1, 2, 4)),
        box=np.array([[10, 10, 10, 0, 0, 0]], dtype=float),
        species=["B", "B"], types=["B"],
    )
    geom = {"B": np.zeros((4, 3))}  # 4 atoms; "carbons" = local 0,1
    idx = species_atom_index(rbt, geom, "B", slice(0, 2))
    assert list(idx) == [0, 1, 4, 5]  # molecule 0 -> 0,1 ; molecule 1 -> 4,5


# --------------------------------------------------------------------------
# RDF + coordination number
# --------------------------------------------------------------------------

def test_rdf_uniform_gas_is_unity():
    rng = np.random.default_rng(0)
    L = 30.0
    n_frames, n_part = 40, 400
    pos = rng.uniform(0, L, size=(n_frames, n_part, 3))
    box = np.array([L, L, L, 0, 0, 0], dtype=float)
    r, g = rdf(pos, box, r_max=12.0, n_bins=60)
    # away from r=0 the ideal gas g(r) should hover near 1
    mid = (r > 4) & (r < 12)
    assert abs(g[mid].mean() - 1.0) < 0.05


def test_rdf_rejects_tilted_box():
    pos = np.zeros((1, 3, 3))
    with pytest.raises(ValueError):
        rdf(pos, np.array([10, 10, 10, 0.5, 0, 0], dtype=float))


def test_rdf_rejects_r_max_beyond_half_box():
    pos = np.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="minimum-image"):
        rdf(pos, np.array([20.0, 20.0, 20.0]), r_max=12.0)   # > L/2 = 10


def test_coordination_number_flat_gr():
    # g(r)=1 everywhere -> n(r) = (4/3) pi r^3 rho
    r = np.linspace(0, 10, 500)
    g = np.ones_like(r)
    rho = 0.01
    n = coordination_number(r, g, rho, r_cut=8.0)
    expected = (4.0 / 3.0) * np.pi * 8.0**3 * rho
    assert np.isclose(n, expected, rtol=5e-3)  # trapezoid discretization


# --------------------------------------------------------------------------
# self-diffusion: unwrap_com, msd, self_diffusion
# --------------------------------------------------------------------------

def test_unwrap_com_crosses_boundary_continuous():
    # One particle taking a constant step of +3 Å each frame in a box of L=10.
    # Wrapped positions jump at the wall; the unwrapped path must be linear.
    L = 10.0
    step = 3.0
    T = 8
    wrapped = np.array([[[ (i * step) % L, 0.0, 0.0]] for i in range(T)])
    box = np.array([L, L, L, 0, 0, 0], dtype=float)
    out = unwrap_com(wrapped, box)
    expected_x = wrapped[0, 0, 0] + step * np.arange(T)
    assert np.allclose(out[:, 0, 0], expected_x)
    # successive steps are exactly constant (continuous, no wall jumps)
    diffs = np.diff(out[:, 0, 0])
    assert np.allclose(diffs, step)


def test_self_diffusion_recovers_linear_slope():
    # Synthetic MSD that is exactly 6*D*t -> recovered D_ang2_ps == D.
    D = 0.37
    dt = 0.5
    n = 100
    t = np.arange(n) * dt
    msd_curve = 6.0 * D * t
    out = self_diffusion(msd_curve, dt)
    assert out["D_ang2_ps"] == pytest.approx(D, rel=1e-9)
    assert out["D_cm2_s"] == pytest.approx(D * 1e-4, rel=1e-9)
    assert out["slope"] == pytest.approx(6.0 * D, rel=1e-9)


def test_msd_ballistic_constant_velocity():
    # Constant-velocity (ballistic) trajectory: MSD(lag) = (v * lag * dt)^2.
    # Here positions step by a fixed displacement per frame, so unwrapped == input.
    v = np.array([0.2, -0.1, 0.05])   # Å per frame
    T = 20
    M = 3
    base = v[None, :] * np.arange(T)[:, None]        # (T, 3)
    unwrapped = np.broadcast_to(base[:, None, :], (T, M, 3)).copy()
    out = msd(unwrapped)
    lags = np.arange(T)
    expected = (np.dot(v, v)) * lags**2              # |v*lag|^2
    assert np.allclose(out, expected)


# --------------------------------------------------------------------------
# angular RDF: plane_normal_from_points, normals_from_orientations, angular_rdf
# --------------------------------------------------------------------------

def test_plane_normal_from_points_hexagon():
    # Regular hexagon in the z=2 plane -> normal is ±z.
    ang = np.deg2rad(np.arange(6) * 60.0)
    pts = np.column_stack([np.cos(ang), np.sin(ang), np.full(6, 2.0)])
    n = plane_normal_from_points(pts)
    assert np.allclose(np.abs(n), [0.0, 0.0, 1.0], atol=1e-12)


def test_normals_from_orientations_identity_quat():
    # Identity quaternion -> lab normal equals the (normalised) body normal.
    body = np.array([0.0, 0.0, 1.0])
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 3, 1))   # (T=2, M=3, 4)
    out = normals_from_orientations(quat, body)
    assert out.shape == (2, 3, 3)
    assert np.allclose(out, np.broadcast_to(body, (2, 3, 3)))


def test_angular_rdf_two_molecules_known_bin():
    # Two molecules 5 Å apart on x; normals at 90° relative angle.
    # The single pair must land in the r-bin containing 5.0 and the theta-bin
    # containing 90°.
    L = 40.0
    com = np.array([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])          # (1, 2, 3)
    normals = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])      # 90° apart
    box = np.array([L, L, L, 0, 0, 0], dtype=float)
    r, theta_edges, g = angular_rdf(com, normals, box, r_max=10.0,
                                    n_r_bins=10, n_theta_bins=18)
    # locate (theta, r) cell of the one pair
    ti, ri = np.unravel_index(np.argmax(g), g.shape)
    assert r[ri] == pytest.approx(5.5, abs=0.5)           # bin centre nearest 5
    assert theta_edges[ti] <= 90.0 < theta_edges[ti + 1]  # 80-90 or 90-100 edge


def test_angular_rdf_isotropic_unity_at_large_r():
    # Many randomly placed, randomly oriented molecules -> g ~ 1 at large r.
    rng = np.random.default_rng(0)
    L = 30.0
    n_frames, M = 20, 200
    com = rng.uniform(0, L, size=(n_frames, M, 3))
    raw = rng.standard_normal((n_frames, M, 3))
    normals = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    box = np.array([L, L, L, 0, 0, 0], dtype=float)
    r, theta_edges, g = angular_rdf(com, normals, box, r_max=12.0,
                                    n_r_bins=24, n_theta_bins=9)
    mid = (r > 5) & (r < 12)
    assert abs(g[:, mid].mean() - 1.0) < 0.1


# --------------------------------------------------------------------------
# npy → Trajectory adapter
# --------------------------------------------------------------------------

def test_trajectory_from_hoomd_npy(tmp_path):
    n = 50
    dt = np.dtype([("step", "f8"), ("time_ps", "f8"), ("temp_K", "f8"),
                   ("ke", "f8"), ("pe", "f8"), ("e_total", "f8"),
                   ("volume_ang3", "f8")])
    a = np.zeros(n, dtype=dt)
    a["time_ps"] = np.arange(n) * 5.0
    a["temp_K"] = 298.0 + np.random.default_rng(1).normal(0, 1, n)
    a["ke"] = 800.0
    a["pe"] = -3000.0
    a["e_total"] = a["ke"] + a["pe"]
    a["volume_ang3"] = 64000.0
    p = tmp_path / "npt.npy"
    np.save(p, a)

    tr = trajectory_from_hoomd_npy(p, n_molecules=450, molar_mass_g_mol=78.1118)
    assert tr.n_frames == n
    assert tr.n_molecules == 450
    assert np.isclose(tr.total_mass, 450 * 78.1118)
    assert np.isclose(tr.dt_ps, 5.0)
    assert 296 < tr.temperature_K < 300
    # enthalpy resolves to PE + KE (== e_total), NOT instantaneous P*V
    assert np.allclose(tr.enthalpy, tr.total_energy)


def test_trajectory_from_hoomd_npy_total_mass_overrides(tmp_path):
    dt = np.dtype([("pe", "f8"), ("ke", "f8"), ("volume_ang3", "f8")])
    a = np.zeros(3, dtype=dt)
    a["volume_ang3"] = 1000.0
    p = tmp_path / "nvt.npy"
    np.save(p, a)
    tr = trajectory_from_hoomd_npy(p, n_molecules=10, molar_mass_g_mol=18.0,
                                   total_mass_amu=123.0, temperature_K=300.0)
    assert np.isclose(tr.total_mass, 123.0)
    assert tr.temperature_K == 300.0


# --------------------------------------------------------------------------
# GSD round-trip (needs the optional `gsd` package)
# --------------------------------------------------------------------------

def test_read_rigid_bodies_roundtrip(tmp_path):
    gsd_hoomd = pytest.importorskip("gsd.hoomd")
    # Two molecules of species "A": each a center + 2 ghost constituents.
    frame = gsd_hoomd.Frame()
    frame.particles.N = 6
    frame.particles.types = ["A", "A_c"]
    frame.particles.typeid = [0, 1, 1, 0, 1, 1]
    frame.particles.position = np.array(
        [[1.0, 0, 0], [1.5, 0, 0], [0.5, 0, 0],
         [-2.0, 0, 0], [-1.5, 0, 0], [-2.5, 0, 0]], dtype=float)
    frame.particles.orientation = np.tile([1.0, 0, 0, 0], (6, 1))
    frame.particles.body = [0, 0, 0, 3, 3, 3]
    frame.configuration.box = [20, 20, 20, 0, 0, 0]
    path = tmp_path / "traj.gsd"
    with gsd_hoomd.open(str(path), "w") as t:
        t.append(frame)

    rbt = read_rigid_bodies(path)
    assert rbt.n_frames == 1
    assert rbt.n_molecules == 2
    assert rbt.species == ["A", "A"]
    assert np.allclose(rbt.com[0], [[1.0, 0, 0], [-2.0, 0, 0]])
    assert np.isclose(rbt.volume[0], 20**3)
