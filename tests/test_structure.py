"""Tests for the structural liquid kernels (RDF exclusion, tetrahedral, H-bonds)."""

import numpy as np
import pytest

from mdforge.liquid.structure import rdf, tetrahedral_order, hydrogen_bonds


def _grid(n_side, spacing):
    """Return the (n_side**3, 3) simple-cubic lattice points at the given spacing."""
    c = np.arange(n_side) * spacing
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def test_rdf_intramolecular_exclusion():
    # Each "molecule" = one O on a wide lattice + two H at +-d0 along x. The only
    # O-H pairs near d0 are intramolecular, so excluding same-molecule pairs must
    # empty the short-range bins while the no-exclusion curve keeps them.
    L = 20.0
    spacing = 10.0
    d0 = 1.0
    O = _grid(2, spacing) + 2.5            # 8 oxygens, away from the cell edge
    M = O.shape[0]
    H = np.empty((2 * M, 3))
    H[0::2] = O + np.array([d0, 0, 0])
    H[1::2] = O + np.array([-d0, 0, 0])

    box = np.array([L, L, L])
    pa = O[None]                            # (1, M, 3)
    pb = H[None]                            # (1, 2M, 3)
    mol_a = np.arange(M)
    mol_b = np.repeat(np.arange(M), 2)

    r, g_full = rdf(pa, box, positions_b=pb, r_max=8.0, n_bins=80)
    _, g_excl = rdf(pa, box, positions_b=pb, mol_a=mol_a, mol_b=mol_b,
                    r_max=8.0, n_bins=80)

    near = r < 3.0
    assert g_full[near].sum() > 0.0        # intramolecular peak present...
    assert g_excl[near].sum() == pytest.approx(0.0, abs=1e-9)  # ...and removed


def test_tetrahedral_order_perfect():
    # A site with four neighbours in perfect tetrahedral directions scores q = 1.
    b = 1.0
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    center = np.array([25.0, 25.0, 25.0])
    pts = [center] + [center + b * d for d in dirs]
    # filler sites, far enough that they never enter the central site's 4-NN
    pts += [[5, 5, 5], [45, 45, 45], [5, 45, 25]]
    pos = np.array(pts)[None]               # (1, N, 3)
    q, q_mean = tetrahedral_order(pos, np.array([50.0, 50.0, 50.0]))
    assert q[0] == pytest.approx(1.0, abs=1e-6)   # the perfectly tetrahedral site
    assert np.all(q <= 1.0 + 1e-9)                # q is bounded above by 1 (can be < 0)


def test_hydrogen_bonds_single_dimer():
    # One donor->acceptor bond: donor H points along the O-O axis (angle ~0),
    # the acceptor's hydrogens point away, so exactly one H-bond exists -> 0.5/mol.
    Od = np.array([10.0, 10.0, 10.0])
    Oa = np.array([10.0, 10.0, 12.8])       # O-O = 2.8 A < 3.5
    O = np.array([Od, Oa])[None]            # (1, 2, 3)
    H = np.array([
        Od + [0.0, 0.0, 0.95],              # donor H toward acceptor -> bonded
        Od + [0.9, 0.0, -0.3],              # donor H away
        Oa + [0.0, 0.9, 0.3],               # acceptor H away from donor
        Oa + [0.0, -0.9, 0.3],
    ])[None]                                 # (1, 4, 3)
    box = np.array([50.0, 50.0, 50.0])
    hb, info = hydrogen_bonds(O, H, box)
    assert hb == pytest.approx(0.5, abs=1e-9)
    assert info["r_oo"] == 3.5


def test_hydrogen_bonds_angle_rejects():
    # Same geometry but both donor hydrogens point sideways -> no bond.
    Od = np.array([10.0, 10.0, 10.0])
    Oa = np.array([10.0, 10.0, 12.8])
    O = np.array([Od, Oa])[None]
    H = np.array([
        Od + [0.95, 0.0, 0.0],
        Od + [-0.95, 0.0, 0.0],
        Oa + [0.95, 0.0, 0.0],
        Oa + [-0.95, 0.0, 0.0],
    ])[None]
    box = np.array([50.0, 50.0, 50.0])
    hb, _ = hydrogen_bonds(O, H, box)
    assert hb == pytest.approx(0.0, abs=1e-9)
