"""Structural liquid kernels: the radial distribution function g(r).

Arrays in, structure out — no file access (parse ⟂ compute). Positions and a
per-frame box come from a parser (e.g. :func:`mdforge.formats.gsd.read_rigid_bodies`
+ :func:`~mdforge.formats.gsd.reconstruct_atoms`, or any other source); these
kernels only see numpy arrays.

The minimum-image convention assumes an orthorhombic cell (the HOOMD tilt
factors ``xy/xz/yz`` are ignored — a ``ValueError`` is raised if they are
non-negligible). Variable-box (NPT) trajectories are handled correctly: each
frame is normalised by its own cell volume before averaging.

Units: length in Angstrom; ``g(r)`` is dimensionless.
"""

from __future__ import annotations

import numpy as np


def _box_lengths(box, n_frames: int) -> np.ndarray:
    """Normalise a box spec to per-frame ``(T, 3)`` edge lengths (orthorhombic)."""
    b = np.asarray(box, dtype=float)
    if b.ndim == 1:
        b = np.broadcast_to(b, (n_frames, b.shape[0])).copy()
    if b.shape[1] >= 6:
        tilt = b[:, 3:6]
        if np.any(np.abs(tilt) > 1e-6):
            raise ValueError(
                "rdf() supports orthorhombic cells only; the trajectory has "
                "non-zero tilt factors (xy/xz/yz)."
            )
    return b[:, :3]


def rdf(
    positions,
    box,
    *,
    positions_b=None,
    mol_a=None,
    mol_b=None,
    r_max: float | None = None,
    n_bins: int = 200,
    frames=None,
):
    """Radial distribution function g(r) from a position trajectory.

    Parameters
    ----------
    positions:
        ``(T, N, 3)`` atom/COM positions in Angstrom (one group). For a single
        frame, pass ``(1, N, 3)``.
    box:
        Per-frame box: ``(T, 3)``/``(T, 6)`` HOOMD ``[Lx,Ly,Lz,(xy,xz,yz)]`` or a
        single ``(3,)``/``(6,)`` broadcast to all frames. Edge lengths in Å.
    positions_b:
        Optional second group ``(T, Nb, 3)`` for a cross g_AB(r). If omitted, the
        self distribution within ``positions`` is computed (pairs ``i < j``).
    mol_a, mol_b:
        Optional per-site molecule ids labelling each site in ``positions``
        (``mol_a``, length N) and ``positions_b`` (``mol_b``, length Nb). When
        given, pairs that share a molecule id are dropped, yielding an
        *inter*-molecular site-site g(r) — needed for partial RDFs such as
        g_OH / g_HH where the intramolecular bonded distances would otherwise
        contaminate the curve. For a self RDF pass only ``mol_a``. The ideal
        normalisation still uses the full group densities, so g → 1 at large r.
    r_max:
        Maximum radius (Å). Defaults to half the smallest box edge seen
        (the minimum-image limit). Values past ``min(L)/2`` are unreliable.
    n_bins:
        Number of radial bins in ``[0, r_max]``.
    frames:
        Optional iterable of frame indices to use (e.g. ``range(equil, T)`` or a
        stride ``slice``); defaults to all frames. Use to drop equilibration or
        subsample expensive atom-atom RDFs.

    Returns
    -------
    (r, g): tuple of ``(n_bins,)`` arrays — bin centres (Å) and g(r).
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    T = pos.shape[0]
    L = _box_lengths(box, T)

    cross = positions_b is not None
    if cross:
        pos_b = np.asarray(positions_b, dtype=float)
        if pos_b.ndim != 3 or pos_b.shape[2] != 3:
            raise ValueError("positions_b must have shape (T, Nb, 3)")

    ma = None if mol_a is None else np.asarray(mol_a)
    mb = None if mol_b is None else np.asarray(mol_b)

    sel = range(T) if frames is None else list(frames)
    if len(sel) == 0:
        raise ValueError("no frames selected")

    if r_max is None:
        r_max = 0.5 * float(L[sel].min())
    edges = np.linspace(0.0, r_max, n_bins + 1)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centers = 0.5 * (edges[1:] + edges[:-1])

    # Intra-molecular exclusion masks (computed once; geometry is fixed per run).
    if cross:
        keep_cross = (
            (ma[:, None] != mb[None, :]).ravel()
            if ma is not None and mb is not None else None
        )
    else:
        i_self, j_self = np.triu_indices(pos.shape[1], k=1)
        keep_self = ma[i_self] != ma[j_self] if ma is not None else None

    g_accum = np.zeros(n_bins, dtype=float)
    used = 0
    for t in sel:
        Lt = L[t]
        a = pos[t]
        V = float(Lt[0] * Lt[1] * Lt[2])
        if cross:
            b = pos_b[t]
            d = a[:, None, :] - b[None, :, :]              # (Na, Nb, 3)
            d -= Lt * np.round(d / Lt)
            r = np.sqrt(np.einsum("ijk,ijk->ij", d, d)).ravel()
            if keep_cross is not None:
                r = r[keep_cross]
            n_pairs = a.shape[0] * b.shape[0]
            norm_dens = n_pairs / V
        else:
            d = a[i_self] - a[j_self]                       # (P, 3)
            d -= Lt * np.round(d / Lt)
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            if keep_self is not None:
                r = r[keep_self]
            n_pairs = a.shape[0] * (a.shape[0] - 1) / 2.0
            norm_dens = n_pairs / V

        hist, _ = np.histogram(r, bins=edges)
        ideal = norm_dens * shell_vol
        with np.errstate(divide="ignore", invalid="ignore"):
            g_accum += np.where(ideal > 0, hist / ideal, 0.0)
        used += 1

    return centers, g_accum / used


def coordination_number(r, g, density: float, r_cut: float) -> float:
    """Running coordination number n(r_cut) = ∫₀^{r_cut} 4π r² ρ g(r) dr.

    ``density`` is the number density of the *target* group (N_b / V for a cross
    RDF, or N / V for a self RDF), in Å⁻³. Integrates the supplied g(r) up to
    ``r_cut`` with the trapezoidal rule.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    mask = r <= r_cut
    integrand = 4.0 * np.pi * r[mask] ** 2 * density * g[mask]
    return float(np.trapezoid(integrand, r[mask]))


def angular_rdf(
    com,
    normals,
    box,
    *,
    r_max: float | None = None,
    n_r_bins: int = 200,
    n_theta_bins: int = 18,
    frames=None,
):
    """Orientation-resolved RDF g(r, theta) from COM positions and plane normals.

    theta is the angle between the two molecules' plane normals (0-180 deg). This
    resolves the ordinary centre-of-mass g(r) by relative orientation, so you can
    see whether close neighbours prefer parallel (face-to-face, theta~0 or 180) or
    perpendicular (edge-to-face / "T-shaped", theta~90) arrangements.

    Normalisation follows Headen, Mol. Phys. 117, 3329 (2019), Eq. 5::

        g(r, theta) = n(r, theta) / [ rho * V_shell(r) * P(theta) ]

    with the orientational prior  P(theta) = (1/2) (cos theta1 - cos theta2)  for
    a bin [theta1, theta2] (this integrates to 1 over [0, pi] and is the fraction
    of solid angle at that relative orientation for randomly oriented planes).
    With this prior, g -> 1 everywhere at large r for an isotropic liquid, and
    departures from 1 at short r are genuine orientational correlations.

    Parameters
    ----------
    com:
        ``(T, M, 3)`` centre-of-mass positions (Angstrom).
    normals:
        ``(T, M, 3)`` unit plane-normal vectors (one per molecule per frame).
        Need not be sign-consistent; the angle is taken on [0, 180] deg.
    box:
        Per-frame ``(T, 3)``/``(T, 6)`` or single ``(3,)``/``(6,)`` edge lengths
        (Angstrom). Variable-box (NPT) trajectories are handled per frame.
    r_max:
        Max radius (Angstrom); defaults to half the smallest box edge.
    n_r_bins, n_theta_bins:
        Radial and angular bin counts. 18 theta bins reproduces the 10-deg bins
        of the experimental EPSR angular RDF.
    frames:
        Optional iterable of frame indices (e.g. ``range(equil, T)``).

    Returns
    -------
    (r_centers, theta_edges, g):
        ``r_centers`` ``(n_r_bins,)`` Angstrom; ``theta_edges`` ``(n_theta_bins+1,)``
        degrees; ``g`` ``(n_theta_bins, n_r_bins)`` with g -> 1 at large r.
    """
    com = np.asarray(com, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if com.ndim != 3 or com.shape[2] != 3:
        raise ValueError("com must be (T, M, 3)")
    if nrm.shape != com.shape:
        raise ValueError("normals must have the same shape as com")
    T, M, _ = com.shape
    L = _box_lengths(box, T)

    sel = list(range(T)) if frames is None else list(frames)
    if not sel:
        raise ValueError("no frames selected")
    if r_max is None:
        r_max = 0.5 * float(L[sel].min())

    r_edges = np.linspace(0.0, r_max, n_r_bins + 1)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    shell_vol = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    theta_edges = np.linspace(0.0, 180.0, n_theta_bins + 1)
    # orientational prior per bin: P = 1/2 (cos t1 - cos t2), sums to 1 over [0,pi]
    ct = np.cos(np.deg2rad(theta_edges))
    p_theta = 0.5 * (ct[:-1] - ct[1:])                      # (n_theta_bins,)

    iu, ju = np.triu_indices(M, k=1)
    g_accum = np.zeros((n_theta_bins, n_r_bins), dtype=float)
    used = 0
    for t in sel:
        Lt = L[t]
        a = com[t]
        nt = nrm[t]
        V = float(Lt[0] * Lt[1] * Lt[2])

        d = a[iu] - a[ju]
        d -= Lt * np.round(d / Lt)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))

        cos = np.einsum("ij,ij->i", nt[iu], nt[ju])
        cos = np.clip(cos, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos))

        hist, _, _ = np.histogram2d(theta, r, bins=[theta_edges, r_edges])
        n_pairs = M * (M - 1) / 2.0
        # ideal pair count per (theta, r) cell for an isotropic system
        ideal = (n_pairs / V) * shell_vol[None, :] * p_theta[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            g_accum += np.where(ideal > 0, hist / ideal, 0.0)
        used += 1

    return r_centers, theta_edges, g_accum / used


def normals_from_orientations(orientation, body_normal):
    """Lab-frame plane normals from rigid-body quaternions and a body normal.

    ``orientation`` is ``(T, M, 4)`` ``[w, x, y, z]``; ``body_normal`` is the
    fixed plane normal in the molecule body frame (3,). Returns ``(T, M, 3)``
    unit normals. Uses the same rotation convention as the GSD reader
    (``p_lab = R(quat) @ p_body``).
    """
    q = np.asarray(orientation, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    n = np.asarray(body_normal, dtype=float)
    n = n / np.linalg.norm(n)
    out = np.einsum("...ij,j->...i", R, n)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def plane_normal_from_points(points):
    """Best-fit plane normal of coplanar points (e.g. the 6 benzene carbons)."""
    p = np.asarray(points, dtype=float)
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p)
    return vt[-1]


def _wrap_into_box(p, Lt):
    """Wrap positions into the half-open cell ``[0, L)`` for a periodic KD-tree."""
    w = np.mod(p, Lt)
    # guard against a coordinate landing exactly on the upper edge (fp round-up),
    # which scipy's periodic cKDTree rejects.
    return np.minimum(w, Lt * (1.0 - 1e-12))


def tetrahedral_order(positions, box, *, frames=None):
    """Errington–Debenedetti tetrahedral order parameter q.

    For each site, ``q = 1 − (3/8) Σ_{j<k} (cos ψ_jk + 1/3)²`` over its four
    nearest neighbours, where ψ_jk is the angle subtended at the site by
    neighbours j and k. ``q = 1`` for a perfect tetrahedron (ideal ice), ``q = 0``
    on average for an ideal gas; ambient liquid water is ≈ 0.5–0.6. Pass the
    ordering sites (e.g. the water oxygens) — one point per molecule.

    positions:
        ``(T, N, 3)`` site positions (Å); needs N ≥ 5.
    box:
        Per-frame ``(T, 3)``/``(T, 6)`` or single ``(3,)``/``(6,)`` orthorhombic
        edge lengths (Å); variable-box (NPT) trajectories handled per frame.
    frames:
        Optional iterable of frame indices (e.g. ``range(equil, T)``).

    Returns
    -------
    (q, q_mean):
        ``q`` is the concatenated per-site values over the selected frames (1-D);
        ``q_mean`` is their mean.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    if pos.shape[1] < 5:
        raise ValueError("tetrahedral_order needs at least 5 sites per frame")
    T = pos.shape[0]
    L = _box_lengths(box, T)
    sel = range(T) if frames is None else list(frames)
    if len(sel) == 0:
        raise ValueError("no frames selected")

    vals = []
    for t in sel:
        Lt = L[t]
        p = _wrap_into_box(pos[t], Lt)
        tree = cKDTree(p, boxsize=Lt)
        _, idx = tree.query(p, k=5)                     # self + four nearest
        v = p[idx[:, 1:5]] - p[:, None, :]              # (N, 4, 3)
        v -= Lt * np.round(v / Lt)                      # minimum image
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        s = np.zeros(p.shape[0])
        for j in range(3):
            for k in range(j + 1, 4):
                c = np.einsum("ij,ij->i", v[:, j], v[:, k])
                s += (c + 1.0 / 3.0) ** 2
        vals.append(1.0 - (3.0 / 8.0) * s)
    q = np.concatenate(vals)
    return q, float(q.mean())


def hydrogen_bonds(oxygens, hydrogens, box, *, frames=None,
                   r_oo: float = 3.5, angle_deg: float = 30.0):
    """Average number of hydrogen bonds per molecule (Luzar–Chandler geometry).

    A donor–acceptor pair (O_d, O_a) is counted as hydrogen bonded when the O–O
    separation is below ``r_oo`` (Å) and, for one of the donor's hydrogens H, the
    angle ∠(O_a–O_d–H) is below ``angle_deg``. Each oxygen owns two hydrogens,
    taken as ``hydrogens[2 m : 2 m + 2]`` for oxygen ``m`` (the contiguous layout
    produced by :func:`mdforge.formats.gsd.reconstruct_atoms` for 3-site water).

    oxygens:
        ``(T, No, 3)`` oxygen positions (Å).
    hydrogens:
        ``(T, 2·No, 3)`` hydrogen positions (Å), grouped two-per-oxygen.
    box:
        Per-frame ``(T, 3)``/``(T, 6)`` or single ``(3,)``/``(6,)`` orthorhombic
        edge lengths (Å).
    frames:
        Optional iterable of frame indices.
    r_oo, angle_deg:
        Geometric cut-offs (defaults 3.5 Å, 30°).

    Returns
    -------
    (hb_per_molecule, info):
        mean hydrogen bonds per molecule, and a dict with the per-frame series.
    """
    from scipy.spatial import cKDTree

    O = np.asarray(oxygens, dtype=float)
    H = np.asarray(hydrogens, dtype=float)
    if O.ndim != 3 or O.shape[2] != 3:
        raise ValueError("oxygens must have shape (T, No, 3)")
    T, No, _ = O.shape
    if H.shape[0] != T or H.shape[1] != 2 * No:
        raise ValueError("hydrogens must have shape (T, 2*No, 3)")
    L = _box_lengths(box, T)
    sel = range(T) if frames is None else list(frames)
    if len(sel) == 0:
        raise ValueError("no frames selected")
    cos_cut = np.cos(np.deg2rad(angle_deg))

    per_frame = []
    for t in sel:
        Lt = L[t]
        Ot, Ht = O[t], H[t]
        p = _wrap_into_box(Ot, Lt)
        tree = cKDTree(p, boxsize=Lt)
        pairs = tree.query_pairs(r_oo, output_type="ndarray")
        count = 0
        for d, a in pairs[:, ::-1].tolist() + pairs.tolist() if len(pairs) else []:
            oo = Ot[a] - Ot[d]
            oo -= Lt * np.round(oo / Lt)
            oo /= np.linalg.norm(oo)
            for h in (Ht[2 * d], Ht[2 * d + 1]):
                oh = h - Ot[d]
                oh -= Lt * np.round(oh / Lt)
                oh /= np.linalg.norm(oh)
                if float(oh @ oo) >= cos_cut:
                    count += 1
                    break                               # one bond per donor pair
        per_frame.append(count / No)
    per_frame = np.asarray(per_frame, dtype=float)
    return float(per_frame.mean()), {
        "per_frame": per_frame.tolist(), "r_oo": r_oo, "angle_deg": angle_deg,
    }


__all__ = [
    "rdf",
    "coordination_number",
    "angular_rdf",
    "normals_from_orientations",
    "plane_normal_from_points",
    "tetrahedral_order",
    "hydrogen_bonds",
]
