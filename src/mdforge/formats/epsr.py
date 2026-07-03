"""mdforge.formats.epsr — readers for EPSR experimental RDF output files.

EPSR (Empirical Potential Structure Refinement) writes radial distribution
functions fit to neutron/X-ray scattering data. Two file shapes appear here:

- ``traj.rdf11``     a plain centre-centre g(r): two ``#`` comment header lines,
  then three columns ``r  g(r)  cumulative_N(r)``.
- ``traj.ardf11zz``  an angular RDF g(r, theta): one r-block per angle bin,
  each preceded by an inline ``# Range lo hi`` marker giving the angle range
  (deg). Columns are ``r, g_raw, ...``.

Pure parse half (file ⟂ compute): these read text from a path and return plain
numpy arrays; the orientational normalisation and any g→1 baselining are left to
the caller (e.g. :func:`mdforge.liquid.angular_rdf`).
"""

from __future__ import annotations

import numpy as np


def read_epsr_rdf(path):
    """Parse an EPSR ``traj.rdf11`` centre-centre g(r) file.

    The file has two leading comment lines starting with ``#``, then three
    whitespace-separated columns: ``r``, ``g(r)`` and the running coordination
    number ``N(r)``. Returns ``(r, g, N)`` as three ``(n_r,)`` numpy arrays.
    """
    r, g, n = [], [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            r.append(float(cols[0]))
            g.append(float(cols[1]))
            n.append(float(cols[2]))
    return (np.array(r, dtype=float),
            np.array(g, dtype=float),
            np.array(n, dtype=float))


def read_epsr_angular_rdf(path):
    """Parse an EPSR ``traj.ardf11zz`` angular-RDF file into per-angle g(r).

    The file stacks one r-block per angle bin, each preceded by an inline
    ``# Range lo hi`` marker giving the angle range (deg). Columns are
    ``r, g_raw, ...``. Returns ``(r, theta_edges, g_raw)`` where ``g_raw`` is
    ``(n_bins, n_r)`` (the file's first value column) and ``theta_edges`` are the
    bin edges in degrees. Empty trailing blocks (all-zero) are dropped.

    The caller decides normalisation. For a unit-baseline comparison with
    :func:`mdforge.liquid.angular_rdf`, divide each row by its large-r mean.
    """
    blocks, cur, rng = [], None, None
    with open(path) as fh:
        for line in fh:
            if "Range" in line:
                head, _, tail = line.partition("#")
                lo, hi = (float(v) for v in tail.split()[1:3])
                if cur is not None:
                    blocks.append((rng, np.array(cur, dtype=float)))
                cur, rng = [], (lo, hi)
                if head.split():
                    cur.append([float(v) for v in head.split()])
            elif line.startswith("#"):
                continue
            else:
                s = line.split()
                if s:
                    cur.append([float(v) for v in s])
    if cur is not None:
        blocks.append((rng, np.array(cur, dtype=float)))

    # drop empty/degenerate blocks (e.g. the trailing 180-190 padding)
    good = [(rng, arr) for rng, arr in blocks if arr.size and arr[:, 1].any()]
    r = good[0][1][:, 0]
    los = [rng[0] for rng, _ in good]
    edges = np.array(los + [good[-1][0][1]], dtype=float)
    g_raw = np.array([arr[:, 1] for _, arr in good], dtype=float)
    return r, edges, g_raw


__all__ = [
    "read_epsr_rdf",
    "read_epsr_angular_rdf",
]
