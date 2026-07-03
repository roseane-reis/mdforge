"""Force-field parameter ↔ flat-vector machinery (goal d).

Ported from ``analyzetool.auxfitting.Auxfit`` ``build_prm_list`` /
``prmlist_to_dict``: flatten a chosen subset of HIPPO parameter terms into the
flat vector a scipy optimizer drives, and write the vector back into a ``prmdict``
(from :func:`mdforge.formats.prm.process_prm`). The prm read/write itself lives
in :mod:`mdforge.formats.prm`.

Supported fit terms: ``chgpen, dispersion, repulsion, polarize, chgtrn,
bond-force, bond-value, angle-force, angle-value``. (Multipole fitting — which
needs the charge-neutrality rule machinery — is deferred.)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

_SUPPORTED = {
    "chgpen", "dispersion", "repulsion", "polarize", "chgtrn",
    "bond-force", "bond-value", "angle-force", "angle-value",
}


@dataclass
class ParameterSpace:
    """Maps a HIPPO ``prmdict`` subset to/from a flat optimization vector.

    Parameters
    ----------
    prmdict:
        The parameter dict from :func:`mdforge.formats.prm.process_prm`.
    termfit:
        Terms to expose as fit variables (see module docstring).
    """

    prmdict: dict
    termfit: list[str]
    _vector: np.ndarray = field(default_factory=lambda: np.array([]), init=False, repr=False)
    _index: dict = field(default_factory=dict, init=False, repr=False)
    _init_chgtrn: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        bad = set(self.termfit) - _SUPPORTED
        if bad:
            raise ValueError(f"Unsupported fit term(s): {sorted(bad)}; supported: {sorted(_SUPPORTED)}")
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        p = self.prmdict
        vec: list[float] = []
        index: dict[str, tuple[int, int]] = {}
        c = 0

        def take(values) -> None:
            nonlocal c
            values = list(values)
            vec.extend(values)
            index[term] = (c, c + len(values))
            c += len(values)

        for term in self.termfit:
            if term == "chgpen":
                take(p["chgpen"][:, 1])
            elif term == "dispersion":
                take(np.atleast_1d(p["dispersion"]))
            elif term == "repulsion":
                take(np.asarray(p["repulsion"]).ravel())
            elif term == "polarize":
                take(p["polarize"][0])
            elif term == "chgtrn":
                self._init_chgtrn = np.asarray(p["chgtrn"]).copy()
                take([v for line in p["chgtrn"] for v in line if v != 0])
            elif term == "bond-force":
                take(p["bond"][1])
            elif term == "bond-value":
                take(p["bond"][2])
            elif term == "angle-force":
                take(p["angle"][1])
            elif term == "angle-value":
                take(p["angle"][2])

        self._vector = np.asarray(vec, dtype=float)
        self._index = index

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._vector.size

    def to_vector(self) -> np.ndarray:
        """Return the flat vector of the current (initial) parameter values."""
        return self._vector.copy()

    def from_vector(self, x) -> dict:
        """Return a new ``prmdict`` with the fit terms replaced by ``x``."""
        x = np.asarray(x, dtype=float)
        out = copy.deepcopy(self.prmdict)
        n_types = len(out["types"])
        for term in self.termfit:
            lo, hi = self._index[term]
            prm = x[lo:hi]
            if term == "chgpen":
                out["chgpen"][:, 1] = prm
            elif term == "dispersion":
                out["dispersion"] = np.asarray(prm)
            elif term == "repulsion":
                out["repulsion"] = np.asarray(prm).reshape(n_types, 3)
            elif term == "polarize":
                out["polarize"][0] = list(prm)
            elif term == "chgtrn":
                z = 0
                for k, line in enumerate(out["chgtrn"]):
                    for i in range(len(line)):
                        if self._init_chgtrn[k][i] != 0:
                            out["chgtrn"][k][i] = prm[z]
                            z += 1
            elif term == "bond-force":
                out["bond"][1] = list(prm)
            elif term == "bond-value":
                out["bond"][2] = list(prm)
            elif term == "angle-force":
                out["angle"][1] = list(prm)
            elif term == "angle-value":
                out["angle"][2] = list(prm)
        return out

    def bounds(self, *, relative: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        """Simple per-parameter box bounds around the initial values.

        ``lower = v - |v|*relative``, ``upper = v + |v|*relative`` (with a small
        floor so zero-valued parameters get a finite window). Term-specific
        physical bounds (the legacy ``wide_range`` table) can be layered on by
        the caller; this is the safe default for local refinement.
        """
        v = self._vector
        span = np.abs(v) * relative
        span = np.where(span < 1e-6, 1e-3, span)
        return v - span, v + span


__all__ = ["ParameterSpace"]
