"""Load the packaged experimental / baseline-model reference data.

Reference data ships as JSON (stdlib — no YAML needed on the scoring path) under
``references/{liquid}_{T}K.json`` and is read via :mod:`importlib.resources`, so
it works from an installed wheel. The experimental partial RDFs (Soper 2013,
revised) ship alongside as ``298_1_g{OO,OH,HH}.txt``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

_PKG = "mdforge.liquid.evaluate.references"


@dataclass(frozen=True)
class Citation:
    key: str
    text: str
    doi: str | None = None
    tables: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PropertyReference:
    """Reference record for one property: experiment + baselines + scoring meta."""

    key: str
    label: str
    unit: str
    computed_unit: str
    rated: bool
    direction: str
    exp_value: float
    exp_uncertainty: float | None
    exp_source: str
    baselines: dict[str, float]
    correction: str | None = None
    note: str | None = None
    unit_aliases: dict = field(default_factory=dict)
    no_baseline_policy: str = "excellent_or_unrated"
    report_only: bool = False   # show the value but assign no verdict / never grade

    def baseline(self, model: str = "tip3p") -> float | None:
        """Baseline value for ``model`` (e.g. the TIP3P bar), or None if absent."""
        return self.baselines.get(model)


@dataclass(frozen=True)
class ReferenceSet:
    liquid: str
    state_point: dict
    properties: dict[str, PropertyReference]
    citations: dict[str, Citation]
    aggregation_defaults: dict
    schema_version: int = 1

    def get(self, key: str) -> PropertyReference:
        try:
            return self.properties[key]
        except KeyError as exc:
            raise KeyError(f"no reference for property {key!r}") from exc

    def rated_keys(self) -> list[str]:
        return [k for k, p in self.properties.items() if p.rated]

    def core_keys(self) -> list[str]:
        core = self.aggregation_defaults.get("core_properties")
        return list(core) if core else self.rated_keys()


def _packaged_path(name: str) -> Path:
    return Path(str(resources.files(_PKG).joinpath(name)))


def load_reference_set(
    liquid: str = "water",
    temperature_K: float = 298.15,
    *,
    path: str | Path | None = None,
) -> ReferenceSet:
    """Load a :class:`ReferenceSet` for ``liquid`` at ``temperature_K``.

    Pass ``path`` to load an out-of-tree JSON file with the same schema.
    """
    if path is not None:
        raw = json.loads(Path(path).read_text())
    else:
        fname = f"{liquid.lower()}_{int(round(temperature_K))}K.json"
        target = resources.files(_PKG).joinpath(fname)
        if not target.is_file():
            avail = ", ".join(f"{lq} @ {T} K" for lq, T in available_reference_sets())
            raise FileNotFoundError(
                f"no packaged reference set for {liquid!r} at {temperature_K} K "
                f"(looked for {fname}). Available: {avail or 'none'}"
            )
        raw = json.loads(target.read_text())

    props: dict[str, PropertyReference] = {}
    for key, p in raw["properties"].items():
        exp = p["experimental"]
        props[key] = PropertyReference(
            key=key,
            label=p["label"],
            unit=p["unit"],
            computed_unit=p.get("computed_unit", p["unit"]),
            rated=bool(p.get("rated", False)),
            direction=p.get("direction", "two_sided"),
            exp_value=float(exp["value"]),
            exp_uncertainty=(None if exp.get("uncertainty") is None
                             else float(exp["uncertainty"])),
            exp_source=exp.get("source", ""),
            baselines={str(m): float(v) for m, v in (p.get("baseline_models") or {}).items()},
            correction=p.get("correction"),
            note=p.get("note"),
            unit_aliases=p.get("unit_aliases", {}) or {},
            no_baseline_policy=p.get("no_baseline_policy", "excellent_or_unrated"),
            report_only=bool(p.get("report_only", False)),
        )

    citations = {
        k: Citation(key=k, text=c.get("text", ""), doi=c.get("doi"),
                    tables=c.get("tables", {}))
        for k, c in (raw.get("citations") or {}).items()
    }

    return ReferenceSet(
        liquid=raw.get("liquid", liquid),
        state_point=raw.get("state_point", {}),
        properties=props,
        citations=citations,
        aggregation_defaults=raw.get("aggregation_defaults", {}),
        schema_version=int(raw.get("schema_version", 1)),
    )


def available_reference_sets() -> list[tuple[str, float]]:
    """Enumerate packaged ``{liquid}_{T}K.json`` reference sets as (liquid, T)."""
    out: list[tuple[str, float]] = []
    for entry in resources.files(_PKG).iterdir():
        name = entry.name
        if name.endswith("K.json") and "_" in name:
            stem = name[:-len("K.json")]
            liquid, _, tstr = stem.rpartition("_")
            try:
                out.append((liquid, float(tstr)))
            except ValueError:
                continue
    return sorted(out)


def _first_peak(r: np.ndarray, g: np.ndarray, after: float) -> tuple[float, float]:
    w = r > after
    idx = int(np.argmax(g[w]) + np.searchsorted(r, after))
    return float(r[idx]), float(g[idx])


def load_experimental_rdf(temperature_K: float = 298.15, pressure_atm: float = 1.0) -> dict:
    """Load the packaged Soper (2013, revised) experimental partial RDFs.

    Returns ``{"gOO"/"gOH"/"gHH": {"r", "g", "peak_r", "peak_g"}}`` (only 298 K /
    1 atm is packaged today). Files are ``Bin no.  r  g(r)  std`` with 4 header
    lines.
    """
    if int(round(temperature_K)) != 298 or int(round(pressure_atm)) != 1:
        raise FileNotFoundError(
            f"only 298 K / 1 atm experimental RDFs are packaged; got "
            f"{temperature_K} K / {pressure_atm} atm"
        )
    out: dict[str, dict] = {}
    for site in ("gOO", "gOH", "gHH"):
        with resources.as_file(resources.files(_PKG).joinpath(f"298_1_{site}.txt")) as p:
            d = np.loadtxt(p, skiprows=4)
        r, g = d[:, 1], d[:, 2]
        # Soper's tables end with a terminator row (r=0, g=0). Keep only the
        # leading block where r increases so a plotted curve doesn't wrap from
        # the last real point back to the origin (spurious diagonal artifact).
        keep = np.ones(len(r), dtype=bool)
        keep[1:] = r[1:] > r[:-1]
        r, g = r[keep], g[keep]
        after = 2.3 if site == "gOO" else 1.5
        peak_r, peak_g = _first_peak(r, g, after)
        out[site] = {"r": r.tolist(), "g": g.tolist(), "peak_r": peak_r, "peak_g": peak_g}
    return out


def load_skinner_rdf(temperature_K: float = 298.15, pressure_atm: float = 1.0) -> dict:
    """Load the packaged Skinner & Benmore (2014) X-ray ``g_OO`` reference.

    A second, independent experimental O-O RDF (high-energy X-ray, APS) plotted
    alongside the neutron Soper (2013) reference. Only the near-ambient 295.1 K
    column is packaged; returns ``{"gOO": {"r", "g", "peak_r", "peak_g"}}``. Same
    ``Bin no.  r  g(r)  std`` 4-header-line layout as the Soper files.
    """
    if int(round(temperature_K)) != 298 or int(round(pressure_atm)) != 1:
        raise FileNotFoundError(
            f"only the ~298 K / 1 atm Skinner g_OO reference is packaged; got "
            f"{temperature_K} K / {pressure_atm} atm"
        )
    with resources.as_file(resources.files(_PKG).joinpath("skinner2014_gOO.txt")) as p:
        d = np.loadtxt(p, skiprows=4)
    r, g = d[:, 1], d[:, 2]
    keep = np.ones(len(r), dtype=bool)
    keep[1:] = r[1:] > r[:-1]           # guard against any non-monotonic tail
    r, g = r[keep], g[keep]
    peak_r, peak_g = _first_peak(r, g, 2.3)
    return {"gOO": {"r": r.tolist(), "g": g.tolist(), "peak_r": peak_r, "peak_g": peak_g}}


__all__ = [
    "Citation", "PropertyReference", "ReferenceSet",
    "load_reference_set", "available_reference_sets", "load_experimental_rdf",
    "load_skinner_rdf",
]
