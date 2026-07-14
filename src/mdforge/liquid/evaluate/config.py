"""Configuration schema + loader for a water-model evaluation run.

The user points at their topology + trajectory files, model name, temperature,
and system info via a small YAML file (see ``EvalConfig.from_yaml``). YAML is
loaded lazily (``pyyaml`` is the optional ``[evaluate]`` extra); everything also
works from a plain ``dict`` via :meth:`EvalConfig.from_dict`, so tests and
TOML/JSON callers need no YAML.

Parsing here only *describes* the run — it opens no trajectory. The ingest layer
(:mod:`mdforge.liquid.evaluate.ingest`) does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

# Reference experimental values are defined at ambient conditions only.
REFERENCE_TEMPERATURE_K = 298.15
REFERENCE_PRESSURE_ATM = 1.0
_DEFAULT_EQUIL_FRAC = {"NPT": 0.5, "NVT": 0.2, "NVE": 0.2}
_ALLOWED_ENSEMBLES = frozenset(_DEFAULT_EQUIL_FRAC)


class EvalConfigError(ValueError):
    """Raised for a malformed evaluation config."""


class EvalStateError(ValueError):
    """Raised when the run's state point is outside the reference validity range."""


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
        raise ImportError(
            "Reading a YAML config requires PyYAML. Install it with: "
            "pip install 'mdforge[evaluate]'  (or pass a dict to EvalConfig.from_dict)"
        ) from exc
    return yaml


def _check_keys(mapping: dict, allowed: set[str], where: str) -> None:
    extra = set(mapping) - allowed
    if extra:
        raise EvalConfigError(f"unknown key(s) in {where}: {sorted(extra)}")


def _dc_from_dict(cls, data: dict | None, where: str):
    """Build a (flat) dataclass from a dict, rejecting unknown keys."""
    data = dict(data or {})
    allowed = {f.name for f in fields(cls)}
    _check_keys(data, allowed, where)
    return cls(**data)


@dataclass
class ModelSpec:
    name: str = "model"
    engine: str = "unknown"


@dataclass
class StateSpec:
    temperature_K: float = REFERENCE_TEMPERATURE_K
    pressure_atm: float = REFERENCE_PRESSURE_ATM


@dataclass
class SystemSpec:
    n_molecules: int | None = None
    molar_mass_g_mol: float = 18.01528
    atoms_per_molecule: int = 3
    charges_e: dict[str, float] | None = None
    molecular_polarizability: float | None = None   # Å³, for eps_inf
    gas_pe_per_molecule: float = 0.0                 # ΔHvap gas reference


@dataclass
class TopologySpec:
    pdb: str | None = None
    txyz: str | None = None


@dataclass
class LegSpec:
    name: str
    ensemble: str
    trajectory: str | None = None   # .gsd / .dcd
    log: str | None = None           # .npy / .csv
    equil_frac: float | None = None  # default resolved by ensemble

    def resolved_equil_frac(self) -> float:
        if self.equil_frac is not None:
            return float(self.equil_frac)
        return _DEFAULT_EQUIL_FRAC.get(self.ensemble.upper(), 0.2)


@dataclass
class RdfKnobs:
    r_max: float = 8.0
    n_bins: int = 200
    stride: int = 3


@dataclass
class DiffusionKnobs:
    dt_ps: float | None = None
    fit_lo: float = 0.2
    fit_hi: float = 0.6
    finite_size_correction: bool = True
    reference_viscosity_pa_s: float = 8.9e-4   # water, 25 °C


@dataclass
class HBondKnobs:
    r_oo: float = 3.5
    angle_deg: float = 30.0


@dataclass
class AnalysisKnobs:
    rdf: RdfKnobs = field(default_factory=RdfKnobs)
    structure_stride: int = 8
    diffusion: DiffusionKnobs = field(default_factory=DiffusionKnobs)
    hbond: HBondKnobs = field(default_factory=HBondKnobs)
    seed: int = 1

    @classmethod
    def from_dict(cls, data: dict | None) -> AnalysisKnobs:
        data = dict(data or {})
        _check_keys(data, {f.name for f in fields(cls)}, "analysis")
        rdf = _dc_from_dict(RdfKnobs, data.get("rdf"), "analysis.rdf")
        diffusion = _dc_from_dict(DiffusionKnobs, data.get("diffusion"), "analysis.diffusion")
        hbond = _dc_from_dict(HBondKnobs, data.get("hbond"), "analysis.hbond")
        return cls(
            rdf=rdf, diffusion=diffusion, hbond=hbond,
            structure_stride=int(data.get("structure_stride", 8)),
            seed=int(data.get("seed", 1)),
        )


@dataclass
class OutputSpec:
    dir: str = "analysis"
    formats: list[str] = field(default_factory=lambda: ["json", "csv", "md"])
    plots: bool = True
    timeseries: bool = False   # retain per-frame thermo series + emit per-leg plots


@dataclass
class EvalConfig:
    model: ModelSpec = field(default_factory=ModelSpec)
    species: str = "water"
    state: StateSpec = field(default_factory=StateSpec)
    system: SystemSpec = field(default_factory=SystemSpec)
    topology: TopologySpec = field(default_factory=TopologySpec)
    legs: list[LegSpec] = field(default_factory=list)
    analysis: AnalysisKnobs = field(default_factory=AnalysisKnobs)
    output: OutputSpec = field(default_factory=OutputSpec)
    base_dir: Path = field(default_factory=Path)   # for relative path resolution

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, *, base_dir: str | Path = ".") -> EvalConfig:
        data = dict(data or {})
        top_allowed = {
            "model", "species", "state", "system", "topology",
            "legs", "analysis", "output",
        }
        _check_keys(data, top_allowed, "config")

        legs_raw = data.get("legs") or []
        if not isinstance(legs_raw, list):
            raise EvalConfigError("'legs' must be a list")
        legs = [_dc_from_dict(LegSpec, leg, f"legs[{i}]") for i, leg in enumerate(legs_raw)]

        cfg = cls(
            model=_dc_from_dict(ModelSpec, data.get("model"), "model"),
            species=str(data.get("species", "water")),
            state=_dc_from_dict(StateSpec, data.get("state"), "state"),
            system=_dc_from_dict(SystemSpec, data.get("system"), "system"),
            topology=_dc_from_dict(TopologySpec, data.get("topology"), "topology"),
            legs=legs,
            analysis=AnalysisKnobs.from_dict(data.get("analysis")),
            output=_dc_from_dict(OutputSpec, data.get("output"), "output"),
            base_dir=Path(base_dir),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        yaml = _import_yaml()
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data, base_dir=path.parent)

    # ------------------------------------------------------------------
    # Validation & path resolution
    # ------------------------------------------------------------------
    def validate(self) -> EvalConfig:
        if not self.legs:
            raise EvalConfigError("config has no 'legs'; at least one is required")
        names = [leg.name for leg in self.legs]
        if len(names) != len(set(names)):
            raise EvalConfigError(f"duplicate leg names: {names}")
        for leg in self.legs:
            ens = leg.ensemble.upper()
            if ens not in _ALLOWED_ENSEMBLES:
                raise EvalConfigError(
                    f"leg {leg.name!r}: ensemble {leg.ensemble!r} not in "
                    f"{sorted(_ALLOWED_ENSEMBLES)}"
                )
            leg.ensemble = ens
            if not leg.trajectory and not leg.log:
                raise EvalConfigError(
                    f"leg {leg.name!r} needs a 'trajectory' and/or a 'log'"
                )
        if not (self.topology.pdb or self.topology.txyz):
            raise EvalConfigError("topology needs at least one of 'pdb' / 'txyz'")
        if self.system.atoms_per_molecule <= 0:
            raise EvalConfigError("system.atoms_per_molecule must be positive")
        return self

    def resolve(self, path: str | None) -> Path | None:
        """Resolve a config-relative path against ``base_dir``."""
        if path is None:
            return None
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)


def state_guard(
    state: StateSpec,
    *,
    temp_tol_K: float = 1.0,
    pressure_tol_atm: float = 0.5,
) -> None:
    """Assert the declared state point is within the reference validity window.

    The packaged experimental references are defined at 298.15 K / 1 atm. NVE /
    NVT legs at that state pass on the config's declared target (not the noisy
    per-frame instantaneous values). Raises :class:`EvalStateError` otherwise.
    """
    dT = abs(state.temperature_K - REFERENCE_TEMPERATURE_K)
    dP = abs(state.pressure_atm - REFERENCE_PRESSURE_ATM)
    if dT > temp_tol_K or dP > pressure_tol_atm:
        raise EvalStateError(
            "experimental references are defined at "
            f"{REFERENCE_TEMPERATURE_K} K / {REFERENCE_PRESSURE_ATM} atm; got "
            f"{state.temperature_K} K / {state.pressure_atm} atm "
            f"(|ΔT|={dT:.2f} K, |ΔP|={dP:.2f} atm). "
            "Pass --no-state-guard to override (results will not be comparable)."
        )


__all__ = [
    "EvalConfig", "ModelSpec", "StateSpec", "SystemSpec", "TopologySpec",
    "LegSpec", "RdfKnobs", "DiffusionKnobs", "HBondKnobs", "AnalysisKnobs",
    "OutputSpec", "EvalConfigError", "EvalStateError", "state_guard",
    "REFERENCE_TEMPERATURE_K", "REFERENCE_PRESSURE_ATM",
]
