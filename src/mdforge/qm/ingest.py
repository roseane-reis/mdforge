"""Ingest model-output payloads into :class:`SpiceMolecule` records.

Lifted from ``prior internal tooling``. Parses a
center-based model-output payload (JSON / Python-literal / in-memory) whose
structure is declared by the caller via ``data_format`` (``single_dict`` /
``list`` / ``dict_by_frame``) and either creates a new record or attaches the
outputs to an existing reference record (verifying coordinates).

The private model's QM-consolidation tooling produces these payloads; this
public ingest accepts the resulting arrays/dicts, not the private importer.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.records import ModelOutputFormat, SpiceMolecule

JsonLike = Any


def _parse_serialized_payload(text: str) -> Any:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        inner = stripped[1:-1].strip()
        if inner.startswith("{") or inner.startswith("["):
            stripped = inner
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Could not parse payload as JSON or Python literal") from exc


def _looks_like_serialized_payload(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[0] in "{["


def _load_json_or_literal(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def load_model_outputs(source: str | Path | JsonLike) -> Any:
    """Load a raw model-output payload without inferring its structure.

    Accepts an in-memory list/dict, a ``Path``, a serialized string, or a file
    path string. Structure normalization happens later via ``data_format``.
    """
    if isinstance(source, (list, dict)):
        return source
    if isinstance(source, Path):
        return _load_json_or_literal(source)
    if isinstance(source, str):
        if _looks_like_serialized_payload(source):
            return _parse_serialized_payload(source)
        maybe_path = Path(source)
        if maybe_path.is_file():
            return _load_json_or_literal(maybe_path)
        raise FileNotFoundError(
            f"Input string is neither an existing file path nor a serialized payload: {source!r}"
        )
    return source


def _parse_atomic_numbers(values) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, str):
        cleaned = [p.strip() for p in values.split(",") if p.strip()]
        return np.asarray([int(p) for p in cleaned], dtype=int)
    return np.asarray(values, dtype=int).flatten()


def model_outputs_to_record(
    source: str | Path | JsonLike,
    *,
    data_format: ModelOutputFormat,
    reference: str | Path | SpiceMolecule | None = None,
    name: str | None = None,
    atomic_numbers=None,
    subset: str = "model-output",
    smiles: str = "",
    gradients: np.ndarray | None = None,
    formation_energy: np.ndarray | None = None,
    source_label: str = "model",
    center_force_unit: str = "unknown",
    center_torque_unit: str = "unknown",
    energy_unit: str = "hartree",
    verify_coordinates: bool = True,
    atol_bohr: float = 1e-4,
) -> SpiceMolecule:
    """Build (or augment a reference) :class:`SpiceMolecule` from model outputs.

    If ``reference`` is given (a joblib path or a record), the outputs are
    attached to it (coordinates verified within ``atol_bohr``). Otherwise a new
    record is created and ``name`` + ``atomic_numbers`` are required.
    """
    raw = load_model_outputs(source)

    if reference is not None:
        mol = reference if isinstance(reference, SpiceMolecule) else SpiceMolecule.load(reference)
        mol.attach_model_outputs(
            raw, data_format=data_format, source_label=source_label,
            center_force_unit=center_force_unit, center_torque_unit=center_torque_unit,
            energy_unit=energy_unit, verify_coordinates=verify_coordinates, atol_bohr=atol_bohr,
        )
        return mol

    z = _parse_atomic_numbers(atomic_numbers)
    if z is None:
        raise ValueError("atomic_numbers is required when no reference is provided")
    if not name or not str(name).strip():
        raise ValueError("name is required when no reference is provided")
    return SpiceMolecule.from_model_outputs(
        name=str(name), subset=subset, smiles=smiles, atomic_numbers=z,
        model_outputs=raw, data_format=data_format, gradients=gradients,
        formation_energy=formation_energy, source_label=source_label,
        center_force_unit=center_force_unit, center_torque_unit=center_torque_unit,
        energy_unit=energy_unit,
    )


def write_model_outputs_to_joblib(
    source: str | Path | JsonLike,
    output_joblib: str | Path,
    *,
    data_format: ModelOutputFormat,
    reference_joblib: str | Path | None = None,
    compress: int = 3,
    **kwargs,
) -> SpiceMolecule:
    """Ingest model outputs and save the resulting record as a joblib file."""
    output_path = Path(output_joblib)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mol = model_outputs_to_record(source, data_format=data_format, reference=reference_joblib, **kwargs)
    mol.save(output_path, compress=compress)
    return mol


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", required=True)
    parser.add_argument("--data-format", required=True, choices=["single_dict", "list", "dict_by_frame"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-joblib", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--atomic-numbers", default=None, help="Comma-separated, e.g. 8,1,1")
    parser.add_argument("--subset", default="model-output")
    parser.add_argument("--smiles", default="")
    parser.add_argument("--source-label", default="model")
    parser.add_argument("--center-force-unit", default="unknown")
    parser.add_argument("--center-torque-unit", default="unknown")
    parser.add_argument("--energy-unit", default="hartree")
    parser.add_argument("--skip-coordinate-check", action="store_true")
    parser.add_argument("--atol-bohr", type=float, default=1e-4)
    parser.add_argument("--compress", type=int, default=3)
    return parser


def main() -> None:  # pragma: no cover
    args = _build_parser().parse_args()
    write_model_outputs_to_joblib(
        source=args.json_path, data_format=args.data_format, output_joblib=args.output,
        reference_joblib=args.reference_joblib, name=args.name, atomic_numbers=args.atomic_numbers,
        subset=args.subset, smiles=args.smiles, source_label=args.source_label,
        center_force_unit=args.center_force_unit, center_torque_unit=args.center_torque_unit,
        energy_unit=args.energy_unit, verify_coordinates=not args.skip_coordinate_check,
        atol_bohr=args.atol_bohr, compress=args.compress,
    )


if __name__ == "__main__":
    main()


__all__ = ["load_model_outputs", "model_outputs_to_record", "write_model_outputs_to_joblib"]
