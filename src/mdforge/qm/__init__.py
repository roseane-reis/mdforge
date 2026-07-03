"""mdforge.qm — QM-vs-model energy/force analysis (goal f).

Array-in metrics + plots, with parsing decoupled from computation. The
comparison contract is ``dict[str, np.ndarray]`` keyed by model name plus a
``reference_key`` naming the tagged reference.

Modules
-------
- :mod:`~mdforge.qm.metrics`     pure-numpy regression metrics (MAE/RMSE/R²/r)
- :mod:`~mdforge.qm.compare`     3-panel reference-vs-model figures (viz extra)
- :mod:`~mdforge.qm.centers`     per-atom → per-center force/torque reduction
- :mod:`~mdforge.qm.interaction` interaction energies (dimer − mon1 − mon2)
- :mod:`~mdforge.qm.ingest`      model-output payloads → SpiceMolecule
- :mod:`~mdforge.qm.report`      records → comparison dicts, metrics, CSV, figures
- :mod:`~mdforge.qm.plots`       interaction-energy profile plots (viz extra)

Like ``liquid``, this subpackage is engine-free — it builds on ``core`` only.
"""

from __future__ import annotations

from ..core.records import SpiceMolecule
from . import centers, compare, ingest, interaction, metrics, plots, report
from .centers import (
    batch_atom_forces_to_center,
    compute_center_magnitude,
    update_record_center_fields_from_reference,
    update_record_dict_center_fields,
)
from .ingest import load_model_outputs, model_outputs_to_record, write_model_outputs_to_joblib
from .interaction import (
    build_interaction_energy_dict,
    compute_model_interactions_no_match,
    compute_pair_interaction_energies,
    monomer_com_distance,
)
from .metrics import (
    energy_metrics,
    gradient_metrics,
    mae,
    pearson,
    r2,
    regression_summary,
    rmse,
)
from .report import build_energy_dict, build_gradient_dict, compare_records, save_metrics_csv

__all__ = [
    # submodules
    "metrics", "compare", "centers", "interaction", "ingest", "report", "plots",
    # records
    "SpiceMolecule",
    # metrics
    "mae", "rmse", "r2", "pearson", "regression_summary",
    "energy_metrics", "gradient_metrics",
    # centers
    "batch_atom_forces_to_center", "update_record_center_fields_from_reference",
    "update_record_dict_center_fields", "compute_center_magnitude",
    # interaction
    "compute_model_interactions_no_match", "compute_pair_interaction_energies",
    "build_interaction_energy_dict", "monomer_com_distance",
    # ingest
    "load_model_outputs", "model_outputs_to_record", "write_model_outputs_to_joblib",
    # report
    "build_energy_dict", "build_gradient_dict", "compare_records", "save_metrics_csv",
]
