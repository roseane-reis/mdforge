"""Tests for mdforge.qm — metrics, centers, interaction, ingest, report, plots."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mdforge.qm import (
    centers,
    interaction,
    metrics,
    report,
)

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_prediction(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        s = metrics.regression_summary(x, x)
        assert s["MAE"] == 0.0
        assert s["RMSE"] == 0.0
        assert s["R2"] == pytest.approx(1.0)
        assert s["Pearson r"] == pytest.approx(1.0)

    def test_known_mae_rmse(self):
        ref = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, -1.0, 1.0])
        assert metrics.mae(ref, pred) == pytest.approx(1.0)
        assert metrics.rmse(ref, pred) == pytest.approx(1.0)

    def test_r2_zero_variance_reference_is_nan(self):
        ref = np.array([5.0, 5.0, 5.0])
        assert np.isnan(metrics.r2(ref, np.array([5.0, 6.0, 4.0])))

    def test_frame_gradient_magnitudes(self):
        # one frame, two atoms, forces (3,4,0)->5 and (0,0,0)->0 → mean 2.5
        g = np.array([[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])
        assert metrics.frame_gradient_magnitudes(g) == pytest.approx([2.5])

    def test_energy_metrics_sorted_and_keyed(self):
        data = {
            "spice": np.array([0.0, 1.0, 2.0]),
            "good": np.array([0.0, 1.0, 2.1]),   # small error
            "bad": np.array([0.0, 5.0, 9.0]),    # big error
        }
        rows = metrics.energy_metrics(data, reference_key="spice")
        assert [r["Model"] for r in rows] == ["good", "bad"]  # sorted by MAE
        assert set(rows[0]) >= {"Model", "MAE", "RMSE", "R2", "Pearson r"}

    def test_reference_key_missing_raises(self):
        with pytest.raises(KeyError):
            metrics.energy_metrics({"a": np.zeros(3)}, reference_key="spice")


# ---------------------------------------------------------------------------
# centers
# ---------------------------------------------------------------------------

def _reference_reduce(center_coords, atom2center, atomic_coords, forces):
    """Plain double-loop reference implementation for golden-master comparison."""
    F, N, _ = atomic_coords.shape
    cc = center_coords if center_coords.ndim == 3 else np.repeat(center_coords[None], F, axis=0)
    a2c = atom2center if atom2center.ndim == 2 else np.repeat(atom2center[None], F, axis=0)
    C = cc.shape[1]
    cf = np.zeros((F, C, 3))
    ct = np.zeros((F, C, 3))
    for f in range(F):
        for a in range(N):
            c = int(a2c[f, a])
            cf[f, c] += forces[f, a]
            ct[f, c] += np.cross(atomic_coords[f, a] - cc[f, c], forces[f, a])
    return cf, ct


class TestCenters:
    def test_vectorized_matches_reference_loop(self):
        rng = np.random.default_rng(0)
        F, N, C = 4, 6, 2
        atomic = rng.standard_normal((F, N, 3))
        forces = rng.standard_normal((F, N, 3))
        center_coords = rng.standard_normal((F, C, 3))
        atom2center = rng.integers(0, C, size=(F, N))
        cf, ct = centers.batch_atom_forces_to_center(center_coords, atom2center, atomic, forces)
        rcf, rct = _reference_reduce(center_coords, atom2center, atomic, forces)
        assert np.allclose(cf, rcf)
        assert np.allclose(ct, rct)

    def test_net_force_conserved(self):
        # Sum of center forces == sum of atom forces (no net force lost).
        rng = np.random.default_rng(1)
        atomic = rng.standard_normal((3, 5, 3))
        forces = rng.standard_normal((3, 5, 3))
        atom2center = np.array([0, 0, 1, 1, 1])
        cc = rng.standard_normal((2, 3))
        cf, _ = centers.batch_atom_forces_to_center(cc, atom2center, atomic, forces)
        assert np.allclose(cf.sum(axis=1), forces.sum(axis=1))

    def test_broadcast_static_center_coords(self):
        atomic = np.zeros((2, 2, 3))
        forces = np.ones((2, 2, 3))
        cc = np.zeros((1, 3))                  # single center, static
        a2c = np.array([0, 0])
        cf, ct = centers.batch_atom_forces_to_center(cc, a2c, atomic, forces)
        assert cf.shape == (2, 1, 3)
        assert np.allclose(cf[:, 0], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

    def test_out_of_range_index_raises(self):
        atomic = np.zeros((1, 2, 3))
        forces = np.zeros((1, 2, 3))
        with pytest.raises(ValueError):
            centers.batch_atom_forces_to_center(np.zeros((1, 3)), np.array([0, 5]), atomic, forces)

    def test_update_record_from_reference(self):
        from mdforge.core.records import SpiceMolecule
        # target record with 2 conformations, 4 atoms
        target = SpiceMolecule(
            name="t", subset="x", smiles="",
            atomic_numbers=np.array([8, 1, 8, 1]),
            conformations=np.zeros((2, 4, 3), dtype=np.float32),
            dft_total_energy=np.zeros(2),
            dft_total_gradient=np.zeros((2, 4, 3), dtype=np.float32),
            formation_energy=np.zeros(2),
        )
        reference = SimpleNamespace(
            center_coords=np.zeros((2, 3)),          # 2 centers
            atom_to_center=np.array([0, 0, 1, 1]),
        )
        grads = np.ones((2, 4, 3))
        centers.update_record_center_fields_from_reference(
            target, reference, grads, values_are_gradients=True
        )
        # forces = -gradients = -1 each; 2 atoms per center → net (-2,-2,-2)
        assert target.forces_per_center.shape == (2, 2, 3)
        assert np.allclose(target.forces_per_center[:, 0], -2.0)


# ---------------------------------------------------------------------------
# interaction
# ---------------------------------------------------------------------------

class TestInteraction:
    def test_infer_n_atoms_explicit_no_rdkit(self):
        # explicit counts must not require rdkit
        assert interaction.infer_n_atoms_per_mol([3, 4]) == (3, 4)

    def test_monomer_com_distance(self):
        # two monomers (1 atom each) at x=0 and x=3 → distance 3
        conf = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        d = interaction.monomer_com_distance(conf, (1, 1))
        assert d == pytest.approx([3.0])

    def test_split_dimer(self):
        conf = np.arange(2 * 4 * 3, dtype=float).reshape(1, 8, 3)
        m1, m2 = interaction.split_dimer_conformations(conf, (3, 5))
        assert m1.shape == (1, 3, 3)
        assert m2.shape == (1, 5, 3)

    def test_compute_model_interactions_no_match(self):
        # interaction = dimer - mon1 - mon2 (from attached monomer totals)
        rec = SimpleNamespace(
            model_total_energy=np.array([-10.0, -12.0]),
            monomer1_total_energy=np.array([-3.0, -3.0]),
            monomer2_total_energy=np.array([-4.0, -4.0]),
            forces_per_center=None,
        )
        interaction.compute_model_interactions_no_match(rec, model_gradient_field=None)
        assert np.allclose(rec.interaction_total_energy, [-3.0, -5.0])

    def test_interaction_gradient_blockwise(self):
        rec = SimpleNamespace(
            model_total_energy=np.array([-10.0]),
            monomer1_total_energy=np.array([-3.0]),
            monomer2_total_energy=np.array([-4.0]),
            forces_per_center=np.ones((1, 5, 3)),
            monomer1_total_gradient=np.ones((1, 2, 3)),
            monomer2_total_gradient=np.full((1, 3, 3), 0.5),
        )
        interaction.compute_model_interactions_no_match(rec, model_gradient_field="forces_per_center")
        # block 0:2 → 1-1=0 ; block 2:5 → 1-0.5=0.5
        assert np.allclose(rec.interaction_total_gradient[0, :2], 0.0)
        assert np.allclose(rec.interaction_total_gradient[0, 2:], 0.5)

    def test_compute_pair_interaction_with_matching(self):
        # dimer block order [mon_a(2 atoms), mon_b(1 atom)]; identical geometries
        za, zb = np.array([8, 1]), np.array([6])
        dimer_conf = np.array([[[0, 0, 0], [1, 0, 0], [0, 0, 5]]], dtype=float)
        dimer = SimpleNamespace(atomic_numbers=np.concatenate([za, zb]),
                                conformations=dimer_conf,
                                dft_total_energy=np.array([-20.0]),
                                model_total_energy=None)
        mon_a = SimpleNamespace(atomic_numbers=za,
                                conformations=dimer_conf[:, :2, :].copy(),
                                dft_total_energy=np.array([-8.0]), model_total_energy=None)
        mon_b = SimpleNamespace(atomic_numbers=zb,
                                conformations=dimer_conf[:, 2:, :].copy(),
                                dft_total_energy=np.array([-9.0]), model_total_energy=None)
        res = interaction.compute_pair_interaction_energies(
            "a_b", {"a_b": dimer}, {"a": mon_a, "b": mon_b},
            dimer_energy_field="dft_total_energy", monomer_energy_field="dft_total_energy",
            rmsd_tol=1e-6,
        )
        assert res.match_mask.all()
        assert res.interaction_energy == pytest.approx([-20.0 - (-8.0) - (-9.0)])


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def _payload(self):
        return [{
            "energy": -1.5,
            "atoms": [
                {"atom": 1, "center": 0, "x": 0.0, "y": 0.0, "z": 0.0},
                {"atom": 2, "center": 0, "x": 1.0, "y": 0.0, "z": 0.0},
            ],
            "centers": [
                {"center": 0, "x": 0.5, "y": 0.0, "z": 0.0,
                 "fx": 0.1, "fy": 0.2, "fz": 0.3, "mx": 0.0, "my": 0.0, "mz": 0.0},
            ],
        }]

    def test_model_outputs_to_record(self):
        from mdforge.qm.ingest import model_outputs_to_record
        mol = model_outputs_to_record(self._payload(), data_format="list",
                                      name="test", atomic_numbers=[8, 1])
        assert mol.n_conformations == 1
        assert mol.n_atoms == 2
        assert mol.forces_per_center.shape == (1, 1, 3)
        assert np.allclose(mol.model_total_energy, [-1.5])
        assert np.allclose(mol.forces_per_center[0, 0], [0.1, 0.2, 0.3])

    def test_write_and_reload_joblib(self, tmp_path):
        from mdforge.core.records import SpiceMolecule
        from mdforge.qm.ingest import write_model_outputs_to_joblib
        fn = tmp_path / "rec.joblib"
        write_model_outputs_to_joblib(self._payload(), fn, data_format="list",
                                      name="test", atomic_numbers=[8, 1])
        reloaded = SpiceMolecule.load(fn)
        assert reloaded.n_conformations == 1
        assert np.allclose(reloaded.model_total_energy, [-1.5])

    def test_load_model_outputs_from_string(self):
        from mdforge.qm.ingest import load_model_outputs
        out = load_model_outputs('[{"energy": 1.0, "atoms": [], "centers": []}]')
        assert isinstance(out, list) and out[0]["energy"] == 1.0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class TestReport:
    def _records(self):
        return {
            "spice": SimpleNamespace(dft_total_energy=np.array([-1.0, 0.0, 1.0])),
            "amber": SimpleNamespace(model_total_energy=np.array([-1.0, 0.1, 1.2])),
            "mace": SimpleNamespace(model_total_energy=np.array([-1.0, 0.0, 1.05])),
        }

    def test_build_energy_dict_field_selection(self):
        edict = report.build_energy_dict(self._records(), reference_key="spice")
        assert np.allclose(edict["spice"], [-1.0, 0.0, 1.0])     # dft field
        assert np.allclose(edict["amber"], [-1.0, 0.1, 1.2])     # model field

    def test_build_energy_dict_relative_to_frame(self):
        edict = report.build_energy_dict(self._records(), reference_key="spice",
                                         relative_to_frame=0)
        # every series offset so frame 0 == 0
        assert all(np.isclose(v[0], 0.0) for v in edict.values())

    def test_compare_records_metrics_only(self):
        # no outdir → metrics lists only, no viz needed
        result = report.compare_records(self._records(), reference_key="spice")
        assert "energy_metrics" in result
        models = [r["Model"] for r in result["energy_metrics"]]
        assert set(models) == {"amber", "mace"}
        assert "energy_fig" not in result  # no figure without outdir

    def test_save_metrics_csv(self, tmp_path):
        rows = [{"Model": "a", "MAE": 0.1}, {"Model": "b", "MAE": 0.2}]
        path = report.save_metrics_csv(rows, tmp_path / "m.csv")
        text = path.read_text()
        assert "Model,MAE" in text and "a,0.1" in text


# ---------------------------------------------------------------------------
# compare + plots  (skip if viz deps absent)
# ---------------------------------------------------------------------------

class TestComparePlots:
    def test_compare_energy_models(self):
        pytest.importorskip("seaborn")
        pytest.importorskip("pandas")
        import matplotlib
        matplotlib.use("Agg")
        from mdforge.qm.compare import compare_energy_models
        rng = np.random.default_rng(0)
        data = {
            "spice": rng.standard_normal(50),
            "amber": rng.standard_normal(50),
        }
        data["amber"] = data["spice"] + 0.1 * rng.standard_normal(50)
        summary, fig = compare_energy_models(data, reference_key="spice")
        assert list(summary["Model"]) == ["amber"]
        assert fig is not None

    def test_interaction_profile_plot(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from mdforge.qm.plots import plot_interaction_energy_profile
        energies = {"spice": np.array([-1.0, -2.0, -1.5]), "model": np.array([-0.9, -2.1, -1.4])}
        x = np.array([5.0, 3.0, 4.0])
        ax = plot_interaction_energy_profile(energies, x, reference_key="spice")
        assert ax is not None
