"""Tests for mdforge.core — units, io, records."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
from mdforge.core.units import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_KCAL_MOL,
    convert_energy,
    convert_gradient,
    convert_length,
    hartree_to_kcal,
    kcal_to_hartree,
)


class TestUnits:
    def test_bohr_angstrom_roundtrip(self):
        val = np.array([1.0, 2.0, 3.0])
        assert np.allclose(convert_length(val, 'bohr', 'angstrom') * ANGSTROM_TO_BOHR, val)

    def test_hartree_kcal_roundtrip(self):
        val = np.array([1.0])
        assert np.allclose(kcal_to_hartree(hartree_to_kcal(val)), val)

    def test_hartree_to_kcal_known_value(self):
        # 1 Hartree ≈ 627.509 kcal/mol
        result = hartree_to_kcal(np.array([1.0]))
        assert abs(result.item() - 627.509) < 0.1

    def test_convert_energy_same_unit_noop(self):
        val = np.array([42.0])
        out = convert_energy(val, 'kcal/mol', 'kcal/mol')
        assert np.allclose(out, val)

    def test_convert_gradient_hartree_bohr_to_kcal_ang(self):
        val = np.array([1.0])
        out = convert_gradient(val, 'Hartree/bohr', 'kcal/mol/Angstrom')
        # 1 Ha/bohr = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM kcal/mol/Å
        expected = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM
        assert abs(out.item() - expected) < 0.1

    def test_unsupported_energy_unit_raises(self):
        with pytest.raises(ValueError, match='Unsupported energy'):
            convert_energy(np.array([1.0]), 'joule', 'kcal/mol')

    def test_unsupported_length_unit_raises(self):
        with pytest.raises(ValueError, match='Unsupported length'):
            convert_length(np.array([1.0]), 'meter', 'bohr')


# ---------------------------------------------------------------------------
# Records — SpiceMolecule
# ---------------------------------------------------------------------------

from mdforge.core.records import (  # noqa: E402
    BulkProperties,
    SpiceMolecule,
    Trajectory,
    _field_groups,
    upgrade_legacy_to_v2,
)


def _make_water() -> SpiceMolecule:
    """Return a minimal 2-conformation water SpiceMolecule for testing."""
    return SpiceMolecule(
        name="water",
        subset="test",
        smiles="O",
        atomic_numbers=np.array([8, 1, 1]),
        conformations=np.zeros((2, 3, 3), dtype=np.float32),
        dft_total_energy=np.array([-76.4, -76.5]),
        dft_total_gradient=np.zeros((2, 3, 3), dtype=np.float32),
        formation_energy=np.array([0.1, 0.2]),
    )


def _pdb_res_seqs(content: str) -> list[int]:
    """Extract per-atom residue sequence numbers (cols 23-26) from PDB text."""
    return [
        int(ln[22:26])
        for ln in content.splitlines()
        if ln.startswith(("ATOM", "HETATM"))
    ]


class TestSpiceMolecule:
    def test_basic_properties(self):
        mol = _make_water()
        assert mol.n_atoms == 3
        assert mol.n_conformations == 2
        assert mol.n_centers == 0

    def test_save_load_roundtrip(self, tmp_path):
        mol = _make_water()
        fn = tmp_path / "water.joblib"
        mol.save(fn)
        loaded = SpiceMolecule.load(fn)
        assert loaded.name == "water"
        assert loaded.n_atoms == 3
        assert loaded.n_conformations == 2
        assert np.allclose(loaded.dft_total_energy, mol.dft_total_energy)

    def test_ensure_optional_fields_on_fresh(self):
        mol = _make_water()
        # all optional fields should be None, not missing
        assert mol.atom_to_center is None
        assert mol.forces_per_center is None
        assert mol.torques_per_center is None
        assert mol.model_total_energy is None
        assert isinstance(mol.metadata, dict)

    def test_legacy_alias_properties(self):
        mol = _make_water()
        mol.monomer_record_keys = ("a", "b")
        assert mol.monomer_record_keys_qm == ("a", "b")

    def test_update_energy_units(self):
        mol = _make_water()
        original_e = mol.dft_total_energy.copy()
        mol.update_energy_units(from_unit='Hartree', to_unit='kcal/mol', fields=['dft_total_energy'])
        expected = original_e * HARTREE_TO_KCAL_MOL
        assert np.allclose(mol.dft_total_energy, expected)

    def test_all_energy_group_does_not_include_torques(self):
        """Regression: torques_per_center must NOT be in the all_energy group."""
        groups = _field_groups()
        assert 'torques_per_center' not in groups['all_energy'], (
            "torques_per_center must not appear in the all_energy field group — "
            "converting torques with update_energy_units() would silently corrupt them"
        )

    def test_to_xyz_template(self):
        mol = _make_water()
        xyz = mol.to_xyz_template()
        lines = xyz.strip().split('\n')
        # 2 frames × (1 count line + 1 comment line + 3 atom lines) = 10 lines
        assert len(lines) == 10
        assert lines[0] == '3'

    def test_to_pdb(self, tmp_path):
        mol = _make_water()
        mol.to_pdb(tmp_path / "pdb", conf_indices=[0])
        pdbs = list((tmp_path / "pdb").glob("*.pdb"))
        assert len(pdbs) == 1
        content = pdbs[0].read_text()
        assert "HETATM" in content
        assert "END" in content

    def test_to_pdb_monomer_single_residue(self, tmp_path):
        """A single-fragment SMILES (no '.') keeps every atom in residue 1."""
        mol = _make_water()  # smiles="O", no fragment separator
        mol.to_pdb(tmp_path / "pdb", conf_indices=[0])
        res_seqs = _pdb_res_seqs(next((tmp_path / "pdb").glob("*.pdb")).read_text())
        assert res_seqs == [1, 1, 1]

    def test_to_pdb_dimer_residue_numbering(self, tmp_path):
        """A dimer ('.'-separated SMILES) puts each monomer in its own residue."""
        dimer = SpiceMolecule(
            name="water water",
            subset="test",
            smiles="[H:2][O:1][H:3].[H:5][O:4][H:6]",
            atomic_numbers=np.array([8, 1, 1, 8, 1, 1]),
            conformations=np.zeros((1, 6, 3), dtype=np.float32),
            dft_total_energy=np.array([-152.9]),
            dft_total_gradient=np.zeros((1, 6, 3), dtype=np.float32),
            formation_energy=np.array([-0.7]),
        )
        dimer.to_pdb(tmp_path / "pdb", conf_indices=[0])
        content = next((tmp_path / "pdb").glob("*.pdb")).read_text()
        # monomer 1 (first 3 atoms) → resSeq 1; monomer 2 (next 3) → resSeq 2.
        assert _pdb_res_seqs(content) == [1, 1, 1, 2, 2, 2]

    def test_to_pdb_atom_count_mismatch_falls_back(self, tmp_path):
        """If the SMILES atom count ≠ record atoms, the split is untrusted → res 1.

        ``CCO.O`` counts 4 heavy atoms but the 3-atom water record has explicit H,
        so the fragment counts can't be trusted and every atom stays in residue 1
        rather than being mis-assigned.
        """
        mol = _make_water()
        mol.smiles = "CCO.O"  # 4 SMILES heavy atoms ≠ 3 record atoms → guard trips
        mol.to_pdb(tmp_path / "pdb", conf_indices=[0])
        res_seqs = _pdb_res_seqs(next((tmp_path / "pdb").glob("*.pdb")).read_text())
        assert res_seqs == [1, 1, 1]


class TestUpgradeLegacy:
    def test_upgrade_adds_missing_fields(self):
        """Upgrading a minimal duck-type object should fill in all optional fields."""
        class Legacy:
            name = "eth"
            subset = "test"
            smiles = "CC"
            atomic_numbers = np.array([6, 6])
            conformations = np.zeros((1, 2, 3), dtype=np.float32)
            dft_total_energy = np.array([-79.0])
            dft_total_gradient = np.zeros((1, 2, 3), dtype=np.float32)
            formation_energy = np.array([0.0])

        upgraded = upgrade_legacy_to_v2(Legacy())
        assert isinstance(upgraded, SpiceMolecule)
        assert upgraded.name == "eth"
        assert upgraded.forces_per_center is None  # added by upgrade


# ---------------------------------------------------------------------------
# Records — Trajectory and BulkProperties stubs
# ---------------------------------------------------------------------------

class TestTrajectoryStub:
    def test_create_trajectory(self):
        t = Trajectory(
            positions=np.zeros((10, 5, 3)),
            potential_energy=np.zeros(10),
        )
        assert t.n_frames == 10
        assert t.n_atoms == 5
        assert t.kinetic_energy is None


class TestBulkPropertiesStub:
    def test_create_bulk_properties(self):
        bp = BulkProperties(temperature_K=298.15, density_kg_m3=997.0)
        assert bp.temperature_K == pytest.approx(298.15)
        assert bp.density_kg_m3 == pytest.approx(997.0)
        assert bp.dielectric is None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

from mdforge.core.io import (  # noqa: E402
    load_center_json,
    load_joblib_records,
    load_pickle,
    save_pickle,
    write_xyz_string,
)


class TestIO:
    def test_write_xyz_string(self):
        atomic_numbers = np.array([8, 1, 1])
        conformations_bohr = np.zeros((2, 3, 3))
        xyz = write_xyz_string(atomic_numbers, conformations_bohr)
        lines = xyz.strip().split('\n')
        # frame 0: lines 0-4  (count + comment + 3 atoms)
        # frame 1: lines 5-9
        assert lines[0] == '3'         # frame 0 atom count
        assert lines[5] == '3'         # frame 1 atom count
        assert lines[2].startswith('O')

    def test_save_load_pickle(self, tmp_path):
        obj = {"key": [1, 2, 3]}
        fn = tmp_path / "test.pkl"
        save_pickle(obj, fn)
        loaded = load_pickle(fn)
        assert loaded == obj

    def test_load_joblib_records_empty_dir(self, tmp_path):
        records = load_joblib_records(tmp_path)
        assert records == {}

    def test_load_joblib_records_with_file(self, tmp_path):
        import joblib
        mol = _make_water()
        joblib.dump(mol, tmp_path / "water.joblib")
        records = load_joblib_records(tmp_path)
        assert "water" in records

    def test_load_center_json(self, tmp_path):
        import json
        data = {
            "energy": -100.0,
            "structure_file": "test.xyz",
            "atoms": [
                {"atom": 1, "center": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"atom": 2, "center": 1, "x": 1.0, "y": 0.0, "z": 0.0},
            ],
            "centers": [
                {"center": 1, "x": 0.5, "y": 0.0, "z": 0.0,
                 "fx": 0.1, "fy": 0.0, "fz": 0.0,
                 "mx": 0.01, "my": 0.0, "mz": 0.0},
            ],
        }
        fn = tmp_path / "mol.json"
        fn.write_text(json.dumps(data))
        result = load_center_json(fn)
        assert result["energy"] == pytest.approx(-100.0)
        assert result["coords"].shape == (2, 3)
        assert result["center_forces"].shape == (1, 3)
        assert result["atom2center"].tolist() == [1, 1]


# ---------------------------------------------------------------------------
# Identity (stub)
# ---------------------------------------------------------------------------

from mdforge.core.identity import IdentityRegistry  # noqa: E402


class TestIdentityRegistry:
    def test_empty_registry(self):
        reg = IdentityRegistry()
        assert len(reg) == 0
        assert reg.get_by_name("water") is None

    def test_from_pickle_dir_missing_raises(self, tmp_path):
        # no full_database.pickle in an empty dir → FileNotFoundError
        with pytest.raises(FileNotFoundError):
            IdentityRegistry.from_pickle_dir(tmp_path)

    def test_from_full_database(self):
        from mdforge.core.identity import IdentityRegistry as IR
        reg = IR.from_full_database(
            {1: ["chloroform", "CHCl3", 5977, 6212]},
            mw={1: 119.37}, name_to_cid={"phosphine": 24404},
        )
        rec = reg.get(1)
        assert rec.name == "chloroform" and rec.formula == "CHCl3"
        assert rec.cid == 5977 and rec.mw == pytest.approx(119.37)
        assert reg.get_by_name("chloroform").molecule_id == 1
        assert reg.get_by_cid(5977).name == "chloroform"
        assert reg.get_by_name("phosphine").cid == 24404
