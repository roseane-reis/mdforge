"""Tests for mdforge.data — bulk properties, dimer databases, identity, build.

Pure parsing/projection logic always runs (synthetic fixtures). Loads of the
real reference-data / databases trees are gated and skip when absent. The 124 MB
DES370K pickle is only loaded behind MDFORGE_TEST_DES370K=1.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from mdforge.core.identity import IdentityRegistry
from mdforge.core.records import BulkProperties
from mdforge.data import (
    Database,
    DimerSet,
    bulk_properties_from_vector,
    load_bulk_table,
    load_des370k_gid,
    load_reference_database,
    load_sapt_dimer,
    parse_ncia_benchmark,
    parse_org_liq_csv,
    project_des370k_components,
)

_REFDATA = Path(os.environ.get("MDFORGE_REFDATA", "/opt/mdforge/reference-data"))
_DBINFO = _REFDATA / "database-info"
_DATABASES = Path(os.environ.get("MDFORGE_DATABASES", "/opt/mdforge/databases"))
_HAS_DBINFO = (_DBINFO / "full_database.pickle").is_file()
_HAS_S101 = (_DATABASES / "S101x7").is_dir()
_DES_PROC = _DATABASES / "DES370k" / "data_per_gid_proc.pickle"

dbinfo_only = pytest.mark.skipif(not _HAS_DBINFO, reason="reference-data/database-info unavailable")
s101_only = pytest.mark.skipif(not _HAS_S101, reason="databases/S101x7 unavailable")


# ---------------------------------------------------------------------------
# bulk
# ---------------------------------------------------------------------------

class TestBulk:
    def test_vector_to_bulk_properties(self):
        # chloroform: [T, density, HV, diel, KT, alphaT, surf]
        bp = bulk_properties_from_vector([298.15, 1479.3, 31.28, 4.71, 1.03, 1.29, 26.67],
                                         molecule_id=1)
        assert bp.temperature_K == pytest.approx(298.15)
        assert bp.density_kg_m3 == pytest.approx(1479.3)
        assert bp.delta_hvap_kcal_mol == pytest.approx(31.28)
        assert bp.dielectric == pytest.approx(4.71)
        assert bp.surface_tension_mN_m == pytest.approx(26.67)
        assert bp.metadata["molecule_id"] == 1

    def test_missing_sentinel_becomes_none(self):
        bp = bulk_properties_from_vector([298.15, 997.0, -1, -1, 0.5, -1, 72.0])
        assert bp.density_kg_m3 == pytest.approx(997.0)
        assert bp.delta_hvap_kcal_mol is None
        assert bp.dielectric is None
        assert bp.kappa_T == pytest.approx(0.5)
        assert bp.alpha_T is None

    def test_load_bulk_table_list_and_dicts(self, tmp_path):
        # list form (org_liq_list)
        p = tmp_path / "list.pickle"
        pickle.dump([[1, 298.15, 1479.3, 31.28, 4.71, 1.03, 1.29, 26.67]], p.open("wb"))
        tbl = load_bulk_table(p)
        assert isinstance(tbl[1], BulkProperties) and tbl[1].density_kg_m3 == pytest.approx(1479.3)

        # dict single-vector form (molinfo_dict)
        p2 = tmp_path / "molinfo.pickle"
        pickle.dump({1: [298.15, 1479.3, 31.28, 4.71, 1.03, 1.29, 26.67]}, p2.open("wb"))
        assert load_bulk_table(p2)[1].dielectric == pytest.approx(4.71)

        # dict multi-T form (org_liq_dict)
        p3 = tmp_path / "dict.pickle"
        pickle.dump({1: [[298.15, 1479.3, 31.28, 4.71, 1.03, 1.29, 26.67],
                         [310.0, 1450.0, 30.0, 4.6, 1.0, 1.3, 25.0]]}, p3.open("wb"))
        multi = load_bulk_table(p3)
        assert isinstance(multi[1], list) and len(multi[1]) == 2
        assert multi[1][1].temperature_K == pytest.approx(310.0)

    def test_parse_org_liq_csv(self, tmp_path):
        csv = tmp_path / "dens.csv"
        csv.write_text(
            "1,chloroform,298.15,1479.30,[9],1375.0,0.3,1373.2,0.3\n"   # 9-col, bracket ref
            "2,water,298.15,997.0,0.5,990.0,0.2\n"                       # 7-col, numeric σ
            "..,310.0,993.0,0.4,985.0,0.2\n"                             # continuation of id 2
        )
        rows = parse_org_liq_csv(csv)
        assert len(rows) == 3
        assert rows[0]["id"] == 1 and rows[0]["ref"] == "[9]" and rows[0]["sigma_exp"] is None
        assert rows[0]["sims"] == [(1375.0, 0.3), (1373.2, 0.3)]
        assert rows[1]["sigma_exp"] == pytest.approx(0.5)
        assert rows[2]["id"] == 2 and rows[2]["T"] == pytest.approx(310.0)  # continuation

    @dbinfo_only
    def test_load_real_molinfo_dict(self):
        tbl = load_bulk_table(_DBINFO / "molinfo_dict.pickle")
        assert len(tbl) > 100
        assert tbl[1].density_kg_m3 == pytest.approx(1479.3, abs=0.1)  # chloroform


# ---------------------------------------------------------------------------
# dimers
# ---------------------------------------------------------------------------

class TestDimers:
    def test_load_sapt_dimer_npy_only(self, tmp_path):
        comps = np.array([[-29.0, 76.1, -15.3, -11.2, 20.4],
                          [-16.0, 35.9, -7.4, -7.1, 5.2]])
        np.save(tmp_path / "d.npy", comps)
        ds = load_sapt_dimer(tmp_path / "d")
        assert isinstance(ds, DimerSet)
        assert ds.n_geometries == 2
        assert ds.qm_components.shape == (2, 5)
        assert np.allclose(ds.interaction_energy, [20.4, 5.2])  # the total column

    def test_project_des370k_components(self):
        gid = {
            "sapt_es": np.array([1.0, 2.0]),
            "sapt_ex": np.array([10.0, 20.0]),
            "sapt_exs2": np.array([1.0, 1.0]),
            "sapt_ind": np.array([-2.0, -3.0]),
            "sapt_exind": np.array([0.5, 0.5]),
            "sapt_delta_HF": np.array([0.1, 0.1]),
            "sapt_disp": np.array([-5.0, -6.0]),
            "sapt_exdisp_os": np.array([0.2, 0.2]),
            "sapt_exdisp_ss": np.array([0.1, 0.1]),
            "cc_CCSD(T)_all": np.array([-3.0, -4.0]),
        }
        comps = project_des370k_components(gid)  # delta_HF in induction
        assert np.allclose(comps[:, 0], [1.0, 2.0])           # electrostatics
        assert np.allclose(comps[:, 1], [11.0, 21.0])         # exchange = ex + exs2
        assert np.allclose(comps[:, 2], [-1.4, -2.4])         # ind + exind + delta_HF
        assert np.allclose(comps[:, 3], [-4.7, -5.7])         # disp + exdisp(os+ss)
        assert np.allclose(comps[:, 4], [-3.0, -4.0])         # CCSD(T) total
        no_dhf = project_des370k_components(gid, delta_hf_in_induction=False)
        assert np.allclose(no_dhf[:, 2], [-1.5, -2.5])

    def test_parse_ncia_benchmark(self, tmp_path):
        f = tmp_path / "bench.txt"
        f.write_text("# CCSD(T)/CBS interaction energies, kcal/mol\n"
                     "1.001_080\t-5.42\n"
                     "1.001_100   -6.13\n")
        out = parse_ncia_benchmark(f)
        assert out["1.001_080"] == pytest.approx(-5.42)
        assert out["1.001_100"] == pytest.approx(-6.13)

    @s101_only
    def test_load_real_s101x7_pair(self):
        npy = next((_DATABASES / "S101x7").glob("*/*.npy"))
        # ensure we don't grab a monomer-split file
        while npy.stem.endswith(("-mol1", "-mol2")):
            npy = next((_DATABASES / "S101x7").glob("*/*.npy"))
        ds = load_sapt_dimer(npy.with_suffix(""), source="S101x7")
        assert ds.qm_components.shape[1] == 5
        if ds.geometries is not None:
            assert ds.geometries.shape[0] == ds.qm_components.shape[0]  # frames align

    @pytest.mark.skipif(not (os.environ.get("MDFORGE_TEST_DES370K") and _DES_PROC.is_file()),
                        reason="set MDFORGE_TEST_DES370K=1 (loads the 124MB DES370K pickle)")
    def test_real_des370k_projection(self):
        proc = pickle.load(_DES_PROC.open("rb"))
        gid = next(iter(proc))
        ds = load_des370k_gid(proc, gid)
        assert ds.qm_components.shape[1] == 5
        assert ds.n_geometries == len(proc[gid]["sapt_es"])


# ---------------------------------------------------------------------------
# identity + build
# ---------------------------------------------------------------------------

class TestBuild:
    def test_database_repr_and_lookup(self):
        reg = IdentityRegistry.from_full_database({1: ["chloroform", "CHCl3", 5977, 6212]})
        db = Database(identity=reg, bulk={1: bulk_properties_from_vector(
            [298.15, 1479.3, 31.28, 4.71, 1.03, 1.29, 26.67], molecule_id=1)})
        assert db.molecule(1).name == "chloroform"
        assert db.bulk_properties(1).density_kg_m3 == pytest.approx(1479.3)
        assert db.molecule_ids() == [1]
        assert "1 molecules" in repr(db) or "molecules" in repr(db)

    @dbinfo_only
    def test_identity_from_real_pickle_dir(self):
        reg = IdentityRegistry.from_pickle_dir(_DBINFO)
        assert len(reg) > 100
        chloroform = reg.get(1)
        assert chloroform.name == "chloroform" and chloroform.formula == "CHCl3"
        assert reg.get_by_name("chloroform").molecule_id == 1

    @dbinfo_only
    def test_load_reference_database(self):
        db = load_reference_database(_REFDATA)
        assert len(db.identity) > 100
        assert len(db.bulk) > 100
        assert db.molecule(1).name == "chloroform"
        assert db.bulk_properties(1).density_kg_m3 == pytest.approx(1479.3, abs=0.1)
