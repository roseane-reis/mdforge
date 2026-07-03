"""Tests for mdforge.formats — txyz, arc, prm, analyze_out, pdb, mol."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mdforge.formats import (
    analyze_out,
    arc,
    epsr,
    pdb,
    prm,
    txyz,
)

# Real HIPPO prm fixtures (set MDFORGE_REFDATA to enable; skipped elsewhere / in CI).
_REF_PRMDIR = Path(os.environ.get("MDFORGE_REFDATA", "/opt/mdforge/reference-data")) / "prmfiles"


# ---------------------------------------------------------------------------
# txyz
# ---------------------------------------------------------------------------

_WATER_TXYZ = """\
3  water
   20.000000   20.000000   20.000000   90.000000   90.000000   90.000000
     1  O      0.000000    0.000000    0.000000      1     2     3
     2  H      0.957000    0.000000    0.000000      1     1
     3  H     -0.240000    0.927000    0.000000      1     1
"""

_WATER_RAW = """\
3
water molecule
O    0.000000    0.000000    0.000000
H    0.957000    0.000000    0.000000
H   -0.240000    0.927000    0.000000
"""


class TestTxyz:
    def test_read_tinker_with_box_and_connectivity(self):
        xyz = txyz.read_txyz(_WATER_TXYZ)
        assert xyz.n_atoms == 3
        assert xyz.is_tinker
        assert xyz.names == ["O", "H", "H"]
        assert list(xyz.types) == [1, 1, 1]
        assert xyz.connectivity[0] == [2, 3]   # O bonded to atoms 2 and 3
        assert xyz.connectivity[1] == [1]      # H bonded to O
        assert np.allclose(xyz.box, [20, 20, 20, 90, 90, 90])
        assert np.allclose(xyz.coords[1], [0.957, 0.0, 0.0])

    def test_tinker_roundtrip(self):
        xyz = txyz.read_txyz(_WATER_TXYZ)
        reparsed = txyz.read_txyz(txyz.write_txyz(xyz))
        assert reparsed.names == xyz.names
        assert np.allclose(reparsed.coords, xyz.coords)
        assert list(reparsed.types) == list(xyz.types)
        assert reparsed.connectivity == xyz.connectivity
        assert np.allclose(reparsed.box, xyz.box)

    def test_read_raw_xyz(self):
        xyz = txyz.read_txyz(_WATER_RAW)
        assert xyz.n_atoms == 3
        assert not xyz.is_tinker
        assert xyz.names == ["O", "H", "H"]
        assert np.allclose(xyz.coords[2], [-0.24, 0.927, 0.0])

    def test_update_coords(self):
        new = np.zeros((3, 3))
        new[1] = [1.0, 1.0, 1.0]
        out = txyz.update_coords(_WATER_TXYZ, new)
        reparsed = txyz.read_txyz(out)
        assert np.allclose(reparsed.coords[1], [1.0, 1.0, 1.0])
        assert reparsed.connectivity[0] == [2, 3]  # connectivity preserved

    def test_update_coords_type_map(self):
        out = txyz.update_coords(_WATER_TXYZ, type_map={1: 99})
        reparsed = txyz.read_txyz(out)
        assert list(reparsed.types) == [99, 99, 99]

    def test_txyz_to_raw_and_back(self):
        raw = txyz.txyz_to_raw(_WATER_TXYZ)
        r = txyz.read_txyz(raw)
        assert not r.is_tinker and r.names == ["O", "H", "H"]


# ---------------------------------------------------------------------------
# arc
# ---------------------------------------------------------------------------

class TestArc:
    def _two_frame_arc(self):
        return _WATER_TXYZ + _WATER_TXYZ.replace("0.957000", "0.960000")

    def test_count_frames(self):
        assert arc.count_frames(self._two_frame_arc()) == 2

    def test_read_arc(self):
        traj = arc.read_arc(self._two_frame_arc())
        assert traj.n_frames == 2
        assert traj.n_atoms == 3
        assert np.isclose(traj.coords[0, 1, 0], 0.957)
        assert np.isclose(traj.coords[1, 1, 0], 0.960)

    def test_volume(self):
        traj = arc.read_arc(self._two_frame_arc())
        vol = traj.volume()
        assert vol is not None
        assert np.allclose(vol, 8000.0)

    def test_roundtrip(self):
        traj = arc.read_arc(self._two_frame_arc())
        reparsed = arc.read_arc(arc.write_arc(traj))
        assert reparsed.n_frames == 2
        assert np.allclose(reparsed.coords, traj.coords)
        assert reparsed.names == traj.names

    def test_frame_extraction(self):
        traj = arc.read_arc(self._two_frame_arc())
        f0 = traj.frame(0)
        assert isinstance(f0, txyz.TinkerXYZ)
        assert np.allclose(f0.coords, traj.coords[0])


# ---------------------------------------------------------------------------
# analyze_out
# ---------------------------------------------------------------------------

_ANALYZE = """\
 Analysis of an X-ray/Tinker structure

 Intermolecular Energy :              -10.5000 Kcal/mole

 Total Potential Energy :             -25.1234 Kcal/mole

 Energy Component Breakdown :         Kcal/mole        Interactions

 Bond Stretching                        1.2345               10
 Angle Bending                          2.3456               12
 Atomic Multipoles                    -20.1234              100
 Polarization                          -5.6789              100
 Repulsion                              8.1234               50
 Dispersion                            -7.2345               50
 Charge Transfer                       -4.5678               30
"""

_TESTGRAD = """\
 Total Potential Energy :             -25.1234 Kcal/mole

 Cartesian Gradient Breakdown over Individual Atoms :

 Type      Atom              dE/dX       dE/dY       dE/dZ          Norm

 Anlyt         1            1.0000      2.0000      3.0000      3.7417
 Anlyt         2           -1.0000     -2.0000     -3.0000      3.7417
"""


class TestAnalyzeOut:
    def test_parse_energy_breakdown(self):
        frames = analyze_out.parse_energy_breakdown(_ANALYZE)
        assert len(frames) == 1
        f = frames[0]
        assert f["Total"] == pytest.approx(-25.1234)
        assert f["Intermolecular"] == pytest.approx(-10.5)
        assert f["Atomic Multipoles"] == pytest.approx(-20.1234)
        assert f["Charge Transfer"] == pytest.approx(-4.5678)

    def test_energy_components_vector(self):
        comps, inter = analyze_out.energy_components(_ANALYZE)
        assert comps.shape == (1, len(analyze_out.ENERGY_TERMS))
        assert inter[0] == pytest.approx(-10.5)
        idx = analyze_out.ENERGY_TERMS.index("Atomic Multipoles")
        assert comps[0, idx] == pytest.approx(-20.1234)

    def test_sapt_components_by_name(self):
        frame = analyze_out.parse_energy_breakdown(_ANALYZE)[0]
        elst, exch, ind, disp, total = analyze_out.sapt_components(frame)
        assert elst == pytest.approx(-20.1234)            # Multipoles
        assert exch == pytest.approx(8.1234)              # Repulsion
        assert ind == pytest.approx(-5.6789 - 4.5678)     # Polarization + Transfer
        assert disp == pytest.approx(-7.2345)             # Dispersion
        assert total == pytest.approx(-25.1234)

    def test_sapt_components_from_array(self):
        comps, _ = analyze_out.energy_components(_ANALYZE)
        out = analyze_out.sapt_components(comps)  # (5, nframes)
        assert out.shape == (5, 1)
        assert out[0, 0] == pytest.approx(-20.1234)

    def test_multiframe(self):
        frames = analyze_out.parse_energy_breakdown(_ANALYZE + _ANALYZE)
        assert len(frames) == 2

    def test_parse_testgrad(self):
        energies, grads = analyze_out.parse_testgrad(_TESTGRAD)
        assert energies.shape == (1,)
        assert energies[0] == pytest.approx(-25.1234)
        assert grads.shape == (1, 2, 3)
        assert np.allclose(grads[0, 0], [1.0, 2.0, 3.0])
        assert np.allclose(grads[0, 1], [-1.0, -2.0, -3.0])


# ---------------------------------------------------------------------------
# prm
# ---------------------------------------------------------------------------

# Minimal synthetic HIPPO-style prm (no multipole block — exercised via real
# files below). Two atom types/classes: O (401) and H (402).
_SYNTH_PRM = """\
forcefield              HIPPO-FORGE_DERIVED

atom          401  401    O     "Water O"           8    15.999     2
atom          402  402    H     "Water H"           1     1.008     1

polarize          401          0.800000     402
polarize          402          0.400000     401

chgpen            401     4.0000     3.500000
chgpen            402     1.0000     2.000000

dispersion        401     8.000000     3.500000
dispersion        402     2.000000     2.000000

repulsion         401     5.000000     4.000000     3.000000
repulsion         402     1.000000     2.000000     1.500000

chgtrn            401     3.000000     4.000000
chgtrn            402     1.000000     2.000000

bond          401  402     500.000000     0.957000

angle         402  401  402      50.000000     104.5
"""


class TestPrm:
    def test_process_prm_parses_terms(self, tmp_path):
        f = tmp_path / "synth.prm"
        f.write_text(_SYNTH_PRM)
        d = prm.process_prm(f)
        assert d["types"] == [401, 402]
        assert d["typcls"] == {401: 401, 402: 402}
        assert np.allclose(d["chgpen"][0], [4.0, 3.5])
        assert np.allclose(d["dispersion"], [8.0, 2.0])
        assert np.allclose(d["repulsion"][0], [5.0, 4.0, 3.0])
        assert np.allclose(d["chgtrn"][1], [1.0, 2.0])
        assert d["polarize"][0][0] == pytest.approx(0.8)
        assert len(d["bond"][0]) == 1
        assert len(d["angle"][0]) == 1

    def test_prm_roundtrip_synthetic(self, tmp_path):
        f = tmp_path / "synth.prm"
        f.write_text(_SYNTH_PRM)
        d1 = prm.process_prm(f)
        prm.write_prm(d1, tmp_path / "out.prm")
        d2 = prm.process_prm(tmp_path / "out.prm")
        assert d1["types"] == d2["types"]
        assert np.allclose(d1["chgpen"], d2["chgpen"])
        assert np.allclose(d1["dispersion"], d2["dispersion"])
        assert np.allclose(d1["repulsion"], d2["repulsion"])
        assert np.allclose(d1["chgtrn"], d2["chgtrn"])
        assert np.allclose(d1["polarize"][0], d2["polarize"][0])
        assert d1["bond"][0] == d2["bond"][0]

    def test_write_key_liquid_and_gas(self, tmp_path):
        liquid = prm.write_key("water.prm", tmp_path / "liquid.key", opt="liquid", a_axis=25.0)
        assert "parameters          water.prm" in liquid
        assert "a-axis            25.0" in liquid
        assert "barostat          langevin" in liquid
        gas = prm.write_key("water.prm", tmp_path / "gas.key", opt="gas")
        assert "BAROSTAT          MONTECARLO" in gas
        assert "fix-chgpen" in gas

    @pytest.mark.skipif(not _REF_PRMDIR.is_dir(), reason="reference prm files unavailable")
    def test_prm_roundtrip_real_files(self, tmp_path):
        files = sorted(_REF_PRMDIR.glob("*.prm"))[:5]
        assert files, "no reference prm files found"
        for src in files:
            d1 = prm.process_prm(src)
            prm.write_prm(d1, tmp_path / "rt.prm")
            d2 = prm.process_prm(tmp_path / "rt.prm")
            assert d1["types"] == d2["types"], src.name
            assert np.allclose(d1["chgpen"], d2["chgpen"]), src.name
            assert np.allclose(d1["repulsion"], d2["repulsion"]), src.name
            assert np.allclose(d1["polarize"][0], d2["polarize"][0]), src.name
            # multipole values (the trickiest section) must survive the round-trip
            if len(d1["multipole"][1]) > 0:
                assert np.allclose(np.array(d1["multipole"][1]),
                                   np.array(d2["multipole"][1])), src.name


# ---------------------------------------------------------------------------
# pdb
# ---------------------------------------------------------------------------

_PDB = """\
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
HETATM    1  O1  HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
END
"""


class TestPdb:
    def test_read_pdb(self):
        s = pdb.read_pdb(_PDB)
        assert s.n_atoms == 3
        assert s.elements == ["O", "H", "H"]
        assert np.allclose(s.box, [20, 20, 20, 90, 90, 90])
        assert np.allclose(s.coords[1], [0.957, 0.0, 0.0])
        assert s.atoms[0].res_name == "HOH"

    def test_pdb_roundtrip(self):
        s = pdb.read_pdb(_PDB)
        reparsed = pdb.read_pdb(pdb.write_pdb(s))
        assert reparsed.elements == s.elements
        assert np.allclose(reparsed.coords, s.coords)
        assert np.allclose(reparsed.box, s.box)

    def test_element_inference_two_letter(self):
        line = "HETATM    1 CL1  MOL A   1       0.000   0.000   0.000  1.00  0.00\n"
        s = pdb.read_pdb(line)
        assert s.elements[0] == "Cl"

    def test_txyz_to_pdb(self):
        xyz = txyz.read_txyz(_WATER_TXYZ)
        s = pdb.txyz_to_pdb(xyz, res_name="HOH")
        assert s.n_atoms == 3
        assert s.elements == ["O", "H", "H"]
        assert s.atoms[1].name == "H1"
        assert s.atoms[2].name == "H2"
        assert np.allclose(s.box, [20, 20, 20, 90, 90, 90])

    def test_pdb_to_txyz(self):
        s = pdb.read_pdb(_PDB)
        xyz = pdb.pdb_to_txyz(s)
        assert xyz.names == ["O", "H", "H"]
        assert not xyz.is_tinker
        assert np.allclose(xyz.coords, s.coords)

    def test_from_arrays_roundtrip(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.24, 0.927, 0.0]])
        s = pdb.PDBStructure.from_arrays(coords, ["O", "H", "H"], res_name="HOH")
        assert s.n_atoms == 3
        assert s.elements == ["O", "H", "H"]
        assert np.allclose(s.coords, coords)
        assert s.atoms[0].name == "O1"  # default name = element + 1-based index
        reparsed = pdb.read_pdb(pdb.write_pdb(s))
        assert reparsed.elements == ["O", "H", "H"]
        assert np.allclose(reparsed.coords, coords)

    def test_from_arrays_two_letter_element(self):
        s = pdb.PDBStructure.from_arrays([[0.0, 0.0, 0.0]], ["cl"])
        assert s.elements == ["Cl"]

    def test_from_arrays_custom_names(self):
        s = pdb.PDBStructure.from_arrays(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["O", "H"], names=["OW", "HW"]
        )
        assert [a.name for a in s.atoms] == ["OW", "HW"]

    def test_to_pdb_string_roundtrip(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        text = pdb.to_pdb_string(coords, ["O", "H"], res_name="MOL")
        s = pdb.read_pdb(text)
        assert s.elements == ["O", "H"]
        assert np.allclose(s.coords, coords)

    def test_from_arrays_validates_shapes(self):
        with pytest.raises(ValueError):
            pdb.PDBStructure.from_arrays(np.zeros((3, 2)), ["O", "H", "H"])
        with pytest.raises(ValueError):
            pdb.PDBStructure.from_arrays(np.zeros((3, 3)), ["O", "H"])

    def test_from_arrays_per_atom_res_seq(self):
        # per-atom res_seq keeps two molecules in distinct residues (survives a round-trip)
        s = pdb.PDBStructure.from_arrays(
            np.zeros((4, 3)), ["O", "H", "S", "H"], res_seq=[2, 2, 1, 1]
        )
        assert [a.res_seq for a in s.atoms] == [2, 2, 1, 1]
        reparsed = pdb.read_pdb(pdb.write_pdb(s))
        assert [a.res_seq for a in reparsed.atoms] == [2, 2, 1, 1]

    def test_from_arrays_per_atom_length_validated(self):
        with pytest.raises(ValueError):
            pdb.PDBStructure.from_arrays(np.zeros((3, 3)), ["O", "H", "H"], res_seq=[1, 2])


# ---------------------------------------------------------------------------
# mol  (skip if rdkit absent)
# ---------------------------------------------------------------------------

class TestMol:
    def test_xyz_to_smiles_water(self):
        pytest.importorskip("rdkit")
        from mdforge.formats.mol import xyz_to_smiles
        # water: O at origin, two H
        z = [8, 1, 1]
        coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        smiles = xyz_to_smiles(z, coords)
        assert "O" in smiles


# ---------------------------------------------------------------------------
# epsr  (experimental RDF readers, tiny synthetic fixtures)
# ---------------------------------------------------------------------------

class TestEpsr:
    def test_read_epsr_rdf(self, tmp_path):
        f = tmp_path / "traj.rdf11"
        f.write_text(
            "# Species  1 calculated using all atoms for COM.\n"
            "# Species  1 calculated using all atoms for COM.\n"
            "    0.0050     0.00000000     0.00000000\n"
            "    0.0150     0.50000000     0.10000000\n"
            "    0.0250     1.20000000     0.40000000\n"
        )
        r, g, n = epsr.read_epsr_rdf(f)
        assert r.shape == g.shape == n.shape == (3,)
        assert np.allclose(r, [0.005, 0.015, 0.025])
        assert np.allclose(g, [0.0, 0.5, 1.2])
        assert np.allclose(n, [0.0, 0.1, 0.4])

    def test_read_epsr_angular_rdf(self, tmp_path):
        # Two angle blocks (0-10, 10-20) with inline Range markers; a third
        # all-zero block (20-30) that must be dropped. Each block has 3 r-points.
        f = tmp_path / "traj.ardf11zz"
        f.write_text(
            "# Species  1 site is axis origin.\n"
            "# Species  1 site is axis origin.\n"
            "    0.0500     1.00000000     0.0   # Range      0.0000    10.0000\n"
            "    0.1500     2.00000000     0.0\n"
            "    0.2500     3.00000000     0.0\n"
            "    0.0500     4.00000000     0.0   # Range     10.0000    20.0000\n"
            "    0.1500     5.00000000     0.0\n"
            "    0.2500     6.00000000     0.0\n"
            "    0.0500     0.00000000     0.0   # Range     20.0000    30.0000\n"
            "    0.1500     0.00000000     0.0\n"
            "    0.2500     0.00000000     0.0\n"
        )
        r, edges, g_raw = epsr.read_epsr_angular_rdf(f)
        assert np.allclose(r, [0.05, 0.15, 0.25])
        # two non-empty blocks -> edges are [0, 10, 20]
        assert np.allclose(edges, [0.0, 10.0, 20.0])
        assert g_raw.shape == (2, 3)
        assert np.allclose(g_raw[0], [1.0, 2.0, 3.0])
        assert np.allclose(g_raw[1], [4.0, 5.0, 6.0])
