"""Tests for mdforge.liquid — stats, thermo, transport, parse, orchestrator.

Strategy: each kernel is checked against an inline reference implementation of
the legacy ``analyzetool`` formula (golden-master, no real trajectory needed),
plus analytic sanity cases. This satisfies the Phase 4 acceptance criterion —
"given a captured Trajectory array set, reproduce the legacy numbers."
"""

from __future__ import annotations

import numpy as np
import pytest

from mdforge.core.records import Trajectory
from mdforge.liquid import (
    compute_bulk_properties,
    stats,
    thermo,
    transport,
)
from mdforge.liquid.constants import (
    AMU_A3_TO_G_CM3,
    DIELECTRIC_PREFACTOR,
    KB_J,
    N_A_LEGACY,
    R_KCAL,
)

# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_bzavg_uniform_equals_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.ones(4)
        assert stats.bzavg(x, b) == pytest.approx(x.mean())

    def test_bzavg_weighted(self):
        x = np.array([1.0, 3.0])
        b = np.array([1.0, 3.0])
        # (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        assert stats.bzavg(x, b) == pytest.approx(2.5)

    def test_bzavg_2d(self):
        obs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3,2)
        b = np.ones(3)
        out = stats.bzavg(obs, b)
        assert np.allclose(out, [3.0, 4.0])  # column means

    def test_statistical_inefficiency_iid_near_one(self):
        rng = np.random.default_rng(0)
        g = stats.statistical_inefficiency(rng.standard_normal(5000))
        assert 1.0 <= g < 1.5

    def test_statistical_inefficiency_correlated_above_one(self):
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.standard_normal(5000))  # strongly autocorrelated
        assert stats.statistical_inefficiency(walk) > 5.0

    def test_statistical_inefficiency_constant(self):
        assert stats.statistical_inefficiency(np.full(100, 3.0)) == 1.0

    def test_legacy_alias(self):
        assert stats.statisticalInefficiency is stats.statistical_inefficiency

    def test_mean_stderr_empty(self):
        mean, err, flag = stats.mean_stderr(np.array([]))
        assert flag is True

    def test_bootstrap_error_reproducible(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(300)
        e1 = stats.bootstrap_error(lambda idx: float(data[idx].mean()), len(data), seed=42)
        e2 = stats.bootstrap_error(lambda idx: float(data[idx].mean()), len(data), seed=42)
        assert e1 == e2 and e1 > 0


# ---------------------------------------------------------------------------
# thermo  (golden-master vs inline legacy formula)
# ---------------------------------------------------------------------------

class TestThermo:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.T = 298.15
        self.vol = 8000.0 + 50.0 * rng.standard_normal(400)
        self.enth = -3000.0 + 20.0 * rng.standard_normal(400)
        self.dip = 5.0 * rng.standard_normal((400, 3))

    def test_density_matches_formula(self):
        mass = 18000.0
        out = thermo.density(self.vol, mass)
        ref = AMU_A3_TO_G_CM3 * mass / self.vol
        assert np.allclose(out, ref)

    def test_density_known_value(self):
        # 100 amu in 100 Å³ → 100/100 * 1.66054 ≈ 1.6605 g/cm³
        assert thermo.density(100.0, 100.0) == pytest.approx(AMU_A3_TO_G_CM3, rel=1e-9)

    def test_thermal_expansion_matches_legacy(self):
        h, v, T = self.enth, self.vol, self.T
        kT = R_KCAL * T
        ref = (1.0 / (kT * T)) * ((h * v).mean() - h.mean() * v.mean()) / v.mean()
        assert thermo.thermal_expansion(h, v, T) == pytest.approx(ref, rel=1e-12)

    def test_compressibility_matches_legacy(self):
        v, T = self.vol, self.T
        V0 = 1e-30 * v
        fluct = (V0 * V0).mean() - V0.mean() ** 2
        ref = 1e11 * fluct / (KB_J * T * V0.mean())
        assert thermo.isothermal_compressibility(v, T) == pytest.approx(ref, rel=1e-12)

    def test_compressibility_constant_volume_is_zero(self):
        v = np.full(100, 8000.0)
        assert thermo.isothermal_compressibility(v, self.T) == pytest.approx(0.0, abs=1e-6)

    def test_heat_capacity_matches_legacy(self):
        h, T, n = self.enth, self.T, 216
        kT = R_KCAL * T
        ref = 1000.0 * (1.0 / (n * kT * T)) * ((h ** 2).mean() - h.mean() ** 2)
        assert thermo.heat_capacity(h, n, T) == pytest.approx(ref, rel=1e-12)

    def test_dielectric_matches_legacy(self):
        d, v, T = self.dip, self.vol, self.T
        L = min(len(d), len(v))
        d, v = d[:L], v[:L]
        D2 = d[:, 0].var() + d[:, 1].var() + d[:, 2].var()
        ref = 1.0 + DIELECTRIC_PREFACTOR * D2 / v.mean() / T
        assert thermo.dielectric_constant(d, v, T, eps_inf=1.0) == pytest.approx(ref, rel=1e-10)

    def test_heat_of_vaporization(self):
        # ΔHvap = gas - liquid_per_mol + RT
        out = thermo.heat_of_vaporization(gas_pe=-5.0, liquid_pe_per_molecule=-15.0, temperature=300.0)
        assert out == pytest.approx(-5.0 - (-15.0) + R_KCAL * 300.0)

    def test_clausius_mossotti(self):
        out = thermo.clausius_mossotti_eps_inf(molpol=1.5, volume_per_molecule=30.0)
        ref = (-np.pi * 8 * 1.5 - 3 * 30.0) / (np.pi * 4 * 1.5 - 3 * 30.0)
        assert out == pytest.approx(ref)


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

class TestTransport:
    def test_pressure_tensor_single_atom(self):
        # 1 frame, 1 atom, m=1 amu, v=(1,0,0) Å/ps, zero virial, V=1000 Å³.
        virial = np.zeros((1, 3, 3))
        vel = np.array([[[1.0, 0.0, 0.0]]])
        masses = np.array([1.0])
        P = transport.pressure_tensor(virial, vel, masses, volume=1000.0)
        expected_00 = 10.0 * 1.0 / (N_A_LEGACY * 1e-30 * 1000.0)
        assert P.shape == (1, 3, 3)
        assert P[0, 0, 0] == pytest.approx(expected_00)
        assert P[0, 1, 1] == pytest.approx(0.0)

    def test_pressure_tensor_subtracts_virial(self):
        virial = np.full((1, 3, 3), 1.0)
        vel = np.zeros((1, 2, 3))
        masses = np.array([1.0, 1.0])
        P = transport.pressure_tensor(virial, vel, masses, volume=1000.0)
        expected = -4184.0 * 1.0 / (N_A_LEGACY * 1e-30 * 1000.0)
        assert P[0, 0, 0] == pytest.approx(expected)

    def test_viscosity_einstein_shape_and_finite(self):
        rng = np.random.default_rng(3)
        P = rng.standard_normal((500, 3, 3)) * 1e7
        visc = transport.viscosity_einstein(P, volume=8000.0, temperature=298.15, dt_ps=0.01)
        assert visc.shape == (499,)
        assert np.all(np.isfinite(visc))

    def test_viscosity_green_kubo_shape(self):
        rng = np.random.default_rng(4)
        P = rng.standard_normal((300, 3, 3)) * 1e7
        visc = transport.viscosity_green_kubo(P, volume=8000.0, temperature=298.15,
                                              dt_ps=0.01, max_lag=150)
        assert visc.shape == (149,)
        assert np.all(np.isfinite(visc))

    def test_einstein_matches_reference(self):
        from scipy.integrate import cumulative_trapezoid
        rng = np.random.default_rng(5)
        P = rng.standard_normal((200, 3, 3)) * 1e7
        T, vol, dt = 298.15, 8000.0, 0.01
        # inline reference
        shear = np.zeros((6, 200))
        shear[0], shear[1], shear[2] = P[:, 0, 1], P[:, 0, 2], P[:, 1, 2]
        shear[3] = (P[:, 0, 0] - P[:, 1, 1]) / 2
        shear[4] = (P[:, 1, 1] - P[:, 2, 2]) / 2
        dt_s = dt * 1e-12
        isq = np.zeros(200)
        for i in range(5):
            cum = cumulative_trapezoid(shear[i], dx=dt_s, initial=0.0)
            isq += cum ** 2 / 5.0
        time = np.arange(200) * dt_s
        ref = isq[1:] * (1e-30 * vol) / (2 * KB_J * T * time[1:])
        out = transport.viscosity_einstein(P, volume=vol, temperature=T, dt_ps=dt)
        assert np.allclose(out, ref)


# ---------------------------------------------------------------------------
# parse  (tiny synthetic fixtures)
# ---------------------------------------------------------------------------

class TestParse:
    def test_parse_analyze_log(self, tmp_path):
        from mdforge.liquid.parse import parse_analyze_log
        log = tmp_path / "analysis.log"
        log.write_text(
            " Total System Mass :    36.0306\n"
            " Total Potential Energy :    -100.0 Kcal/mole\n"
            " Dipole X,Y,Z-Components :    1.0 2.0 3.0\n"
            " Cell Volume :    8000.0\n"
            " Total Potential Energy :    -101.0 Kcal/mole\n"
            " Dipole X,Y,Z-Components :    1.1 2.1 3.1\n"
            " Cell Volume :    8010.0\n"
        )
        out = parse_analyze_log(log)
        assert out["mass"] == pytest.approx(36.0306)
        assert np.allclose(out["potential_energy"], [-100.0, -101.0])
        assert out["dipole"].shape == (2, 3)
        assert np.allclose(out["volume"], [8000.0, 8010.0])

    def test_parse_dynamics_log(self, tmp_path):
        from mdforge.liquid.parse import parse_dynamics_log
        log = tmp_path / "liquid.log"
        log.write_text(
            " Current Potential       -100.0 Kcal/mole\n"
            " Current Kinetic           50.0 Kcal/mole\n"
            " Lattice Lengths        20.0 20.0 20.0\n"
            " Current Potential       -101.0 Kcal/mole\n"
            " Current Kinetic           51.0 Kcal/mole\n"
            " Lattice Lengths        20.0 20.0 20.1\n"
        )
        out = parse_dynamics_log(log)
        assert np.allclose(out["potential_energy"], [-100.0, -101.0])
        assert np.allclose(out["kinetic_energy"], [50.0, 51.0])
        assert out["volume"][0] == pytest.approx(8000.0)

    def test_parse_box_xyz_pbc(self, tmp_path):
        from mdforge.liquid.parse import parse_box_xyz
        box = tmp_path / "liquid.xyz"
        box.write_text(
            "2  water\n"
            " 20.0 20.0 20.0 90.0 90.0 90.0\n"
            "  1  O   0.0 0.0 0.0  1\n"
            "  2  H   1.0 0.0 0.0  2\n"
        )
        out = parse_box_xyz(box)
        assert out["n_atoms"] == 2
        assert out["volume"] == pytest.approx(8000.0)
        assert out["masses"][0] == pytest.approx(15.9994)
        assert out["masses"][1] == pytest.approx(1.00794)

    def test_parse_velocity_dump(self, tmp_path):
        from mdforge.liquid.parse import parse_velocity_dump
        vel = tmp_path / "liquid.vel"
        vel.write_text(
            "2\n"
            "  1  O   0.1 0.2 0.3\n"
            "  2  H   0.4 0.5 0.6\n"
            "2\n"
            "  1  O   0.7 0.8 0.9\n"
            "  2  H   1.0 1.1 1.2\n"
        )
        out = parse_velocity_dump(vel, n_atoms=2)
        assert out.shape == (2, 2, 3)
        assert np.allclose(out[0, 0], [0.1, 0.2, 0.3])
        assert np.allclose(out[1, 1], [1.0, 1.1, 1.2])

    def test_parse_velocity_fortran_exponent(self, tmp_path):
        from mdforge.liquid.parse import parse_velocity_dump
        vel = tmp_path / "x.vel"
        vel.write_text("1\n  1  O   1.0D-01 2.0D-01 3.0D-01\n")
        out = parse_velocity_dump(vel, n_atoms=1)
        assert np.allclose(out[0, 0], [0.1, 0.2, 0.3])

    def test_parse_virial(self, tmp_path):
        from mdforge.liquid.parse import parse_virial
        log = tmp_path / "v.log"
        log.write_text(
            " Internal Virial Tensor :    1.0 2.0 3.0\n"
            "                             2.0 4.0 5.0\n"
            "                             3.0 5.0 6.0\n"
        )
        out = parse_virial(log)
        assert out.shape == (1, 3, 3)
        assert np.allclose(out[0], [[1, 2, 3], [2, 4, 5], [3, 5, 6]])

    def test_trajectory_from_tinker(self, tmp_path):
        from mdforge.liquid.parse import trajectory_from_tinker
        (tmp_path / "liquid.xyz").write_text(
            "2  water\n 20.0 20.0 20.0 90.0 90.0 90.0\n"
            "  1  O   0.0 0.0 0.0  1\n  2  H   1.0 0.0 0.0  2\n"
        )
        (tmp_path / "analysis.log").write_text(
            " Total System Mass :    18.0153\n"
            " Total Potential Energy :    -100.0 Kcal/mole\n"
            " Dipole X,Y,Z-Components :    1.0 2.0 3.0\n"
            " Cell Volume :    8000.0\n"
        )
        traj = trajectory_from_tinker(tmp_path, n_atoms_per_molecule=2)
        assert traj.n_atoms == 2
        assert traj.n_molecules == 1
        assert np.allclose(traj.potential_energy, [-100.0])
        assert traj.dipole.shape == (1, 3)
        assert traj.total_mass == pytest.approx(15.9994 + 1.00794)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

class TestComputeBulkProperties:
    def _make_traj(self, seed=11):
        rng = np.random.default_rng(seed)
        n = 500
        return Trajectory(
            potential_energy=-3000.0 + 20.0 * rng.standard_normal(n),
            kinetic_energy=1500.0 + 10.0 * rng.standard_normal(n),
            volume=8000.0 + 50.0 * rng.standard_normal(n),
            dipole=5.0 * rng.standard_normal((n, 3)),
            masses=np.full(216 * 3, 6.0),  # dummy masses, total set below
            n_molecules=216,
            temperature_K=298.15,
        )

    def test_full_properties(self):
        traj = self._make_traj()
        props = compute_bulk_properties(traj, equil=100, molpol=1.5,
                                        gas_pe_per_molecule=-10.0)
        assert props.density_kg_m3 is not None and props.density_kg_m3 > 0
        assert props.alpha_T is not None
        assert props.kappa_T is not None
        assert props.dielectric is not None and props.dielectric > 1.0
        assert props.delta_hvap_kcal_mol is not None
        assert "cp" in props.metadata

    def test_density_only_trajectory(self):
        # Only volume + masses → density computed, others None.
        traj = Trajectory(
            volume=np.full(100, 8000.0),
            masses=np.full(648, 6.0),
            n_molecules=216,
        )
        props = compute_bulk_properties(traj)
        assert props.density_kg_m3 is not None
        assert props.alpha_T is None
        assert props.dielectric is None

    def test_bootstrap_errors_attached(self):
        traj = self._make_traj()
        props = compute_bulk_properties(traj, equil=100, bootstrap=True, seed=0)
        assert "errors" in props.metadata
        assert props.metadata["errors"]["density_g_cm3"] > 0

    def test_equilibration_trim(self):
        traj = self._make_traj()
        full = compute_bulk_properties(traj, equil=0)
        trimmed = compute_bulk_properties(traj, equil=250)
        # Different equilibration windows give different density estimates.
        assert full.density_kg_m3 != trimmed.density_kg_m3


# ---------------------------------------------------------------------------
# plots  (skip if matplotlib absent)
# ---------------------------------------------------------------------------

class TestPlots:
    def test_plots_importable_and_run(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from mdforge.liquid import plots

        ax = plots.plot_running_average(np.random.default_rng(0).standard_normal(100),
                                        dt_ps=1.0, label="PE")
        assert ax is not None
        ax2 = plots.plot_property_vs_experiment([1.0, 2.0], [1.1, 1.9],
                                                property_name="density")
        assert ax2 is not None
