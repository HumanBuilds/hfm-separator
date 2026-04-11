"""Unit tests for the cross-flow solver."""

import numpy as np
import pytest

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.solvers.crossflow import (
    CrossflowSolver,
    solve_eq18_permeate_composition,
)
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin, gpu_to_si


def _air_module() -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=2.0e-3,
        feed_side="bore",
    )


def _air_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            name="N2",
            feed_mole_fraction=0.7841,
            permeance=from_gpu(3.57),
            molar_mass_kg_per_kmol=28.0134,
            pure_viscosity_pa_s=1.85e-5,
        ),
        ComponentSpec(
            name="O2",
            feed_mole_fraction=0.2084,
            permeance=from_gpu(20.0),
            molar_mass_kg_per_kmol=31.9988,
            pure_viscosity_pa_s=2.08e-5,
        ),
        ComponentSpec(
            name="CO2",
            feed_mole_fraction=0.0003,
            permeance=from_gpu(60.0),
            molar_mass_kg_per_kmol=44.01,
            pure_viscosity_pa_s=1.55e-5,
        ),
        ComponentSpec(
            name="H2O",
            feed_mole_fraction=0.0072,
            permeance=from_gpu(1000.0),
            molar_mass_kg_per_kmol=18.0153,
            pure_viscosity_pa_s=1.01e-5,
        ),
    ]


class TestEq18Solver:
    def test_pure_component_degenerate(self) -> None:
        x = np.array([1.0])
        q = np.array([gpu_to_si(20.0)])
        y = solve_eq18_permeate_composition(
            feed_composition=x,
            permeances=q,
            feed_pressure_pa=1.0e6,
            permeate_pressure_pa=1.0e5,
        )
        assert y[0] == pytest.approx(1.0, abs=1e-9)

    def test_binary_o2_enriched(self) -> None:
        """O₂ has higher permeance than N₂ → permeate must be O₂-enriched."""
        x = np.array([0.79, 0.21])
        q = np.array([gpu_to_si(3.57), gpu_to_si(20.0)])
        y = solve_eq18_permeate_composition(
            feed_composition=x,
            permeances=q,
            feed_pressure_pa=1.0e6,
            permeate_pressure_pa=1.0e5,
        )
        assert sum(y) == pytest.approx(1.0, abs=1e-10)
        assert y[1] > x[1]
        assert y[0] < x[0]

    def test_sum_to_one(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.uniform(0.05, 0.95, size=5)
        x /= x.sum()
        q = rng.uniform(0.1, 100.0, size=5) * gpu_to_si(1.0)
        y = solve_eq18_permeate_composition(
            feed_composition=x,
            permeances=q,
            feed_pressure_pa=2.0e6,
            permeate_pressure_pa=2.0e5,
        )
        assert y.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)


class TestCrossflowSolver:
    def test_feeds_validate(self) -> None:
        components = _air_components()
        CrossflowSolver(module=_air_module(), components=components)

    def test_mass_conservation(self) -> None:
        solver = CrossflowSolver(module=_air_module(), components=_air_components())
        result = solver.solve()
        assert (
            result.residue_flow_kmol_s + result.permeate_flow_kmol_s
            == pytest.approx(result.feed_flow_kmol_s, rel=1e-6)
        )
        assert result.residue_composition.sum() == pytest.approx(1.0, abs=1e-8)
        assert result.permeate_composition.sum() == pytest.approx(1.0, abs=1e-8)

    def test_fast_components_enriched_in_permeate(self) -> None:
        solver = CrossflowSolver(module=_air_module(), components=_air_components())
        result = solver.solve()
        # H2O (1000 GPU) and O2 (20 GPU) should be enriched in permeate.
        assert result.permeate_composition[1] > result.feed_composition[1]  # O2
        assert result.permeate_composition[3] > result.feed_composition[3]  # H2O
        # N2 (3.57 GPU) should be enriched in residue.
        assert result.residue_composition[0] > result.feed_composition[0]

    def test_converged_flag_and_pattern(self) -> None:
        solver = CrossflowSolver(module=_air_module(), components=_air_components())
        result = solver.solve()
        assert result.converged is True
        assert result.pattern == "crossflow"
        assert result.n_stages >= 100

    def test_rejects_bad_feed_sum(self) -> None:
        comps = _air_components()
        comps[0] = ComponentSpec(
            name=comps[0].name,
            feed_mole_fraction=0.5,
            permeance=comps[0].permeance,
            molar_mass_kg_per_kmol=comps[0].molar_mass_kg_per_kmol,
            pure_viscosity_pa_s=comps[0].pure_viscosity_pa_s,
        )
        with pytest.raises(ValueError, match="sum to 1"):
            CrossflowSolver(module=_air_module(), components=comps)
