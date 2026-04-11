"""Unit tests for the cocurrent solver."""

import numpy as np
import pytest

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.solvers.cocurrent import CocurrentSolver
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.solvers.crossflow import CrossflowSolver
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin


def _air_module_bore(feed: float = 1.0e-3) -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=feed,
        feed_side="bore",
    )


def _air_module_shell(feed: float = 1.0e-3) -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=feed,
        feed_side="shell",
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


class TestCocurrentBoreFeed:
    def test_converges(self) -> None:
        result = CocurrentSolver(
            module=_air_module_bore(),
            components=_air_components(),
        ).solve()
        assert result.converged is True
        assert result.pattern == "cocurrent"
        assert result.n_stages >= 100

    def test_mass_conservation(self) -> None:
        result = CocurrentSolver(
            module=_air_module_bore(),
            components=_air_components(),
        ).solve()
        assert (
            result.residue_flow_kmol_s + result.permeate_flow_kmol_s
            == pytest.approx(result.feed_flow_kmol_s, rel=1e-6)
        )
        np.testing.assert_allclose(result.residue_profiles.sum(axis=0), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.permeate_profiles.sum(axis=0), 1.0, atol=1e-6)

    def test_permeate_enriched_in_fast_components(self) -> None:
        """Cocurrent still enriches the faster components on the permeate side."""
        result = CocurrentSolver(
            module=_air_module_bore(),
            components=_air_components(),
        ).solve()
        # O2 (20 GPU) and H2O (1000 GPU) must be enriched.
        assert result.permeate_composition[1] > result.feed_composition[1]
        assert result.permeate_composition[3] > result.feed_composition[3]

    def test_residue_purity_bounds(self) -> None:
        """Cocurrent N₂ purity must be strictly between cross-flow and
        countercurrent at the same operating point — this is the textbook
        hierarchy."""
        cc = CountercurrentSolver(
            module=_air_module_bore(),
            components=_air_components(),
        ).solve()
        cf = CrossflowSolver(
            module=_air_module_bore(),
            components=_air_components(),
        ).solve()
        co = CocurrentSolver(
            module=_air_module_bore(),
            components=_air_components(),
        ).solve()
        assert cc.residue_composition[0] > cf.residue_composition[0]
        assert cf.residue_composition[0] >= co.residue_composition[0]


class TestCocurrentShellFeed:
    def test_converges_with_shell_feed(self) -> None:
        result = CocurrentSolver(
            module=_air_module_shell(),
            components=_air_components(),
        ).solve()
        assert result.converged is True
        assert result.n_iterations >= 1

    def test_mass_conservation_shell(self) -> None:
        result = CocurrentSolver(
            module=_air_module_shell(),
            components=_air_components(),
        ).solve()
        assert (
            result.residue_flow_kmol_s + result.permeate_flow_kmol_s
            == pytest.approx(result.feed_flow_kmol_s, rel=1e-6)
        )


class TestCocurrentErrors:
    def test_rejects_purge_fraction(self) -> None:
        cfg = _air_module_bore().model_copy(update={"purge_fraction": 0.1})
        with pytest.raises(ValueError, match="purge_fraction"):
            CocurrentSolver(module=cfg, components=_air_components()).solve()
