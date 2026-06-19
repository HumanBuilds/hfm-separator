"""Tests for the ``hfm_separator.models`` layer."""

import numpy as np
import pytest
from pydantic import ValidationError

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import (
    PermeanceFn,
    constant,
    from_gpu,
)
from hfm_separator.models.results import SimulationResults
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin, gpu_to_si


class TestPermeance:
    def test_constant_returns_supplied_value(self) -> None:
        q = constant(5.0e-12)
        x = np.array([0.5, 0.5])
        assert q(1.0e5, 298.0, x) == 5.0e-12

    def test_constant_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            constant(-1.0)

    def test_from_gpu_matches_manual_conversion(self) -> None:
        q = from_gpu(20.0)
        x = np.array([1.0])
        assert q(1.0e5, 298.0, x) == pytest.approx(gpu_to_si(20.0))

    def test_runtime_protocol_accepts_callable(self) -> None:
        q = constant(1.0e-13)
        assert isinstance(q, PermeanceFn)

    def test_custom_pressure_dependent_permeance(self) -> None:
        def q(p: float, t: float, x: np.ndarray) -> float:
            return gpu_to_si(60.0) * (1.0 + 0.01 * p / 1.0e5)

        assert q(2.0e5, 298.0, np.array([1.0])) == pytest.approx(gpu_to_si(60.0) * 1.02)


class TestComponentSpec:
    def _make(self, **kwargs: object) -> ComponentSpec:
        base: dict[str, object] = {
            "name": "N2",
            "feed_mole_fraction": 0.5,
            "permeance": from_gpu(3.57),
            "molar_mass_kg_per_kmol": 28.0134,
            "pure_viscosity_pa_s": 1.78e-5,
        }
        base.update(kwargs)
        return ComponentSpec(**base)  # type: ignore[arg-type]

    def test_constructs_cleanly(self) -> None:
        spec = self._make()
        assert spec.name == "N2"
        assert spec.feed_mole_fraction == 0.5

    def test_rejects_fraction_of_zero(self) -> None:
        with pytest.raises(ValidationError):
            self._make(feed_mole_fraction=0.0)

    def test_rejects_fraction_of_one(self) -> None:
        with pytest.raises(ValidationError):
            self._make(feed_mole_fraction=1.0)

    def test_rejects_negative_molar_mass(self) -> None:
        with pytest.raises(ValidationError):
            self._make(molar_mass_kg_per_kmol=-1.0)

    def test_rejects_zero_viscosity(self) -> None:
        with pytest.raises(ValidationError):
            self._make(pure_viscosity_pa_s=0.0)

    def test_rejects_blank_name(self) -> None:
        with pytest.raises(ValidationError):
            self._make(name="   ")

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_nonfinite_molar_mass(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._make(molar_mass_kg_per_kmol=bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_nonfinite_viscosity(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            self._make(pure_viscosity_pa_s=bad)

    def test_is_frozen(self) -> None:
        spec = self._make()
        with pytest.raises(ValidationError):
            spec.feed_mole_fraction = 0.9  # type: ignore[misc]


class TestModuleConfig:
    def _make(self, **kwargs: object) -> ModuleConfig:
        base: dict[str, object] = {
            "n_fibers": 300_000,
            "fiber_od_m": 300e-6,
            "fiber_id_m": 150e-6,
            "active_length_m": 1.0,
            "pot_length_m": 0.1,
            "feed_pressure_pa": bara_to_pa(10.0),
            "permeate_pressure_pa": bara_to_pa(1.0),
            "feed_temp_k": celsius_to_kelvin(40.0),
            "feed_flow_kmol_s": 1.0e-3,
            "feed_side": "bore",
        }
        base.update(kwargs)
        return ModuleConfig(**base)  # type: ignore[arg-type]

    def test_constructs_cleanly(self) -> None:
        cfg = self._make()
        assert cfg.pressure_ratio == pytest.approx(10.0)
        assert cfg.fiber_inner_radius_m == pytest.approx(75e-6)
        assert cfg.fiber_outer_radius_m == pytest.approx(150e-6)
        assert cfg.total_membrane_area_m2 == pytest.approx(
            np.pi * 300e-6 * 1.0 * 300_000
        )

    def test_rejects_permeate_above_feed(self) -> None:
        with pytest.raises(ValidationError):
            self._make(permeate_pressure_pa=bara_to_pa(11.0))

    def test_rejects_id_above_od(self) -> None:
        with pytest.raises(ValidationError):
            self._make(fiber_id_m=400e-6)

    def test_rejects_zero_fibers(self) -> None:
        with pytest.raises(ValidationError):
            self._make(n_fibers=0)

    def test_rejects_negative_pot_length(self) -> None:
        with pytest.raises(ValidationError):
            self._make(pot_length_m=-0.01)

    def test_rejects_purge_fraction_at_one(self) -> None:
        with pytest.raises(ValidationError):
            self._make(purge_fraction=1.0)

    def test_default_feed_side_is_shell(self) -> None:
        cfg = ModuleConfig(
            n_fibers=100,
            fiber_od_m=300e-6,
            fiber_id_m=150e-6,
            active_length_m=0.8,
            pot_length_m=0.1,
            feed_pressure_pa=1.0e6,
            permeate_pressure_pa=1.0e5,
            feed_temp_k=300.0,
            feed_flow_kmol_s=1.0e-3,
        )
        assert cfg.feed_side == "shell"


class TestSimulationResults:
    def _make(self) -> SimulationResults:
        n_components = 2
        n_stages = 4
        return SimulationResults(
            component_names=("A", "B"),
            feed_composition=np.array([0.5, 0.5]),
            residue_composition=np.array([0.7, 0.3]),
            permeate_composition=np.array([0.3, 0.7]),
            feed_flow_kmol_s=1.0,
            residue_flow_kmol_s=0.6,
            permeate_flow_kmol_s=0.4,
            stage_cut=0.4,
            residue_recovery=0.6,
            axial_positions=np.linspace(0.0, 1.0, n_stages),
            residue_profiles=np.zeros((n_components, n_stages)),
            permeate_profiles=np.zeros((n_components, n_stages)),
            feed_side_pressure_pa=np.full(n_stages, 1.0e6),
            permeate_side_pressure_pa=np.full(n_stages, 1.0e5),
            feed_side_flow_kmol_s=np.linspace(1.0, 0.6, n_stages),
            permeate_side_flow_kmol_s=np.linspace(0.0, 0.4, n_stages),
            n_iterations=12,
            n_stages=n_stages,
            converged=True,
            residuals=(1e-9, 1e-9),
            pattern="countercurrent",
        )

    def test_to_dataframe_columns(self) -> None:
        df = self._make().to_dataframe()
        assert list(df.columns) == [
            "z_over_L",
            "feed_pressure_pa",
            "permeate_pressure_pa",
            "feed_flow_kmol_s",
            "permeate_flow_kmol_s",
            "x_A",
            "y_A",
            "x_B",
            "y_B",
        ]

    def test_frozen(self) -> None:
        result = self._make()
        with pytest.raises(AttributeError):
            result.n_iterations = 99  # type: ignore[misc]
