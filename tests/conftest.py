"""Shared fixtures for integration tests.

Module parameters come from Table 1 of Coker et al. (1998). Permeances
come from Tables 2-4. Pure-component viscosities at module temperature
are standard engineering values (NIST / Perry's) within the tolerance
of the Wilke mixing rule.
"""

import pytest

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin


@pytest.fixture
def air_module_10bar() -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=1.0e-3,
        feed_side="bore",
        purge_fraction=0.0,
    )


@pytest.fixture
def air_module_5bar() -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(5.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=1.0e-3,
        feed_side="bore",
        purge_fraction=0.0,
    )


@pytest.fixture
def air_components() -> list[ComponentSpec]:
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


@pytest.fixture
def ternary_module() -> ModuleConfig:
    # Feed flow: paper p.10 says "283.2 m³(STP)/h (10,000 SCFH)" for Fig 13.
    paper_feed_kmol_s = 283.2 / 22.414 / 3600.0  # ≈ 3.51e-3
    return ModuleConfig(
        n_fibers=350_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(25.0),
        feed_flow_kmol_s=paper_feed_kmol_s,
        feed_side="shell",
        purge_fraction=0.0,
    )


@pytest.fixture
def ternary_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            name="fast",
            feed_mole_fraction=0.3333,
            permeance=from_gpu(500.0),
            molar_mass_kg_per_kmol=30.0,
            pure_viscosity_pa_s=1.5e-5,
        ),
        ComponentSpec(
            name="mid",
            feed_mole_fraction=0.3333,
            permeance=from_gpu(100.0),
            molar_mass_kg_per_kmol=30.0,
            pure_viscosity_pa_s=1.5e-5,
        ),
        ComponentSpec(
            name="slow",
            feed_mole_fraction=0.3334,
            permeance=from_gpu(10.0),
            molar_mass_kg_per_kmol=30.0,
            pure_viscosity_pa_s=1.5e-5,
        ),
    ]


@pytest.fixture
def h2_module_case1() -> ModuleConfig:
    """High pressure ratio: 42.4/7.9 bara ≈ 5.3."""
    return ModuleConfig(
        n_fibers=500_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(42.4),
        permeate_pressure_pa=bara_to_pa(7.9),
        feed_temp_k=celsius_to_kelvin(50.0),
        feed_flow_kmol_s=2.5e-3,
        feed_side="shell",
        purge_fraction=0.0,
    )


@pytest.fixture
def h2_module_case2() -> ModuleConfig:
    """Low pressure ratio: 76.9/42.4 bara ≈ 1.8."""
    return ModuleConfig(
        n_fibers=500_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(76.9),
        permeate_pressure_pa=bara_to_pa(42.4),
        feed_temp_k=celsius_to_kelvin(50.0),
        feed_flow_kmol_s=2.5e-3,
        feed_side="shell",
        purge_fraction=0.0,
    )


@pytest.fixture
def h2_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            name="H2",
            feed_mole_fraction=0.650,
            permeance=from_gpu(100.0),
            molar_mass_kg_per_kmol=2.016,
            pure_viscosity_pa_s=0.93e-5,
        ),
        ComponentSpec(
            name="C2H4",
            feed_mole_fraction=0.025,
            permeance=from_gpu(3.03),
            molar_mass_kg_per_kmol=28.054,
            pure_viscosity_pa_s=1.07e-5,
        ),
        ComponentSpec(
            name="CH4",
            feed_mole_fraction=0.210,
            permeance=from_gpu(2.86),
            molar_mass_kg_per_kmol=16.043,
            pure_viscosity_pa_s=1.17e-5,
        ),
        ComponentSpec(
            name="C2H6",
            feed_mole_fraction=0.080,
            permeance=from_gpu(2.00),
            molar_mass_kg_per_kmol=30.07,
            pure_viscosity_pa_s=1.02e-5,
        ),
        ComponentSpec(
            name="C3H8",
            feed_mole_fraction=0.035,
            permeance=from_gpu(1.89),
            molar_mass_kg_per_kmol=44.097,
            pure_viscosity_pa_s=0.90e-5,
        ),
    ]
