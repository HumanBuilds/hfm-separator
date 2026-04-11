"""Tests for ``hfm_separator.numerics.pressure``."""

import math

import numpy as np
import pytest

from hfm_separator.numerics.constants import R_GAS_KMOL
from hfm_separator.numerics.pressure import (
    compressibility_group_a9,
    hagen_poiseuille_pressure_drop,
    permeability_group_a16,
)


def test_hagen_poiseuille_matches_manual_calculation() -> None:
    flow_total = 1.0e-3  # kmol/s across the bundle
    pressure = 1.0e6
    viscosity = 1.85e-5
    r_inner = 75e-6
    n_fibers = 300_000
    dz = 0.01
    temperature = 313.15

    expected = (
        (8.0 * viscosity / (math.pi * r_inner**4))
        * (flow_total / n_fibers)
        * (R_GAS_KMOL * temperature / pressure)
        * dz
    )

    observed = hagen_poiseuille_pressure_drop(
        flow_kmol_s=flow_total,
        upstream_pressure_pa=pressure,
        mixture_viscosity_pa_s=viscosity,
        fiber_inner_radius_m=r_inner,
        n_fibers=n_fibers,
        stage_length_m=dz,
        temperature_k=temperature,
    )

    assert observed == pytest.approx(expected, rel=1e-12)


def test_hagen_poiseuille_linear_in_flow() -> None:
    common = dict(
        upstream_pressure_pa=1.0e6,
        mixture_viscosity_pa_s=1.85e-5,
        fiber_inner_radius_m=75e-6,
        n_fibers=300_000,
        stage_length_m=0.01,
        temperature_k=313.15,
    )
    dp1 = hagen_poiseuille_pressure_drop(flow_kmol_s=1.0e-3, **common)
    dp2 = hagen_poiseuille_pressure_drop(flow_kmol_s=2.0e-3, **common)
    assert dp2 == pytest.approx(2.0 * dp1, rel=1e-12)


def test_hagen_poiseuille_rejects_zero_pressure() -> None:
    with pytest.raises(ValueError):
        hagen_poiseuille_pressure_drop(
            flow_kmol_s=1e-3,
            upstream_pressure_pa=0.0,
            mixture_viscosity_pa_s=1e-5,
            fiber_inner_radius_m=1e-4,
            n_fibers=1,
            stage_length_m=1.0,
            temperature_k=300.0,
        )


def test_hagen_poiseuille_rejects_negative_flow() -> None:
    with pytest.raises(ValueError):
        hagen_poiseuille_pressure_drop(
            flow_kmol_s=-1e-3,
            upstream_pressure_pa=1e6,
            mixture_viscosity_pa_s=1e-5,
            fiber_inner_radius_m=1e-4,
            n_fibers=1,
            stage_length_m=1.0,
            temperature_k=300.0,
        )


def test_compressibility_group_large_for_air_case() -> None:
    """Paper reports order ~200 for the air separation case (p. 1301)."""
    bore_area = math.pi * (75e-6) ** 2 * 300_000
    g = compressibility_group_a9(
        pressure_pa=1.0e6,
        temperature_k=313.15,
        molar_mass_kg_per_kmol=28.0,
        flow_kmol_s=1.0e-3,
        bore_area_m2=bore_area,
    )
    assert g > 50.0


def test_permeability_group_large_for_air_case() -> None:
    """Paper reports order ~1.4e7 for the air separation case (p. 1302)."""
    g = permeability_group_a16(
        viscosity_pa_s=1.85e-5,
        molar_flow_kmol_s=1.0e-3,
        fiber_inner_radius_m=75e-6,
        fiber_outer_radius_m=150e-6,
        molar_mass_kg_per_kmol=28.0,
        pressure_pa=1.0e6,
        permeance_si=3.346e-13 * 20.0,
    )
    assert g > 1.0e3


def test_pressure_drop_is_small_fraction_of_feed() -> None:
    """Sanity check that Eq. 21 gives a tiny drop across one stage for the
    paper's air case conditions."""
    dp = hagen_poiseuille_pressure_drop(
        flow_kmol_s=np.asarray(1.0e-3),
        upstream_pressure_pa=1.0e6,
        mixture_viscosity_pa_s=1.85e-5,
        fiber_inner_radius_m=75e-6,
        n_fibers=300_000,
        stage_length_m=1.0 / 100.0,
        temperature_k=313.15,
    )
    assert dp / 1.0e6 < 0.01
