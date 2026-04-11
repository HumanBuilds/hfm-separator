"""Tests for ``hfm_separator.utils.unit_conv``."""

import math

import pytest

from hfm_separator.utils.unit_conv import (
    GPU_TO_SI,
    STP_MOLAR_VOLUME_M3_PER_KMOL,
    bara_to_pa,
    celsius_to_kelvin,
    gpu_to_si,
    kelvin_to_celsius,
    kmol_s_to_stp_m3h,
    pa_to_bara,
    si_to_gpu,
    stp_m3h_to_kmol_s,
)


class TestPermeance:
    def test_one_gpu_matches_reference(self) -> None:
        assert gpu_to_si(1.0) == pytest.approx(3.346e-13, rel=1e-12)

    def test_zero_gpu_is_zero(self) -> None:
        assert gpu_to_si(0.0) == 0.0

    def test_linear_in_gpu(self) -> None:
        assert gpu_to_si(100.0) == pytest.approx(100.0 * GPU_TO_SI)

    def test_round_trip(self) -> None:
        assert si_to_gpu(gpu_to_si(20.0)) == pytest.approx(20.0)


class TestPressure:
    def test_one_bara_is_1e5_pa(self) -> None:
        assert bara_to_pa(1.0) == 1.0e5

    def test_ten_bara_is_1e6_pa(self) -> None:
        assert bara_to_pa(10.0) == 1.0e6

    def test_round_trip(self) -> None:
        assert pa_to_bara(bara_to_pa(42.0)) == pytest.approx(42.0)


class TestTemperature:
    def test_zero_celsius_is_273p15_kelvin(self) -> None:
        assert celsius_to_kelvin(0.0) == 273.15

    def test_forty_celsius(self) -> None:
        assert celsius_to_kelvin(40.0) == pytest.approx(313.15)

    def test_round_trip(self) -> None:
        assert kelvin_to_celsius(celsius_to_kelvin(25.0)) == pytest.approx(25.0)


class TestFlowRate:
    def test_one_molar_volume_per_hour_is_one_kmol_per_hour(self) -> None:
        result = stp_m3h_to_kmol_s(STP_MOLAR_VOLUME_M3_PER_KMOL)
        assert result == pytest.approx(1.0 / 3600.0, rel=1e-12)

    def test_round_trip(self) -> None:
        assert kmol_s_to_stp_m3h(stp_m3h_to_kmol_s(123.456)) == pytest.approx(123.456)

    def test_zero_flow(self) -> None:
        assert stp_m3h_to_kmol_s(0.0) == 0.0


def test_constants_are_finite() -> None:
    assert math.isfinite(GPU_TO_SI)
    assert math.isfinite(STP_MOLAR_VOLUME_M3_PER_KMOL)
    assert GPU_TO_SI > 0
    assert STP_MOLAR_VOLUME_M3_PER_KMOL > 0
