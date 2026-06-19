"""Tests for ``hfm_separator.numerics.staging``."""

import numpy as np
import pytest

from hfm_separator.numerics.staging import choose_n_stages, stages_per_component
from hfm_separator.utils.unit_conv import bara_to_pa, gpu_to_si


def _air_args() -> dict[str, object]:
    return {
        "fiber_outer_radius_m": 150e-6,
        "active_length_m": 1.0,
        "n_fibers": 300_000,
        "feed_mole_fractions": np.array([0.7841, 0.2084, 0.0003, 0.0072]),
        "permeances_si": np.array(
            [
                gpu_to_si(3.57),
                gpu_to_si(20.0),
                gpu_to_si(60.0),
                gpu_to_si(1000.0),
            ]
        ),
        "feed_pressure_pa": bara_to_pa(10.0),
        "feed_flow_kmol_s": 1.0e-3,
    }


def test_air_minimum_stages_met() -> None:
    n = choose_n_stages(**_air_args())  # type: ignore[arg-type]
    assert n >= 100


def test_rounding_to_next_100() -> None:
    n = choose_n_stages(**_air_args())  # type: ignore[arg-type]
    assert n % 100 == 0


def test_delta_x_max_halved_approximately_doubles_stages() -> None:
    """Eq. 22 is dominated by ``(1 - xF)/Δxmax`` so halving ``Δxmax``
    approximately (not exactly) doubles ``N`` — the ``Δxmax`` term in the
    numerator drops out in the limit ``Δxmax → 0``."""
    args = _air_args()
    per_default = stages_per_component(**args, delta_x_max=0.005)  # type: ignore[arg-type]
    per_half = stages_per_component(**args, delta_x_max=0.0025)  # type: ignore[arg-type]
    ratio = per_half / per_default
    assert np.allclose(ratio, 2.0, rtol=0.03)


def test_highest_permeance_sets_stage_count() -> None:
    """H2O at 1000 GPU should dominate the air mixture."""
    args = _air_args()
    per = stages_per_component(**args)  # type: ignore[arg-type]
    # Component 3 is H2O (index 3).
    assert np.argmax(per) == 3


def test_default_floor_is_100() -> None:
    """Tiny module → formula gives a value below 100 → clamped to 100."""
    n = choose_n_stages(
        fiber_outer_radius_m=1e-4,
        active_length_m=0.01,
        n_fibers=10,
        feed_mole_fractions=np.array([0.5, 0.5]),
        permeances_si=np.array([gpu_to_si(1.0), gpu_to_si(1.0)]),
        feed_pressure_pa=bara_to_pa(2.0),
        feed_flow_kmol_s=1.0e-3,
    )
    assert n == 100


@pytest.mark.parametrize("bad_delta", [0.0, -0.005])
def test_nonpositive_delta_x_max_raises(bad_delta: float) -> None:
    with pytest.raises(ValueError, match="delta_x_max must be positive"):
        stages_per_component(**_air_args(), delta_x_max=bad_delta)  # type: ignore[arg-type]


@pytest.mark.parametrize("delta", [0.002, 0.005, 0.01])
def test_formula_scales_linearly_in_geometry(delta: float) -> None:
    args = _air_args()
    per_small = stages_per_component(**args, delta_x_max=delta)  # type: ignore[arg-type]
    args_big = {**args, "n_fibers": 600_000}
    per_big = stages_per_component(**args_big, delta_x_max=delta)  # type: ignore[arg-type]
    np.testing.assert_allclose(per_big / per_small, 2.0, rtol=1e-6)
