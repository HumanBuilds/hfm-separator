"""Adaptive stage count from Eq. 22.

``docs/numerics.md`` §"Adaptive Stage Count"::

    N_j = 2π·R₀·L·Nf·(1 − xF_j + Δxmax)·Q_j·PF·xF_j / (F·Δxmax)

The number of stages is chosen per component; the module-wide ``N`` is the
maximum across components, rounded up to the next 100, with a floor of 100.
"""

import math

import numpy as np

_STAGES_CEIL_TO: int = 100
_MIN_STAGES: int = 100
_DEFAULT_DELTA_X_MAX: float = 0.005


def stages_per_component(
    fiber_outer_radius_m: float,
    active_length_m: float,
    n_fibers: int,
    feed_mole_fractions: np.ndarray,
    permeances_si: np.ndarray,
    feed_pressure_pa: float,
    feed_flow_kmol_s: float,
    delta_x_max: float = _DEFAULT_DELTA_X_MAX,
) -> np.ndarray:
    """Return the per-component stage count from Eq. 22 as a float array."""
    if delta_x_max <= 0.0:
        raise ValueError(f"delta_x_max must be positive, got {delta_x_max}")
    x_f = np.asarray(feed_mole_fractions, dtype=float)
    q = np.asarray(permeances_si, dtype=float)
    geom = 2.0 * math.pi * fiber_outer_radius_m * active_length_m * float(n_fibers)
    driver = q * feed_pressure_pa * x_f * (1.0 - x_f + delta_x_max)
    return geom * driver / (feed_flow_kmol_s * delta_x_max)


def choose_n_stages(
    fiber_outer_radius_m: float,
    active_length_m: float,
    n_fibers: int,
    feed_mole_fractions: np.ndarray,
    permeances_si: np.ndarray,
    feed_pressure_pa: float,
    feed_flow_kmol_s: float,
    delta_x_max: float = _DEFAULT_DELTA_X_MAX,
) -> int:
    """Return the module-wide ``N`` — per-component max, rounded up to 100."""
    per = stages_per_component(
        fiber_outer_radius_m=fiber_outer_radius_m,
        active_length_m=active_length_m,
        n_fibers=n_fibers,
        feed_mole_fractions=feed_mole_fractions,
        permeances_si=permeances_si,
        feed_pressure_pa=feed_pressure_pa,
        feed_flow_kmol_s=feed_flow_kmol_s,
        delta_x_max=delta_x_max,
    )
    raw = float(np.max(per))
    rounded = int(math.ceil(raw / _STAGES_CEIL_TO) * _STAGES_CEIL_TO)
    return max(_MIN_STAGES, rounded)
