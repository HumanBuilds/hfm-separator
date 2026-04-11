"""Callable permeance interface and factory helpers.

The solver layer *always* treats permeance as a callable so the simple case
(constant permeance) and the complex case (pressure/temperature/composition
dependence) share one call site. See ``docs/design.md`` §1.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PermeanceFn(Protocol):
    """Callable returning a component permeance at local conditions.

    Parameters
    ----------
    pressure_pa : float
        Local feed-side (high-pressure) value in Pa.
    temp_k : float
        Local temperature in K.
    mole_fractions : np.ndarray
        Local feed-side mole fractions with shape ``(n_components,)``.

    Returns
    -------
    float
        Permeance in SI: ``kmol / (m² · s · Pa)``.
    """

    def __call__(
        self,
        pressure_pa: float,
        temp_k: float,
        mole_fractions: np.ndarray,
    ) -> float: ...


def constant(permeance_si: float) -> PermeanceFn:
    """Return a ``PermeanceFn`` that ignores inputs and returns ``permeance_si``."""
    if permeance_si < 0.0:
        raise ValueError(f"permeance must be non-negative, got {permeance_si}")

    def _fn(
        pressure_pa: float,
        temp_k: float,
        mole_fractions: np.ndarray,
    ) -> float:
        del pressure_pa, temp_k, mole_fractions
        return permeance_si

    return _fn


def from_gpu(gpu_value: float) -> PermeanceFn:
    """Return a constant ``PermeanceFn`` from a permeance expressed in GPU."""
    from hfm_separator.utils.unit_conv import gpu_to_si

    return constant(gpu_to_si(gpu_value))
