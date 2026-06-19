"""Wilke mixing rule for low-pressure gas-mixture viscosity.

Follows ``docs/numerics.md``::

    μ_mix = Σ_i (y_i · μ_i) / Σ_j (y_j · φ_ij)

    φ_ij = [1 + (μ_i/μ_j)^0.5 · (M_j/M_i)^0.25]^2 / [√8 · (1 + M_i/M_j)^0.5]
"""

import math

import numpy as np

_SQRT_EIGHT: float = math.sqrt(8.0)


def wilke_mixture_viscosity(
    mole_fractions: np.ndarray,
    pure_viscosities: np.ndarray,
    molar_masses: np.ndarray,
) -> float:
    """Return the Wilke mixing-rule viscosity for a gas mixture.

    Parameters
    ----------
    mole_fractions : np.ndarray
        Component mole fractions, shape ``(R,)``. Need not sum to 1 exactly
        but should be non-negative.
    pure_viscosities : np.ndarray
        Pure-component viscosities in Pa·s, shape ``(R,)``.
    molar_masses : np.ndarray
        Molar masses in kg/kmol, shape ``(R,)``. The ratio ``M_i/M_j`` is
        what matters, so any consistent mass unit works.

    Returns
    -------
    float
        Mixture viscosity in Pa·s.
    """
    y = np.asarray(mole_fractions, dtype=float)
    mu = np.asarray(pure_viscosities, dtype=float)
    m = np.asarray(molar_masses, dtype=float)
    if not (y.shape == mu.shape == m.shape):
        raise ValueError(
            "mole_fractions, pure_viscosities, molar_masses must share shape"
        )
    if np.any(mu <= 0.0) or np.any(m <= 0.0):
        raise ValueError("pure_viscosities and molar_masses must be strictly positive")
    if np.any(y < 0.0) or y.sum() <= 0.0:
        # A degenerate composition has no defined mixture viscosity. Fail loudly
        # rather than silently returning a finite-but-meaningless number that
        # would then feed the pressure-drop correlation.
        raise ValueError("mole_fractions must be non-negative with a positive sum")

    # φ_ij matrix — shape (R, R), diagonal is 1.
    mu_ratio = mu[:, None] / mu[None, :]
    m_ratio = m[:, None] / m[None, :]
    numerator = (1.0 + np.sqrt(mu_ratio) * np.power(1.0 / m_ratio, 0.25)) ** 2
    denominator = _SQRT_EIGHT * np.sqrt(1.0 + m_ratio)
    phi = numerator / denominator

    # Row-sums Σ_j y_j · φ_ij are strictly positive: φ > 0 and Σ y > 0.
    row_sums = phi @ y
    return float(np.sum(y * mu / row_sums))
