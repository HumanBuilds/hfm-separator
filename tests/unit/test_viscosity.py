"""Tests for ``hfm_separator.numerics.viscosity``."""

import numpy as np
import pytest

from hfm_separator.numerics.viscosity import wilke_mixture_viscosity


def test_pure_component() -> None:
    mu = wilke_mixture_viscosity(
        mole_fractions=np.array([1.0]),
        pure_viscosities=np.array([1.789e-5]),
        molar_masses=np.array([28.0134]),
    )
    assert mu == pytest.approx(1.789e-5)


def test_limit_to_component_one() -> None:
    mu = wilke_mixture_viscosity(
        mole_fractions=np.array([1.0 - 1e-9, 1e-9]),
        pure_viscosities=np.array([1.789e-5, 2.031e-5]),
        molar_masses=np.array([28.0134, 31.9988]),
    )
    assert mu == pytest.approx(1.789e-5, rel=1e-6)


def test_limit_to_component_two() -> None:
    mu = wilke_mixture_viscosity(
        mole_fractions=np.array([1e-9, 1.0 - 1e-9]),
        pure_viscosities=np.array([1.789e-5, 2.031e-5]),
        molar_masses=np.array([28.0134, 31.9988]),
    )
    assert mu == pytest.approx(2.031e-5, rel=1e-6)


def test_equal_binary_n2_o2_at_298k() -> None:
    """Reference: 50/50 N₂/O₂ mixture at 298 K, expected ~1.91e-5 Pa·s."""
    mu = wilke_mixture_viscosity(
        mole_fractions=np.array([0.5, 0.5]),
        pure_viscosities=np.array([1.789e-5, 2.031e-5]),
        molar_masses=np.array([28.0134, 31.9988]),
    )
    assert 1.87e-5 < mu < 1.95e-5  # ±2% envelope around 1.91e-5


def test_wilke_between_pure_values() -> None:
    pure = np.array([1.789e-5, 2.031e-5])
    mu_mix = wilke_mixture_viscosity(
        mole_fractions=np.array([0.3, 0.7]),
        pure_viscosities=pure,
        molar_masses=np.array([28.0134, 31.9988]),
    )
    assert pure.min() < mu_mix < pure.max()


def test_rejects_nonpositive_viscosity() -> None:
    with pytest.raises(ValueError):
        wilke_mixture_viscosity(
            mole_fractions=np.array([0.5, 0.5]),
            pure_viscosities=np.array([1.0e-5, 0.0]),
            molar_masses=np.array([28.0, 32.0]),
        )


def test_rejects_degenerate_composition() -> None:
    """An all-zero composition has no defined mixture viscosity."""
    with pytest.raises(ValueError, match="non-negative with a positive sum"):
        wilke_mixture_viscosity(
            mole_fractions=np.array([0.0, 0.0]),
            pure_viscosities=np.array([1.789e-5, 2.031e-5]),
            molar_masses=np.array([28.0134, 31.9988]),
        )


def test_rejects_negative_mole_fraction() -> None:
    with pytest.raises(ValueError, match="non-negative with a positive sum"):
        wilke_mixture_viscosity(
            mole_fractions=np.array([-0.1, 1.1]),
            pure_viscosities=np.array([1.789e-5, 2.031e-5]),
            molar_masses=np.array([28.0134, 31.9988]),
        )


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        wilke_mixture_viscosity(
            mole_fractions=np.array([0.5, 0.5]),
            pure_viscosities=np.array([1.0e-5]),
            molar_masses=np.array([28.0, 32.0]),
        )
