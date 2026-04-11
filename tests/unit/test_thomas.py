"""Tests for ``hfm_separator.numerics.thomas``."""

import numpy as np
import pytest

from hfm_separator.numerics.thomas import thomas_solve


def test_trivial_2x2() -> None:
    """[[2,1],[1,2]] x = [3,3] → x = [1,1]."""
    lower = np.array([0.0, 1.0])
    diag = np.array([2.0, 2.0])
    upper = np.array([1.0, 0.0])
    rhs = np.array([3.0, 3.0])
    x = thomas_solve(lower, diag, upper, rhs)
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-12)


def test_5x5_known_solution() -> None:
    """Construct from a known solution vector and verify round-trip."""
    n = 5
    rng = np.random.default_rng(42)
    lower = np.concatenate(([0.0], rng.uniform(-1.0, 1.0, size=n - 1)))
    upper = np.concatenate((rng.uniform(-1.0, 1.0, size=n - 1), [0.0]))
    diag = np.abs(lower) + np.abs(upper) + 2.0  # diagonally dominant
    x_true = rng.uniform(-5.0, 5.0, size=n)

    rhs = np.zeros(n)
    rhs[0] = diag[0] * x_true[0] + upper[0] * x_true[1]
    for i in range(1, n - 1):
        rhs[i] = (
            lower[i] * x_true[i - 1] + diag[i] * x_true[i] + upper[i] * x_true[i + 1]
        )
    rhs[-1] = lower[-1] * x_true[-2] + diag[-1] * x_true[-1]

    x = thomas_solve(lower, diag, upper, rhs)
    np.testing.assert_allclose(x, x_true, atol=1e-12)


def test_large_random_vs_numpy_solve() -> None:
    """Random diagonally dominant tridiagonal compared to np.linalg.solve."""
    n = 200
    rng = np.random.default_rng(7)
    lower = np.concatenate(([0.0], rng.uniform(-1.0, 1.0, size=n - 1)))
    upper = np.concatenate((rng.uniform(-1.0, 1.0, size=n - 1), [0.0]))
    diag = np.abs(lower) + np.abs(upper) + 5.0
    rhs = rng.uniform(-10.0, 10.0, size=n)

    dense = np.zeros((n, n))
    for i in range(n):
        dense[i, i] = diag[i]
        if i > 0:
            dense[i, i - 1] = lower[i]
        if i < n - 1:
            dense[i, i + 1] = upper[i]

    x_expected = np.linalg.solve(dense, rhs)
    x = thomas_solve(lower, diag, upper, rhs)
    np.testing.assert_allclose(x, x_expected, atol=1e-10)


def test_zero_pivot_raises() -> None:
    lower = np.array([0.0, 1.0])
    diag = np.array([0.0, 2.0])
    upper = np.array([1.0, 0.0])
    rhs = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="zero pivot"):
        thomas_solve(lower, diag, upper, rhs)


def test_propagating_zero_pivot_raises() -> None:
    """A singular system where the pivot becomes zero after elimination."""
    lower = np.array([0.0, 2.0])
    diag = np.array([1.0, 2.0])
    upper = np.array([1.0, 0.0])
    rhs = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="zero pivot"):
        thomas_solve(lower, diag, upper, rhs)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        thomas_solve(
            np.zeros(3),
            np.ones(4),
            np.zeros(3),
            np.zeros(3),
        )


def test_empty_system_returns_empty() -> None:
    out = thomas_solve(np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    assert out.shape == (0,)
