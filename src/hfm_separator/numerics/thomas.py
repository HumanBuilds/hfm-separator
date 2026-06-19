"""Thomas algorithm for tridiagonal linear systems.

Implemented directly (not via ``scipy.linalg.solve_banded``) so the solver
is hot-path-friendly, easy to inspect in isolation, and testable against
analytic solutions for small systems. See ``docs/design.md`` §5.
"""

import numpy as np

_ZERO_PIVOT_EPS: float = 1.0e-300


def thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system ``A x = rhs`` using the Thomas algorithm.

    The system has the form::

        diag[0]  upper[0]     0          ...
        lower[1] diag[1]  upper[1]       ...
        0        lower[2] diag[2] ...
        ...

    Runs in ``O(N)`` time and space. Raises ``ValueError`` on an exact or
    numerically-zero pivot (a structurally singular system). This is a
    structural guard, not a conditioning estimate: a well-posed but
    ill-conditioned system is not rejected here.

    Parameters
    ----------
    lower : np.ndarray
        Sub-diagonal of length ``N``. ``lower[0]`` is ignored.
    diag : np.ndarray
        Main diagonal of length ``N``.
    upper : np.ndarray
        Super-diagonal of length ``N``. ``upper[-1]`` is ignored.
    rhs : np.ndarray
        Right-hand side vector of length ``N``.

    Returns
    -------
    np.ndarray
        Solution vector ``x`` with length ``N``.
    """
    n = diag.shape[0]
    if not (lower.shape == upper.shape == rhs.shape == (n,)):
        raise ValueError("lower, diag, upper, rhs must all have identical shape (N,)")
    if n == 0:
        return np.empty(0, dtype=float)

    c_prime = np.empty(n, dtype=float)
    d_prime = np.empty(n, dtype=float)

    if abs(diag[0]) < _ZERO_PIVOT_EPS:
        raise ValueError(f"zero pivot at row 0 (diag[0] = {diag[0]})")

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if abs(denom) < _ZERO_PIVOT_EPS:
            raise ValueError(f"zero pivot at row {i} (denominator = {denom})")
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x
