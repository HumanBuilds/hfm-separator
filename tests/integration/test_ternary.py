"""Integration tests for the ternary model case (Figure 13).

Equimolar feed with Q₁ : Q₂ : Q₃ = 500 : 100 : 10 GPU. The signature
finding is that the *intermediate* permeance component shows an interior
axial maximum in its residue mole fraction. The slowest permeance is
monotonically enriched, and the fastest permeance is monotonically
depleted.
"""

import numpy as np
import pytest

from hfm_separator.solvers.countercurrent import CountercurrentSolver


class TestTernaryFigure13:
    def _solve(self, module, components):
        return CountercurrentSolver(module=module, components=components).solve()

    def test_mass_balance(self, ternary_module, ternary_components) -> None:
        result = self._solve(ternary_module, ternary_components)
        assert (
            result.residue_flow_kmol_s + result.permeate_flow_kmol_s
            == pytest.approx(result.feed_flow_kmol_s, rel=1e-6)
        )
        np.testing.assert_allclose(result.residue_profiles.sum(axis=0), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.permeate_profiles.sum(axis=0), 1.0, atol=1e-6)

    def test_most_permeable_monotonically_depletes(
        self, ternary_module, ternary_components
    ) -> None:
        """Component 1 (fastest, 500 GPU) mole fraction must decrease
        monotonically from feed end (z/L=1) to residue end (z/L=0)."""
        result = self._solve(ternary_module, ternary_components)
        profile = result.residue_profiles[0, :]
        # z/L=0 is index 0 (residue end), z/L=1 is index -1 (feed end).
        # Expect monotone decrease as index decreases.
        diffs = np.diff(profile[::-1])  # feed → residue
        assert np.all(diffs <= 1e-9), (
            f"fast component not monotonically decreasing: {diffs[diffs > 0]}"
        )

    def test_least_permeable_monotonically_enriches(
        self, ternary_module, ternary_components
    ) -> None:
        """Component 3 (slowest, 10 GPU) mole fraction must increase
        monotonically from feed end to residue end."""
        result = self._solve(ternary_module, ternary_components)
        profile = result.residue_profiles[2, :]
        diffs = np.diff(profile[::-1])  # feed → residue
        assert np.all(diffs >= -1e-9), (
            f"slow component not monotonically increasing: {diffs[diffs < 0]}"
        )

    def test_intermediate_has_interior_axial_maximum(
        self, ternary_module, ternary_components
    ) -> None:
        """Component 2 (intermediate, 100 GPU) rises from feed end (as
        component 1 strips away), reaches an interior peak, then falls
        near the residue end (as component 2 itself becomes the most
        permeable remaining species)."""
        result = self._solve(ternary_module, ternary_components)
        profile = result.residue_profiles[1, :]
        peak_idx = int(np.argmax(profile))
        # Peak must be strictly interior — not at either endpoint.
        assert 0 < peak_idx < profile.shape[0] - 1
        assert profile[peak_idx] > profile[0]
        assert profile[peak_idx] > profile[-1]
        # Peak height is operating-point dependent. Qualitatively it must
        # exceed the feed mole fraction by a clear margin.
        assert profile[peak_idx] > 0.35
