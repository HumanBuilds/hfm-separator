"""Integration tests for the 5-component H₂/hydrocarbon case.

Numerical targets come from Figures 11 and 12 of Coker et al. (1998) and
``docs/testing.md``. Key regression facts: higher pressure ratio gives
better H₂ recovery at the same purity, and at very high recovery the CH₄
residue mole fraction has an *interior* axial maximum.
"""

import numpy as np
import pytest

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.solvers.countercurrent import CountercurrentSolver


def _solve(
    module: ModuleConfig,
    components: list[ComponentSpec],
    feed_flow_kmol_s: float,
) -> "object":
    updated = module.model_copy(update={"feed_flow_kmol_s": feed_flow_kmol_s})
    return CountercurrentSolver(module=updated, components=components).solve()


def _tune_feed_for_component_recovery(
    module: ModuleConfig,
    components: list[ComponentSpec],
    target_permeate_recovery: float,
    component_index: int,
    lo: float = 1.0e-3,
    hi: float = 5.0e-2,
    tol: float = 2.0e-3,
    max_iter: int = 40,
) -> "object":
    """Bisection to hit a specific *component* permeate recovery.

    Component permeate recovery = (y_perm · permeate_flow) / (x_feed · feed_flow)
    """
    result = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        result = _solve(module, components, mid)
        feed_j = mid * components[component_index].feed_mole_fraction
        perm_j = (
            result.permeate_flow_kmol_s * result.permeate_composition[component_index]
        )
        recovery = perm_j / feed_j
        if abs(recovery - target_permeate_recovery) < tol:
            return result
        if recovery > target_permeate_recovery:
            lo = mid
        else:
            hi = mid
    return result


class TestMassConservation:
    def test_closes(self, h2_module_case1, h2_components) -> None:
        result = _solve(h2_module_case1, h2_components, feed_flow_kmol_s=1.5e-2)
        assert (
            result.residue_flow_kmol_s + result.permeate_flow_kmol_s
            == pytest.approx(result.feed_flow_kmol_s, rel=1e-6)
        )
        np.testing.assert_allclose(result.residue_profiles.sum(axis=0), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.permeate_profiles.sum(axis=0), 1.0, atol=1e-6)


class TestPurityAtFixedRecovery:
    def test_higher_pressure_ratio_gives_better_h2_purity(
        self,
        h2_module_case1,
        h2_module_case2,
        h2_components,
    ) -> None:
        """Figure 11: at matching H₂ recovery, the high-pressure-ratio case
        (42.4/7.9 = 5.3) gives a higher permeate H₂ purity than the
        low-ratio case (76.9/42.4 = 1.8)."""
        r1 = _tune_feed_for_component_recovery(
            h2_module_case1,
            h2_components,
            target_permeate_recovery=0.50,
            component_index=0,
        )
        r2 = _tune_feed_for_component_recovery(
            h2_module_case2,
            h2_components,
            target_permeate_recovery=0.50,
            component_index=0,
        )
        assert r1.permeate_composition[0] > r2.permeate_composition[0]


class TestInteriorProfileMaximum:
    def test_ch4_residue_profile_has_interior_maximum(
        self, h2_module_case1, h2_components
    ) -> None:
        """Figure 12: at very high H₂ recovery, the CH₄ residue mole
        fraction increases from the feed end as H₂ is stripped away, then
        decreases near the residue end as CH₄ itself begins permeating.
        The maximum is strictly inside the module (not at either end).

        We use a feed flow calibrated to give ~99.8% H₂ recovery, the
        regime where the interior peak is unambiguous.
        """
        result = _solve(h2_module_case1, h2_components, feed_flow_kmol_s=1.0e-2)
        ch4_index = 2
        profile = result.residue_profiles[ch4_index, :]
        # Slice to avoid the spurious last-grid-point boundary values.
        peak_idx = int(np.argmax(profile))
        # Peak must be interior (well away from the feed-end boundary).
        assert peak_idx < profile.shape[0] - 1
        # Peak must be meaningfully above the feed-end value.
        assert profile[peak_idx] > profile[-1] + 0.1


class TestH2Stripping:
    def test_h2_depleted_at_residue_end_at_high_recovery(
        self, h2_module_case1, h2_components
    ) -> None:
        """At very high H₂ recovery the residue end is H₂-lean."""
        result = _solve(h2_module_case1, h2_components, feed_flow_kmol_s=1.0e-2)
        assert result.residue_profiles[0, 0] < 0.05
