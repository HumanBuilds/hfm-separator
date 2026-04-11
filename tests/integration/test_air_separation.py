"""Integration tests for the 4-component air separation case.

Targets are read from Figures 4, 5, 6, 8, 10 of Coker et al. (1998) and
encoded per ``docs/testing.md``. These are regression tests against the
published paper — qualitative (signs and monotonicity) where quantitative
tolerances are tight, quantitative where the paper gives a specific
numerical value.
"""

import numpy as np
import pytest

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.solvers.crossflow import CrossflowSolver


def _solve_with_feed(
    module: ModuleConfig,
    components: list[ComponentSpec],
    feed_flow_kmol_s: float,
    purge_fraction: float = 0.0,
) -> "object":
    updated = module.model_copy(
        update={
            "feed_flow_kmol_s": feed_flow_kmol_s,
            "purge_fraction": purge_fraction,
        }
    )
    return CountercurrentSolver(module=updated, components=components).solve()


def _tune_feed_for_recovery(
    module: ModuleConfig,
    components: list[ComponentSpec],
    target_recovery: float,
    purge_fraction: float = 0.0,
    tol: float = 1.0e-4,
    max_iter: int = 40,
) -> "object":
    """Bisection search on feed flow to hit a specified residue recovery."""
    lo, hi = 5.0e-5, 1.0e-2
    # Ensure bracketing: lower feed → more permeation per unit → lower recovery
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        result = _solve_with_feed(module, components, mid, purge_fraction)
        if abs(result.residue_recovery - target_recovery) < tol:
            return result
        if result.residue_recovery < target_recovery:
            lo = mid
        else:
            hi = mid
    return result


class TestMassConservation:
    def test_no_purge(self, air_module_10bar, air_components) -> None:
        result = CountercurrentSolver(
            module=air_module_10bar, components=air_components
        ).solve()
        assert (
            result.residue_flow_kmol_s + result.permeate_flow_kmol_s
            == pytest.approx(result.feed_flow_kmol_s, rel=1e-6)
        )
        assert result.residue_composition.sum() == pytest.approx(1.0, abs=1e-8)
        assert result.permeate_composition.sum() == pytest.approx(1.0, abs=1e-8)

    def test_per_stage_compositions_sum_to_one(
        self, air_module_10bar, air_components
    ) -> None:
        result = CountercurrentSolver(
            module=air_module_10bar, components=air_components
        ).solve()
        np.testing.assert_allclose(result.residue_profiles.sum(axis=0), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.permeate_profiles.sum(axis=0), 1.0, atol=1e-6)


class TestN2PurityVsRecovery:
    def test_n2_purity_at_half_recovery_10bar(
        self, air_module_10bar, air_components
    ) -> None:
        """Paper Fig. 4: ~90% N₂ purity at ~50% residue recovery, 10 bar."""
        result = _tune_feed_for_recovery(
            air_module_10bar, air_components, target_recovery=0.50
        )
        # Paper Figure 4 shows ~0.9 at this operating point within a few %.
        assert 0.85 < result.residue_composition[0] < 0.98

    def test_higher_pressure_improves_purity(
        self, air_module_10bar, air_module_5bar, air_components
    ) -> None:
        """Paper Fig. 5: doubling feed pressure improves residue purity at the
        same recovery."""
        r10 = _tune_feed_for_recovery(
            air_module_10bar, air_components, target_recovery=0.50
        )
        r5 = _tune_feed_for_recovery(
            air_module_5bar, air_components, target_recovery=0.50
        )
        assert r10.residue_composition[0] > r5.residue_composition[0]


class TestPurgeEffects:
    def test_purge_decreases_recovery_at_fixed_flow(
        self, air_module_10bar, air_components
    ) -> None:
        """Figure 6: increasing purge reduces the R'/F ratio at fixed feed."""
        base_feed = 1.0e-3
        r_no_purge = _solve_with_feed(
            air_module_10bar, air_components, base_feed, purge_fraction=0.0
        )
        r_with_purge = _solve_with_feed(
            air_module_10bar, air_components, base_feed, purge_fraction=0.10
        )
        assert r_with_purge.residue_recovery < r_no_purge.residue_recovery

    def test_residue_is_dried_below_feed_water_content(
        self, air_module_10bar, air_components
    ) -> None:
        """With H₂O at 1000 GPU, the residue should be drier than the feed
        at any non-trivial stage cut — the core dehydration claim of Fig. 10."""
        result = _solve_with_feed(
            air_module_10bar, air_components, feed_flow_kmol_s=1.0e-3
        )
        # H2O is component index 3.
        assert result.residue_composition[3] < result.feed_composition[3] * 0.01

    def test_purge_reduces_water_in_net_permeate(
        self, air_module_10bar, air_components
    ) -> None:
        """Figure 8/10 qualitative: purging the dry residue into the permeate
        dilutes the water content of the permeate stream."""
        base_feed = 1.0e-3
        r_no_purge = _solve_with_feed(
            air_module_10bar, air_components, base_feed, purge_fraction=0.0
        )
        r_purged = _solve_with_feed(
            air_module_10bar, air_components, base_feed, purge_fraction=0.10
        )
        assert r_purged.permeate_composition[3] < r_no_purge.permeate_composition[3]


class TestComparativeHierarchy:
    def test_countercurrent_beats_crossflow(
        self, air_module_10bar, air_components
    ) -> None:
        """Countercurrent always gives equal or better residue purity than
        cross-flow at the same operating point."""
        cc = CountercurrentSolver(
            module=air_module_10bar, components=air_components
        ).solve()
        cf = CrossflowSolver(module=air_module_10bar, components=air_components).solve()
        assert cc.residue_composition[0] >= cf.residue_composition[0] - 1e-6
