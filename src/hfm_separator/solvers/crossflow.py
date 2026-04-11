"""Cross-flow solver (Eq. 18) and initial-guess generator.

The cross-flow pattern is both a standalone flow model and — per
``docs/design.md`` §4 — the mandatory initial-guess provider for
:class:`CountercurrentSolver` and :class:`CocurrentSolver`.

Algorithm (march from feed end k=N to residue end k=1):

1. Choose N from Eq. 22.
2. At each stage, take the upstream composition as the local feed-side
   composition, evaluate permeances, solve Eq. 18 for the local
   permeate composition ``y_j`` via Newton's method on a scalar total
   flux variable, compute the transmembrane flux, and subtract it from
   the upstream component flow to get the downstream component flow.
3. The overall permeate composition is the flow-weighted average of all
   per-stage local permeate compositions.

When ``feed_side == "bore"`` the feed-side pressure is updated via
Hagen-Poiseuille stage by stage. When ``feed_side == "shell"`` the
shell-side pressure is held constant (bore side is the permeate side,
whose flow profile is unknown until the march is complete).
"""

from dataclasses import dataclass

import numpy as np

from hfm_separator.models.results import SimulationResults
from hfm_separator.numerics.staging import choose_n_stages
from hfm_separator.solvers.base import BaseSolver


@dataclass(frozen=True)
class CrossflowProfiles:
    """Converged axial profiles from a cross-flow march.

    All arrays are indexed so that ``[..., 0]`` is the residue end
    (paper k=1) and ``[..., -1]`` is the feed end (paper k=N).
    """

    n_stages: int
    L: np.ndarray
    ell: np.ndarray
    x: np.ndarray
    V: np.ndarray
    up: np.ndarray
    y: np.ndarray
    m: np.ndarray
    P_feed: np.ndarray
    P_perm: np.ndarray
    sweep_flow: float
    sweep_vector: np.ndarray


_NEWTON_MAX_ITER: int = 200
_NEWTON_TOL: float = 1.0e-13
_MIN_FLUX: float = 1.0e-300


def solve_eq18_permeate_composition(
    feed_composition: np.ndarray,
    permeances: np.ndarray,
    feed_pressure_pa: float,
    permeate_pressure_pa: float,
) -> np.ndarray:
    """Solve Eq. 18 for the local permeate composition at a stage.

    Uses the reformulation ``y_j = Q_j P_L x_j / (M + Q_j P_V)`` with
    ``M = Σ Q_j (P_L x_j − P_V y_j)`` (total volumetric driving flux
    scaled by Q), solved with Newton iteration on the scalar ``M``.
    ``g(M) = Σ_j Q_j P_L x_j / (M + Q_j P_V) − 1 = 0`` is monotonically
    decreasing and has exactly one positive root.
    """
    q_pl_x = permeances * feed_pressure_pa * feed_composition
    q_pv = permeances * permeate_pressure_pa

    # Initial guess from Eq. 19 (selectivity-limited).
    qx_sum = float(np.sum(permeances * feed_composition))
    if qx_sum <= 0.0:
        return np.asarray(feed_composition, dtype=float).copy()
    y0 = permeances * feed_composition / qx_sum
    m0 = permeances * (feed_pressure_pa * feed_composition - permeate_pressure_pa * y0)
    m_total = float(np.sum(m0))
    if m_total <= 0.0:
        m_total = qx_sum * feed_pressure_pa
    m_total = max(m_total, _MIN_FLUX)

    for _ in range(_NEWTON_MAX_ITER):
        denom = m_total + q_pv
        y = q_pl_x / denom
        g = float(np.sum(y)) - 1.0
        if abs(g) < _NEWTON_TOL:
            return y
        dg = -float(np.sum(q_pl_x / denom**2))
        if dg == 0.0:
            break
        step = g / dg
        m_next = m_total - step
        if m_next <= 0.0:
            m_next = 0.5 * m_total
        m_total = m_next

    return q_pl_x / (m_total + q_pv)


class CrossflowSolver(BaseSolver):
    """Standalone cross-flow solver and iterative-solver initial-guess provider."""

    def solve(self) -> SimulationResults:
        profiles = self._march()
        return self._profiles_to_results(profiles)

    def initial_guess_profiles(self) -> CrossflowProfiles:
        """Return marching profiles suitable for initializing an iterative solver.

        ``L[i]`` / ``ell[:, i]`` are the values at grid point ``k = i + 1``
        (feed-side output of stage ``k``). The feed boundary values are
        accessible as ``self.module.feed_flow_kmol_s`` and
        ``self.feed_mole_fractions``.
        """
        return self._march()

    def _march(self) -> CrossflowProfiles:
        module = self.module
        n_stages = choose_n_stages(
            fiber_outer_radius_m=module.fiber_outer_radius_m,
            active_length_m=module.active_length_m,
            n_fibers=module.n_fibers,
            feed_mole_fractions=self.feed_mole_fractions,
            permeances_si=self._permeances_at(
                module.feed_pressure_pa, self.feed_mole_fractions
            ),
            feed_pressure_pa=module.feed_pressure_pa,
            feed_flow_kmol_s=module.feed_flow_kmol_s,
        )
        r = self.n_components
        delta_a = module.total_membrane_area_m2 / n_stages
        dz = module.active_length_m / n_stages
        feed_total = module.feed_flow_kmol_s
        feed_vector = feed_total * self.feed_mole_fractions

        ell_cells = np.zeros((r, n_stages))
        l_cells = np.zeros(n_stages)
        y_cells = np.zeros((r, n_stages))
        m_cells = np.zeros((r, n_stages))
        p_feed_cells = np.zeros(n_stages)
        p_perm_cells = np.zeros(n_stages)

        upstream_ell = feed_vector.copy()
        upstream_l = feed_total
        upstream_p_feed = module.feed_pressure_pa

        for k in range(n_stages, 0, -1):
            i = k - 1
            x_local = upstream_ell / upstream_l
            permeances = self._permeances_at(upstream_p_feed, x_local)

            if module.feed_side == "bore":
                dp = self._pressure_drop_stage(
                    bore_flow_kmol_s=upstream_l,
                    upstream_pressure_pa=upstream_p_feed,
                    mole_fractions=x_local,
                    stage_length_m=dz,
                )
                p_feed_k = upstream_p_feed - dp
            else:
                p_feed_k = module.feed_pressure_pa
            p_perm_k = module.permeate_pressure_pa

            y_local = solve_eq18_permeate_composition(
                feed_composition=x_local,
                permeances=permeances,
                feed_pressure_pa=p_feed_k,
                permeate_pressure_pa=p_perm_k,
            )
            driving = p_feed_k * x_local - p_perm_k * y_local
            driving = np.maximum(driving, 0.0)
            m_local = permeances * delta_a * driving
            m_local = np.minimum(m_local, upstream_ell)  # cannot over-strip

            ell_k = upstream_ell - m_local
            ell_k = np.maximum(ell_k, 1.0e-30)
            l_k = float(np.sum(ell_k))

            ell_cells[:, i] = ell_k
            l_cells[i] = l_k
            y_cells[:, i] = y_local
            m_cells[:, i] = m_local
            p_feed_cells[i] = p_feed_k
            p_perm_cells[i] = p_perm_k

            upstream_ell = ell_k
            upstream_l = l_k
            upstream_p_feed = p_feed_k

        x_profiles = ell_cells / l_cells[np.newaxis, :]

        # Derive V_k profile from flow balance: V_k = V_0 + L_{k+1} − L_1.
        sweep_flow = module.purge_fraction * feed_total
        l_next = np.concatenate([l_cells[1:], [feed_total]])
        v_cells = sweep_flow + l_next - l_cells[0]
        ell_next = np.concatenate(
            [ell_cells[:, 1:], feed_vector[:, np.newaxis]], axis=1
        )
        ell_residue = ell_cells[:, 0:1]
        # For purging, sweep composition is the un-permeated residue (conservative).
        sweep_vector = sweep_flow * (
            ell_residue[:, 0] / l_cells[0] if sweep_flow > 0 else np.zeros(r)
        )
        up_cells = sweep_vector[:, np.newaxis] + ell_next - ell_residue
        up_cells = np.maximum(up_cells, 0.0)

        return CrossflowProfiles(
            n_stages=n_stages,
            L=l_cells,
            ell=ell_cells,
            x=x_profiles,
            V=v_cells,
            up=up_cells,
            y=y_cells,
            m=m_cells,
            P_feed=p_feed_cells,
            P_perm=p_perm_cells,
            sweep_flow=sweep_flow,
            sweep_vector=sweep_vector,
        )

    def _profiles_to_results(self, profiles: CrossflowProfiles) -> SimulationResults:
        module = self.module
        n = profiles.n_stages

        # Residue properties are at stage 1 (grid point 1, Python index 0).
        residue_flow = float(profiles.L[0])
        residue_composition = profiles.ell[:, 0] / residue_flow

        total_m = profiles.m.sum(axis=1)
        permeate_flow = float(total_m.sum())
        if permeate_flow <= 0.0:
            permeate_composition = self.feed_mole_fractions.copy()
        else:
            permeate_composition = total_m / permeate_flow

        feed_flow = module.feed_flow_kmol_s
        stage_cut = permeate_flow / feed_flow
        residue_recovery = residue_flow / feed_flow
        axial = (np.arange(n, dtype=float) + 0.5) / n

        return SimulationResults(
            component_names=self.component_names,
            feed_composition=self.feed_mole_fractions.copy(),
            residue_composition=residue_composition,
            permeate_composition=permeate_composition,
            feed_flow_kmol_s=feed_flow,
            residue_flow_kmol_s=residue_flow,
            permeate_flow_kmol_s=permeate_flow,
            stage_cut=stage_cut,
            residue_recovery=residue_recovery,
            axial_positions=axial,
            residue_profiles=profiles.x,
            permeate_profiles=profiles.y,
            feed_side_pressure_pa=profiles.P_feed,
            permeate_side_pressure_pa=profiles.P_perm,
            feed_side_flow_kmol_s=profiles.L,
            permeate_side_flow_kmol_s=profiles.V,
            n_iterations=1,
            n_stages=n,
            converged=True,
            residuals=(0.0, float("nan")),
            pattern="crossflow",
        )
