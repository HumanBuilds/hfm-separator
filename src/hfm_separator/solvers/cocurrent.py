"""Cocurrent solver — direct stage marching (Eq. 26).

Both feed and permeate flow in the same direction: k = N+1 → k = 1.
Each iteration re-marches the explicit Eq. 26 using the previous
iteration's ``L_k`` and ``V_k`` profiles, then refreshes them. The
convergence target is Eq. 27 ``|ΔV₁/V₁| < 1e-8``.

No sweep on the cocurrent side (``purge_fraction`` must be zero). Sweep
at the feed end is physically unusual for cocurrent and is not
supported; the solver raises ``ValueError`` otherwise.
"""

import numpy as np

from hfm_separator.models.results import SimulationResults
from hfm_separator.solvers.base import BaseSolver
from hfm_separator.solvers.crossflow import CrossflowSolver

_MAX_ITER: int = 500
_CONVERGE_TOL: float = 1.0e-8
_FLOW_FLOOR: float = 1.0e-30


class CocurrentSolver(BaseSolver):
    """Cocurrent HFM solver via explicit marching of Eq. 26."""

    def solve(self) -> SimulationResults:
        module = self.module
        if module.purge_fraction != 0.0:
            raise ValueError("cocurrent solver does not support purge_fraction > 0")

        guess_solver = CrossflowSolver(module=module, components=self.components)
        guess = guess_solver.initial_guess_profiles()

        n_stages = guess.n_stages
        r = self.n_components
        feed_total = module.feed_flow_kmol_s
        feed_vector = feed_total * self.feed_mole_fractions
        dz = module.active_length_m / n_stages
        delta_a = module.total_membrane_area_m2 / n_stages

        ell = guess.ell.copy()
        l_cells = guess.L.copy()

        # Cocurrent υ_j,k = Σ_{k'=k..N} m_j,k'  (reverse-cumulative sum).
        up_cells = np.cumsum(guess.m[:, ::-1], axis=1)[:, ::-1]
        v_cells = np.maximum(up_cells.sum(axis=0), _FLOW_FLOOR)

        p_feed_cells = guess.P_feed.copy()
        p_perm_cells = guess.P_perm.copy()

        sweep_vector = np.zeros(r)
        converged = False
        res_v1 = float("inf")
        n_iter = 0

        for iteration in range(1, _MAX_ITER + 1):
            n_iter = iteration
            v1_old = float(v_cells[0])

            p_feed_cells, p_perm_cells = self._update_pressures(
                l_cells=l_cells,
                v_cells=v_cells,
                ell=ell,
                up=up_cells,
                dz=dz,
            )

            x_profiles = ell / l_cells[np.newaxis, :]
            q_matrix = np.empty((r, n_stages))
            for i in range(n_stages):
                q_matrix[:, i] = self._permeances_at(p_feed_cells[i], x_profiles[:, i])

            alpha_mat = v_cells[np.newaxis, :] / (
                p_perm_cells[np.newaxis, :] * delta_a * q_matrix
            )
            eta_mat = (p_feed_cells[np.newaxis, :] * v_cells[np.newaxis, :]) / (
                p_perm_cells[np.newaxis, :] * l_cells[np.newaxis, :]
            )

            new_ell = np.empty_like(ell)
            new_up = np.empty_like(up_cells)

            ell_upstream = feed_vector.copy()
            up_upstream = sweep_vector.copy()

            for i in range(n_stages - 1, -1, -1):
                alpha_k = alpha_mat[:, i]
                denominator = 1.0 + alpha_k + eta_mat[:, i]
                numerator = up_upstream + (1.0 + alpha_k) * ell_upstream
                ell_new_k = numerator / denominator
                ell_new_k = np.maximum(ell_new_k, 0.0)
                up_new_k = np.maximum(ell_upstream + up_upstream - ell_new_k, 0.0)
                new_ell[:, i] = ell_new_k
                new_up[:, i] = up_new_k
                ell_upstream = ell_new_k
                up_upstream = up_new_k

            ell = new_ell
            up_cells = new_up
            l_cells = np.maximum(ell.sum(axis=0), _FLOW_FLOOR)
            v_cells = np.maximum(up_cells.sum(axis=0), _FLOW_FLOOR)

            v1_new = float(v_cells[0])
            res_v1 = abs((v1_new - v1_old) / v1_new) if v1_new > 0 else float("inf")
            if res_v1 < _CONVERGE_TOL:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"cocurrent solver did not converge in {_MAX_ITER} iterations; "
                f"|ΔV1/V1|={res_v1:.3e}"
            )

        return self._build_results(
            n_stages=n_stages,
            ell=ell,
            l_cells=l_cells,
            v_cells=v_cells,
            up_cells=up_cells,
            p_feed=p_feed_cells,
            p_perm=p_perm_cells,
            n_iter=n_iter,
            residuals=(res_v1, float("nan")),
        )

    def _update_pressures(
        self,
        l_cells: np.ndarray,
        v_cells: np.ndarray,
        ell: np.ndarray,
        up: np.ndarray,
        dz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        module = self.module
        n_stages = l_cells.shape[0]
        p_feed = np.empty(n_stages)
        p_perm = np.empty(n_stages)

        if module.feed_side == "bore":
            upstream_p = module.feed_pressure_pa
            upstream_l = module.feed_flow_kmol_s
            upstream_x = self.feed_mole_fractions
            for i in range(n_stages - 1, -1, -1):
                dp = self._pressure_drop_stage(
                    bore_flow_kmol_s=upstream_l,
                    upstream_pressure_pa=upstream_p,
                    mole_fractions=upstream_x,
                    stage_length_m=dz,
                )
                p_feed[i] = upstream_p - dp
                upstream_p = p_feed[i]
                upstream_l = float(l_cells[i])
                upstream_x = ell[:, i] / max(upstream_l, _FLOW_FLOOR)
            p_perm[:] = module.permeate_pressure_pa
        else:
            p_feed[:] = module.feed_pressure_pa
            p_perm[0] = module.permeate_pressure_pa
            for i in range(1, n_stages):
                downstream_p = p_perm[i - 1]
                downstream_v = float(v_cells[i - 1])
                downstream_v_safe = max(downstream_v, _FLOW_FLOOR)
                downstream_y = up[:, i - 1] / downstream_v_safe
                dp = self._pressure_drop_stage(
                    bore_flow_kmol_s=downstream_v,
                    upstream_pressure_pa=downstream_p,
                    mole_fractions=downstream_y,
                    stage_length_m=dz,
                )
                p_perm[i] = downstream_p + dp
        return p_feed, p_perm

    def _build_results(
        self,
        n_stages: int,
        ell: np.ndarray,
        l_cells: np.ndarray,
        v_cells: np.ndarray,
        up_cells: np.ndarray,
        p_feed: np.ndarray,
        p_perm: np.ndarray,
        n_iter: int,
        residuals: tuple[float, float],
    ) -> SimulationResults:
        module = self.module
        residue_flow = float(l_cells[0])
        residue_composition = ell[:, 0] / residue_flow
        permeate_flow = float(v_cells[0])
        if permeate_flow <= 0.0:
            permeate_composition = self.feed_mole_fractions.copy()
        else:
            permeate_composition = up_cells[:, 0] / permeate_flow

        feed_flow = module.feed_flow_kmol_s
        stage_cut = permeate_flow / feed_flow
        residue_recovery = residue_flow / feed_flow
        axial = (np.arange(n_stages, dtype=float) + 0.5) / n_stages
        x_profiles = ell / l_cells[np.newaxis, :]
        y_profiles = up_cells / np.maximum(v_cells[np.newaxis, :], _FLOW_FLOOR)

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
            residue_profiles=x_profiles,
            permeate_profiles=y_profiles,
            feed_side_pressure_pa=p_feed,
            permeate_side_pressure_pa=p_perm,
            feed_side_flow_kmol_s=l_cells,
            permeate_side_flow_kmol_s=v_cells,
            n_iterations=n_iter,
            n_stages=n_stages,
            converged=True,
            residuals=residuals,
            pattern="cocurrent",
        )
