"""Countercurrent solver — Thomas algorithm + successive substitution.

Implements the tridiagonal formulation from Eqs. 10-13 of Coker et al.
(1998). Uses the mandatory cross-flow initial guess, then iterates until
Eqs. 16-17 tolerances are satisfied::

    |ΔL₁/L₁| < 1e-8  and  |ΔV_N/V_N| < 1e-8

For purging, a fraction ``purge_fraction`` of the residue flow ``L₁`` is
routed back to the permeate side at ``k=0`` as a sweep with the residue's
local composition — updated each iteration because ``L₁`` itself moves.
"""

import numpy as np

from hfm_separator.models.results import SimulationResults
from hfm_separator.numerics.thomas import thomas_solve
from hfm_separator.solvers.base import BaseSolver
from hfm_separator.solvers.crossflow import CrossflowSolver

_MAX_ITER: int = 500
_CONVERGE_TOL: float = 1.0e-8
_FLOW_FLOOR: float = 1.0e-30


class CountercurrentSolver(BaseSolver):
    """Countercurrent HFM solver using Thomas + successive substitution."""

    def solve(self) -> SimulationResults:
        module = self.module
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
        v_cells = np.maximum(guess.V.copy(), _FLOW_FLOOR)
        up_cells = guess.up.copy()
        p_feed_cells = guess.P_feed.copy()
        p_perm_cells = guess.P_perm.copy()

        converged = False
        res_l = float("inf")
        res_v = float("inf")
        n_iter = 0

        for iteration in range(1, _MAX_ITER + 1):
            n_iter = iteration
            l1_old = float(l_cells[0])
            vN_old = float(v_cells[-1])

            sweep_flow = module.purge_fraction * l1_old
            residue_composition = ell[:, 0] / l1_old
            sweep_vector = sweep_flow * residue_composition

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
            beta_mat = (
                1.0
                + q_matrix
                * delta_a
                * p_feed_cells[np.newaxis, :]
                / l_cells[np.newaxis, :]
            )

            new_ell = np.zeros_like(ell)
            for j in range(r):
                alpha = alpha_mat[j]
                beta = beta_mat[j]
                lower = np.zeros(n_stages)
                diag = np.zeros(n_stages)
                upper = np.zeros(n_stages)
                rhs = np.zeros(n_stages)

                diag[0] = 1.0 + alpha[0] * beta[0]
                upper[0] = -(1.0 + alpha[0])
                rhs[0] = sweep_vector[j]

                if n_stages >= 3:
                    lower[1:-1] = -alpha[:-2] * beta[:-2]
                    diag[1:-1] = 1.0 + alpha[:-2] + alpha[1:-1] * beta[1:-1]
                    upper[1:-1] = -(1.0 + alpha[1:-1])

                lower[-1] = -alpha[-2] * beta[-2]
                diag[-1] = 1.0 + alpha[-2] + alpha[-1] * beta[-1]
                rhs[-1] = (1.0 + alpha[-1]) * feed_vector[j]

                new_ell[j, :] = thomas_solve(lower, diag, upper, rhs)

            new_ell = np.maximum(new_ell, _FLOW_FLOOR)
            ell = new_ell
            l_cells = np.maximum(ell.sum(axis=0), _FLOW_FLOOR)

            sweep_flow = module.purge_fraction * float(l_cells[0])
            l_next = np.concatenate([l_cells[1:], [feed_total]])
            v_cells = np.maximum(sweep_flow + l_next - l_cells[0], _FLOW_FLOOR)

            residue_composition = ell[:, 0] / l_cells[0]
            sweep_vector_new = sweep_flow * residue_composition
            ell_next = np.concatenate([ell[:, 1:], feed_vector[:, np.newaxis]], axis=1)
            up_cells = np.maximum(
                sweep_vector_new[:, np.newaxis] + ell_next - ell[:, 0:1],
                0.0,
            )

            l1_new = float(l_cells[0])
            vN_new = float(v_cells[-1])
            res_l = abs((l1_new - l1_old) / l1_new) if l1_new > 0 else float("inf")
            res_v = abs((vN_new - vN_old) / vN_new) if vN_new > 0 else float("inf")

            if res_l < _CONVERGE_TOL and res_v < _CONVERGE_TOL:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"countercurrent solver did not converge in {_MAX_ITER} iterations; "
                f"|ΔL1/L1|={res_l:.3e}, |ΔVN/VN|={res_v:.3e}"
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
            residuals=(res_l, res_v),
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
            p_perm[-1] = module.permeate_pressure_pa
            for i in range(n_stages - 2, -1, -1):
                downstream_p = p_perm[i + 1]
                downstream_v = float(v_cells[i + 1])
                downstream_v_safe = max(downstream_v, _FLOW_FLOOR)
                downstream_y = up[:, i + 1] / downstream_v_safe
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
        residue_flow_internal = float(l_cells[0])
        residue_composition = ell[:, 0] / residue_flow_internal
        permeate_port_flow = float(v_cells[-1])
        if permeate_port_flow <= 0.0:
            permeate_composition = self.feed_mole_fractions.copy()
        else:
            permeate_composition = up_cells[:, -1] / permeate_port_flow

        sweep_flow = module.purge_fraction * residue_flow_internal
        net_residue_flow = residue_flow_internal - sweep_flow
        feed_flow = module.feed_flow_kmol_s
        stage_cut = permeate_port_flow / feed_flow
        residue_recovery = net_residue_flow / feed_flow

        axial = (np.arange(n_stages, dtype=float) + 0.5) / n_stages
        x_profiles = ell / l_cells[np.newaxis, :]
        y_profiles = up_cells / np.maximum(v_cells[np.newaxis, :], _FLOW_FLOOR)

        return SimulationResults(
            component_names=self.component_names,
            feed_composition=self.feed_mole_fractions.copy(),
            residue_composition=residue_composition,
            permeate_composition=permeate_composition,
            feed_flow_kmol_s=feed_flow,
            residue_flow_kmol_s=net_residue_flow,
            permeate_flow_kmol_s=permeate_port_flow,
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
            pattern="countercurrent",
        )
