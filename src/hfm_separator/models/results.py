"""Immutable simulation result container."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationResults:
    """Results from a single HFM module simulation.

    All compositions are mole fractions, all flows are in ``kmol/s``,
    all pressures are in Pa.

    Attributes
    ----------
    component_names : tuple[str, ...]
        Component names in the order they appear in all profile arrays.
    feed_composition : np.ndarray
        Feed mole fractions, shape ``(n_components,)``.
    residue_composition : np.ndarray
        Residue (non-permeating) mole fractions, shape ``(n_components,)``.
    permeate_composition : np.ndarray
        Overall permeate mole fractions, shape ``(n_components,)``.
    feed_flow_kmol_s : float
        Total molar feed flow rate.
    residue_flow_kmol_s : float
        Total molar residue flow rate leaving the module.
    permeate_flow_kmol_s : float
        Total molar permeate flow rate leaving the module.
    stage_cut : float
        ``permeate_flow / feed_flow`` — fraction of feed that permeated.
    residue_recovery : float
        ``residue_flow / feed_flow`` — fraction of feed in the residue.
    axial_positions : np.ndarray
        ``z/L`` positions for the axial profiles, shape ``(n_stages,)``.
        ``z/L = 0`` is the residue end and ``z/L = 1`` is the feed end.
    residue_profiles : np.ndarray
        Feed-side mole fractions at each stage, shape ``(n_components, n_stages)``.
    permeate_profiles : np.ndarray
        Permeate-side mole fractions at each stage, shape ``(n_components, n_stages)``.
    feed_side_pressure_pa : np.ndarray
        Feed-side pressure per stage, shape ``(n_stages,)``.
    permeate_side_pressure_pa : np.ndarray
        Permeate-side pressure per stage, shape ``(n_stages,)``.
    feed_side_flow_kmol_s : np.ndarray
        Feed-side total flow per stage, shape ``(n_stages,)``.
    permeate_side_flow_kmol_s : np.ndarray
        Permeate-side total flow per stage, shape ``(n_stages,)``.
    n_iterations : int
        Number of successive-substitution iterations executed.
    n_stages : int
        Number of axial stages ``N``.
    converged : bool
        Whether the successive-substitution loop reached the paper's
        tolerances (Eqs. 16-17 / Eq. 27).
    residuals : tuple[float, float]
        Final ``(|ΔL₁/L₁|, |ΔV_{N+1}/V_{N+1}|)`` — second entry is
        ``nan`` for cocurrent (Eq. 27 only uses the first).
    pattern : str
        One of ``"crossflow"``, ``"countercurrent"``, ``"cocurrent"``.
    """

    component_names: tuple[str, ...]
    feed_composition: np.ndarray
    residue_composition: np.ndarray
    permeate_composition: np.ndarray
    feed_flow_kmol_s: float
    residue_flow_kmol_s: float
    permeate_flow_kmol_s: float
    stage_cut: float
    residue_recovery: float
    axial_positions: np.ndarray
    residue_profiles: np.ndarray
    permeate_profiles: np.ndarray
    feed_side_pressure_pa: np.ndarray
    permeate_side_pressure_pa: np.ndarray
    feed_side_flow_kmol_s: np.ndarray
    permeate_side_flow_kmol_s: np.ndarray
    n_iterations: int
    n_stages: int
    converged: bool
    residuals: tuple[float, float]
    pattern: str

    def to_dataframe(self) -> pd.DataFrame:
        """Return axial profiles as a tidy ``pandas.DataFrame``.

        Columns: ``z_over_L``, ``feed_pressure_pa``, ``permeate_pressure_pa``,
        ``feed_flow_kmol_s``, ``permeate_flow_kmol_s``, then one column
        per component for both residue (``x_{name}``) and permeate
        (``y_{name}``) mole fractions.
        """
        data: dict[str, np.ndarray] = {
            "z_over_L": self.axial_positions,
            "feed_pressure_pa": self.feed_side_pressure_pa,
            "permeate_pressure_pa": self.permeate_side_pressure_pa,
            "feed_flow_kmol_s": self.feed_side_flow_kmol_s,
            "permeate_flow_kmol_s": self.permeate_side_flow_kmol_s,
        }
        for j, name in enumerate(self.component_names):
            data[f"x_{name}"] = self.residue_profiles[j, :]
            data[f"y_{name}"] = self.permeate_profiles[j, :]
        return pd.DataFrame(data)
