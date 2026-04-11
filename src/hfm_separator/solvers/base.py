"""Abstract base solver shared by all flow patterns.

Validates inputs, caches numpy-friendly component arrays, and provides
helpers for pressure profiles and per-stage permeance evaluation.
"""

from abc import ABC, abstractmethod

import numpy as np

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.results import SimulationResults
from hfm_separator.numerics.pressure import hagen_poiseuille_pressure_drop
from hfm_separator.numerics.viscosity import wilke_mixture_viscosity

_FEED_COMPOSITION_TOL: float = 1.0e-6


class BaseSolver(ABC):
    """Base class for all HFM separation solvers.

    Subclasses must implement :meth:`solve` which returns a
    :class:`SimulationResults` frozen dataclass.
    """

    def __init__(self, module: ModuleConfig, components: list[ComponentSpec]) -> None:
        self.module = module
        self.components = components
        self._validate_feed_composition()
        self.n_components: int = len(components)
        self.component_names: tuple[str, ...] = tuple(c.name for c in components)
        self.feed_mole_fractions: np.ndarray = np.array(
            [c.feed_mole_fraction for c in components], dtype=float
        )
        self.molar_masses: np.ndarray = np.array(
            [c.molar_mass_kg_per_kmol for c in components], dtype=float
        )
        self.pure_viscosities: np.ndarray = np.array(
            [c.pure_viscosity_pa_s for c in components], dtype=float
        )

    def _validate_feed_composition(self) -> None:
        if not self.components:
            raise ValueError("components list must be non-empty")
        total = sum(c.feed_mole_fraction for c in self.components)
        if abs(total - 1.0) > _FEED_COMPOSITION_TOL:
            raise ValueError(
                f"feed mole fractions must sum to 1.0 (within {_FEED_COMPOSITION_TOL}),"
                f" got {total:.10f}"
            )

    def _permeances_at(
        self,
        pressure_pa: float,
        mole_fractions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate every component permeance at local conditions."""
        temp_k = self.module.feed_temp_k
        return np.array(
            [c.permeance(pressure_pa, temp_k, mole_fractions) for c in self.components],
            dtype=float,
        )

    def _pressure_drop_stage(
        self,
        bore_flow_kmol_s: float,
        upstream_pressure_pa: float,
        mole_fractions: np.ndarray,
        stage_length_m: float,
    ) -> float:
        """Hagen-Poiseuille pressure drop across one stage on the bore side.

        The result is clamped to at most 99% of ``upstream_pressure_pa``.
        If Hagen-Poiseuille says more, the laminar-incompressible
        assumptions have broken down and the solver would produce a
        negative pressure downstream — the clamp keeps the iteration
        numerically stable at the price of a small approximation error
        in a regime the paper says should never be reached anyway.
        """
        if bore_flow_kmol_s <= 0.0 or upstream_pressure_pa <= 0.0:
            return 0.0
        mu = wilke_mixture_viscosity(
            mole_fractions=mole_fractions,
            pure_viscosities=self.pure_viscosities,
            molar_masses=self.molar_masses,
        )
        dp = hagen_poiseuille_pressure_drop(
            flow_kmol_s=bore_flow_kmol_s,
            upstream_pressure_pa=upstream_pressure_pa,
            mixture_viscosity_pa_s=mu,
            fiber_inner_radius_m=self.module.fiber_inner_radius_m,
            n_fibers=self.module.n_fibers,
            stage_length_m=stage_length_m,
            temperature_k=self.module.feed_temp_k,
        )
        return min(dp, 0.99 * upstream_pressure_pa)

    @abstractmethod
    def solve(self) -> SimulationResults: ...
