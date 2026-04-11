"""Hollow-fiber membrane module configuration."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ModuleConfig(BaseModel):
    """Physical configuration of a hollow-fiber membrane module.

    All values in SI. Helpers in ``hfm_separator.utils.unit_conv`` convert
    from engineering units before construction.

    Attributes
    ----------
    n_fibers : int
        Total number of hollow fibers in the module bundle.
    fiber_od_m : float
        Outer diameter of a single fiber (m). The active separation layer
        lives on the outside.
    fiber_id_m : float
        Inner bore diameter of a single fiber (m). Sets the bore-side
        cross-section used by Hagen-Poiseuille.
    active_length_m : float
        Active (permeating) fiber length in m. Equals total length minus
        twice the pot length.
    pot_length_m : float
        Length of non-permeating potted fiber at each end (m).
    feed_pressure_pa : float
        Feed-side inlet pressure (Pa).
    permeate_pressure_pa : float
        Permeate-side outlet pressure (Pa).
    feed_temp_k : float
        Feed gas temperature (K). The module is assumed isothermal.
    feed_side : {"bore", "shell"}
        Which side of the fibers receives the feed gas.
    purge_fraction : float
        Fraction of the feed flow rate used as a permeate-side sweep,
        ``0`` means no purge.
    feed_flow_kmol_s : float
        Total molar feed flow rate (kmol/s).
    """

    model_config = ConfigDict(frozen=True)

    n_fibers: int
    fiber_od_m: float
    fiber_id_m: float
    active_length_m: float
    pot_length_m: float
    feed_pressure_pa: float
    permeate_pressure_pa: float
    feed_temp_k: float
    feed_flow_kmol_s: float
    feed_side: Literal["bore", "shell"] = "shell"
    purge_fraction: float = 0.0

    @model_validator(mode="after")
    def _check_pressures(self) -> "ModuleConfig":
        if self.feed_pressure_pa <= 0.0:
            raise ValueError("feed_pressure_pa must be positive")
        if self.permeate_pressure_pa <= 0.0:
            raise ValueError("permeate_pressure_pa must be positive")
        if self.permeate_pressure_pa >= self.feed_pressure_pa:
            raise ValueError("permeate_pressure_pa must be less than feed_pressure_pa")
        return self

    @model_validator(mode="after")
    def _check_geometry(self) -> "ModuleConfig":
        if self.fiber_id_m <= 0.0 or self.fiber_od_m <= 0.0:
            raise ValueError("fiber diameters must be positive")
        if self.fiber_id_m >= self.fiber_od_m:
            raise ValueError("fiber_id_m must be less than fiber_od_m")
        if self.active_length_m <= 0.0:
            raise ValueError("active_length_m must be positive")
        if self.pot_length_m < 0.0:
            raise ValueError("pot_length_m must be non-negative")
        if self.n_fibers <= 0:
            raise ValueError("n_fibers must be positive")
        return self

    @model_validator(mode="after")
    def _check_feed(self) -> "ModuleConfig":
        if self.feed_temp_k <= 0.0:
            raise ValueError("feed_temp_k must be positive")
        if self.feed_flow_kmol_s <= 0.0:
            raise ValueError("feed_flow_kmol_s must be positive")
        if not 0.0 <= self.purge_fraction < 1.0:
            raise ValueError("purge_fraction must be in [0, 1)")
        return self

    @property
    def pressure_ratio(self) -> float:
        """Dimensionless ratio of feed to permeate pressure."""
        return self.feed_pressure_pa / self.permeate_pressure_pa

    @property
    def fiber_outer_radius_m(self) -> float:
        return self.fiber_od_m / 2.0

    @property
    def fiber_inner_radius_m(self) -> float:
        return self.fiber_id_m / 2.0

    @property
    def total_membrane_area_m2(self) -> float:
        """Active membrane area referenced to the outer surface (Eq. 1 context)."""
        return math.pi * self.fiber_od_m * self.active_length_m * float(self.n_fibers)
