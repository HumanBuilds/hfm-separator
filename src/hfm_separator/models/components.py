"""User-facing component specification."""

import math

from pydantic import BaseModel, ConfigDict, field_validator

from hfm_separator.models.permeance import PermeanceFn


class ComponentSpec(BaseModel):
    """Physical and transport properties of a single feed component.

    Values are in SI. Use ``from_gpu`` to supply permeance from GPU values.

    Attributes
    ----------
    name : str
        Human-readable label (used in result DataFrames).
    feed_mole_fraction : float
        Mole fraction in the feed stream, strictly in ``(0, 1)``.
    permeance : PermeanceFn
        Callable returning permeance in ``kmol / (m²·s·Pa)``.
    molar_mass_kg_per_kmol : float
        Molar mass used by the Wilke viscosity mixing rule.
    pure_viscosity_pa_s : float
        Pure-component gas viscosity at module temperature (Pa·s).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    feed_mole_fraction: float
    permeance: PermeanceFn
    molar_mass_kg_per_kmol: float
    pure_viscosity_pa_s: float

    @field_validator("feed_mole_fraction")
    @classmethod
    def _fraction_in_open_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(
                f"feed_mole_fraction must be in the open interval (0, 1), got {v}"
            )
        return v

    @field_validator("molar_mass_kg_per_kmol")
    @classmethod
    def _positive_molar_mass(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"molar_mass_kg_per_kmol must be positive and finite, got {v}"
            )
        return v

    @field_validator("pure_viscosity_pa_s")
    @classmethod
    def _positive_viscosity(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"pure_viscosity_pa_s must be positive and finite, got {v}"
            )
        return v

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must be a non-empty string")
        return v
