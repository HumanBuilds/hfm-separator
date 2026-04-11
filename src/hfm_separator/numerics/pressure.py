"""Bore-side pressure drop models.

Primary: Hagen-Poiseuille (Eq. 21) — valid for laminar, Newtonian,
incompressible, impermeable-wall flow. ``docs/numerics.md`` §"Bore-side
Pressure Drop" shows the paper demonstrates this holds for every case in
the paper via criteria in Eqs. A9 and A16.

Secondary: full mechanical energy balance (Eq. A5) is supplied for
extensibility and verification — ``verify_hagen_poiseuille_regime``
computes the dimensionless groups from the appendix.
"""

import numpy as np

from hfm_separator.numerics.constants import R_GAS_KMOL


def hagen_poiseuille_pressure_drop(
    flow_kmol_s: float,
    upstream_pressure_pa: float,
    mixture_viscosity_pa_s: float,
    fiber_inner_radius_m: float,
    n_fibers: int,
    stage_length_m: float,
    temperature_k: float,
) -> float:
    """Pressure drop across one axial stage of length ``stage_length_m``.

    From Eq. 21::

        P_upstream − P_downstream = (8 μ_mix / (π R_i^4)) · V · (RT/P) · Δz

    This is the drop for a *single* representative fiber, but Eq. 21 uses
    the *per-fiber* volumetric flow. The caller supplies the *total* flow
    through ``n_fibers`` tubes; we divide before plugging in.

    Parameters
    ----------
    flow_kmol_s : float
        Total bore-side molar flow across all fibers at this stage (kmol/s).
    upstream_pressure_pa : float
        Upstream (higher) absolute pressure (Pa).
    mixture_viscosity_pa_s : float
        Local mixture viscosity (Pa·s).
    fiber_inner_radius_m : float
        Bore inner radius of a single fiber (m).
    n_fibers : int
        Number of fibers sharing the total flow.
    stage_length_m : float
        Length of the axial stage (m).
    temperature_k : float
        Temperature (K).

    Returns
    -------
    float
        Positive pressure drop in Pa.
    """
    if upstream_pressure_pa <= 0.0:
        raise ValueError("upstream_pressure_pa must be positive")
    if flow_kmol_s < 0.0:
        raise ValueError("flow_kmol_s must be non-negative")

    flow_per_fiber = flow_kmol_s / float(n_fibers)
    volumetric_per_fiber = (
        flow_per_fiber * R_GAS_KMOL * temperature_k / upstream_pressure_pa
    )
    prefactor = 8.0 * mixture_viscosity_pa_s / (np.pi * fiber_inner_radius_m**4)
    return float(prefactor * volumetric_per_fiber * stage_length_m)


def compressibility_group_a9(
    pressure_pa: float,
    temperature_k: float,
    molar_mass_kg_per_kmol: float,
    flow_kmol_s: float,
    bore_area_m2: float,
) -> float:
    """Eq. A9 dimensionless group — large means compressibility negligible.

    ``PL² / (RTM · (L/A)²)``
    """
    mass_flux = (flow_kmol_s * molar_mass_kg_per_kmol) / bore_area_m2
    return float(
        pressure_pa**2
        / (R_GAS_KMOL * temperature_k * molar_mass_kg_per_kmol * mass_flux**2)
    )


def permeability_group_a16(
    viscosity_pa_s: float,
    molar_flow_kmol_s: float,
    fiber_inner_radius_m: float,
    fiber_outer_radius_m: float,
    molar_mass_kg_per_kmol: float,
    pressure_pa: float,
    permeance_si: float,
) -> float:
    """Eq. A16 group — large means wall-permeability effects are negligible."""
    numerator = 8.0 * viscosity_pa_s * molar_flow_kmol_s**2
    denominator = (
        np.pi**2
        * fiber_inner_radius_m**2
        * fiber_outer_radius_m
        * molar_mass_kg_per_kmol
        * (permeance_si * pressure_pa) ** 3
    )
    return float(numerator / denominator)
