"""Unit conversions between common engineering units and SI.

All solver-internal quantities are SI. Conversion happens only here and in
Pydantic model constructors via factory helpers such as ``from_gpu``.

Constants
---------
GPU_TO_SI : float
    1 GPU = 1e-6 cm³(STP) / (cm²·s·cmHg) expressed in SI kmol / (m²·s·Pa).
    The numerical value 3.346e-13 is taken from ``docs/design.md``.
STP_MOLAR_VOLUME_M3_PER_KMOL : float
    Molar volume of an ideal gas at STP (0 °C, 1 atm) used by the paper
    when converting between m³(STP)/h and kmol/s.
"""

GPU_TO_SI: float = 3.346e-13

STP_MOLAR_VOLUME_M3_PER_KMOL: float = 22.414

_CELSIUS_KELVIN_OFFSET: float = 273.15
_BARA_TO_PA: float = 1.0e5
_SECONDS_PER_HOUR: float = 3600.0


def gpu_to_si(gpu: float) -> float:
    """Convert a permeance value from GPU to kmol / (m²·s·Pa)."""
    return gpu * GPU_TO_SI


def si_to_gpu(permeance_si: float) -> float:
    """Convert a permeance value from kmol / (m²·s·Pa) to GPU."""
    return permeance_si / GPU_TO_SI


def bara_to_pa(bara: float) -> float:
    """Convert an absolute pressure from bara to Pa."""
    return bara * _BARA_TO_PA


def pa_to_bara(pa: float) -> float:
    """Convert an absolute pressure from Pa to bara."""
    return pa / _BARA_TO_PA


def celsius_to_kelvin(celsius: float) -> float:
    """Convert a temperature from degrees Celsius to kelvin."""
    return celsius + _CELSIUS_KELVIN_OFFSET


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert a temperature from kelvin to degrees Celsius."""
    return kelvin - _CELSIUS_KELVIN_OFFSET


def stp_m3h_to_kmol_s(flow_stp_m3h: float) -> float:
    """Convert a volumetric flow at STP from m³/h to kmol/s."""
    return flow_stp_m3h / STP_MOLAR_VOLUME_M3_PER_KMOL / _SECONDS_PER_HOUR


def kmol_s_to_stp_m3h(flow_kmol_s: float) -> float:
    """Convert a molar flow from kmol/s to m³(STP)/h."""
    return flow_kmol_s * STP_MOLAR_VOLUME_M3_PER_KMOL * _SECONDS_PER_HOUR
