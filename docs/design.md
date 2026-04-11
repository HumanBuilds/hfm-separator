# Design Decisions

---

## 1. Callable Permeance (not a scalar float)

### Decision
Permeance is always a `PermeanceFn` callable, never a raw float, at the solver layer.

### Rationale
The paper explicitly lists pressure-dependent permeability as a supported feature.
If we store permeance as a scalar, adding pressure/temperature/composition dependence
later requires changing all call sites. The callable design makes the simple case
(constant permeance) and the complex case (dual-mode sorption, plasticisation, etc.)
identical at the solver layer — the solver always calls `Q(P, T, x)`.

### Implementation

```python
# models/permeance.py

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class PermeanceFn(Protocol):
    """
    Callable interface for permeance, optionally dependent on local conditions.

    Parameters
    ----------
    pressure_pa : float
        Local feed-side pressure (Pa).
    temp_k : float
        Local temperature (K).
    mole_fractions : np.ndarray
        Local mole fractions, shape (n_components,).

    Returns
    -------
    float
        Permeance in SI: kmol / (m² · s · Pa).
    """
    def __call__(
        self,
        pressure_pa: float,
        temp_k: float,
        mole_fractions: np.ndarray,
    ) -> float: ...


def constant(permeance_si: float) -> PermeanceFn:
    """
    Factory for pressure/composition-independent permeance.
    This is the correct choice for the paper's case studies.
    """
    def _fn(
        pressure_pa: float,   # noqa: ARG001
        temp_k: float,        # noqa: ARG001
        mole_fractions: np.ndarray,  # noqa: ARG001
    ) -> float:
        return permeance_si
    return _fn


def from_gpu(gpu_value: float) -> PermeanceFn:
    """Convenience: GPU value → constant PermeanceFn in SI."""
    from hfm_separator.utils.unit_conv import gpu_to_si
    return constant(gpu_to_si(gpu_value))
```

### Usage at call sites

```python
# Simple case — constant permeance from GPU value
o2 = ComponentSpec(name="O2", feed_mole_fraction=0.2084, permeance=from_gpu(20.0))

# Complex case — pressure-dependent (e.g. CO2 plasticisation in glassy polymer)
def co2_plasticised(pressure_pa: float, temp_k: float, x: np.ndarray) -> float:
    p_bar = pressure_pa / 1e5
    return gpu_to_si(60.0) * (1.0 + 0.015 * p_bar)

co2 = ComponentSpec(name="CO2", feed_mole_fraction=0.0003, permeance=co2_plasticised)
```

Because `PermeanceFn` is a `Protocol`, `ty` will structurally validate any callable
passed as `permeance` — wrong argument types are caught at check time.

---

## 2. Pydantic v2 for all user-facing models

### Decision
`ModuleConfig`, `ComponentSpec`, and `FiberGeometry` are Pydantic BaseModels.
`SimulationResults` is a plain dataclass (it's output, not validated input).

### Rationale
- Free validation with clear error messages when users pass bad inputs
- Serialisation to/from JSON/dict for saving simulation setups
- `arbitrary_types_allowed=True` in `ConfigDict` is required for `PermeanceFn`

### Key validators to implement

```python
# In ComponentSpec
@field_validator("feed_mole_fraction")
@classmethod
def must_be_fraction(cls, v: float) -> float:
    if not 0.0 < v < 1.0:
        raise ValueError(f"feed_mole_fraction must be in (0, 1), got {v}")
    return v

# In ModuleConfig — validate after all fields are set
@model_validator(mode="after")
def pressures_must_be_positive_and_ordered(self) -> "ModuleConfig":
    if self.permeate_pressure_pa >= self.feed_pressure_pa:
        raise ValueError("Permeate pressure must be less than feed pressure")
    return self
```

### Feed composition normalisation
The sum of all `feed_mole_fraction` values must equal 1.0 (within tolerance 1e-6).
This check belongs on a `FeedSpec` or `Simulation` model that holds a list of
`ComponentSpec`, not on the individual component.

---

## 3. SI units internally; conversion at the boundary only

### Decision
All internal computation uses SI. Conversion happens only in:
- `utils/unit_conv.py` (conversion functions)
- Pydantic model constructors (via `from_gpu()`, `bara_to_pa()`)

### Rationale
Mixed units are a common source of silent bugs. Enforcing SI internally means
any dimensional error produces a physically nonsensical result rather than
a plausible-looking wrong answer.

### Unit conversion reference

| Quantity | User-facing unit | SI unit | Conversion |
|----------|-----------------|---------|------------|
| Permeance | GPU | kmol/(m²·s·Pa) | × 3.346×10⁻¹³ |
| Pressure | bara | Pa | × 1×10⁵ |
| Flow | m³(STP)/h | kmol/s | × 1/22.414/3600 |
| Temperature | °C | K | + 273.15 |
| Length | μm | m | × 1×10⁻⁶ |

```python
# utils/unit_conv.py

GPU_TO_SI = 3.346e-13  # kmol / (m² · s · Pa) per GPU

def gpu_to_si(gpu: float) -> float:
    """Convert permeance from GPU to SI (kmol/(m²·s·Pa))."""
    return gpu * GPU_TO_SI

def bara_to_pa(bara: float) -> float:
    """Convert pressure from bara to Pa."""
    return bara * 1.0e5

def celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15

def stp_m3h_to_kmol_s(flow_stp_m3h: float) -> float:
    """Convert volumetric flow at STP (m³/h) to molar flow (kmol/s)."""
    return flow_stp_m3h / 22.414 / 3600.0
```

---

## 4. Cross-flow as mandatory initial guess

### Decision
`CountercurrentSolver` and `CocurrentSolver` always call `CrossflowSolver` first
and use the result as the starting point for iteration. This is not optional.

### Rationale
The paper explicitly states this (p. 1293):
> "The cross-flow model generally provides an excellent starting point for the
> countercurrent or cocurrent model and, therefore, largely alleviates potential
> problems associated with the limited radius of convergence of our successive
> substitution method."

The successive substitution scheme (Thomas algorithm loop) can fail to converge
if the initial guess is poor, particularly at high stage cuts (>90%). The
cross-flow model provides an initial guess that already respects the residue-side
concentration profile, which is much better than a complete-mixing assumption.

---

## 5. Thomas algorithm — explicit, not scipy

### Decision
Implement the Thomas algorithm directly in `numerics/thomas.py`.
Do not use `scipy.linalg.solve_banded` for this.

### Rationale
- The tridiagonal structure is fixed and known — no benefit from the generality
  of scipy's banded solver
- Direct implementation is O(N) with no overhead
- Testable in isolation with exact analytical solutions for small systems
- Easier to inspect and debug intermediate values

```python
# numerics/thomas.py

import numpy as np

def thomas_solve(
    lower: np.ndarray,   # shape (N,) — lower diagonal, lower[0] unused
    diag: np.ndarray,    # shape (N,) — main diagonal
    upper: np.ndarray,   # shape (N,) — upper diagonal, upper[-1] unused
    rhs: np.ndarray,     # shape (N,) — right-hand side
) -> np.ndarray:
    """
    Solve a tridiagonal system Ax = rhs using the Thomas algorithm.

    O(N) time and O(N) space.
    Raises ValueError if a zero pivot is encountered (singular system).

    Parameters
    ----------
    lower : shape (N,)
        Sub-diagonal. lower[k] is the coefficient of x[k-1] in equation k.
        lower[0] is ignored.
    diag : shape (N,)
        Main diagonal.
    upper : shape (N,)
        Super-diagonal. upper[k] is the coefficient of x[k+1] in equation k.
        upper[-1] is ignored.
    rhs : shape (N,)
        Right-hand side vector.

    Returns
    -------
    np.ndarray, shape (N,)
        Solution vector x.
    """
    ...
```

---

## 6. Results as a dataclass with `to_dataframe()`

### Decision
`SimulationResults` is a `dataclass`, not a Pydantic model.
It exposes a `to_dataframe()` method for post-processing.

### Rationale
Results are output — they don't need validation. A plain dataclass is lighter and
has no risk of coercing computed values. Pandas earns its place in `to_dataframe()`
for tabular export and downstream plotting.

```python
# models/results.py

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationResults:
    """
    Results from a single hollow-fiber membrane module simulation.

    All compositions are mole fractions.
    All flows are in kmol/s.
    All pressures are in Pa.
    """
    component_names: tuple[str, ...]

    # Bulk outlet conditions
    feed_composition: np.ndarray       # shape: (n_components,)
    residue_composition: np.ndarray    # shape: (n_components,)
    permeate_composition: np.ndarray   # shape: (n_components,)
    residue_flow_kmol_s: float
    permeate_flow_kmol_s: float
    stage_cut: float                   # permeate / feed
    residue_recovery: float            # residue / feed

    # Axial profiles — shape: (n_components, n_stages)
    axial_positions: np.ndarray        # z/L, shape: (n_stages,)
    residue_profiles: np.ndarray       # xⱼ,ₖ
    permeate_profiles: np.ndarray      # yⱼ,ₖ
    pressure_profile: np.ndarray       # Pa, shape: (n_stages,)

    # Solver diagnostics
    n_iterations: int
    n_stages: int
    converged: bool

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return axial profiles as a tidy DataFrame.

        Columns: z_over_L, pressure_pa, then one column per component
        for both residue (x_{name}) and permeate (y_{name}) mole fractions.
        """
        ...
```

---

## 7. Solver as a stateless class

### Decision
Solvers are instantiated with config and components, then `solve()` is called.
No internal state is mutated after `solve()` returns. `solve()` can be called
multiple times and will return identical results.

```python
# solvers/base.py

from abc import ABC, abstractmethod
from ..models.module import ModuleConfig
from ..models.components import ComponentSpec
from ..models.results import SimulationResults


class BaseSolver(ABC):
    def __init__(
        self,
        module: ModuleConfig,
        components: list[ComponentSpec],
    ) -> None:
        self.module = module
        self.components = components
        self._validate_feed_composition()

    def _validate_feed_composition(self) -> None:
        total = sum(c.feed_mole_fraction for c in self.components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Feed mole fractions must sum to 1.0, got {total:.8f}"
            )

    @abstractmethod
    def solve(self) -> SimulationResults: ...
```
