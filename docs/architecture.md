# Architecture

---

## Module Map

```
hfm_separator/
│
├── models/                     ← User-facing input/output types
│   ├── permeance.py            Protocol + factory functions
│   ├── components.py           ComponentSpec (Pydantic)
│   ├── module.py               ModuleConfig, FiberGeometry (Pydantic)
│   └── results.py              SimulationResults (dataclass)
│
├── numerics/                   ← Mathematical primitives, no I/O, no Pydantic
│   ├── thomas.py               Tridiagonal solver
│   ├── pressure.py             Hagen-Poiseuille + full Eq. A5
│   ├── viscosity.py            Wilke mixing rule
│   └── staging.py              Adaptive stage-count selection (Eq. 22)
│
├── solvers/                    ← Simulation orchestration
│   ├── base.py                 Abstract BaseSolver
│   ├── crossflow.py            Algebraic per-stage (also: initial-guess generator)
│   ├── countercurrent.py       Thomas algorithm + successive substitution
│   └── cocurrent.py            Direct stage marching
│
└── utils/
    └── unit_conv.py            GPU↔SI, bara↔Pa, STP flow conversions
```

---

## Data Flow

```
User input (GPU, bara, °C, m³(STP)/h)
        │
        ▼
   ModuleConfig          ComponentSpec(s)
   (Pydantic, SI)        (Pydantic, SI, PermeanceFn)
        │                       │
        └──────────┬────────────┘
                   │
                   ▼
           BaseSolver.__init__()
           └── validates feed composition sums to 1
                   │
                   ▼
           CrossflowSolver.solve()        ← always runs first
           └── Newton on Eq. 18          ← provides initial guess
           └── Eq. 22 → N stages
                   │
                   ▼ (initial guess)
           CountercurrentSolver  or  CocurrentSolver
           └── Thomas algorithm (Eq. 14) per component per iteration
           └── successive substitution until Eqs. 16-17 satisfied
           └── Hagen-Poiseuille pressure update (Eq. 21)
                   │
                   ▼
           SimulationResults (frozen dataclass, SI)
                   │
                   ▼
           .to_dataframe()  →  pandas DataFrame  →  plotting / export
```

---

## Key Boundaries

### models/ ↔ solvers/
Solvers receive `ModuleConfig` and `list[ComponentSpec]` and return
`SimulationResults`. All values are in SI at this boundary. Solvers never
call unit conversion functions.

### models/ ↔ numerics/
Numerics functions receive and return plain numpy arrays and floats (SI).
They have no knowledge of `ComponentSpec` or `ModuleConfig`. This makes
them independently testable.

### solvers/ ↔ numerics/
Solvers orchestrate calls to numerics functions, manage iteration loops,
assemble the tridiagonal system, and track convergence. The separation
means you can swap in a different pressure model or viscosity correlation
without touching the solver logic.

---

## ModuleConfig

```python
# models/module.py

from typing import Literal
from pydantic import BaseModel, model_validator


class ModuleConfig(BaseModel):
    """
    Physical configuration of a hollow-fiber membrane module.

    All values in SI units. Use utility functions in unit_conv.py
    to convert from common engineering units before constructing.

    Attributes
    ----------
    n_fibers : int
        Total number of hollow fibers in the module bundle.
    fiber_od_m : float
        Fiber outer diameter (m). Separation membrane is on the exterior.
    fiber_id_m : float
        Fiber inner diameter (m). Sets bore cross-section for pressure drop.
    active_length_m : float
        Active (permeating) fiber length = total length − 2 × pot_length_m.
    pot_length_m : float
        Length of potted (non-permeating) fiber at each end (m).
    feed_pressure_pa : float
        Feed-side inlet pressure (Pa).
    permeate_pressure_pa : float
        Permeate-side outlet pressure (Pa).
    feed_temp_k : float
        Feed gas temperature (K). Assumed isothermal through module.
    feed_side : {"bore", "shell"}
        Which side of the fiber receives the feed gas.
    purge_fraction : float
        Fraction of residue flow used as permeate sweep (0 = no purge).
    """
    n_fibers: int
    fiber_od_m: float
    fiber_id_m: float
    active_length_m: float
    pot_length_m: float
    feed_pressure_pa: float
    permeate_pressure_pa: float
    feed_temp_k: float
    feed_side: Literal["bore", "shell"] = "shell"
    purge_fraction: float = 0.0

    @model_validator(mode="after")
    def check_pressures(self) -> "ModuleConfig":
        if self.permeate_pressure_pa >= self.feed_pressure_pa:
            raise ValueError(
                "permeate_pressure_pa must be less than feed_pressure_pa"
            )
        return self

    @model_validator(mode="after")
    def check_geometry(self) -> "ModuleConfig":
        if self.fiber_id_m >= self.fiber_od_m:
            raise ValueError("fiber_id_m must be less than fiber_od_m")
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
        """Active membrane area based on outer diameter (m²). Eq. 1 context."""
        import math
        return math.pi * self.fiber_od_m * self.active_length_m * self.n_fibers
```

---

## Solver Pattern

All solvers follow this pattern:

```python
# Instantiate with config
solver = CountercurrentSolver(module=cfg, components=specs)

# Run simulation — returns immutable results
result = solver.solve()

# Inspect outputs
print(f"N2 purity: {result.residue_composition[0]:.4f}")
print(f"Recovery:  {result.residue_recovery:.4f}")

# Export to DataFrame for plotting
df = result.to_dataframe()
```

The solver is reusable — you can change module config and components by
constructing a new solver instance. This makes parameter sweeps clean:

```python
purge_fractions = [0.0, 0.02, 0.05, 0.10]
results = []
for pf in purge_fractions:
    cfg = base_module.model_copy(update={"purge_fraction": pf})
    results.append(CountercurrentSolver(cfg, components).solve())
```

---

## Error Handling

| Condition | Where caught | Exception type |
|-----------|-------------|----------------|
| Invalid mole fraction | `ComponentSpec` validator | `ValueError` |
| Mole fractions don't sum to 1 | `BaseSolver.__init__` | `ValueError` |
| Permeate pressure ≥ feed pressure | `ModuleConfig` validator | `ValueError` |
| Zero pivot in Thomas algorithm | `thomas_solve` | `ValueError` |
| Non-convergence after max iterations | Solver `solve()` | `RuntimeError` |
| Negative computed flow rate | Solver iteration | `RuntimeError` |

Non-convergence should include the iteration count and final residuals in the
error message to help with debugging.
