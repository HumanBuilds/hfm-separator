# Testing Strategy

---

## Philosophy

The paper provides exact numerical targets for three case studies. This is unusual
and valuable — use it. The integration tests are regression tests against published
results. If a refactor breaks a paper figure, the tests will catch it.

Unit tests target the mathematical primitives. Integration tests target the
published figures. Together they should reach ≥ 90% coverage.

---

## Unit Tests

### `test_thomas.py`

Test the Thomas algorithm against systems with known exact solutions.

**Test 1 — 2×2 system (trivial)**
```
[2  1] [x1]   [3]
[1  2] [x2] = [3]
→ x = [1, 1]
```

**Test 2 — 5×5 system (known solution)**
Construct from a known solution vector, verify round-trip.

**Test 3 — near-singular system**
Verify that a near-zero pivot raises `ValueError` rather than returning
silently wrong results.

**Test 4 — large random system**
Generate a random diagonally dominant tridiagonal system, solve with Thomas,
verify against `np.linalg.solve`. Tolerance: 1e-10.

---

### `test_viscosity.py`

Test the Wilke mixing rule.

**Test 1 — pure component**
Single component → mixture viscosity equals pure viscosity.

**Test 2 — equal binary**
50/50 N₂/O₂ mixture at 298 K. Reference values from NIST:
- μ(N₂, 298 K) ≈ 1.789×10⁻⁵ Pa·s
- μ(O₂, 298 K) ≈ 2.031×10⁻⁵ Pa·s
- μ(mixture) ≈ 1.91×10⁻⁵ Pa·s (tolerance ±2%)

**Test 3 — limiting behaviour**
As y₁ → 1, μmix → μ₁.

---

### `test_pressure.py`

**Test 1 — Hagen-Poiseuille (Eq. 21)**
Known geometry, known flow → verify pressure drop against manual calculation.

**Test 2 — Eq. A5 reduces to Eq. 21**
At conditions satisfying A9 and A16, verify that Eq. A5 gives the same result as
Eq. 21 to within 0.1%.

**Test 3 — Compressibility criterion (Eq. A9)**
Use paper's worked example values and verify the dimensionless group >> 1.
Expected: ~200 (paper p. 1301).

**Test 4 — Permeability criterion (Eq. A16)**
Use paper's worked example values.
Expected: ~1.4×10⁷ (paper p. 1302).

---

### `test_staging.py`

**Test 1 — Air separation, N₂**
Parameters from Table 1 and 2 of the paper. Verify N ≥ 100.

**Test 2 — Δxmax sensitivity**
Verify that halving Δxmax doubles N (approximately linear relationship).

**Test 3 — High-permeability component dominates**
In the air mixture, H₂O has permeance 1000 GPU. Verify it produces the largest N
and therefore sets the stage count for the simulation.

---

### `test_unit_conv.py`

Round-trip tests for all conversions. Exact expected values:

| Input | Function | Expected output |
|-------|----------|-----------------|
| 1 GPU | `gpu_to_si` | 3.346×10⁻¹³ kmol/(m²·s·Pa) |
| 10 bara | `bara_to_pa` | 1.0×10⁶ Pa |
| 1 bara | `bara_to_pa` | 1.0×10⁵ Pa |
| 0 °C | `celsius_to_kelvin` | 273.15 K |
| 22.414 m³(STP)/h | `stp_m3h_to_kmol_s` | 1/3600 kmol/s |

---

## Integration Tests

### Simulation parameters from Table 1

| Parameter | Air | Ternary | H₂ |
|-----------|-----|---------|-----|
| Feed side | Bore | Shell | Shell |
| Feed pressure (bara) | 10 | 10 | 76.9 / 42.4 |
| Permeate pressure (bara) | 1 | 1 | 42.4 / 7.9 |
| Feed temp (°C) | 40 | 25 | 50 |
| Fiber OD/ID (μm) | 300/150 | 300/150 | 300/150 |
| Active length (m) | 1 | 1 | 1 |
| Pot length (m) | 0.1 | 0.1 | 0.1 |
| N fibers | 300,000 | 350,000 | 500,000 |

---

### `test_air_separation.py`

Permeances from Table 2 (polysulfone-like, 0.1 μm effective layer):

| Component | Feed mole fraction | Permeance (GPU) |
|-----------|-------------------|-----------------|
| N₂ | 0.7841 | 3.57 |
| O₂ | 0.2084 | 20.0 |
| CO₂ | 0.0003 | 60.0 |
| H₂O | 0.0072 | 1000.0 |

**Test targets (read from Figures 4, 5, 8, 10):**

`test_n2_purity_no_purge_10bar`:
- At ~50% residue recovery, 10 bar, 0% purge → N₂ purity ≈ 0.90 (±0.01)

`test_n2_purity_no_purge_5bar`:
- At ~50% residue recovery, 5 bar, 0% purge → N₂ purity ≈ 0.85 (±0.01)

`test_purge_reduces_overall_recovery`:
- At fixed 98% N₂ purity, increasing purge from 0% to 10% must DECREASE R'/F
- This is the key finding from Figure 6

`test_dew_point_decreases_with_purge`:
- At 86% N₂ purity, 1000 GPU water permeance:
  - 0% purge → dew point ≈ −41.6°C (±3°C)
  - 10% purge → dew point ≈ −89.7°C (±3°C)
- Source: paper p. 1298

`test_dew_point_lower_at_higher_permeance`:
- 1000 GPU water permeance always gives lower dew point than 350 GPU at same conditions

---

### `test_h2_separation.py`

Permeances from Table 3:

| Component | Feed mole fraction | Permeance (GPU) |
|-----------|-------------------|-----------------|
| H₂ | 0.650 | 100.0 |
| C₂H₄ | 0.025 | 3.03 |
| CH₄ | 0.210 | 2.86 |
| C₂H₆ | 0.080 | 2.00 |
| C₃H₈ | 0.035 | 1.89 |

**Test targets (read from Figures 11, 12):**

`test_higher_pressure_ratio_better_purity`:
- Case 1: pf=42.4, pp=7.9 bara → pf/pp = 5.3
- Case 2: pf=76.9, pp=42.4 bara → pf/pp = 1.8
- At 60% H₂ recovery: Case 1 purity > Case 2 purity (Figure 11)

`test_ch4_composition_profile_has_maximum`:
- At 94.6% permeate recovery, pf=42.4, pp=7.9:
  - CH₄ mole fraction in residue has an interior maximum vs axial position
  - Maximum is not at z/L=0 or z/L=1
- This is the key finding from Figure 12

`test_h2_depleted_at_residue_end`:
- At very high recovery (>90%), H₂ mole fraction at residue end (z/L=0) < 0.05

`test_high_recovery_convergence`:
- At 94.6% permeate recovery, solver must converge (no divergence)
- This was a case where shooting methods failed (paper p. 1298)

---

### `test_ternary.py`

Permeances from Table 4 (equimolar feed, max selectivity 50):

| Component | Feed mole fraction | Permeance (GPU) |
|-----------|-------------------|-----------------|
| 1 (most permeable) | 0.3333 | 500.0 |
| 2 (intermediate) | 0.3333 | 100.0 |
| 3 (least permeable) | 0.3334 | 10.0 |

Module parameters: Shell feed, 10/1 bara, 25°C, 350,000 fibers (Table 1 ternary column).

**Test targets (read from Figure 13):**

`test_most_permeable_depletes_monotonically`:
- Component 1 residue mole fraction must decrease monotonically from z/L=1 to z/L=0

`test_least_permeable_enriches_monotonically`:
- Component 3 residue mole fraction must increase monotonically from z/L=1 to z/L=0

`test_intermediate_component_has_axial_maximum`:
- Component 2 residue mole fraction must have an interior maximum
- Maximum must not be at z/L=0 or z/L=1
- Maximum value should be > 0.45 (estimated from Figure 13, ±0.05)

`test_mass_balance_closure`:
- At every stage: Σⱼ xⱼ,ₖ = 1.0 (±1e-6)
- At every stage: Σⱼ yⱼ,ₖ = 1.0 (±1e-6)
- Global: F = R'' + V (feed = residue + permeate, kmol/s)

---

## Shared Fixtures (`conftest.py`)

```python
import pytest
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.permeance import from_gpu
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin


@pytest.fixture
def air_module_10bar() -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=1.0,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_side="bore",
        purge_fraction=0.0,
    )


@pytest.fixture
def air_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(name="N2",  feed_mole_fraction=0.7841, permeance=from_gpu(3.57)),
        ComponentSpec(name="O2",  feed_mole_fraction=0.2084, permeance=from_gpu(20.0)),
        ComponentSpec(name="CO2", feed_mole_fraction=0.0003, permeance=from_gpu(60.0)),
        ComponentSpec(name="H2O", feed_mole_fraction=0.0072, permeance=from_gpu(1000.0)),
    ]


@pytest.fixture
def ternary_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(name="comp1", feed_mole_fraction=0.3333, permeance=from_gpu(500.0)),
        ComponentSpec(name="comp2", feed_mole_fraction=0.3333, permeance=from_gpu(100.0)),
        ComponentSpec(name="comp3", feed_mole_fraction=0.3334, permeance=from_gpu(10.0)),
    ]
```

---

## Coverage Requirements

- Overall: ≥ 90%
- `numerics/`: 100% (pure functions, no I/O)
- `solvers/`: ≥ 85% (some edge cases are hard to trigger)
- `models/`: ≥ 95% (validators should all be tested)
- `utils/`: 100%
