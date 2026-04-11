# Numerical Model

All equation numbers refer to:
> Coker, D.T., Freeman, B.D., Fleming, G.K. (1998). AIChE Journal, 44(6), 1289–1302.

---

## Overview

The simulator divides a single representative hollow fiber into N stages in the axial
direction and enforces component mass balances on each stage. This is formally
equivalent to a first-order finite-difference discretisation of the underlying
differential mass and pressure distribution equations.

Results for a single fiber are scaled by the number of fibers to give module totals.

---

## Notation

| Symbol | Description | Units |
|--------|-------------|-------|
| R | Number of components | — |
| N | Number of stages | — |
| k | Stage index (1 = residue end, N = feed end) | — |
| Lₖ | Total feed-side flow leaving stage k | kmol/s |
| Vₖ | Total permeate-side flow leaving stage k | kmol/s |
| ℓⱼ,ₖ | Component j flow on feed side of stage k | kmol/s |
| υⱼ,ₖ | Component j flow on permeate side of stage k | kmol/s |
| xⱼ,ₖ | Mole fraction of j on feed side of stage k | — |
| yⱼ,ₖ | Mole fraction of j on permeate side of stage k | — |
| ṁⱼ,ₖ | Transmembrane mass flow of j at stage k | kmol/s |
| Qⱼ | Permeance of component j | kmol/(m²·s·Pa) |
| ΔAₖ | Membrane area of stage k | m² |
| PL | Feed-side (high) pressure | Pa |
| PV | Permeate-side (low) pressure | Pa |
| Rᵢ | Fiber inner radius | m |
| R₀ | Fiber outer radius | m |
| L | Active fiber length | m |
| Nf | Number of fibers | — |

---

## Core Relationships (all flow patterns)

**Mole fraction to component flow:**
```
ℓⱼ,ₖ = xⱼ,ₖ · Lₖ                          (Eq. 2)
υⱼ,ₖ = yⱼ,ₖ · Vₖ                          (Eq. 3)
```

**Total flows from component flows:**
```
Lₖ = Σⱼ ℓⱼ,ₖ                              (Eq. 4)
Vₖ = Σⱼ υⱼ,ₖ                              (Eq. 5)
```

**Membrane area per stage:**
```
ΔAₖ = 2π·R₀·L·Nf / N                      (Eq. 1)
```

**Transmembrane flux (solution-diffusion):**
```
ṁⱼ,ₖ = Qⱼ · ΔAₖ · (PL,k · xⱼ,ₖ − PV,k · yⱼ,ₖ)   (Eq. 8)
```

---

## Countercurrent Flow

### Mass balance on stage k
```
ℓⱼ,ₖ₊₁ − ℓⱼ,ₖ + υⱼ,ₖ₋₁ − υⱼ,ₖ = 0       (Eq. 6)
ṁⱼ,ₖ = ℓⱼ,ₖ₊₁ − ℓⱼ,ₖ                     (Eq. 7)
```

### Elimination to tridiagonal form
Substituting Eqs. 2, 3, 7, 8 into Eq. 6 and rearranging:
```
Bⱼ,ₖ · ℓⱼ,ₖ₋₁ + Cⱼ,ₖ · ℓⱼ,ₖ + Dⱼ,ₖ · ℓⱼ,ₖ₊₁ = 0    (Eq. 10)
```

**Coefficients (Eqs. 11–13):**
```
Bⱼ,ₖ = −Vₖ₋₁ / (PV,k-1 · ΔAₖ₋₁ · Qⱼ) · (1 + Qⱼ·ΔAₖ₋₁·PL,k-1 / Lₖ₋₁)

Cⱼ,ₖ = 1 + Vₖ₋₁/(PV,k-1·ΔAₖ₋₁·Qⱼ) + Vₖ/(PV,k·ΔAₖ·Qⱼ) · (1 + Qⱼ·ΔAₖ·PL,k / Lₖ)

Dⱼ,ₖ = −Vₖ / (PV,k · ΔAₖ · Qⱼ) − 1
```

For k=1 (residue end), V₀ is the sweep/purge flow rate.

### Tridiagonal matrix system (Eq. 14)
Written once per component j, yielding N equations in N unknowns {ℓⱼ,₁ … ℓⱼ,N}.
Solved with the Thomas algorithm.

### Permeate side flows (Eq. 15)
```
Vₖ = Vₖ₋₁ + Lₖ₊₁ − Lₖ
```

### Convergence criteria (Eqs. 16–17)
```
|ΔL₁ / L₁|       < 1e-8
|ΔV_{N+1}/V_{N+1}| < 1e-8
```
where Δ denotes the change between successive iterations.

---

## Cocurrent Flow

### Mass balance on stage k
```
ℓⱼ,ₖ + υⱼ,ₖ − ℓⱼ,ₖ₊₁ − υⱼ,ₖ₊₁ = 0      (Eq. 23)
ṁⱼ,ₖ = ℓⱼ,ₖ₊₁ − ℓⱼ,ₖ                     (Eq. 24)
```

### Direct stage calculation (Eq. 26)
No tridiagonal solve needed. Integration proceeds from feed end (k=N) to residue end (k=1):
```
         υⱼ,ₖ₊₁ + (1 + Vₖ/(Qⱼ·ΔAₖ·PV,k)) · ℓⱼ,ₖ₊₁
ℓⱼ,ₖ = ─────────────────────────────────────────────────
         1 + Vₖ/(Qⱼ·ΔAₖ·PV,k) + PL,k·Vₖ/(PV,k·Lₖ)
```

### Convergence criterion (Eq. 27)
```
|ΔV₁ / V₁| < 1e-8
```

---

## Cross-flow

Permeate composition on a stage depends only on upstream composition, permeances,
and pressures — not on downstream stages. This makes it algebraically explicit
stage-by-stage starting from the feed end.

### Permeate composition (Eq. 18)
```
yⱼ,ₖ − Qⱼ·(PL,k·xⱼ,ₖ − PV,k·yⱼ,ₖ) / Σₘ Qₘ·(PL,k·xₘ,ₖ − PV,k·yₘ,ₖ) = 0
```
This is a set of R−1 nonlinear equations per stage. Solve with Newton's method.

### Initial guess for stage N (feed stage)

**Selectivity-limited** (pressure ratio >> selectivity):
```
yⱼ,N = Qⱼ·xⱼ,N₊₁ / Σₘ Qₘ·xₘ,N₊₁                (Eq. 19)
```

**Pressure-ratio-limited** (selectivity >> pressure ratio):
```
yⱼ,N = xⱼ,N · PL,N / PV,N                          (Eq. 20)
```

If Eq. 19 predicts a partial pressure in the permeate exceeding that in the feed for
any component, use Eq. 20 for that component. If Σyⱼ > 1, renormalise.

---

## Bore-side Pressure Drop

### Hagen-Poiseuille (Eq. 21) — primary model
```
PV,k-1 − PV,k = (8·μmix / π·Rᵢ⁴) · Vₖ · (RT/PV,k) · Δz
```
where Δz = L/N and μmix is computed by the Wilke mixing rule.

Valid for: steady-state, laminar, incompressible, Newtonian, impermeable-wall flow.
The appendix (Eq. A5) shows this holds for all cases considered in the paper.

### Full mechanical energy balance (Eq. A5) — for extensibility
```
ln(PL,k+1/PL,k) = (8μmix·v̄ₖ/Rᵢ²) · (Lₖ/PL,k·Lₖ₊₁) · Δz
                 + (RT/2M) · (Lₖ/A)²/PL,k² · [Lₖ/Lₖ₊₁ − (PL,k/PL,k+1)²]
                 + (M/2RT) · v̄w² · (Lₖ₊₁−Lₖ)/Lₖ₊₁
```

**Criterion for gas compressibility effects to be negligible (Eq. A9):**
```
PL,k² / (RTM · (Lₖ/A)²) >> 1
```

**Criterion for fiber-wall permeability effects to be negligible (Eq. A16):**
```
(8/π²) · μmix · Lₖ² / (Rᵢ²·R₀·M) · 1/(Q·PL,k)³ >> 1
```

The paper demonstrates both criteria are satisfied for all cases considered.
Implement `pressure.py` to support both Eq. 21 and Eq. A5, selecting via a flag.

---

## Adaptive Stage Count (Eq. 22)

The number of stages N is chosen per-component, and the maximum is used:
```
N = 2π·R₀·L·Nf·(1 − xFⱼ + Δxmax)·Qⱼ·PF·xFⱼ / (F·Δxmax)
```

- Set Δxmax = 0.005 (as used in the paper)
- Round up to the nearest 100
- Minimum N = 100; at stage cuts > 90%, N may reach 1,000

---

## Wilke Viscosity Mixing Rule

```
μmix = Σᵢ (yᵢ·μᵢ) / Σⱼ yⱼ·φᵢⱼ

φᵢⱼ = [1 + (μᵢ/μⱼ)^0.5 · (Mⱼ/Mᵢ)^0.25]² / [2√2 · (1 + Mᵢ/Mⱼ)^0.5]
```

Pure-component viscosities μᵢ are properties of `ComponentSpec` and must be
supplied by the user. Values for the case studies are given in `docs/testing.md`.

---

## Principal Model Assumptions

1. Shell-side pressure change is negligible
2. Bore-side pressure given by Hagen-Poiseuille
3. Hollow fibers: very thin dense membrane on porous support; all mass-transfer
   resistance confined to separation membrane
4. No axial mixing on either shell or lumen side
5. Plug flow on both sides
6. Single representative fiber scaled to module total by Nf
7. No fiber deformation under pressure
8. Uniform fiber geometry throughout module
9. Steady-state operation
