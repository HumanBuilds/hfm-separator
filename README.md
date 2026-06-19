# hfm-separator

[![CI](https://github.com/HumanBuilds/hfm-separator/actions/workflows/ci.yml/badge.svg)](https://github.com/HumanBuilds/hfm-separator/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#commands)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Multicomponent gas separation simulator for hollow-fiber membrane contactors.

Implements the numerical model from:

> Coker, D.T., Freeman, B.D., Fleming, G.K. (1998). "Modeling Multicomponent Gas
> Separation Using Hollow-Fiber Membrane Contactors." *AIChE Journal*, 44(6), 1289–1302.

Supports cocurrent, countercurrent, and cross-flow contacting patterns with optional
permeate purging. Permeance can be constant or a function of local pressure,
temperature, and composition.

---

## Quickstart

```bash
uv sync
uv run python examples/air_separation.py
```

---

## Commands

```bash
uv run pytest                         # full test suite
uv run pytest tests/unit/             # unit tests only
uv run pytest tests/integration/      # integration tests (slower)
uv run ruff check src/ tests/         # lint
uv run ruff format src/ tests/        # format
uv run ty check src/                  # type check
```

---

## Example

```python
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.permeance import from_gpu
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin

module = ModuleConfig(
    n_fibers=300_000,
    fiber_od_m=300e-6,
    fiber_id_m=150e-6,
    active_length_m=1.0,
    pot_length_m=0.1,
    feed_pressure_pa=bara_to_pa(10.0),
    permeate_pressure_pa=bara_to_pa(1.0),
    feed_temp_k=celsius_to_kelvin(40.0),
    feed_side="bore",
)

components = [
    ComponentSpec(name="N2",  feed_mole_fraction=0.7841, permeance=from_gpu(3.57)),
    ComponentSpec(name="O2",  feed_mole_fraction=0.2084, permeance=from_gpu(20.0)),
    ComponentSpec(name="CO2", feed_mole_fraction=0.0003, permeance=from_gpu(60.0)),
    ComponentSpec(name="H2O", feed_mole_fraction=0.0072, permeance=from_gpu(1000.0)),
]

result = CountercurrentSolver(module=module, components=components).solve()
print(f"N2 purity:  {result.residue_composition[0]:.4f}")
print(f"O2 purity:  {result.permeate_composition[1]:.4f}")
print(f"Recovery:   {result.residue_recovery:.4f}")
```

---

## Documentation

Full mathematical model, design decisions, and testing strategy are in `docs/`:

- `docs/numerics.md` — all equations with paper references
- `docs/architecture.md` — module structure and data flow
- `docs/design.md` — design decisions with rationale and code sketches
- `docs/testing.md` — test strategy and numerical targets from paper figures

---

## Development

```bash
uv sync --all-extras       # install runtime + dev + docs dependencies
uv run pre-commit install  # enable lint/format/type hooks on commit
uv run pytest              # run the full suite (coverage gate: 90%)
```

CI (lint, format, type-check, tests on Python 3.12 & 3.13, and a packaging
build) runs on every push and pull request via GitHub Actions.

---

## Project Structure

```
src/hfm_separator/
├── models/        # Pydantic input models + results dataclass
├── numerics/      # Mathematical primitives (Thomas, Wilke, Hagen-Poiseuille)
├── solvers/       # Cross-flow, countercurrent, cocurrent
└── utils/         # Unit conversions
```

## License

MIT
