"""Visual comparison of simulation results against Coker et al. (1998).

Reproduces the signature figures from the paper using this package's
solvers, and annotates the paper's reported numerical targets inline so
the PNG outputs can be placed side-by-side with the paper's PDF.

Covered:
    Figs 4 & 5 — N₂ residue purity vs residue recovery (10 bar and 5 bar)
    Figs 6/8/10 — purge effect on recovery, N₂ purity and permeate dew point
    Fig 11      — H₂ permeate purity vs H₂ recovery at two pressure ratios
    Fig 12      — H₂/hydrocarbon axial profiles (CH₄ interior maximum)
    Fig 13      — Ternary axial profiles (intermediate component interior maximum)

Run from the repo root::

    uv run python examples/paper_figures.py
    uv run python examples/paper_figures.py --show     # interactive
    uv run python examples/paper_figures.py --output-dir my_outputs/

Sweeps are priced for speed not precision — expect ~60–90 seconds.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.models.results import SimulationResults
from hfm_separator.solvers.base import BaseSolver
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.solvers.crossflow import CrossflowSolver
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin

# ── Component fixtures (standalone — not imported from tests/) ─────────────


def air_components() -> list[ComponentSpec]:
    """Air + trace CO₂/H₂O with Table 2 permeances (polysulfone-like)."""
    return [
        ComponentSpec(
            name="N2",
            feed_mole_fraction=0.7841,
            permeance=from_gpu(3.57),
            molar_mass_kg_per_kmol=28.0134,
            pure_viscosity_pa_s=1.85e-5,
        ),
        ComponentSpec(
            name="O2",
            feed_mole_fraction=0.2084,
            permeance=from_gpu(20.0),
            molar_mass_kg_per_kmol=31.9988,
            pure_viscosity_pa_s=2.08e-5,
        ),
        ComponentSpec(
            name="CO2",
            feed_mole_fraction=0.0003,
            permeance=from_gpu(60.0),
            molar_mass_kg_per_kmol=44.01,
            pure_viscosity_pa_s=1.55e-5,
        ),
        ComponentSpec(
            name="H2O",
            feed_mole_fraction=0.0072,
            permeance=from_gpu(1000.0),
            molar_mass_kg_per_kmol=18.0153,
            pure_viscosity_pa_s=1.01e-5,
        ),
    ]


def h2_components() -> list[ComponentSpec]:
    """5-component H₂/hydrocarbon feed with Table 3 permeances."""
    return [
        ComponentSpec(
            name="H2",
            feed_mole_fraction=0.650,
            permeance=from_gpu(100.0),
            molar_mass_kg_per_kmol=2.016,
            pure_viscosity_pa_s=0.93e-5,
        ),
        ComponentSpec(
            name="C2H4",
            feed_mole_fraction=0.025,
            permeance=from_gpu(3.03),
            molar_mass_kg_per_kmol=28.054,
            pure_viscosity_pa_s=1.07e-5,
        ),
        ComponentSpec(
            name="CH4",
            feed_mole_fraction=0.210,
            permeance=from_gpu(2.86),
            molar_mass_kg_per_kmol=16.043,
            pure_viscosity_pa_s=1.17e-5,
        ),
        ComponentSpec(
            name="C2H6",
            feed_mole_fraction=0.080,
            permeance=from_gpu(2.00),
            molar_mass_kg_per_kmol=30.07,
            pure_viscosity_pa_s=1.02e-5,
        ),
        ComponentSpec(
            name="C3H8",
            feed_mole_fraction=0.035,
            permeance=from_gpu(1.89),
            molar_mass_kg_per_kmol=44.097,
            pure_viscosity_pa_s=0.90e-5,
        ),
    ]


def ternary_components() -> list[ComponentSpec]:
    """Equimolar model ternary with Q₁:Q₂:Q₃ = 500:100:10 GPU (Table 4)."""
    return [
        ComponentSpec(
            name="fast",
            feed_mole_fraction=0.3333,
            permeance=from_gpu(500.0),
            molar_mass_kg_per_kmol=30.0,
            pure_viscosity_pa_s=1.5e-5,
        ),
        ComponentSpec(
            name="mid",
            feed_mole_fraction=0.3333,
            permeance=from_gpu(100.0),
            molar_mass_kg_per_kmol=30.0,
            pure_viscosity_pa_s=1.5e-5,
        ),
        ComponentSpec(
            name="slow",
            feed_mole_fraction=0.3334,
            permeance=from_gpu(10.0),
            molar_mass_kg_per_kmol=30.0,
            pure_viscosity_pa_s=1.5e-5,
        ),
    ]


# ── Module fixtures ────────────────────────────────────────────────────────


def air_module(
    feed_flow_kmol_s: float,
    pressure_bar: float = 10.0,
    purge_fraction: float = 0.0,
) -> ModuleConfig:
    return ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(pressure_bar),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=feed_flow_kmol_s,
        feed_side="bore",
        purge_fraction=purge_fraction,
    )


def h2_module(feed_flow_kmol_s: float, high_ratio: bool = True) -> ModuleConfig:
    if high_ratio:
        feed_pa, perm_pa = bara_to_pa(42.4), bara_to_pa(7.9)
    else:
        feed_pa, perm_pa = bara_to_pa(76.9), bara_to_pa(42.4)
    return ModuleConfig(
        n_fibers=500_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=feed_pa,
        permeate_pressure_pa=perm_pa,
        feed_temp_k=celsius_to_kelvin(50.0),
        feed_flow_kmol_s=feed_flow_kmol_s,
        feed_side="shell",
    )


def ternary_module(feed_flow_kmol_s: float) -> ModuleConfig:
    return ModuleConfig(
        n_fibers=350_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(25.0),
        feed_flow_kmol_s=feed_flow_kmol_s,
        feed_side="shell",
    )


# ── Helpers ────────────────────────────────────────────────────────────────

_PHYSICAL_FLOW_THRESHOLD = 1.0e-20


def dew_point_celsius(partial_pressure_pa: float) -> float:
    """Dew point from water vapor partial pressure via the Magnus formula.

    Uses the liquid-water regime coefficients when the answer is above 0 °C
    and switches to the ice-sublimation coefficients below — valid down to
    about −90 °C which covers the paper's dehydration targets.
    """
    if partial_pressure_pa <= 0.0:
        return float("-inf")
    p_hpa = partial_pressure_pa / 100.0
    # Over liquid water
    a, b, c = 6.112, 17.62, 243.12
    y = np.log(p_hpa / a)
    t_liquid = c * y / (b - y)
    if t_liquid >= 0.0:
        return float(t_liquid)
    # Over ice
    a, b, c = 6.112, 22.46, 272.62
    y = np.log(p_hpa / a)
    return float(c * y / (b - y))


def sweep_feed_flows(
    module_factory: Callable[[float], ModuleConfig],
    components: list[ComponentSpec],
    solver_cls: type[BaseSolver],
    feed_flows: np.ndarray,
    label: str,
) -> list[SimulationResults]:
    """Solve the same module at a range of feed flows. Skip failures and
    unphysical results where the bore flow hit the numerical floor."""
    results: list[SimulationResults] = []
    print(f"    {label}: ", end="", flush=True)
    for feed in feed_flows:
        module = module_factory(feed)
        try:
            result = solver_cls(module=module, components=components).solve()
        except Exception as exc:
            print(f"x({type(exc).__name__})", end="", flush=True)
            continue
        if float(np.min(result.feed_side_flow_kmol_s)) <= _PHYSICAL_FLOW_THRESHOLD:
            print("x(floored)", end="", flush=True)
            continue
        results.append(result)
        print(".", end="", flush=True)
    print(f"  [{len(results)}/{len(feed_flows)} converged]")
    return results


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 110,
            "lines.linewidth": 1.8,
        }
    )


# ── Figure functions ───────────────────────────────────────────────────────


def figure_4_5_n2_purity_vs_recovery(output_dir: Path) -> None:
    print("[Figs 4-5] N₂ residue purity vs residue recovery")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    components = air_components()
    feed_flows = np.logspace(-3.6, -2.5, 12)

    for ax, pressure in zip(axes, [10.0, 5.0], strict=True):
        for solver_cls, style, color, label in [
            (CountercurrentSolver, "-", "tab:blue", "Countercurrent"),
            (CrossflowSolver, "--", "tab:orange", "Cross-flow"),
        ]:
            results = sweep_feed_flows(
                lambda f, p=pressure: air_module(f, pressure_bar=p),
                components,
                solver_cls,
                feed_flows,
                label=f"{label} @ {pressure:.0f} bar",
            )
            recs = [r.residue_recovery for r in results]
            purities = [r.residue_composition[0] for r in results]
            ax.plot(
                recs,
                purities,
                style,
                color=color,
                marker="o",
                markersize=4,
                label=label,
            )

        # Paper Fig 4 values read by eye from paper_pages/paper_page_07.png
        # at 10 bar, 0 % purge, 50 % recovery → N₂ ≈ 0.93–0.94.
        # At 5 bar, 0 % purge, 50 % recovery → N₂ ≈ 0.88.
        # Plotted as a shaded band to honestly represent the reading uncertainty.
        paper_band_center = 0.935 if pressure == 10.0 else 0.88
        ax.axvline(0.50, color="gray", linestyle=":", alpha=0.5)
        ax.fill_between(
            [0.48, 0.52],
            paper_band_center - 0.01,
            paper_band_center + 0.01,
            color="red",
            alpha=0.25,
            label=f"Paper Fig 4 (read by eye): ≈{paper_band_center:.3f}",
        )

        ax.set_xlabel("Overall residue recovery  R/F")
        ax.set_ylabel("N₂ mole fraction in residue")
        ax.set_title(f"{pressure:.0f} bar feed, 40 °C, 300 k fibers")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.78, 1.0)

    fig.suptitle(
        "Reproduction of Coker et al. 1998, Figures 4 & 5 — air separation",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out = output_dir / "fig_4_5_n2_purity_vs_recovery.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"    wrote {out}\n")


def figure_6_purge_effect(output_dir: Path) -> None:
    print("[Figs 6/8/10] Purge effect on recovery and permeate dew point")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    components = air_components()
    feed = 1.0e-3
    purges = np.linspace(0.0, 0.15, 11)

    recs: list[float] = []
    purities: list[float] = []
    dew_perm: list[float] = []
    actual_purges: list[float] = []

    print("    ", end="", flush=True)
    for p in purges:
        module = air_module(feed, pressure_bar=10.0, purge_fraction=p)
        try:
            result = CountercurrentSolver(module=module, components=components).solve()
        except Exception:
            print("x", end="", flush=True)
            continue
        actual_purges.append(p)
        recs.append(result.residue_recovery)
        purities.append(float(result.residue_composition[0]))
        y_h2o = float(result.permeate_composition[3])
        p_perm = float(result.permeate_side_pressure_pa[-1])
        dew_perm.append(dew_point_celsius(y_h2o * p_perm))
        print(".", end="", flush=True)
    print(f"  [{len(recs)}/{len(purges)}]")

    actual_purges_arr = np.asarray(actual_purges) * 100.0

    ax = axes[0]
    ax.plot(
        actual_purges_arr, recs, "o-", color="tab:blue", label="Residue recovery R′/F"
    )
    ax.plot(
        actual_purges_arr, purities, "s-", color="tab:orange", label="N₂ residue purity"
    )
    ax.set_xlabel("Purge fraction (% of residue)")
    ax.set_ylabel("Fraction")
    ax.set_title("Fig 6 — purge shifts the recovery/purity tradeoff")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    ax = axes[1]
    ax.plot(
        actual_purges_arr, dew_perm, "o-", color="tab:green", label="Permeate dew point"
    )
    ax.axhline(-41.6, color="red", linestyle=":", alpha=0.6)
    ax.axhline(-89.7, color="red", linestyle=":", alpha=0.6)
    ax.text(
        actual_purges_arr.max() * 0.98,
        -41.6,
        "Paper: −41.6 °C at 0 % purge, 86 % N₂",
        color="red",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.text(
        actual_purges_arr.max() * 0.98,
        -89.7,
        "Paper: −89.7 °C at 10 % purge, 86 % N₂",
        color="red",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("Purge fraction (% of residue)")
    ax.set_ylabel("Dew point (°C)")
    ax.set_title("Figs 8/10 — permeate dew point vs purge")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Reproduction of Coker et al. 1998, Figures 6 / 8 / 10 — purge effect",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out = output_dir / "fig_6_purge_effect.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"    wrote {out}\n")


def figure_11_h2_purity_vs_recovery(output_dir: Path) -> None:
    print("[Fig 11] H₂ permeate purity vs H₂ recovery at two pressure ratios")
    fig, ax = plt.subplots(figsize=(9, 6))
    components = h2_components()
    feed_flows = np.logspace(-2.6, -1.2, 12)

    for high_ratio, style, color, label in [
        (True, "-o", "tab:blue", "High ratio 42.4/7.9 = 5.37"),
        (False, "-s", "tab:orange", "Low ratio 76.9/42.4 = 1.81"),
    ]:
        results = sweep_feed_flows(
            lambda f, hr=high_ratio: h2_module(f, high_ratio=hr),
            components,
            CountercurrentSolver,
            feed_flows,
            label=label,
        )
        h2_recs: list[float] = []
        h2_purities: list[float] = []
        for result in results:
            h2_feed = result.feed_flow_kmol_s * components[0].feed_mole_fraction
            h2_perm = result.permeate_flow_kmol_s * float(
                result.permeate_composition[0]
            )
            h2_recs.append(h2_perm / h2_feed)
            h2_purities.append(float(result.permeate_composition[0]))
        ax.plot(h2_recs, h2_purities, style, color=color, markersize=5, label=label)

    ax.axvline(0.60, color="red", linestyle=":", alpha=0.6)
    ax.text(
        0.61,
        0.68,
        "Paper Fig 11:\nat 60 % H₂ recovery,\nhigh-ratio > low-ratio",
        color="red",
        fontsize=9,
    )
    ax.set_xlabel("H₂ recovery in permeate")
    ax.set_ylabel("H₂ mole fraction in permeate")
    ax.set_title(
        "Reproduction of Coker et al. 1998, Figure 11 — H₂/hydrocarbon, 50 °C",
        fontweight="bold",
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    out = output_dir / "fig_11_h2_purity_vs_recovery.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"    wrote {out}\n")


def figure_12_h2_axial_profiles(output_dir: Path) -> None:
    print("[Fig 12] H₂/hydrocarbon axial profiles with CH₄ interior maximum")
    components = h2_components()
    # Paper p.10: "The feed flow rate is 283.2 m³(STP)/h (10,000 SCFH), and
    # the total permeate recovery is 94.6%."
    paper_feed_kmol_s = 283.2 / 22.414 / 3600.0  # ≈ 3.51e-3
    module = h2_module(feed_flow_kmol_s=paper_feed_kmol_s, high_ratio=True)
    result = CountercurrentSolver(module=module, components=components).solve()

    fig, ax = plt.subplots(figsize=(10, 6))
    names_and_colors = [
        ("H₂", "tab:blue"),
        ("C₂H₄", "tab:cyan"),
        ("CH₄", "tab:green"),
        ("C₂H₆", "tab:red"),
        ("C₃H₈", "tab:purple"),
    ]
    for j, (name, color) in enumerate(names_and_colors):
        ax.plot(
            result.axial_positions,
            result.residue_profiles[j],
            color=color,
            label=name,
        )

    ch4 = result.residue_profiles[2]
    peak_idx = int(np.argmax(ch4))
    peak_pos = float(result.axial_positions[peak_idx])
    peak_val = float(ch4[peak_idx])
    ax.plot(peak_pos, peak_val, "k*", markersize=18, zorder=10)
    ax.annotate(
        f"CH₄ interior peak\n{peak_val:.3f} at z/L = {peak_pos:.2f}\n"
        f"(feed x_CH₄ = 0.21)",
        xy=(peak_pos, peak_val),
        xytext=(0.15, 0.82),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=10,
        ha="left",
    )

    ax.set_xlabel("Axial position  z/L  (0 = residue end, 1 = feed end)")
    ax.set_ylabel("Feed-side mole fraction")
    ax.set_title(
        "Reproduction of Coker et al. 1998, Figure 12 — H₂/hydrocarbon separation\n"
        f"42.4 → 7.9 bar, F = 283.2 m³(STP)/h (paper p.10), "
        f"total permeate recovery = {result.stage_cut:.3f}  "
        f"(paper quotes 0.946)",
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    out = output_dir / "fig_12_h2_axial_profiles.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"    wrote {out}\n")


def figure_13_ternary_axial_profiles(output_dir: Path) -> None:
    print("[Fig 13] Ternary axial profiles with intermediate-component interior max")
    components = ternary_components()
    # Paper p.10: "Typical composition profiles are presented in Figure 13 for
    # a feed flow rate of 283.2 m³(STP)/h (10,000 SCFH)."
    paper_feed_kmol_s = 283.2 / 22.414 / 3600.0  # ≈ 3.51e-3
    module = ternary_module(feed_flow_kmol_s=paper_feed_kmol_s)
    result = CountercurrentSolver(module=module, components=components).solve()

    fig, ax = plt.subplots(figsize=(10, 6))
    labels_and_colors = [
        ("Component 1 — fast (500 GPU)", "tab:blue"),
        ("Component 2 — intermediate (100 GPU)", "tab:orange"),
        ("Component 3 — slow (10 GPU)", "tab:green"),
    ]
    for j, (label, color) in enumerate(labels_and_colors):
        ax.plot(
            result.axial_positions,
            result.residue_profiles[j],
            color=color,
            label=label,
        )

    mid = result.residue_profiles[1]
    peak_idx = int(np.argmax(mid))
    peak_pos = float(result.axial_positions[peak_idx])
    peak_val = float(mid[peak_idx])
    ax.plot(peak_pos, peak_val, "k*", markersize=18, zorder=10)
    ax.annotate(
        f"Intermediate interior peak\n{peak_val:.3f} at z/L = {peak_pos:.2f}\n"
        "(feed x = 0.3333)",
        xy=(peak_pos, peak_val),
        xytext=(0.12, 0.75),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=10,
        ha="left",
    )

    ax.set_xlabel("Axial position  z/L  (0 = residue end, 1 = feed end)")
    ax.set_ylabel("Feed-side mole fraction")
    ax.set_title(
        "Reproduction of Coker et al. 1998, Figure 13 — ternary model case\n"
        f"10 → 1 bar equimolar feed, 25 °C, F = 283.2 m³(STP)/h (paper p.10), "
        f"stage cut = {result.stage_cut:.3f}",
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out = output_dir / "fig_13_ternary_axial_profiles.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"    wrote {out}\n")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "paper_figures",
        help="Directory for PNG outputs (default: examples/paper_figures)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after writing",
    )
    parser.add_argument(
        "--only",
        choices=["4_5", "6", "11", "12", "13"],
        help="Render only one figure (useful while iterating)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.resolve()}\n")

    _setup_plot_style()
    figures: dict[str, Callable[[Path], None]] = {
        "4_5": figure_4_5_n2_purity_vs_recovery,
        "6": figure_6_purge_effect,
        "11": figure_11_h2_purity_vs_recovery,
        "12": figure_12_h2_axial_profiles,
        "13": figure_13_ternary_axial_profiles,
    }
    selected = {args.only: figures[args.only]} if args.only else figures

    t0 = time.perf_counter()
    for fn in selected.values():
        fn(args.output_dir)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.1f} s.")
    print("Open the PNGs next to the paper PDF for side-by-side comparison.")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
