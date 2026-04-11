"""Ternary model separation example (Coker et al. 1998, Table 4).

Reproduces the three-component equimolar feed with 500/100/10 GPU
permeances. The intermediate component has an interior axial maximum
in its residue mole fraction (Figure 13).
"""

from pathlib import Path

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin


def main() -> None:
    # Paper p.10: Figure 13 uses F = 283.2 m³(STP)/h (10,000 SCFH).
    paper_feed_kmol_s = 283.2 / 22.414 / 3600.0
    module = ModuleConfig(
        n_fibers=350_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(25.0),
        feed_flow_kmol_s=paper_feed_kmol_s,
        feed_side="shell",
    )
    components = [
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

    result = CountercurrentSolver(module=module, components=components).solve()

    import numpy as np

    mid_profile = result.residue_profiles[1, :]
    mid_peak_idx = int(np.argmax(mid_profile))

    print("=" * 60)
    print("Ternary model — countercurrent, 10 bar, 25 °C, equimolar feed")
    print("=" * 60)
    print(f"N stages            : {result.n_stages}")
    print(f"Iterations          : {result.n_iterations}")
    print(f"Stage cut           : {result.stage_cut:.4f}")
    print(f"Residue recovery    : {result.residue_recovery:.4f}")
    print(f"Fast residue        : {result.residue_composition[0]:.4f}")
    print(f"Mid residue         : {result.residue_composition[1]:.4f}")
    print(f"Slow residue        : {result.residue_composition[2]:.4f}")
    peak_val = mid_profile[mid_peak_idx]
    peak_pos = result.axial_positions[mid_peak_idx]
    print(f"Mid peak            : {peak_val:.4f} at z/L={peak_pos:.3f}")
    print()

    out_csv = Path(__file__).with_suffix(".csv")
    result.to_dataframe().to_csv(out_csv, index=False)
    print(f"Axial profile written to {out_csv}")


if __name__ == "__main__":
    main()
