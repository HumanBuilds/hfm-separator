"""H₂/hydrocarbon separation example (Coker et al. 1998, Table 3).

Reproduces the high pressure-ratio case (42.4 → 7.9 bara, shell-feed,
500,000 fibers) at ~99.8% H₂ recovery. This is the operating point
where the CH₄ axial profile has a clear interior maximum (Figure 12).
"""

from pathlib import Path

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin


def main() -> None:
    module = ModuleConfig(
        n_fibers=500_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(42.4),
        permeate_pressure_pa=bara_to_pa(7.9),
        feed_temp_k=celsius_to_kelvin(50.0),
        feed_flow_kmol_s=1.0e-2,
        feed_side="shell",
    )
    components = [
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

    result = CountercurrentSolver(module=module, components=components).solve()

    h2_permeate_recovery = (
        result.permeate_flow_kmol_s
        * result.permeate_composition[0]
        / (module.feed_flow_kmol_s * components[0].feed_mole_fraction)
    )

    print("=" * 60)
    print("H₂ separation — countercurrent, 42.4→7.9 bar, 50 °C")
    print("=" * 60)
    print(f"N stages            : {result.n_stages}")
    print(f"Iterations          : {result.n_iterations}")
    print(f"Stage cut           : {result.stage_cut:.4f}")
    print(f"H₂ recovery         : {h2_permeate_recovery:.4f}")
    print(f"H₂ permeate purity  : {result.permeate_composition[0]:.4f}")
    print(f"H₂ residue x (end)  : {result.residue_profiles[0, 0]:.2e}")
    print()

    out_csv = Path(__file__).with_suffix(".csv")
    result.to_dataframe().to_csv(out_csv, index=False)
    print(f"Axial profile written to {out_csv}")


if __name__ == "__main__":
    main()
