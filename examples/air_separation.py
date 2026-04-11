"""Air separation example (Coker et al. 1998, Table 1/2).

Reproduces the 10-bar, 40 °C, bore-feed air dehydration case at a
moderate stage cut. Prints the summary and writes a tidy CSV of axial
profiles alongside this script.
"""

from pathlib import Path

from hfm_separator.models.components import ComponentSpec
from hfm_separator.models.module import ModuleConfig
from hfm_separator.models.permeance import from_gpu
from hfm_separator.solvers.countercurrent import CountercurrentSolver
from hfm_separator.utils.unit_conv import bara_to_pa, celsius_to_kelvin


def main() -> None:
    module = ModuleConfig(
        n_fibers=300_000,
        fiber_od_m=300e-6,
        fiber_id_m=150e-6,
        active_length_m=0.8,
        pot_length_m=0.1,
        feed_pressure_pa=bara_to_pa(10.0),
        permeate_pressure_pa=bara_to_pa(1.0),
        feed_temp_k=celsius_to_kelvin(40.0),
        feed_flow_kmol_s=1.0e-3,
        feed_side="bore",
    )
    components = [
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

    result = CountercurrentSolver(module=module, components=components).solve()

    print("=" * 60)
    print("Air separation — countercurrent, 10 bar, 40 °C")
    print("=" * 60)
    print(f"N stages            : {result.n_stages}")
    print(f"Iterations          : {result.n_iterations}")
    print(f"Final residuals     : {result.residuals}")
    print(f"Stage cut           : {result.stage_cut:.4f}")
    print(f"Residue recovery    : {result.residue_recovery:.4f}")
    print(f"N₂ residue purity   : {result.residue_composition[0]:.4f}")
    print(f"O₂ permeate purity  : {result.permeate_composition[1]:.4f}")
    print(f"H₂O residue x       : {result.residue_composition[3]:.2e}")
    print()

    out_csv = Path(__file__).with_suffix(".csv")
    result.to_dataframe().to_csv(out_csv, index=False)
    print(f"Axial profile written to {out_csv}")


if __name__ == "__main__":
    main()
