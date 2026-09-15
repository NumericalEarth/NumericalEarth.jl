using Test, Oceananigans, NumericalEarth, Breeze, CUDA
using NumericalEarth.Atmospheres: breeze_prognostic_state
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIceDensityState,
                             temperature, mixture_gas_constant

architectures = parse(Bool, get(ENV, "GPU_TEST", "false")) ? (GPU(),) : (CPU(),)
for arch in architectures, FT in (Float32, Float64)
    @testset "Parent moist EOS round trip $arch $FT" begin
        grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(1, 1, 1))
        constants = Breeze.ThermodynamicConstants(FT)
        fields = ntuple(_ -> CenterField(grid), 5)
        T, vapor, liquid, ice, pressure = fields
        for (qᵛ, qˡ, qⁱ) in ((0, 0, 0), (0.02, 0, 0), (0.01, 0.001, 0.002))
            values = FT.((280, qᵛ, qˡ, qⁱ, 70000))
            for (field, value) in zip(fields, values)
                set!(field, value)
            end
            state = breeze_prognostic_state(constants, FT(1e5), T, vapor, liquid, ice, pressure)
            ρ = only(Array(interior(state.ρ)))
            θ = only(Array(interior(state.θˡⁱ)))
            moisture = MoistureMassFractions(FT(qᵛ), FT(qˡ), FT(qⁱ))
            recovered = LiquidIceDensityState(θ, moisture, FT(1e5), ρ)
            reconstructed_temperature = temperature(recovered, constants)
            reconstructed_pressure = ρ * mixture_gas_constant(moisture, constants) * reconstructed_temperature
            @test reconstructed_temperature ≈ FT(280) rtol=20eps(FT)
            @test reconstructed_pressure ≈ FT(70000) rtol=20eps(FT)
        end
    end
end
