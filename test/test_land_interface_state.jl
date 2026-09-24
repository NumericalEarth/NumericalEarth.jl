include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations, saturation_specific_humidity
using Thermodynamics: Thermodynamics
using Oceananigans.TimeSteppers: update_state!

# A canopy defined outside NumericalEarth: a bulk canopy conductance gᶜ = λˡᵉᵃᶠ gₛ in
# series with the aerodynamic conductance, reading the leaf area index from the land state.
struct BigLeafHumidity{FT, Φ}
    stomatal_conductance :: FT
    phase :: Φ
end

BigLeafHumidity(gₛ) = BigLeafHumidity(gₛ, Thermodynamics.Liquid())

InterfaceComputations.land_state_names(::BigLeafHumidity) = (:T, :λˡᵉᵃᶠ)

const leaf_area_index = Ref(1.0)
InterfaceComputations.land_state_field(::SlabLand, ::Val{:λˡᵉᵃᶠ}) = leaf_area_index[]

# Vapor-flux balance gᶜ (qᵛ⁺(Tₛ) - qˢ) = J with J = -u★ q★ from the previous iterate.
@inline function InterfaceComputations.compute_interface_humidity(q::BigLeafHumidity, Tₛ, Ψₛ, Ψᵃᵗ, Ψˡᵃ, ℙᵃᵗ)
    qᵛ⁺ = saturation_specific_humidity(ℙᵃᵗ.thermodynamics_parameters, Tₛ, Ψᵃᵗ.p, q.phase)
    gᶜ  = Ψˡᵃ.λˡᵉᵃᶠ * q.stomatal_conductance
    J   = - Ψₛ.fluxes.u★ * Ψₛ.fluxes.q★
    Δq  = Ψₛ.specific_humidity - Ψᵃᵗ.q
    D   = gᶜ * Δq + J
    return ifelse(D == 0, Ψₛ.specific_humidity, (gᶜ * qᵛ⁺ * Δq + J * Ψᵃᵗ.q) / D)
end

struct RoughnessHumidity end
InterfaceComputations.land_state_names(::RoughnessHumidity) = (:T, :ℓᵐ)

function column_latent_heat(grid; kw...)
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    model = AtmosphereLandModel(atmosphere, SlabLand(grid); kw...)
    fill!(parent(model.atmosphere.velocities.u), 5)
    fill!(parent(model.atmosphere.temperature), 300)
    fill!(parent(model.atmosphere.specific_humidity), 0.005)
    fill!(parent(model.atmosphere.pressure), 101_325)
    set!(model.land; T = 303, M = 150)
    update_state!(model)
    return only(Array(interior(model.interfaces.atmosphere_land_interface.fluxes.latent_heat)))
end

@testset "Land state declared by the interface formulation" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = 1, x = (0, 1), y = (0, 1), z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        model = AtmosphereLandModel(atmosphere, SlabLand(grid))
        @test keys(model.interfaces.exchanger.land.state) == (:T, :𝒮)

        model = AtmosphereLandModel(atmosphere, SlabLand(grid);
                                    atmosphere_land_interface_specific_humidity = BigLeafHumidity(0.005))
        @test keys(model.interfaces.exchanger.land.state) == (:T, :λˡᵉᵃᶠ)

        # A land that cannot provide a requested quantity fails at construction.
        @test_throws MethodError AtmosphereLandModel(atmosphere, SlabLand(grid);
                                                     atmosphere_land_interface_specific_humidity = RoughnessHumidity())

        # Transpiration grows with leaf area toward the saturated bare surface.
        bare = column_latent_heat(grid)

        leaf_area_index[] = 0.5
        sparse = column_latent_heat(grid; atmosphere_land_interface_specific_humidity = BigLeafHumidity(0.005))

        leaf_area_index[] = 5.0
        dense = column_latent_heat(grid; atmosphere_land_interface_specific_humidity = BigLeafHumidity(0.005))
        open_stomata = column_latent_heat(grid; atmosphere_land_interface_specific_humidity = BigLeafHumidity(1e3))

        @test 0 < sparse < dense < bare
        @test open_stomata ≈ bare rtol = 1e-3
    end
end
