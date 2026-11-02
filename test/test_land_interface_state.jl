include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations, saturation_specific_humidity, land_temperature
using Thermodynamics: Thermodynamics
using Oceananigans.TimeSteppers: update_state!

# A canopy defined outside NumericalEarth: a bulk canopy conductance gᶜⁿᵖʸ = λˡᵉᵃᶠ gₛ in
# series with the aerodynamic conductance, reading the leaf area index from the land state.
struct BigLeafHumidity{FT, Φ}
    stomatal_conductance :: FT
    phase :: Φ
end

BigLeafHumidity(gₛ) = BigLeafHumidity(gₛ, Thermodynamics.Liquid())

const leaf_area = Ref(1.0)
leaf_area_index(::SlabLand) = leaf_area[]

InterfaceComputations.land_state_accessors(::BigLeafHumidity) = (T = land_temperature, λˡᵉᵃᶠ = leaf_area_index)

# Vapor-flux balance gᶜⁿᵖʸ (qᵛ⁺(Tⁱⁿ) - qˢ) = J with J = -u★ q★ from the previous iterate.
@inline function InterfaceComputations.compute_interface_humidity(q::BigLeafHumidity, Tⁱⁿ, Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, Ψᵣ, ℙᵃᵗ)
    qᵛ⁺  = saturation_specific_humidity(ℙᵃᵗ.thermodynamics_parameters, Tⁱⁿ, Ψᵃᵗ.p, q.phase)
    gᶜⁿᵖʸ = Ψˡᵃ.λˡᵉᵃᶠ * q.stomatal_conductance
    J    = - Ψⁱⁿ.fluxes.u★ * Ψⁱⁿ.fluxes.q★
    Δq   = Ψⁱⁿ.specific_humidity - Ψᵃᵗ.q
    D    = gᶜⁿᵖʸ * Δq + J
    return ifelse(D == 0, Ψⁱⁿ.specific_humidity, (gᶜⁿᵖʸ * qᵛ⁺ * Δq + J * Ψᵃᵗ.q) / D)
end

# A quantity no land provides
function momentum_roughness_length end

struct RoughnessHumidity end
InterfaceComputations.land_state_accessors(::RoughnessHumidity) = (T = land_temperature, ℓᵐ = momentum_roughness_length)

humidity_state(specific_humidity) = (temperature = BulkTemperature(), specific_humidity)

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
                                    atmosphere_land_interface_state = humidity_state(BigLeafHumidity(0.005)))
        @test keys(model.interfaces.exchanger.land.state) == (:T, :λˡᵉᵃᶠ)

        # A land that cannot provide a requested quantity fails at construction.
        @test_throws MethodError AtmosphereLandModel(atmosphere, SlabLand(grid);
                                                     atmosphere_land_interface_state = humidity_state(RoughnessHumidity()))

        # Transpiration grows with leaf area toward the saturated bare surface.
        bare = column_latent_heat(grid)

        leaf_area[] = 0.5
        sparse = column_latent_heat(grid; atmosphere_land_interface_state = humidity_state(BigLeafHumidity(0.005)))

        leaf_area[] = 5.0
        dense = column_latent_heat(grid; atmosphere_land_interface_state = humidity_state(BigLeafHumidity(0.005)))
        open_stomata = column_latent_heat(grid; atmosphere_land_interface_state = humidity_state(BigLeafHumidity(1e3)))

        @test 0 < sparse < dense < bare
        @test open_stomata ≈ bare rtol = 1e-3
    end
end
