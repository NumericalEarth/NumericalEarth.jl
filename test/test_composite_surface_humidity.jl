include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    compute_interface_humidity, AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities,
    saturation_specific_humidity, dry_layer_terms, canopy_conductance_terms, atmospheric_vapor_flux,
    evaporation_partition,
    DryLayerHumidity, StorageBasedDryLayerDepth, DryLayerVaporPistonVelocity,
    ConstantTortuosity,
    CanopyConductanceHumidity, CompositeSurfaceHumidity, BulkHumidity
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Atmospheres: PrescribedAtmosphere, AtmosphereThermodynamicsParameters
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

make_soil(; molecular_diffusivity = 2.5e-5, tortuosity = ConstantTortuosity()) =
    DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(;
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity, tortuosity),
        thermal_exchange_depth = 0.10, porosity = 0.4)

# Mirror `compute_interface_humidity(formulation, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)`. The
# radiation state Ψᵣ only drives InteractiveAbsorbedPAR; these cases use the
# prescribed-PAR default, so it is `nothing`.
function _make_call_args(; Tˡᵃ, Tⁱⁿ, 𝒮, pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻, leaf_area_index = 2.0)
    ℂ  = AtmosphereThermodynamicsParameters(Float64)
    Ψₐ = (T = Tᵃᵗ, p = pᵃᵗ, q = qᵃᵗ, u = 1.0, v = 0.0, z = 10.0, h_bℓ = 1000.0)
    Ψₛ = AirLandInterfaceState(InterfaceFluxScales(u★, 0.0, q★), InterfaceVelocities(0.0, 0.0),
                               Tⁱⁿ, qⁱⁿ⁻, (saturation=𝒮,), (temperature=Tˡᵃ,),
                               (leaf_area_index=leaf_area_index,))
    Ψᵢ = (T = Tˡᵃ,)
    ℙₐ = (thermodynamics_parameters = ℂ, surface_layer_height = 10.0, gravitational_acceleration = 9.81)
    return ℂ, Ψₛ, Ψₐ, Ψᵢ, nothing, ℙₐ
end

@testset "CompositeSurfaceHumidity limits and divider" begin
    st = (Tˡᵃ=295.0, Tⁱⁿ=300.0, pᵃᵗ=1.0e5, qᵃᵗ=1.0e-2, Tᵃᵗ=295.0, u★=0.3, q★=-2.0e-4, qⁱⁿ⁻=0.008)
    soil   = make_soil()
    canopy = CanopyConductanceHumidity(Float64; leaf_area_index = 2.0)

    # Limit 1: no canopy (g_c = 0) reproduces DryLayerHumidity bit-for-bit.
    comp0 = CompositeSurfaceHumidity(soil, CanopyConductanceHumidity(Float64; leaf_area_index = 0.0))
    _, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(; 𝒮=0.2, leaf_area_index=0.0, st...)
    @test isapprox(compute_interface_humidity(comp0, st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ),
                   compute_interface_humidity(soil,  st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ); atol = 1e-15)

    # Limit 2: negligible soil diffusivity (Gᵉ → 0) with a dry soil (σ = 1)
    # reproduces CanopyConductanceHumidity.
    comp_dry = CompositeSurfaceHumidity(make_soil(molecular_diffusivity = 1e-14), canopy)
    _, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(; 𝒮=0.0, st...)
    @test isapprox(compute_interface_humidity(comp_dry, st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ),
                   compute_interface_humidity(canopy,   st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ); rtol = 1e-6)

    # Two-source divider matches the analytic conductance-weighted average, and
    # the total flux partitions exactly into soil-evaporation + transpiration.
    comp = CompositeSurfaceHumidity(soil, canopy)
    _, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(; 𝒮=0.05, st...)   # 𝒮 ≪ 𝒮ᶜ ⇒ σ = 1
    Gᵉ, qᵉ, σ, q_wet = dry_layer_terms(comp.soil, st.Tⁱⁿ, Ψₛ, Ψₐ, ℙₐ)
    g_c, q_leaf = canopy_conductance_terms(comp.canopy, st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    Jᵃ, Δq = atmospheric_vapor_flux(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)
    qˢ = ((Gᵉ * qᵉ + g_c * q_leaf) * Δq + Jᵃ * st.qᵃᵗ) / ((Gᵉ + g_c) * Δq + Jᵃ)
    @test σ ≈ 1
    @test compute_interface_humidity(comp, st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ) ≈ qˢ

    E_total = (Jᵃ / Δq) * (qˢ - st.qᵃᵗ)
    E_soil  = Gᵉ * (qᵉ - qˢ)
    E_can   = g_c * (q_leaf - qˢ)
    @test isapprox(E_total, E_soil + E_can; rtol = 1e-10)

    # The `evaporation_partition` diagnostic returns exactly this split.
    part = evaporation_partition(comp, qˢ, st.Tⁱⁿ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    @test part.soil_evaporation ≈ E_soil
    @test part.transpiration ≈ E_can
    @test isapprox(part.soil_evaporation + part.transpiration, E_total; rtol = 1e-10)

    # Type stability.
    for FT in (Float32, Float64)
        ℂ  = AtmosphereThermodynamicsParameters(FT)
        Ψₐ = (T=FT(295), p=FT(1e5), q=FT(1e-2), u=FT(1), v=FT(0), z=FT(10), h_bℓ=FT(1000))
        Ψₛ = AirLandInterfaceState(InterfaceFluxScales(FT(0.3), FT(0), FT(-2e-4)),
                                   InterfaceVelocities(FT(0), FT(0)),
                                   FT(300), FT(8e-3), (saturation=FT(0.3),), (temperature=FT(295),),
                                   (leaf_area_index=FT(2),))
        Ψᵢ = (T=FT(295),)
        Ψᵣ = nothing
        ℙₐ = (thermodynamics_parameters=ℂ, surface_layer_height=FT(10), gravitational_acceleration=FT(9.81))
        cFT = CompositeSurfaceHumidity(
                DryLayerHumidity(;
                    dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth=0.05,
                        dry_layer_onset_saturation=0.5, dry_layer_exponent=1.0),
                    vapor_exchange = DryLayerVaporPistonVelocity(FT;
                        minimum_dry_layer_depth=1e-4, molecular_diffusivity=2.5e-5),
                    thermal_exchange_depth=0.10, porosity=0.4),
                CanopyConductanceHumidity(FT; leaf_area_index=2))
        @test eltype(compute_interface_humidity(cFT, FT(300), Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)) == FT
        @inferred compute_interface_humidity(cFT, FT(300), Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    end
end

@testset "CompositeSurfaceHumidity coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        soil   = make_soil()
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = 2.0)
        comp   = CompositeSurfaceHumidity(soil, canopy)

        function latent_heat(q_formulation; water_storage = 45.0)
            atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
            fill!(parent(atmosphere.temperature),       295.0)
            fill!(parent(atmosphere.specific_humidity), 0.006)
            fill!(parent(atmosphere.velocities.u),      5.0)
            fill!(parent(atmosphere.pressure),          101325.0)
            land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0),
                                  energy = SlabEnergy(FT))
            set!(land; T = 300.0)
            fill!(parent(land.water_storage), water_storage)   # 𝒮 = water_storage / 150
            model = AtmosphereLandModel(atmosphere, land; radiation = nothing,
                                        atmosphere_land_interface_specific_humidity = q_formulation)
            update_state!(model.land)   # refresh diagnostic 𝒮 = M/M_max from the storage we set
            update_state!(model)
            return Array(interior(model.interfaces.atmosphere_land_interface.fluxes.latent_heat))[1, 1, 1]
        end

        # Two parallel sources evaporate more than either branch alone at an
        # intermediate saturation, and every flux stays finite.
        LE_soil = latent_heat(soil)
        LE_can  = latent_heat(canopy)
        LE_comp = latent_heat(comp)
        @test LE_comp > LE_soil
        @test LE_comp > LE_can
        @test all(isfinite, (LE_soil, LE_can, LE_comp))

        # Drydown throttles the composite through the ground saturation 𝒮.
        @test latent_heat(comp; water_storage = 120.0) > latent_heat(comp; water_storage = 15.0)
    end
end
