include("runtests_setup.jl")

using ClimaSeaIce.Rheologies
using ClimaSeaIce.SeaIceDynamics
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: hours, days
using NumericalEarth.DataWrangling: all_dates
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentInterfaces,
                                                              celsius_to_kelvin,
                                                              SimilarityScales,
                                                              surface_specific_humidity,
                                                              SkinTemperature,
                                                              BulkTemperature,
                                                              DiffusiveFlux,
                                                              InteriorDiffusivity,
                                                              assemble_interior_fields,
                                                              SkinHumidity,
                                                              FractionalHumidity,
                                                              CriticalSaturation,
                                                              evaporation_efficiency,
                                                              AirLandInterfaceState,
                                                              compute_interface_humidity,
                                                              saturation_specific_humidity
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Statistics: mean, std
using Thermodynamics

struct FixedSpecificHumidity{FT}
    qᵒᶜ :: FT
end

@inline NumericalEarth.EarthSystemModels.InterfaceComputations.surface_specific_humidity(h::FixedSpecificHumidity, args...) = h.qᵒᶜ

@testset "Test surface fluxes" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch, Float32;
                                     size = 1,
                                     latitude = 10,
                                     longitude = 10,
                                     z = (-1, 0),
                                     topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 bottom_drag_coefficient = 0)

        dates = all_dates(RepeatYearJRA55(), :temperature)
        atmosphere = JRA55PrescribedAtmosphere(arch; end_date=dates[2])

        @allowscalar begin
            h  = atmosphere.surface_layer_height
            pᵃᵗ = atmosphere.pressure[1][1, 1, 1]

            Tᵃᵗ = 15 + celsius_to_kelvin
            qᵃᵗ = Float32(0.003)

            uᵃᵗ = atmosphere.velocities.u[1][1, 1, 1]
            vᵃᵗ = atmosphere.velocities.v[1][1, 1, 1]

            ℂᵃᵗ = atmosphere.thermodynamics_parameters

            fill!(parent(atmosphere.tracers.T),    Tᵃᵗ)
            fill!(parent(atmosphere.tracers.q),    qᵃᵗ)
            fill!(parent(atmosphere.velocities.u), uᵃᵗ)
            fill!(parent(atmosphere.velocities.v), vᵃᵗ)
            fill!(parent(atmosphere.pressure),     pᵃᵗ)

            # Force the saturation humidity of the ocean to be
            # equal to the atmospheric saturation humidity
            atmosphere_ocean_interface_specific_humidity = FixedSpecificHumidity(qᵃᵗ)

            # Thermodynamic parameters of the atmosphere
            cp = Thermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
            ρᵃᵗ = Thermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
            ℰv = Thermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)

            # No radiation: pass `radiation = nothing` to disable radiative
            # contributions wholesale.
            for atmosphere_ocean_interface_temperature in (BulkTemperature(),
                                                           SkinTemperature(DiffusiveFlux(1e-2, 1)),
                                                           SkinTemperature(DiffusiveFlux(InteriorDiffusivity(), 1)))
                @info " Testing zero fluxes with $(atmosphere_ocean_interface_temperature)..."

                interfaces = ComponentInterfaces(atmosphere, ocean;
                                                 atmosphere_ocean_interface_specific_humidity,
                                                 atmosphere_ocean_interface_temperature)

                g = ocean.model.buoyancy.formulation.gravitational_acceleration

                # Ensure that the ΔT between atmosphere and ocean is zero
                # Note that the Δθ accounts for the "lapse rate" at height h
                Tᵒᶜ = Tᵃᵗ - celsius_to_kelvin + h / cp * g

                fill!(parent(ocean.model.velocities.u), uᵃᵗ)
                fill!(parent(ocean.model.velocities.v), vᵃᵗ)
                fill!(parent(ocean.model.tracers.T), Tᵒᶜ)

                # Compute the turbulent fluxes (neglecting radiation)
                coupled_model    = OceanOnlyModel(ocean; atmosphere, interfaces)
                turbulent_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

                # Make sure all fluxes are (almost) zero!
                @test turbulent_fluxes.x_momentum[1, 1, 1]    < eps(eltype(grid))
                @test turbulent_fluxes.y_momentum[1, 1, 1]    < eps(eltype(grid))
                @test turbulent_fluxes.sensible_heat[1, 1, 1] < eps(eltype(grid))
                @test turbulent_fluxes.latent_heat[1, 1, 1]   < eps(eltype(grid))
                @test turbulent_fluxes.water_vapor[1, 1, 1]   < eps(eltype(grid))
            end

            @info " Testing interior diffusivity assessment..."

            column_grid = RectilinearGrid(arch, Float32;
                                          size = 1,
                                          x = 10,
                                          y = 10,
                                          z = (-1, 0),
                                          topology = (Flat, Flat, Bounded))

            closure = (VerticalScalarDiffusivity(κ=1e-3), VerticalScalarDiffusivity(κ=2e-3))
            diffusive_ocean = ocean_simulation(column_grid;
                                               momentum_advection = nothing,
                                               tracer_advection = nothing,
                                               closure,
                                               bottom_drag_coefficient = 0)

            set!(diffusive_ocean.model, T = 15, S = 30)

            skin_temperature = SkinTemperature(DiffusiveFlux(InteriorDiffusivity(), 0.5))
            interfaces = ComponentInterfaces(atmosphere, diffusive_ocean;
                                             atmosphere_ocean_interface_temperature = skin_temperature)

            coupled_model = OceanOnlyModel(diffusive_ocean; atmosphere, interfaces)
            assessed_diffusivity = compute!(Field(interfaces.exchanger.ocean.state.κ))
            @test all(Array(interior(assessed_diffusivity)) .≈ 3e-3)

            # The skin temperature solve runs with the assessed diffusivity
            Tₛ = interfaces.atmosphere_ocean_interface.temperature
            @test all(isfinite, Array(interior(Tₛ)))

            # The diffusivity operation is stripped from the kernel arguments
            # unless the temperature formulation needs it
            exchanged_state = interfaces.exchanger.ocean.state
            @test !haskey(assemble_interior_fields(exchanged_state, BulkTemperature()), :κ)
            @test haskey(assemble_interior_fields(exchanged_state, skin_temperature), :κ)

            @info " Testing neutral fluxes..."

            # Constructing very special fluxes that do not account for stability of
            # the atmosphere, have zero gustiness and a constant roughness length of
            # `1e-4` for momentum, water vapor and temperature.
            # For this case, we can compute the fluxes by hand.
            ℓ = 1e-4

            @inline zero_stability_function(ζ) = zero(ζ)

            stability_functions = SimilarityScales(zero_stability_function,
                                                   zero_stability_function,
                                                   zero_stability_function)

            similarity_theory = SimilarityTheoryFluxes(; momentum_roughness_length = ℓ,
                                                         temperature_roughness_length = ℓ,
                                                         water_vapor_roughness_length = ℓ,
                                                         gustiness_parameter = 0,
                                                         minimum_gustiness = 0,
                                                         stability_functions)

            interfaces = ComponentInterfaces(atmosphere, ocean;
                                             atmosphere_ocean_fluxes=similarity_theory)

            # mid-latitude ocean conditions
            set!(ocean.model, u = 0, v = 0, T = 15, S = 30)

            coupled_model = OceanOnlyModel(ocean; atmosphere, interfaces)

            # Now manually compute the fluxes:
            Tᵒᶜ = ocean.model.tracers.T[1, 1, 1] + celsius_to_kelvin
            Sᵒᶜ = ocean.model.tracers.S[1, 1, 1]

            interface_properties = interfaces.atmosphere_ocean_interface.properties
            q_formulation = interface_properties.specific_humidity_formulation
            qᵒᶜ = surface_specific_humidity(q_formulation, ℂᵃᵗ, pᵃᵗ, Tᵒᶜ, Sᵒᶜ)
            g = ocean.model.buoyancy.formulation.gravitational_acceleration

            # Differences!
            Δu = uᵃᵗ
            Δv = vᵃᵗ
            ΔU = sqrt(Δu^2 + Δv^2)
            Δθ = Tᵃᵗ - Tᵒᶜ + h / cp * g
            Δq = qᵃᵗ - qᵒᶜ
            ϰ  = similarity_theory.von_karman_constant

            # Characteristic scales
            u★ = ϰ / log(h / ℓ) * ΔU
            θ★ = ϰ / log(h / ℓ) * Δθ
            q★ = ϰ / log(h / ℓ) * Δq

            τˣ = - ρᵃᵗ * u★^2 * Δu / sqrt(Δu^2 + Δv^2)
            τʸ = - ρᵃᵗ * u★^2 * Δv / sqrt(Δu^2 + Δv^2)
            𝒬ᵀ = - ρᵃᵗ * cp * u★ * θ★
            Jᵛ = - ρᵃᵗ * u★ * q★
            𝒬ᵛ = - ρᵃᵗ * u★ * q★ * ℰv

            turbulent_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

            # Make sure fluxes agree with the hand-calculated ones
            @test turbulent_fluxes.x_momentum[1, 1, 1]    ≈ τˣ
            @test turbulent_fluxes.y_momentum[1, 1, 1]    ≈ τʸ
            @test turbulent_fluxes.sensible_heat[1, 1, 1] ≈ 𝒬ᵀ
            @test turbulent_fluxes.latent_heat[1, 1, 1]   ≈ 𝒬ᵛ
            @test turbulent_fluxes.water_vapor[1, 1, 1]   ≈ Jᵛ
        end

        @info " Testing surface fluxes with land component..."

        # Test that fluxes work with and without land
        ocean_no_land = ocean_simulation(grid;
                                         momentum_advection = nothing,
                                         tracer_advection = nothing,
                                         closure = nothing,
                                         bottom_drag_coefficient = 0)

        set!(ocean_no_land.model, T = 15, S = 30)
        model_no_land = OceanOnlyModel(ocean_no_land; atmosphere)

        ocean_with_land = ocean_simulation(grid;
                                           momentum_advection = nothing,
                                           tracer_advection = nothing,
                                           closure = nothing,
                                           bottom_drag_coefficient = 0)

        set!(ocean_with_land.model, T = 15, S = 30)
        land_dates = all_dates(RepeatYearJRA55(), :river_freshwater_flux)
        land = JRA55PrescribedLand(arch; end_date=land_dates[2])
        model_with_land = OceanOnlyModel(ocean_with_land; atmosphere, land)

        # Verify land exchanger is wired up
        @test isnothing(model_no_land.interfaces.exchanger.land)
        @test !isnothing(model_with_land.interfaces.exchanger.land)

        # The model coerces every component clock to its own time type, so the stored
        # land is a reclocked copy of `land` sharing all other fields by reference.
        @test model_with_land.land.freshwater_flux === land.freshwater_flux
        @test typeof(model_with_land.land.clock.time) === typeof(model_with_land.clock.time)

        # Test PrescribedLand display methods
        @test summary(land) isa String
        @test contains(sprint(show, land), "PrescribedLand")
        @test contains(sprint(show, land), "freshwater_flux")

        # update_state! exercises the new flux-assembly paths without invoking
        # the ocean RK step, which trips an upstream Oceananigans bug in Azᶠᶜᵃ
        # on this size=1 (Flat, Flat, Bounded) grid; see
        # https://github.com/CliMA/Oceananigans.jl/issues/5547
        update_state!(model_no_land)        # get_land_freshwater_flux(::Nothing) path
        update_state!(model_with_land)      # _interpolate_land_freshwater_flux! kernel

        @info " Testing FreezingLimitedOceanTemperature..."

        grid = LatitudeLongitudeGrid(arch;
                                    size = (2, 2, 10),
                                latitude = (-0.5, 0.5),
                               longitude = (-0.5, 0.5),
                                       z = (-1, 0),
                                topology = (Bounded, Bounded, Bounded))

        ocean = ocean_simulation(grid; momentum_advection = nothing,
                                         tracer_advection = nothing,
                                                 coriolis = nothing,
                                                  closure = nothing,
                                  bottom_drag_coefficient = 0.0)

        dates = all_dates(RepeatYearJRA55(), :temperature)
        atmosphere = JRA55PrescribedAtmosphere(arch; end_date=dates[2])

        fill!(ocean.model.tracers.T, -2.0)

        @allowscalar begin
            ocean.model.tracers.T[1, 2, 10] = 1.0
            ocean.model.tracers.T[2, 1, 10] = 1.0

            # Cap all fluxes except for heating ones where T < 0
            sea_ice = FreezingLimitedOceanTemperature()

            # Always cooling!
            fill!(atmosphere.tracers.T, 273.15 - 20)

            coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere)

            # Test that the temperature has snapped up to freezing
            @test minimum(ocean.model.tracers.T) == 0
        end

        @info "Testing Surface Fluxes with sea ice..."

        grid = RectilinearGrid(arch;
                               size = (2, 2, 2),
                             extent = (1, 1, 1),
                           topology = (Periodic, Periodic, Bounded))

        ocean = ocean_simulation(grid; momentum_advection = nothing,
                                         tracer_advection = nothing,
                                                 coriolis = nothing,
                                                  closure = nothing,
                                  bottom_drag_coefficient = 0.0)

        SSU = view(ocean.model.velocities.u, :, :, grid.Nz)
        SSV = view(ocean.model.velocities.v, :, :, grid.Nz)

        τo  = SemiImplicitStress(uₑ=SSU, vₑ=SSV, Cᴰ=0.001, ρₑ=1000.0)
        τua = Field{Face, Center, Nothing}(grid)
        τva = Field{Center, Face, Nothing}(grid)

        dynamics = SeaIceMomentumEquation(grid;
                                          top_momentum_stress = (u=τua, v=τva),
                                          bottom_momentum_stress = τo,
                                          rheology = nothing,
                                          solver = ExplicitSolver())

        sea_ice = sea_ice_simulation(grid; dynamics, advection=Centered())

        # Set a velocity for the ocean
        fill!(ocean.model.velocities.u, 0.1)
        fill!(ocean.model.velocities.v, 0.2)
        fill!(ocean.model.tracers.T,   -2.0)

        # Test that we populate the sea-ice ocean stress
        earth = OceanSeaIceModel(ocean, sea_ice; atmosphere)

        τˣ = earth.interfaces.sea_ice_ocean_interface.fluxes.x_momentum
        τʸ = earth.interfaces.sea_ice_ocean_interface.fluxes.y_momentum

        @allowscalar begin
            @test τˣ[1, 1, 1] == sqrt(0.1^2 + 0.2^2) * 0.1
            @test τʸ[1, 1, 1] == sqrt(0.1^2 + 0.2^2) * 0.2
        end
    end
end

@testset "SkinHumidity vapor-flux balance" begin
    ℂ = AtmosphereThermodynamicsParameters(Float64)
    ℙₐ = (; thermodynamics_parameters = ℂ)

    pᵃᵗ = 101325.0
    Tᵃᵗ = 290.0
    qᵃᵗ = 0.005
    Ψₐ = (; p = pᵃᵗ, q = qᵃᵗ, T = Tᵃᵗ)

    # The saturated reservoir is at the bulk (energy) temperature `Tᵈ`; the skin
    # temperature does not enter the vapor balance.
    Tᵈ = 295.0
    Ψᵢ = (; T = Tᵈ) # interior state (unused by SkinHumidity, passed for signature)
    Tₛ = 310.0 # skin temperature, deliberately ≠ Tᵈ — qˢ must be independent of it
    qᵛ⁺ = saturation_specific_humidity(ℂ, Tᵈ, pᵃᵗ, Thermodynamics.Liquid())
    @test qᵛ⁺ > qᵃᵗ # reservoir saturation exceeds the sub-saturated air

    # AirLandInterfaceState with an upward moisture flux (q★ < 0 ⟹ qˢ > qᵃᵗ);
    # bulk reservoir temperature carried in the energy component.
    mkΨₛ(q) = AirLandInterfaceState(0.3, -0.01, -1e-4, 0.0, 0.0, Tₛ, q,
                                    (saturation = 1.0,), (temperature = Tᵈ,))

    # Drive the fixed point to convergence for a few saturation depths
    converge(d) = begin
        sh = SkinHumidity(surface_thickness=d, vapor_diffusivity=2e-2)
        q = qᵛ⁺
        for _ in 1:100
            q = compute_interface_humidity(sh, Tₛ, mkΨₛ(q), Ψₐ, Ψᵢ, ℙₐ)
        end
        return q
    end

    q_thin  = converge(1e-3) # high soil conductance
    q_mid   = converge(1e-1)
    q_thick = converge(1e2)  # vanishing soil conductance

    # qˢ is always bounded between the atmospheric and saturated values
    for q in (q_thin, q_mid, q_thick)
        @test qᵃᵗ ≤ q ≤ qᵛ⁺
    end

    # Thin dry layer ⟹ surface ≈ saturated; thick dry layer ⟹ surface ≈ air;
    # and qˢ decreases monotonically as the dry layer deepens.
    @test isapprox(q_thin, qᵛ⁺; rtol=1e-2)
    @test isapprox(q_thick, qᵃᵗ; rtol=1e-2)
    @test q_thin > q_mid > q_thick

    # Zero turbulent flux (first iterate) ⟹ saturated surface qˢ = qᵛ⁺
    sh = SkinHumidity(surface_thickness=0.1, vapor_diffusivity=2e-2)
    Ψₛ⁰ = AirLandInterfaceState(0.0, 0.0, 0.0, 0.0, 0.0, Tₛ, 0.0,
                                (saturation = 1.0,), (temperature = Tᵈ,))
    @test compute_interface_humidity(sh, Tₛ, Ψₛ⁰, Ψₐ, Ψᵢ, ℙₐ) ≈ qᵛ⁺
end

@testset "FractionalHumidity (Manabe critical wetness)" begin
    ℂ = AtmosphereThermodynamicsParameters(Float64)
    ℙₐ = (; thermodynamics_parameters = ℂ)
    pᵃᵗ = 101325.0
    Ψₐ = (; p = pᵃᵗ, q = 0.005, T = 290.0)
    Ψᵢ = (; T = 295.0) # unused by FractionalHumidity, passed for signature
    Tₛ = 295.0
    qᵛ⁺ = saturation_specific_humidity(ℂ, Tₛ, pᵃᵗ, Thermodynamics.Liquid())

    # Manabe efficiency: β = min(𝒮/𝒮ᶜ, 1), 𝒮ᶜ = 0.75
    cs = CriticalSaturation(0.75)
    @test evaporation_efficiency(cs, (saturation = 0.0,))   == 0.0
    @test evaporation_efficiency(cs, (saturation = 0.375,)) ≈ 0.5
    @test evaporation_efficiency(cs, (saturation = 0.75,))  ≈ 1.0
    @test evaporation_efficiency(cs, (saturation = 1.0,))   ≈ 1.0   # saturated above 𝒮ᶜ

    # Constant efficiency ignores the land state
    @test evaporation_efficiency(0.3, (saturation = 0.9,)) == 0.3

    # qˢ = β · qᵛ⁺(Tₛ), with β derived from the materialized hydrology state
    mkΨₛ(𝒮) = AirLandInterfaceState(0.3, 0.0, 0.0, 0.0, 0.0, Tₛ, 0.0, (saturation = 𝒮,), (;))
    fh = FractionalHumidity(efficiency = cs)
    @test compute_interface_humidity(fh, Tₛ, mkΨₛ(0.0),   Ψₐ, Ψᵢ, ℙₐ) ≈ 0.0
    @test compute_interface_humidity(fh, Tₛ, mkΨₛ(0.375), Ψₐ, Ψᵢ, ℙₐ) ≈ 0.5 * qᵛ⁺
    @test compute_interface_humidity(fh, Tₛ, mkΨₛ(1.0),   Ψₐ, Ψᵢ, ℙₐ) ≈ qᵛ⁺ # saturated

    # Constant-efficiency FractionalHumidity is a uniform fraction of saturation
    fc = FractionalHumidity(efficiency = 0.4)
    @test compute_interface_humidity(fc, Tₛ, mkΨₛ(0.1), Ψₐ, Ψᵢ, ℙₐ) ≈ 0.4 * qᵛ⁺
end

#=
@testset "Fluxes regression" begin
    for arch in test_architectures
        @info "Testing fluxes regression..."

        grid = LatitudeLongitudeGrid(arch;
                                     size = (20, 20, 20),
                                 latitude = (-60, 60),
                                longitude = (0, 360),
                                        z = (-5000, 0))

        # Speed up compilation by removing all the unnecessary stuff
        momentum_advection = nothing
        tracer_advection   = nothing
        tracers  = (:T, :S)
        buoyancy = nothing
        closure  = nothing
        coriolis = nothing

        ocean = ocean_simulation(grid; momentum_advection, tracer_advection, closure, tracers, coriolis)

        date = DateTimeProlepticGregorian(1993, 1, 1)
        dataset = ECCO4Monthly()
        T_metadata = Metadatum(:temperature; date, dataset)
        S_metadata = Metadatum(:salinity; date, dataset)

        set!(ocean.model; T=T_metadata, S=S_metadata)

        end_date   = all_dates(RepeatYearJRA55(), :temperature)[10]
        atmosphere = JRA55PrescribedAtmosphere(arch; end_date, backend = InMemory())
        radiation  = Radiation(ocean_albedo=0.1, ocean_emissivity=1.0)
        sea_ice    = nothing

        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
        times = 0:1hours:1days
        Ntimes = length(times)

        # average the fluxes over one day
        Jᵀ = interior(ocean.model.tracers.T.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
        Jˢ = interior(ocean.model.tracers.S.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
        τˣ = interior(ocean.model.velocities.u.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
        τʸ = interior(ocean.model.velocities.v.boundary_conditions.top.condition, :, :, 1) ./ Ntimes

        for time in times[2:end]
            coupled_model.clock.time = time
            update_state!(coupled_model)
            Jᵀ .+= interior(ocean.model.tracers.T.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
            Jˢ .+= interior(ocean.model.tracers.S.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
            τˣ .+= interior(ocean.model.velocities.u.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
            τʸ .+= interior(ocean.model.velocities.v.boundary_conditions.top.condition, :, :, 1) ./ Ntimes
        end

        Jᵀ_mean = mean(Jᵀ)
        Jˢ_mean = mean(Jˢ)
        τˣ_mean = mean(τˣ)
        τʸ_mean = mean(τʸ)

        Jᵀ_std = std(Jᵀ)
        Jˢ_std = std(Jˢ)
        τˣ_std = std(τˣ)
        τʸ_std = std(τʸ)

        # Regression test
        @test_broken Jᵀ_mean ≈ -3.526464713488678e-5
        @test_broken Jˢ_mean ≈ 1.1470078542716042e-6
        @test_broken τˣ_mean ≈ -1.0881334225579832e-5
        @test_broken τʸ_mean ≈ 5.653281786086694e-6

        @test_broken Jᵀ_std ≈ 7.477575901188957e-5
        @test_broken Jˢ_std ≈ 3.7416720607945508e-6
        @test_broken τˣ_std ≈ 0.00011349625113971719
        @test_broken τʸ_std ≈ 7.627885224680635e-5
    end
end
=#
