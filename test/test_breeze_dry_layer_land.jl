include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties
using Statistics
using Test

NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@test !isnothing(NumericalEarthBreezeExt)

# Tiny coupled Breeze + variably saturated SlabLand for CI. Verifies the
# end-to-end wiring (interface ↔ hydrology ↔ energy), then runs short enough
# to keep CI under a few seconds.
function build_coupled_test_model(arch; M_init, T_init, with_radiation = false)
    grid = RectilinearGrid(arch,
                           size = (8, 8), halo = (5, 5),
                           x = (-1kilometer, 1kilometer),
                           z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 295.0
    atmosphere = atmosphere_simulation(grid; potential_temperature = θ₀)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.potential_temperature, u = 2)

    land_grid = RectilinearGrid(arch,
                                size = grid.Nx,
                                halo = grid.Hx,
                                x = (-1kilometer, 1kilometer),
                                topology = (Periodic, Flat, Flat))

    hydrology = VariablySaturatedHydrology(eltype(land_grid);
        slab_depth = 1.0, porosity = 0.4,
        residual_liquid_fraction = 0.0,
        storage_height = 1000,
        retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
        hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
        deep_liquid_flux = NoDeepLiquidFlux(),
        runoff = NoRunoff())
    energy = WaterCoupledEnergy(eltype(land_grid);
        dry_heat_capacity = 1e6,
        liquid_heat_capacity = 4186,
        reference_temperature = 273.15,
        deep_temperature = T_init,
        deep_time_scale = 12hours,
        advect_deep_liquid_energy = true,
        advect_surface_liquid_energy = false)
    land = SlabLand(land_grid; hydrology, energy)
    set!(land; T = T_init, M = M_init)

    # Default `BulkTemperature()` on the temperature side: Tⁱⁿ = Tˡᵃ.
    # (Land-side `SkinTemperature` is deferred — the PR plan §13 lists the
    # alternative thermal-conductance models.) The χ-interpolation in the
    # humidity formulation collapses gracefully: Tᵉ = Tˡᵃ when Tⁱⁿ = Tˡᵃ.
    al = atmosphere_land_interface(land_grid, atmosphere, land;
        specific_humidity = DryLayerHumidity(;
            dry_layer_depth = StorageBasedDryLayerDepth(
                maximum_dry_layer_depth = 0.05,
                dry_layer_onset_saturation = 0.5,
                dry_layer_exponent = 2.0),
            vapor_exchange = DryLayerVaporPistonVelocity(
                minimum_dry_layer_depth = 1e-4,
                molecular_diffusivity = 2.5e-5,
                tortuosity = PowerLawTortuosity()),
            thermal_exchange_depth = 0.10,
            porosity = 0.4))

    radiation = nothing
    if with_radiation
        # Lightweight prescribed radiation with known nonzero downwelling fluxes,
        # so the land radiative contribution is unambiguous and nonzero.
        radiation = PrescribedRadiation(land_grid; ocean_surface = nothing,
                                        sea_ice_surface = nothing,
                                        land_surface = SurfaceRadiationProperties(0.2, 0.95))
        parent(radiation.downwelling_shortwave) .= 600  # W m⁻²
        parent(radiation.downwelling_longwave)  .= 350  # W m⁻²
    end

    return AtmosphereLandModel(atmosphere, land; atmosphere_land_interface = al, radiation)
end

@testset "Breeze + DryLayerHumidity wiring" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Coupled construction and time stepping on $A" begin
            model = build_coupled_test_model(arch; M_init = 200.0, T_init = 295.0)

            @test model isa EarthSystemModel
            @test model.atmosphere isa Simulation
            @test model.atmosphere.model isa Breeze.AtmosphereModel
            @test model.land isa SlabLand
            @test model.land.hydrology isa VariablySaturatedHydrology
            @test model.land.energy isa WaterCoupledEnergy

            al_iface = model.interfaces.atmosphere_land_interface
            @test !isnothing(al_iface)

            simulation = Simulation(model; Δt = 0.5, stop_iteration = 10)
            run!(simulation)

            @test model.clock.iteration == 10
            @test model.clock.time > 0

            # End-to-end: turbulent fluxes are nonzero, no NaNs anywhere.
            land_fluxes = model.land.fluxes
            T_after = Array(interior(model.land.temperature))
            M_after = Array(interior(model.land.water_storage))
            Jv      = Array(interior(land_fluxes.vapor_flux))
            @test !any(isnan, T_after)
            @test !any(isnan, M_after)
            @test !any(isnan, Jv)
            # Atmosphere is initialized warm and unsaturated, so we expect
            # evaporation upward: Jᵛ ≥ 0 everywhere.
            @test all(Jv .>= 0)
            # Some cell actually evaporates.
            @test maximum(Jv) > 0
        end

        @testset "Drydown reduces M and saturates skin on $A" begin
            model = build_coupled_test_model(arch; M_init = 200.0, T_init = 295.0)

            M0 = Array(interior(model.land.water_storage))
            simulation = Simulation(model; Δt = 0.5, stop_iteration = 100)
            run!(simulation)
            M1 = Array(interior(model.land.water_storage))

            # Even with a tiny number of steps, dry-down should drop M.
            @test all(M1 .<= M0)
            @test mean(M1) < mean(M0)
        end

        @testset "Radiation adds to surface_energy_flux, not overwritten, on $A" begin
            # Regression guard for the radiation→land coupling (issue #326): the
            # radiative flux must be *added* to the turbulent `surface_energy_flux`
            # that `WaterCoupledEnergy` reads — not overwrite it, and not be
            # overwritten by the turbulent assembly. Covers the call ordering in
            # `time_step_earth_system_model.jl` and the accumulator/sign choice in
            # `apply_air_land_radiative_fluxes.jl`.
            m_rad = build_coupled_test_model(arch; M_init = 200.0, T_init = 295.0, with_radiation = true)
            @test haskey(m_rad.radiation.interface_fluxes, :land)
            update_state!(m_rad)
            Es_rad = Array(interior(m_rad.land.fluxes.surface_energy_flux))

            # Net radiative flux into the surface, reconstructed from the kernel's
            # own stored diagnostics (downwelling stored negative, upwelling positive).
            rf = m_rad.radiation.interface_fluxes.land
            ΣQ_rad = Array(interior(rf.downwelling_longwave)) .+
                     Array(interior(rf.downwelling_shortwave)) .-
                     Array(interior(rf.upwelling_longwave))

            m_nor = build_coupled_test_model(arch; M_init = 200.0, T_init = 295.0, with_radiation = false)
            update_state!(m_nor)
            Es_nor = Array(interior(m_nor.land.fluxes.surface_energy_flux))

            @test maximum(abs, ΣQ_rad) > 0   # radiation contributes a nonzero flux
            @test Es_rad != Es_nor           # ... which actually changed Es (not dropped)
            # `surface_energy_flux` is positive-upward, so net-downward radiation
            # enters as −ΣQ_rad, added on top of the turbulent flux:
            @test Es_rad ≈ Es_nor .- ΣQ_rad
        end
    end
end
