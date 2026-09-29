include("runtests_setup.jl")

using NumericalEarth: IceShelfOceanInterface,
                      PressureDependentLiquidus,
                      TEOS10Liquidus,
                      VelocityBasedFrictionVelocity,
                      compute_ice_shelf_fluxes!,
                      ice_shelf_boundary_conditions,
                      ice_shelf_tracer_forcing

using NumericalEarth.IceShelves: at_depth, ice_shelf_friction_velocity, ice_shelf_liquidus, ice_shelf_interface_heat_flux,
                                 melting_temperature_salinity_derivative

using NumericalEarth.EarthSystemModels: ThreeEquationHeatFlux
using NumericalEarth.EarthSystemModels.InterfaceComputations: compute_interface_heat_flux

using ClimaSeaIce.SeaIceThermodynamics: LinearLiquidus, melting_temperature

using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

# Ice wedge whose draft rises from -0.95 at x = 0 to the surface at x ≈ 0.59.
cavity_test_ceiling(x) = min(0.0, -0.95 + 1.6x)

@testset "PressureDependentLiquidus" begin
    liquidus = PressureDependentLiquidus()

    # ISOMIP+ surface values
    @test melting_temperature(liquidus, 0.0, 0.0) ≈ 0.0832
    @test melting_temperature(liquidus, 34.5, 0.0) ≈ 0.0832 - 0.0573 * 34.5

    # Melting temperature decreases with depth (z < 0)
    z = -500.0
    @test melting_temperature(liquidus, 34.5, z) ≈ 0.0832 - 0.0573 * 34.5 + liquidus.depth_slope * z
    @test melting_temperature(liquidus, 34.5, -1000.0) < melting_temperature(liquidus, 34.5, 0.0)

    # ISOMIP+ magnitude: λ ≈ 7.59e-4 °C/m
    @test liquidus.depth_slope ≈ 7.53e-8 * 1028 * 9.81

    # at_depth returns the equivalent LinearLiquidus at fixed interface depth
    effective = at_depth(liquidus, z)
    @test effective isa LinearLiquidus
    @test melting_temperature(effective, 34.5) ≈ melting_temperature(liquidus, 34.5, z)

    # at_depth on a LinearLiquidus is the identity
    linear = LinearLiquidus(Float64)
    @test at_depth(linear, z) === linear

    # Float32 construction propagates the float type
    liquidus32 = PressureDependentLiquidus(Float32)
    @test liquidus32.slope isa Float32
    @test at_depth(liquidus32, -100) isa LinearLiquidus{Float32}
end

@testset "TEOS10Liquidus" begin
    # Reference values from GibbsSeaWater's gsw_ct_freezing_poly, with p = -1020 g z / 10⁴ dbar
    liquidus = TEOS10Liquidus()
    @test melting_temperature(liquidus, 34.5, -500.0) ≈ -2.267114932771106 atol = 1e-12
    @test melting_temperature(liquidus, 20.0, -2000.0) ≈ -2.652151046532734 atol = 1e-12
    @test melting_temperature(liquidus, 35.0, 0.0) ≈ -1.908782848445280 atol = 1e-12

    air_free = TEOS10Liquidus(; saturation_fraction = 0)
    @test melting_temperature(air_free, 34.5, -500.0) ≈ -2.265205873885675 atol = 1e-12
    @test melting_temperature(air_free, 0.0, 0.0) ≈ 0.017947064327969 atol = 1e-12

    # The salinity derivative matches a central difference
    for S in (5.0, 34.5), z in (0.0, -1000.0)
        δ = 1e-5
        ∂S = (melting_temperature(liquidus, S + δ, z) - melting_temperature(liquidus, S - δ, z)) / 2δ
        @test melting_temperature_salinity_derivative(liquidus, S, z) ≈ ∂S rtol = 1e-8
    end

    # Relinearizing converges onto the nonlinear liquidus, melting or freezing
    flux = ThreeEquationHeatFlux()
    ocean_properties = (reference_density = 1020.0, heat_capacity = 3991.0)
    ice_state = (; S = 0.0, h = 0.0, hc = 0.0, ℵ = 1.0, T = 0.0)

    for ocean_state in ((; T = 1.0, S = 34.5), (; T = -2.5, S = 34.5)), z in (-100.0, -1000.0)
        𝒬, Tᵦ, Sᵦ = ice_shelf_interface_heat_flux(flux, ocean_state, ice_state, liquidus, z,
                                                  ocean_properties, 334e3, 0.002)
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, z) atol = 1e-12
    end

    once = TEOS10Liquidus(; iterations = 1)
    _, Tᵦ, Sᵦ = ice_shelf_interface_heat_flux(flux, (; T = 1.0, S = 34.5), ice_state, once, -100.0,
                                              ocean_properties, 334e3, 0.002)
    @test abs(Tᵦ - melting_temperature(once, Sᵦ, -100.0)) > 1e-3

    liquidus32 = TEOS10Liquidus(Float32)
    @test melting_temperature(liquidus32, 34.5f0, -500f0) isa Float32
    @test melting_temperature(liquidus32, 34.5f0, -500f0) ≈ -2.267114932771106 atol = 1e-5

    # The default follows the ocean's equation of state
    interface = (; liquidus = nothing)
    teos10 = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState(reference_density = 1030))
    @test ice_shelf_liquidus(interface, (; grid = RectilinearGrid(size = (1, 1, 1), extent = (1, 1, 1)), buoyancy = teos10)) isa TEOS10Liquidus{Float64}
    @test ice_shelf_liquidus(interface.liquidus, Float64, teos10).reference_density == 1030
    @test ice_shelf_liquidus(nothing, Float64, SeawaterBuoyancy()) isa PressureDependentLiquidus
    @test ice_shelf_liquidus(nothing, Float64, nothing) isa PressureDependentLiquidus
    @test ice_shelf_liquidus(liquidus32, Float64, teos10) === liquidus32
end

@testset "Three-equation solve with pressure-dependent liquidus" begin
    liquidus = PressureDependentLiquidus()
    flux = ThreeEquationHeatFlux() # constant friction velocity u★ = 0.002

    ℰ = 334e3
    u★ = 0.002
    ocean_properties = (reference_density = 1028.0, heat_capacity = 3991.0)

    ocean_state = (; T = 1.0, S = 34.5)
    ice_state = (; S = 0.0, h = 0.0, hc = 0.0, ℵ = 1.0, T = 0.0)

    𝒬₁, Tᵦ₁, Sᵦ₁ = compute_interface_heat_flux(flux, ocean_state, ice_state,
                                               at_depth(liquidus, -100.0),
                                               ocean_properties, ℰ, u★)
    q₁ = 𝒬₁ / ℰ

    # Warm water under shallow draft: melting
    @test q₁ > 0
    @test 𝒬₁ > 0

    # Interface sits on the liquidus, cooler and fresher than the ambient ocean
    @test Tᵦ₁ ≈ melting_temperature(liquidus, Sᵦ₁, -100.0)
    @test Tᵦ₁ < ocean_state.T
    @test Sᵦ₁ < ocean_state.S

    # Deeper draft: lower freezing point, larger thermal driving, more melt
    𝒬₂, Tᵦ₂, Sᵦ₂ = compute_interface_heat_flux(flux, ocean_state, ice_state,
                                               at_depth(liquidus, -900.0),
                                               ocean_properties, ℰ, u★)
    @test 𝒬₂ > 𝒬₁
    @test Tᵦ₂ < Tᵦ₁

    # Supercooled ocean: freezing (negative melt rate)
    cold_state = (; T = -3.0, S = 34.5)
    𝒬₃, _, _ = compute_interface_heat_flux(flux, cold_state, ice_state,
                                           at_depth(liquidus, 0.0),
                                           ocean_properties, ℰ, u★)
    @test 𝒬₃ < 0
end

@testset "VelocityBasedFrictionVelocity" begin
    # Constant friction velocity passes through untouched
    @test ice_shelf_friction_velocity(0.01, 1, 1, 1, nothing, nothing, nothing, nothing) == 0.01

    fv = VelocityBasedFrictionVelocity()
    @test fv.drag_coefficient == 0.0015
    @test VelocityBasedFrictionVelocity(Float32).drag_coefficient isa Float32

    grid = RectilinearGrid(size = (4, 4, 4), extent = (1, 1, 1))
    u = XFaceField(grid)
    v = YFaceField(grid)
    set!(u, 0.1)
    set!(v, 0.2)
    fill_halo_regions!(u)
    fill_halo_regions!(v)

    # Uniform velocities: interpolation and boundary-layer averaging are both
    # exact for a uniform field, u★ = √(Cd (u² + v²))
    H_TBL = 1.0
    u★ = ice_shelf_friction_velocity(fv, 2, 2, 2, grid, u, v, H_TBL)
    @test u★ ≈ sqrt(0.0015 * (0.1^2 + 0.2^2))

    # Tidal RMS velocity keeps u★ from vanishing at rest (ISOMIP+, Eq. 27)
    fv_tidal = VelocityBasedFrictionVelocity(drag_coefficient = 2.5e-3, tidal_velocity = 0.01)
    u₀ = XFaceField(grid)
    v₀ = YFaceField(grid)
    u★₀ = ice_shelf_friction_velocity(fv_tidal, 2, 2, 2, grid, u₀, v₀, H_TBL)
    @test u★₀ ≈ sqrt(2.5e-3) * 0.01
    u★₁ = ice_shelf_friction_velocity(fv_tidal, 2, 2, 2, grid, u, v, H_TBL)
    @test u★₁ ≈ sqrt(2.5e-3 * (0.1^2 + 0.2^2 + 0.01^2))
end

for arch in test_architectures
    @testset "IceShelfOceanInterface k_draft map [$(typeof(arch))]" begin
        # Four column types along x, z ∈ (-1, 0) with Δz = 1/4:
        # 1. open ocean          → k_draft = 0
        # 2. iced, draft -0.5    → topmost wet cell k = 2
        # 3. closed, draft -0.99 → no wet cell, k_draft = 0
        # 4. raised bottom -0.6, no ice → k_draft = 0
        underlying_grid = RectilinearGrid(arch, size = (4, 3, 4),
                                          x = (0, 4), y = (0, 3), z = (-1, 0),
                                          topology = (Bounded, Periodic, Bounded))

        bottom(x, y)  = x < 3 ? -1.0 : -0.6
        ceiling(x, y) = x < 1 ? 0.0 : (x < 2 ? -0.5 : (x < 3 ? -0.99 : 0.0))

        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(bottom, ceiling))
        interface = IceShelfOceanInterface(grid)

        @test Array(interior(interface.k_draft))[:, 2, 1] == [0, 2, 0, 0]

        # Melt fluxes with a constant friction velocity and quiescent warm water
        interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux())
        model = HydrostaticFreeSurfaceModel(grid; tracers = (:T, :S))
        set!(model, T = 1.0, S = 34.5)

        compute_ice_shelf_fluxes!(interface, model)

        melt_rate      = Array(interior(interface.fluxes.melt_rate))[:, 2, 1]
        interface_heat = Array(interior(interface.fluxes.interface_heat))[:, 2, 1]
        heat_flux      = Array(interior(interface.fluxes.temperature))[:, 2, 1]
        salt_flux      = Array(interior(interface.fluxes.salt))[:, 2, 1]

        # Melting only in the iced column
        @test melt_rate[2] > 0
        @test interface_heat[2] > 0
        @test heat_flux[2] > 0
        @test salt_flux[2] > 0

        for i in (1, 3, 4)
            @test melt_rate[i] == 0
            @test heat_flux[i] == 0
            @test salt_flux[i] == 0
        end

        # Interface state on the liquidus at the discrete ice base (z = -0.5)
        T★ = Array(interior(interface.temperature))[2, 2, 1]
        S★ = Array(interior(interface.salinity))[2, 2, 1]
        @test T★ ≈ melting_temperature(ice_shelf_liquidus(interface, model), S★, -0.5)
        @test Array(interior(interface.friction_velocity))[2, 2, 1] == 0.002

        ρᵒᶜ = interface.properties.reference_density
        cᵒᶜ = interface.properties.heat_capacity
        @test heat_flux[2] ≈ interface_heat[2] / (ρᵒᶜ * cᵒᶜ)

        # With a TEOS-10 ocean the interface sits on the TEOS-10 liquidus
        buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState())
        teos10_model = HydrostaticFreeSurfaceModel(grid; tracers = (:T, :S), buoyancy)
        set!(teos10_model, T = 1.0, S = 34.5)

        compute_ice_shelf_fluxes!(interface, teos10_model)

        liquidus = ice_shelf_liquidus(interface, teos10_model)
        @test liquidus isa TEOS10Liquidus
        T★ = Array(interior(interface.temperature))[2, 2, 1]
        S★ = Array(interior(interface.salinity))[2, 2, 1]
        @test T★ ≈ melting_temperature(liquidus, S★, -0.5) atol = 1e-10
        @test Array(interior(interface.fluxes.melt_rate))[2, 2, 1] > 0
    end

    @testset "Melting cavity simulation [$(typeof(arch))]" begin
        Nx, Nz = 16, 20
        underlying_grid = RectilinearGrid(arch, size = (Nx, Nz),
                                          x = (0, 1), z = (-1, 0),
                                          topology = (Bounded, Flat, Bounded),
                                          halo = (4, 4))

        T₀, S₀ = 1.0, 34.5
        buoyancy = SeawaterBuoyancy()
        ice_load = CavityLoad(buoyancy, (; T = (x, z) -> T₀, S = (x, z) -> S₀))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-1, cavity_test_ceiling; ice_load))

        # Constant u★: melt starts from rest and spins up the cavity circulation
        interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux())
        boundary_conditions = ice_shelf_boundary_conditions(interface)
        forcing = ice_shelf_tracer_forcing(interface)

        closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1e-3, κ = 1e-4)

        model = HydrostaticFreeSurfaceModel(grid;
                                            tracers = (:T, :S),
                                            buoyancy,
                                            closure,
                                            boundary_conditions,
                                            forcing)

        set!(model, T = T₀, S = S₀)

        simulation = Simulation(model; Δt = 0.05, stop_iteration = 200, verbose = false)
        add_callback!(simulation, sim -> compute_ice_shelf_fluxes!(interface, sim), IterationInterval(1))

        compute_ice_shelf_fluxes!(interface, model)
        run!(simulation)

        u, v, w = model.velocities
        T, S = model.tracers.T, model.tracers.S

        @test all(isfinite, interior(T))
        @test all(isfinite, interior(S))
        @test all(isfinite, interior(u))
        @test all(isfinite, interior(w))
        @test all(isfinite, interior(interface.fluxes.melt_rate))

        k_draft   = Array(interior(interface.k_draft))[:, 1, 1]
        melt_rate = Array(interior(interface.fluxes.melt_rate))[:, 1, 1]
        Tᵢ = Array(interior(T))[:, 1, :]
        Sᵢ = Array(interior(S))[:, 1, :]

        iced = findall(k_draft .> 0)
        @test length(iced) > 5

        # melt_rate is in kg m⁻² s⁻¹; convert to m of ice per year
        melt_rate_m_yr = melt_rate[iced] ./ 918 .* 3.156e7
        @test all(melt_rate_m_yr .> 0.1)
        @test all(melt_rate_m_yr .< 1000)

        # Meltwater cools and freshens the cell beneath the ice
        @test all(Tᵢ[i, k_draft[i]] < T₀ for i in iced)
        @test all(Sᵢ[i, k_draft[i]] < S₀ for i in iced)

        # The buoyant meltwater plume drives a circulation
        @test maximum(abs, interior(u)) > 0
    end
end
