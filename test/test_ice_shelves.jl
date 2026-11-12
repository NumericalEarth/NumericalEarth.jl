include("runtests_setup.jl")

using NumericalEarth: IceShelfOceanInterface,
                      TEOS10Liquidus,
                      VelocityBasedFrictionVelocity,
                      StratFeedback,
                      RotatingBoundaryLayerTransfer,
                      ConvectiveMeltLimit,
                      IceHeatConduction,
                      compute_ice_shelf_fluxes!,
                      ice_shelf_boundary_conditions,
                      ice_shelf_tracer_forcing

using NumericalEarth.IceShelves: ice_shelf_friction_velocity, ice_shelf_liquidus, ice_ocean_interface_heat_flux,
                                 ice_ocean_interface_fluxes, melting_temperature_salinity_derivative, U²ᶜᶜᶜ

using NumericalEarth.EarthSystemModels: ThreeEquationHeatFlux

using ClimaSeaIce.SeaIceThermodynamics: melting_temperature

using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

# Ice wedge whose draft rises from -0.95 at x = 0 to the surface at x ≈ 0.59.
cavity_test_top_height(x) = min(0.0, -0.95 + 1.6x)

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
        𝒬, Tᵦ, Sᵦ = ice_ocean_interface_heat_flux(flux, ocean_state, ice_state, liquidus, z,
                                                  ocean_properties, 334e3, 0.002)
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, z) atol = 1e-12
    end

    once = TEOS10Liquidus(; iterations = 1)
    _, Tᵦ, Sᵦ = ice_ocean_interface_heat_flux(flux, (; T = 1.0, S = 34.5), ice_state, once, -100.0,
                                              ocean_properties, 334e3, 0.002)
    @test abs(Tᵦ - melting_temperature(once, Sᵦ, -100.0)) > 1e-3

    liquidus32 = TEOS10Liquidus(Float32)
    @test melting_temperature(liquidus32, 34.5f0, -500f0) isa Float32
    @test melting_temperature(liquidus32, 34.5f0, -500f0) ≈ -2.267114932771106 atol = 1e-5

    # The default is TEOS-10, consistent with the ocean's equation of state and gravity
    interface = (; liquidus = nothing)
    teos10 = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState(reference_density = 1030))
    @test ice_shelf_liquidus(interface, (; grid = RectilinearGrid(size = (1, 1, 1), extent = (1, 1, 1)), buoyancy = teos10)) isa TEOS10Liquidus{Float64}
    @test ice_shelf_liquidus(interface.liquidus, Float64, teos10).reference_density == 1030
    @test ice_shelf_liquidus(nothing, Float64, SeawaterBuoyancy(gravitational_acceleration = 9.8)).gravitational_acceleration == 9.8
    @test ice_shelf_liquidus(nothing, Float64, nothing) isa TEOS10Liquidus{Float64}
    @test ice_shelf_liquidus(liquidus32, Float64, teos10) === liquidus32
end

@testset "Three-equation solve with pressure-dependent liquidus" begin
    liquidus = TEOS10Liquidus()
    flux = ThreeEquationHeatFlux() # constant friction velocity u★ = 0.002

    ℰ = 334e3
    u★ = 0.002
    ocean_properties = (reference_density = 1028.0, heat_capacity = 3991.0)

    ocean_state = (; T = 1.0, S = 34.5)
    ice_state = (; S = 0.0, h = 0.0, hc = 0.0, ℵ = 1.0, T = 0.0)

    𝒬₁, Tᵦ₁, Sᵦ₁ = ice_ocean_interface_heat_flux(flux, ocean_state, ice_state, liquidus, -100.0,
                                                 ocean_properties, ℰ, u★)

    # Warm water under shallow draft: melting
    @test 𝒬₁ > 0

    # Interface sits on the liquidus, cooler and fresher than the ambient ocean
    @test Tᵦ₁ ≈ melting_temperature(liquidus, Sᵦ₁, -100.0) atol = 1e-10
    @test Tᵦ₁ < ocean_state.T
    @test Sᵦ₁ < ocean_state.S

    # Deeper draft: lower freezing point, larger thermal driving, more melt
    𝒬₂, Tᵦ₂, _ = ice_ocean_interface_heat_flux(flux, ocean_state, ice_state, liquidus, -900.0,
                                               ocean_properties, ℰ, u★)
    @test 𝒬₂ > 𝒬₁
    @test Tᵦ₂ < Tᵦ₁

    # Supercooled ocean: freezing (negative melt rate)
    cold_state = (; T = -3.0, S = 34.5)
    𝒬₃, _, _ = ice_ocean_interface_heat_flux(flux, cold_state, ice_state, liquidus, 0.0,
                                             ocean_properties, ℰ, u★)
    @test 𝒬₃ < 0
end

@testset "VelocityBasedFrictionVelocity" begin
    # Constant friction velocity passes through untouched
    @test ice_shelf_friction_velocity(0.01, 1, 1, 1, nothing, nothing, nothing) == 0.01

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
    U² = KernelFunctionOperation{Center, Center, Center}(U²ᶜᶜᶜ, grid, u, v)
    u★ = ice_shelf_friction_velocity(fv, 2, 2, 2, grid, U², H_TBL)
    @test u★ ≈ sqrt(0.0015 * (0.1^2 + 0.2^2))

    # Tidal RMS velocity keeps u★ from vanishing at rest (ISOMIP+, Eq. 27)
    fv_tidal = VelocityBasedFrictionVelocity(drag_coefficient = 2.5e-3, tidal_velocity = 0.01)
    u₀ = XFaceField(grid)
    v₀ = YFaceField(grid)
    U²₀ = KernelFunctionOperation{Center, Center, Center}(U²ᶜᶜᶜ, grid, u₀, v₀)
    u★₀ = ice_shelf_friction_velocity(fv_tidal, 2, 2, 2, grid, U²₀, H_TBL)
    @test u★₀ ≈ sqrt(2.5e-3) * 0.01
    u★₁ = ice_shelf_friction_velocity(fv_tidal, 2, 2, 2, grid, U², H_TBL)
    @test u★₁ ≈ sqrt(2.5e-3 * (0.1^2 + 0.2^2 + 0.01^2))

    # Minimum friction velocity
    fv_min = VelocityBasedFrictionVelocity(minimum_friction_velocity = 1e-4)
    @test ice_shelf_friction_velocity(fv_min, 2, 2, 2, grid, U²₀, H_TBL) == 1e-4
    @test ice_shelf_friction_velocity(fv_min, 2, 2, 2, grid, U², H_TBL) ≈ sqrt(0.0015 * (0.1^2 + 0.2^2))
    @test fv.minimum_friction_velocity == 0
    @test VelocityBasedFrictionVelocity(Float32; minimum_friction_velocity = 1e-4).minimum_friction_velocity isa Float32
end

melt_physics(; transfer_coefficients = nothing, convective_limit = nothing,
               ice_heat_conduction = nothing, basal_slope = nothing) =
    (; transfer_coefficients, convective_limit, ice_heat_conduction, basal_slope)

@testset "Optional ice shelf melt physics" begin
    flux = ThreeEquationHeatFlux()
    ℰ = 334e3
    zᵈ = -500.0
    properties = (reference_density = 1028.0, heat_capacity = 3991.0)
    ice_state = (; S = 0.0, h = 0.0, hc = 0.0, ℵ = 1.0, T = 0.0)
    warm = (; T = 1.0, S = 34.5)
    cold = (; T = -3.0, S = 34.5)
    steep = fill(1.0, 1, 1, 1)
    flat = fill(0.0, 1, 1, 1)

    fluxes(physics, ocean_state, liquidus; u★ = 0.002) =
        ice_ocean_interface_fluxes(physics, 1, 1, flux, ocean_state, ice_state, liquidus, zᵈ, properties, ℰ, u★)

    let liquidus = TEOS10Liquidus()
        𝒬₀, q₀, Tᵦ₀, Sᵦ₀ = fluxes(melt_physics(), warm, liquidus)
        𝒬ᵒ, Tᵦᵒ, Sᵦᵒ = ice_ocean_interface_heat_flux(flux, warm, ice_state, liquidus, zᵈ, properties, ℰ, 0.002)
        @test (𝒬₀, q₀, Tᵦ₀, Sᵦ₀) === (𝒬ᵒ, 𝒬ᵒ / ℰ, Tᵦᵒ, Sᵦᵒ)

        # The general solver reproduces the default when no option is active
        𝒬, q, Tᵦ, Sᵦ = fluxes(melt_physics(convective_limit = ConvectiveMeltLimit(), basal_slope = flat), warm, liquidus)
        @test 𝒬 ≈ 𝒬₀
        @test q ≈ q₀
        @test Tᵦ ≈ Tᵦ₀
        @test Sᵦ ≈ Sᵦ₀

        # Meltwater stratification suppresses melting
        𝒬, q, Tᵦ, Sᵦ = fluxes(melt_physics(transfer_coefficients = StratFeedback()), warm, liquidus)
        @test 0 < q < q₀
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, zᵈ) atol = 1e-10

        # Strong shear swamps the stratification and recovers constant coefficients
        _, q₀, _, _ = fluxes(melt_physics(), warm, liquidus; u★ = 0.1)
        _, q, _, _  = fluxes(melt_physics(transfer_coefficients = StratFeedback()), warm, liquidus; u★ = 0.1)
        @test q ≈ q₀

        # Freezing is statically unstable, so the coefficients stay constant
        _, q₀, _, _ = fluxes(melt_physics(), cold, liquidus)
        _, q, _, _  = fluxes(melt_physics(transfer_coefficients = StratFeedback()), cold, liquidus)
        @test q₀ < 0
        @test q ≈ q₀

        # Convection on a steep base enhances melting under weak shear
        _, q₀, _, _ = fluxes(melt_physics(), warm, liquidus; u★ = 1e-4)
        _, q, _, _  = fluxes(melt_physics(convective_limit = ConvectiveMeltLimit(), basal_slope = steep), warm, liquidus; u★ = 1e-4)
        @test q > q₀

        _, qˢ, _, _ = fluxes(melt_physics(transfer_coefficients = StratFeedback(),
                                          convective_limit = ConvectiveMeltLimit(), basal_slope = steep), warm, liquidus; u★ = 1e-4)
        @test qˢ ≥ q * (1 - 1e-6)

        # Conduction into cold ice takes heat away from melting
        _, q₀, _, _ = fluxes(melt_physics(), warm, liquidus)
        𝒬, q, Tᵦ, _ = fluxes(melt_physics(ice_heat_conduction = IceHeatConduction()), warm, liquidus)
        γᵀ = flux.heat_transfer_coefficient * 0.002
        @test 𝒬 ≈ properties.reference_density * properties.heat_capacity * γᵀ * (warm.T - Tᵦ)
        @test 0 < q < 𝒬 / ℰ
        @test q < q₀
    end

    # Velocity- and stability-dependent coefficients in a rotating boundary layer
    f = -1.4e-4
    u★ = 0.002
    hj = RotatingBoundaryLayerTransfer()
    ξᴺ, k, ν = hj.stability_constant, hj.von_karman_constant, hj.molecular_viscosity
    Γᵀ = 12.5 * hj.prandtl_number^(2/3) - 6
    ρc = properties.reference_density * properties.heat_capacity

    let liquidus = TEOS10Liquidus()
        warm_f = (; warm..., f)
        neutral = melt_physics(transfer_coefficients = RotatingBoundaryLayerTransfer(critical_richardson_number = Inf))

        𝒬ⁿ, qⁿ, Tᵦ, Sᵦ = fluxes(neutral, warm_f, liquidus; u★)
        Γᵗ = log(ξᴺ * u★^2 / (5ν * abs(f))) / k + 1 / (2ξᴺ) - 1 / k
        @test 𝒬ⁿ / (ρc * (warm.T - Tᵦ)) ≈ u★ / (Γᵗ + Γᵀ)
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, zᵈ) atol = 1e-10

        # Without rotation the log-layer term vanishes
        𝒬, _, Tᵦ, _ = fluxes(neutral, (; warm..., f = 0.0), liquidus; u★)
        @test 𝒬 / (ρc * (warm.T - Tᵦ)) ≈ u★ / (1 / (2ξᴺ) - 1 / k + Γᵀ)
        _, q, _, _ = fluxes(melt_physics(transfer_coefficients = hj), (; warm..., f = 0.0), liquidus; u★)
        @test isfinite(q) && q > 0

        # Meltwater stratification reduces η★ and so the melt rate
        _, q, _, _ = fluxes(melt_physics(transfer_coefficients = hj), warm_f, liquidus; u★)
        @test 0 < q < qⁿ

        # Freezing is unstable, so η★ = 1
        _, qⁿ, _, _ = fluxes(neutral, (; cold..., f), liquidus; u★)
        _, q, _, _  = fluxes(melt_physics(transfer_coefficients = hj), (; cold..., f), liquidus; u★)
        @test qⁿ < 0
        @test q ≈ qⁿ
    end

    # Advective-diffusive ice heat flux
    let liquidus = TEOS10Liquidus()
        advective = IceHeatConduction(advective = true)
        cᴵ, Tˢ = advective.ice_heat_capacity, advective.surface_temperature

        _, q₀, _, _ = fluxes(melt_physics(), warm, liquidus)
        𝒬, q, Tᵦ, Sᵦ = fluxes(melt_physics(ice_heat_conduction = advective), warm, liquidus)
        @test 𝒬 ≈ q * (ℰ + cᴵ * (Tᵦ - Tˢ))
        @test 0 < q < q₀
        @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ, zᵈ) atol = 1e-10

        _, q, _, _ = fluxes(melt_physics(ice_heat_conduction = IceHeatConduction(advective = true, ice_heat_capacity = 0)), warm, liquidus)
        @test q ≈ q₀

        # No ice heat flux when the ocean is below freezing
        _, q₀, _, _ = fluxes(melt_physics(), cold, liquidus)
        _, q, _, _  = fluxes(melt_physics(ice_heat_conduction = advective), cold, liquidus)
        @test q ≈ q₀
    end

    @test RotatingBoundaryLayerTransfer(Float32).stability_constant isa Float32
    @test IceHeatConduction(Float32; advective = true).ice_heat_capacity isa Float32
    @test StratFeedback(Float32).heat_prefactor isa Float32
    @test ConvectiveMeltLimit(Float32).coefficient isa Float32
    @test IceHeatConduction(Float32).ice_diffusivity isa Float32
end

# Immersed-top grids need an Oceananigans release with CliMA/Oceananigans.jl#6175
if isdefined(Oceananigans, :TopLoad)

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
        top_height(x, y) = x < 1 ? 0.0 : (x < 2 ? -0.5 : (x < 3 ? -0.99 : 0.0))

        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedVerticalBoundaries(; bottom_height = bottom, top_height))
        interface = IceShelfOceanInterface(grid)

        @test Array(interior(interface.k_draft))[:, 2, 1] == [0, 2, 0, 0]

        # An immersed top with zero ice concentration is inert; fluxes scale with the concentration
        rock_interface = IceShelfOceanInterface(grid; ice_concentration = (x, y) -> x > 2,
                                                flux_formulation = ThreeEquationHeatFlux())
        half_interface = IceShelfOceanInterface(grid; ice_concentration = 1/2,
                                                flux_formulation = ThreeEquationHeatFlux())

        # Melt fluxes with a constant friction velocity and quiescent warm water
        interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux())
        model = HydrostaticFreeSurfaceModel(grid; tracers = (:T, :S))
        set!(model, T = 1.0, S = 34.5)

        compute_ice_shelf_fluxes!(interface, model)
        compute_ice_shelf_fluxes!(rock_interface, model)
        compute_ice_shelf_fluxes!(half_interface, model)
        for name in keys(rock_interface.fluxes)
            @test all(Array(interior(rock_interface.fluxes[name])) .== 0)
        end

        for name in (:interface_heat, :freshwater)
            @test Array(interior(half_interface.fluxes[name]))[2, 2, 1] ≈
                  Array(interior(interface.fluxes[name]))[2, 2, 1] / 2
        end

        interface_heat = Array(interior(interface.fluxes.interface_heat))[:, 2, 1]
        freshwater     = Array(interior(interface.fluxes.freshwater))[:, 2, 1]
        heat_content   = Array(interior(interface.fluxes.freshwater_heat_content))[:, 2, 1]

        # Melting only in the iced column
        @test interface_heat[2] > 0
        @test freshwater[2] > 0

        for i in (1, 3, 4)
            @test interface_heat[i] == 0
            @test freshwater[i] == 0
        end

        # Interface state on the liquidus at the discrete ice base (z = -0.5)
        T★ = Array(interior(interface.temperature))[2, 2, 1]
        S★ = Array(interior(interface.salinity))[2, 2, 1]
        @test T★ ≈ melting_temperature(ice_shelf_liquidus(interface, model), S★, -0.5) atol = 1e-10
        @test Array(interior(interface.friction_velocity))[2, 2, 1] == 0.002

        # Meltwater enters at the interface temperature
        @test heat_content[2] ≈ T★ * freshwater[2]

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
        @test Array(interior(interface.fluxes.freshwater))[2, 2, 1] > 0

        # Optional melt physics through the kernel
        @test_throws ArgumentError IceShelfOceanInterface(grid; convective_limit = ConvectiveMeltLimit())

        reduced_interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux(),
                                                   transfer_coefficients = StratFeedback(),
                                                   ice_heat_conduction = IceHeatConduction())
        convective_interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux(friction_velocity = 1e-4),
                                                      convective_limit = ConvectiveMeltLimit(), basal_slope = 1)
        weak_interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux(friction_velocity = 1e-4))

        for i in (interface, reduced_interface, convective_interface, weak_interface)
            compute_ice_shelf_fluxes!(i, model)
        end

        melt_rate = Array(interior(interface.fluxes.freshwater))[:, 2, 1]
        reduced_melt_rate = Array(interior(reduced_interface.fluxes.freshwater))[:, 2, 1]
        @test 0 < reduced_melt_rate[2] < melt_rate[2]
        @test reduced_melt_rate[[1, 3, 4]] == [0, 0, 0]
        @test Array(interior(convective_interface.fluxes.freshwater))[2, 2, 1] >
              Array(interior(weak_interface.fluxes.freshwater))[2, 2, 1]

        hj_interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux(),
                                              transfer_coefficients = RotatingBoundaryLayerTransfer(),
                                              ice_heat_conduction = IceHeatConduction(advective = true))
        rotating_model = HydrostaticFreeSurfaceModel(grid; tracers = (:T, :S), coriolis = FPlane(f = -1.4e-4))
        set!(rotating_model, T = 1.0, S = 34.5)

        for m in (model, rotating_model)
            compute_ice_shelf_fluxes!(hj_interface, m)
            hj_melt_rate = Array(interior(hj_interface.fluxes.freshwater))[:, 2, 1]
            @test hj_melt_rate[2] > 0
            @test hj_melt_rate[[1, 3, 4]] == [0, 0, 0]
        end
    end

    @testset "Partial-cell drafts [$(typeof(arch))]" begin
        # Open; thin draft cutting k = Nz; draft cutting k = 2; shallow partial bottom without ice
        underlying_grid = RectilinearGrid(arch, size = (4, 3, 4),
                                          x = (0, 4), y = (0, 3), z = (-1, 0),
                                          topology = (Bounded, Periodic, Bounded))

        bottom(x, y)  = x < 3 ? -1.0 : -0.15
        top_height(x, y) = x < 1 ? 0.0 : (x < 2 ? -0.1 : (x < 3 ? -0.6 : 0.0))

        grid = ImmersedBoundaryGrid(underlying_grid, PartialCellVerticalBoundaries(; bottom_height = bottom, top_height))
        interface = IceShelfOceanInterface(grid; flux_formulation = ThreeEquationHeatFlux())

        @test Array(interior(interface.k_draft))[:, 2, 1] == [0, 4, 2, 0]

        model = HydrostaticFreeSurfaceModel(grid; tracers = (:T, :S))
        set!(model, T = 1.0, S = 34.5)
        compute_ice_shelf_fluxes!(interface, model)

        Jʷ = Array(interior(interface.fluxes.freshwater))[:, 2, 1]
        T★ = Array(interior(interface.temperature))[:, 2, 1]
        S★ = Array(interior(interface.salinity))[:, 2, 1]
        liquidus = ice_shelf_liquidus(interface, model)

        @test all(Jʷ[2:3] .> 0)
        @test Jʷ[1] == Jʷ[4] == 0
        @test T★[2] ≈ melting_temperature(liquidus, S★[2], -0.1) atol = 1e-10
        @test T★[3] ≈ melting_temperature(liquidus, S★[3], -0.6) atol = 1e-10
    end

    @testset "Melting cavity simulation [$(typeof(arch))]" begin
        Nx, Nz = 16, 20
        underlying_grid = RectilinearGrid(arch, size = (Nx, Nz),
                                          x = (0, 1), z = (-1, 0),
                                          topology = (Bounded, Flat, Bounded),
                                          halo = (4, 4))

        T₀, S₀ = 1.0, 34.5
        buoyancy = SeawaterBuoyancy()
        top_load = TopLoad(buoyancy, (; T = (x, z) -> T₀, S = (x, z) -> S₀))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedVerticalBoundaries(; bottom_height = -1, top_height = cavity_test_top_height, top_load))

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
        @test all(isfinite, interior(interface.fluxes.freshwater))

        k_draft    = Array(interior(interface.k_draft))[:, 1, 1]
        freshwater = Array(interior(interface.fluxes.freshwater))[:, 1, 1]
        Tᵢ = Array(interior(T))[:, 1, :]
        Sᵢ = Array(interior(S))[:, 1, :]

        iced = findall(k_draft .> 0)
        @test length(iced) > 5

        # Meltwater volume flux in m s⁻¹ to m of ice per year
        ρᵒᶜ = interface.properties.reference_density
        melt_rate_m_yr = freshwater[iced] .* ρᵒᶜ ./ 918 .* 3.156e7
        @test all(melt_rate_m_yr .> 0.1)
        @test all(melt_rate_m_yr .< 1000)

        # Meltwater cools and freshens the cell beneath the ice
        @test all(Tᵢ[i, k_draft[i]] < T₀ for i in iced)
        @test all(Sᵢ[i, k_draft[i]] < S₀ for i in iced)

        # The buoyant meltwater plume drives a circulation
        @test maximum(abs, interior(u)) > 0
    end
end

else
    @warn "Skipping the ice shelf cavity tests: this Oceananigans has no immersed tops"
end
