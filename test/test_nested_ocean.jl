include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.Bathymetry: bottom_height_from_mask
using NumericalEarth.EarthSystemModels: ocean_surface_temperature, ocean_surface_velocities,
                                        ocean_temperature
using NumericalEarth.Grids: is_three_dimensional
using NumericalEarth.NestedOceans: parent_ocean_variables
using Oceananigans
using Oceananigans.BoundaryConditions: GravityWaveRadiation, NormalFlow, NormalRadiation,
                                       SurfaceWaveRadiation, Value
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.TimeSteppers: time_step!, update_state!
using Oceananigans.Units: days
using Test

const SIDES = (:west, :east, :south, :north)

# A resting parent bracketing the child, with a uniform zonal throughflow.
function nesting_test_parent(; u = 0.1, T = 15, S = 35, η = 0)
    grid = LatitudeLongitudeGrid(size = (12, 12, 6),
                                 longitude = (-2, 2), latitude = (28, 32), z = (-1000, 0),
                                 topology = (Bounded, Bounded, Bounded))

    parent_ocean = PrescribedOcean(grid, [0.0, 10days])
    set!(parent_ocean; T, S, u, v = 0, η)

    return parent_ocean
end

# The halo must cover the default WENO(order=7) tracer advection stencil.
nesting_test_child_grid() = LatitudeLongitudeGrid(size = (20, 20, 6), halo = (7, 7, 7),
                                                  longitude = (-1, 1), latitude = (29, 31), z = (-1000, 0),
                                                  topology = (Bounded, Bounded, Bounded))

@testset "PrescribedOcean: grid vertical size selects surface vs volumetric fields" begin
    surface_grid = LatitudeLongitudeGrid(size = (8, 8, 1), longitude = (-1, 1), latitude = (-1, 1),
                                         z = (-1, 0), topology = (Bounded, Bounded, Bounded))
    surface_ocean = PrescribedOcean(surface_grid)

    @test !is_three_dimensional(surface_grid)
    @test location(surface_ocean.temperature)  == (Center, Center, Nothing)
    @test location(surface_ocean.salinity)     == (Center, Center, Nothing)
    @test location(surface_ocean.velocities.u) == (Center, Center, Nothing)
    @test location(surface_ocean.free_surface) == (Center, Center, Nothing)

    volumetric_grid = LatitudeLongitudeGrid(size = (8, 8, 4), longitude = (-1, 1), latitude = (-1, 1),
                                            z = (-100, 0), topology = (Bounded, Bounded, Bounded))
    volumetric_ocean = PrescribedOcean(volumetric_grid)

    @test is_three_dimensional(volumetric_grid)
    @test location(volumetric_ocean.temperature)  == (Center, Center, Center)
    @test location(volumetric_ocean.velocities.v) == (Center, Center, Center)

    # The free surface stays two-dimensional either way.
    @test location(volumetric_ocean.free_surface) == (Center, Center, Nothing)

    # Salinity defaults to 35 in both forms.
    @test all(interior(volumetric_ocean.salinity) .== 35)

    @test keys(parent_ocean_variables(volumetric_ocean)) == (:u, :v, :T, :S, :η)

    # `ocean_temperature` is the full state; `ocean_surface_temperature` is its uppermost level, and both
    # forms carry the same `(Nx, Ny, 1, Nt)` surface shape so the coupling interface sees one layout.
    @test size(ocean_temperature(volumetric_ocean)) == (8, 8, 4, 1)
    @test size(ocean_surface_temperature(volumetric_ocean)) == (8, 8, 1, 1)
    @test size(ocean_surface_temperature(surface_ocean)) == (8, 8, 1, 1)
    @test size(first(ocean_surface_velocities(volumetric_ocean))) == (8, 8, 1, 1)
end

@testset "nested_ocean_model: boundary conditions land where the nest needs them" begin
    model = nested_ocean_model(nesting_test_parent(), nesting_test_child_grid())
    child = model.child

    free_surface = child.free_surface
    U_bcs = free_surface.barotropic_velocities.U.boundary_conditions
    V_bcs = free_surface.barotropic_velocities.V.boundary_conditions

    # Flather on the barotropic transport, on the sides each component is normal to.
    for side in (:west, :east)
        @test getproperty(U_bcs, side).classification isa NormalFlow
        @test getproperty(U_bcs, side).classification.scheme isa GravityWaveRadiation
    end

    for side in (:south, :north)
        @test getproperty(V_bcs, side).classification isa NormalFlow
        @test getproperty(V_bcs, side).classification.scheme isa GravityWaveRadiation
    end

    # Oceananigans pairs the Chapman companion on the free surface at exactly the Flather sides.
    η_bcs = free_surface.displacement.boundary_conditions
    for side in SIDES
        @test getproperty(η_bcs, side).classification isa Value
        @test getproperty(η_bcs, side).classification.scheme isa SurfaceWaveRadiation
    end

    # Tracers radiate as Value; momentum is NormalFlow on its own side and Value on the tangential ones.
    for side in SIDES
        for name in (:T, :S)
            bc = getproperty(child.tracers[name].boundary_conditions, side)
            @test bc.classification isa Value
            @test bc.classification.scheme isa NormalRadiation
        end

        u_bc = getproperty(child.velocities.u.boundary_conditions, side)
        v_bc = getproperty(child.velocities.v.boundary_conditions, side)
        @test u_bc.classification isa (side in (:west, :east) ? NormalFlow : Value)
        @test v_bc.classification isa (side in (:south, :north) ? NormalFlow : Value)
        @test u_bc.classification.scheme isa NormalRadiation
        @test v_bc.classification.scheme isa NormalRadiation
    end

    # The coupled interface reaches the child through the wrapper.
    @test NumericalEarth.underlying_ocean_model(model) === child
end

@testset "OceanStateExchanger: barotropic exterior integrates over the child's column" begin
    u₀ = 0.1
    depth = 1000
    model = nested_ocean_model(nesting_test_parent(; u = u₀), nesting_test_child_grid())
    update_state!(model)

    for side in SIDES
        slabs = getproperty(model.exchanger.boundaries, side)

        # Zonal sides see the full transport u₀ * H; meridional sides see none (the parent v is zero).
        expected = side in (:west, :east) ? u₀ * depth : 0
        @test all(isapprox.(Array(slabs.U), expected; atol = 1e-10))
        @test all(Array(slabs.η) .== 0)
    end

    # One slab entry per boundary-face cell, indexed as `condition[i, k, 1]` by `getbc`.
    Nx, Ny, _ = size(model.child.grid)
    @test size(model.exchanger.boundaries.west.U)  == (Ny, 1, 1)
    @test size(model.exchanger.boundaries.south.U) == (Nx, 1, 1)
end

@testset "nested_ocean_model: a quiescent parent leaves the child unchanged" begin
    T₀, S₀ = 15.0, 35.0
    model = nested_ocean_model(nesting_test_parent(; u = 0, T = T₀, S = S₀), nesting_test_child_grid())
    set!(model, T = T₀, S = S₀, u = 0, v = 0)
    update_state!(model)

    for n in 1:20
        time_step!(model, 60.0)
    end

    T = Array(interior(model.child.tracers.T))
    S = Array(interior(model.child.tracers.S))

    # Radiating boundaries on all four sides must not manufacture a gradient out of a uniform state.
    @test maximum(abs, T .- T₀) < 1e-10
    @test maximum(abs, S .- S₀) < 1e-10

    # `NestedModel` advances the parent to the child's time.
    @test model.parent.clock.time == model.clock.time == 20 * 60.0
end

@testset "nested_ocean_model: Flather holds the volume budget without a transport correction" begin
    u₀ = 0.1
    model = nested_ocean_model(nesting_test_parent(; u = u₀), nesting_test_child_grid(); coriolis = nothing)
    set!(model, T = 15, S = 35, u = u₀, v = 0)
    update_state!(model)

    grid = model.child.grid
    Nx, Ny, _ = size(grid)
    cell_area = [Azᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny]
    displacement = model.child.free_surface.displacement
    total_volume() = sum(Array(interior(displacement))[:, :, 1] .* cell_area)

    initial_volume = total_volume()
    for n in 1:200
        time_step!(model, 60.0)
    end

    # The Flather condition generates a compensating transport whenever the child's free surface drifts
    # from the parent's, so a steady throughflow leaves the domain volume alone to within a nanometer of
    # mean sea level over ~3 hours.
    mean_sea_level_drift = (total_volume() - initial_volume) / sum(cell_area)
    @test abs(mean_sea_level_drift) < 1e-6
end

@testset "bottom_height_from_mask: the seafloor a dataset marks with missing values" begin
    # z runs bottom-to-top: k = 1 is the deepest cell, whose bottom face sits at -100.
    grid = LatitudeLongitudeGrid(size = (3, 1, 4), longitude = (-1, 1), latitude = (-1, 1),
                                 z = (-100, 0), topology = (Bounded, Bounded, Bounded))

    mask = Field{Center, Center, Center}(grid, Bool)

    # Column 1 wet throughout, column 2 wet only in the top two cells, column 3 dry throughout.
    interior(mask)[1, 1, :] .= false
    interior(mask)[2, 1, :] .= (true, true, false, false)
    interior(mask)[3, 1, :] .= true

    bottom_height = Array(interior(bottom_height_from_mask(mask)))

    @test bottom_height[1, 1, 1] == -100   # full column
    @test bottom_height[2, 1, 1] == -50    # deepest wet cell is k = 3, whose bottom face is at -50
    @test bottom_height[3, 1, 1] == 0      # a dry column sits at sea level
end
