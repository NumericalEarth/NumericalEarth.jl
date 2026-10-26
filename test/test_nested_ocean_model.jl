include("runtests_setup.jl")

using NumericalEarth.Bathymetry: bottom_height_from_mask
using NumericalEarth.DataWrangling: bracketing_dates
using NumericalEarth.NestedModels: open_boundary_sides, parent_ocean_sources, relaxation_masks, BoundaryTimeSeries,
                                   BarotropicBoundaryTransport, ConsistentBoundaryVelocity, OceanBoundaryExchanger
using NumericalEarth.Oceans: fill_prescribed_halo_regions!
using NumericalEarth.EarthSystemModels: BoundaryPrescribedComponent, ocean_surface_temperature,
                                        ocean_temperature, prognostic_model
using Oceananigans.BoundaryConditions: getbc, GravityWaveRadiation, NormalFlow, ObliqueRadiation,
                                       SurfaceWaveRadiation, TracerReservoir, Value
using Oceananigans.Grids: inactive_cell
using Oceananigans.ImmersedBoundaries: bottom_height_field
using Oceananigans.Operators: Azᶜᶜᶜ, Δzᶠᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: days

# The synthetic ocean is 10° resolution, with land over λ ∈ [-60, 20], φ ∈ [-30, 40]. Its velocities
# (u = 0.1 m/s, v = -0.05 m/s) and free surface (η = 0.2 m) are uniform.
synthetic_dates = (DateTime(2000, 1, 1), DateTime(2000, 3, 1))

function synthetic_regional_grid(arch; longitude, latitude, size = (16, 12, 6))
    grid = LatitudeLongitudeGrid(arch; size, longitude, latitude, z = (-4000, 0), halo = (7, 7, 7))
    return ImmersedBoundaryGrid(grid, GridFittedBottom(synthetic_bottom_height(grid)))
end

# The western edge of this box, λ = 10°, runs over land from 30°S to 40°N.
coastal_grid(arch; size = (16, 12, 6)) = synthetic_regional_grid(arch; longitude = (10, 50), latitude = (-25, 35), size)

wet_maximum(field) = maximum(abs, filter(!iszero, Array(interior(field))))

@testset "Open boundary sides follow the coastline" begin
    for arch in test_architectures
        open_ocean = synthetic_regional_grid(arch; longitude = (100, 140), latitude = (-40, 0))
        @test open_boundary_sides(open_ocean) == (:west, :east, :south, :north)
        @test open_boundary_sides(coastal_grid(arch)) == (:east, :south, :north)

        periodic = LatitudeLongitudeGrid(arch; size = (16, 12, 6), longitude = (0, 360), latitude = (-40, 0),
                                         z = (-4000, 0), topology = (Periodic, Bounded, Bounded))
        @test open_boundary_sides(periodic) == (:south, :north)
    end
end

@testset "Volumetric PrescribedOcean from a dataset on $(arch)" for arch in test_architectures
    region = BoundingBox(longitude = (0, 60), latitude = (-35, 45))
    ocean = PrescribedOcean(region, synthetic_dates, SyntheticOcean(); architecture = arch)
    @test ocean isa PrescribedOcean
    @test size(ocean.temperature, 3) == 4
    @test location(ocean.free_surface) == (Center, Center, Nothing)
    @test length(ocean.times) == 3

    time_step!(ocean, 86400)
    @test ocean.clock.time == 86400
end

function check_nested_ocean(nested, grid)
    model = nested.child
    u, v, w = model.velocities
    T, S = model.tracers
    Nx, Ny, Nz = size(grid)
    j = Ny ÷ 2

    # Five conditions on each open side, none on the closed (land) west side
    @test u.boundary_conditions.east.classification isa NormalFlow
    @test u.boundary_conditions.east.condition isa Union{BoundaryTimeSeries, ConsistentBoundaryVelocity}
    @test u.boundary_conditions.south.condition isa BoundaryTimeSeries    # tangential
    @test v.boundary_conditions.north.classification isa NormalFlow
    @test v.boundary_conditions.east.condition isa BoundaryTimeSeries     # tangential
    @test T.boundary_conditions.south.condition isa BoundaryTimeSeries
    @test !(T.boundary_conditions.west.condition isa BoundaryTimeSeries)
    @test !(u.boundary_conditions.west.condition isa BoundaryTimeSeries)

    clock = model.clock
    @allowscalar begin
        @test getbc(u.boundary_conditions.east.condition, j, Nz, grid, clock) ≈ 0.1 rtol = 1e-6
        @test getbc(v.boundary_conditions.north.condition, Nx ÷ 2, Nz, grid, clock) ≈ -0.05 rtol = 1e-6

        # Flather: the parent transport through the child's wet column, and the parent's η
        cpu_grid = on_architecture(CPU(), grid)
        U_bcs = model.free_surface.barotropic_velocities.U.boundary_conditions
        exterior = on_architecture(CPU(), U_bcs.east.condition)
        @test first(exterior) isa BarotropicBoundaryTransport
        Uᵉˣᵗ, ηᵉˣᵗ = getbc(exterior, j, 1, cpu_grid, clock)
        wet_depth = sum(Δzᶠᶜᶜ(Nx + 1, j, k, cpu_grid) for k in 1:Nz if !inactive_cell(Nx, j, k, cpu_grid))
        @test Uᵉˣᵗ ≈ 0.1 * wet_depth rtol = 1e-6
        @test ηᵉˣᵗ ≈ 0.2 rtol = 1e-6

        @test S[Nx, j, Nz] > 33   # a wet cell; immersed cells hold zeros
    end

    return nothing
end

@testset "nested_ocean_model in a synthetic dataset on $(arch)" for arch in test_architectures
    grid = coastal_grid(arch)
    nested = nested_ocean_model(grid, SyntheticOcean(); dates = synthetic_dates, parent_padding = 10, relaxation_width = 3)

    @test nested isa NestedModel
    @test nested.parent isa BoundaryPrescribedComponent
    @test keys(nested.exchanger.bands) == (:east, :south, :north)
    check_nested_ocean(nested, grid)

    # The interior starts from the parent's full-domain snapshot
    @test wet_maximum(nested.child.velocities.u) ≈ 0.1 rtol = 1e-6
    @test wet_maximum(nested.child.free_surface.displacement) ≈ 0.2 rtol = 1e-6

    # The parent steps first, and the bands then bracket the child's step
    time_step!(nested, 60)
    @test nested.parent.clock.time == nested.child.clock.time == 60
    @test (nested.exchanger.times.previous, nested.exchanger.times.current) == (0, 60)
    time_step!(nested, 60)
    @test !any(isnan, interior(nested.child.velocities.u))
    @test !any(isnan, interior(nested.child.tracers.T))
end

@testset "nested_ocean_model in a full-domain PrescribedOcean on $(arch)" for arch in test_architectures
    child_grid = coastal_grid(arch)
    parent_region = BoundingBox(longitude = (0, 60), latitude = (-35, 45))

    # A full-domain prescribed parent also initializes the child
    prescribed = PrescribedOcean(parent_region, synthetic_dates, SyntheticOcean(); architecture = arch)
    nested = nested_ocean_model(prescribed, child_grid; relaxation_width = 3)
    @test nested.parent === prescribed
    check_nested_ocean(nested, child_grid)
    @test wet_maximum(nested.child.velocities.u) ≈ 0.1 rtol = 1e-6

    time_step!(nested, 60)
    time_step!(nested, 60)
    @test prescribed.clock.time == nested.child.clock.time == 120
    @test nested.exchanger.times.current == 120
    @test !any(isnan, interior(nested.child.velocities.u))
end

@testset "A nested ocean couples like an ocean on $(arch)" for arch in test_architectures
    grid = coastal_grid(arch)
    nested = nested_ocean_model(grid, SyntheticOcean(); dates = synthetic_dates, parent_padding = 10, relaxation_width = 3)
    ocean = Simulation(nested; Δt = 60)

    atmosphere = synthetic_prescribed_atmosphere(arch)
    radiation = synthetic_prescribed_radiation(arch)
    coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

    for _ in 1:2
        time_step!(coupled_model, 60)
    end

    @test nested.child.clock.time == nested.parent.clock.time == 120
    @test !any(isnan, interior(nested.child.tracers.T))
end

#####
##### A resting or uniformly flowing prescribed parent around an open-ocean child
#####

const lateral_sides = (:west, :east, :south, :north)

function nesting_test_parent(arch; u = 0.1, T = 15, S = 35, η = 0)
    grid = LatitudeLongitudeGrid(arch; size = (12, 12, 6),
                                 longitude = (-2, 2), latitude = (28, 32), z = (-1000, 0),
                                 topology = (Bounded, Bounded, Bounded))

    parent_ocean = PrescribedOcean(grid, [0.0, 10days])
    set!(parent_ocean; T, S, u, v = 0, η)

    return parent_ocean
end

# The halo must cover the default WENO(order=7) tracer advection stencil.
nesting_test_child_grid(arch) = LatitudeLongitudeGrid(arch; size = (20, 20, 6), halo = (7, 7, 7),
                                                      longitude = (-1, 1), latitude = (29, 31), z = (-1000, 0),
                                                      topology = (Bounded, Bounded, Bounded))

@testset "PrescribedOcean: grid vertical size selects surface vs volumetric fields" begin
    surface_grid = LatitudeLongitudeGrid(size = (8, 8, 1), longitude = (-1, 1), latitude = (-1, 1),
                                         z = (-1, 0), topology = (Bounded, Bounded, Bounded))
    surface_ocean = PrescribedOcean(surface_grid)

    @test location(surface_ocean.temperature)  == (Center, Center, Nothing)
    @test location(surface_ocean.salinity)     == (Center, Center, Nothing)
    @test location(surface_ocean.velocities.u) == (Center, Center, Nothing)
    @test isnothing(surface_ocean.free_surface)

    volumetric_grid = LatitudeLongitudeGrid(size = (8, 8, 4), longitude = (-1, 1), latitude = (-1, 1),
                                            z = (-100, 0), topology = (Bounded, Bounded, Bounded))
    volumetric_ocean = PrescribedOcean(volumetric_grid)

    @test location(volumetric_ocean.temperature)  == (Center, Center, Center)
    @test location(volumetric_ocean.velocities.v) == (Center, Center, Center)
    @test location(volumetric_ocean.free_surface) == (Center, Center, Nothing)
    @test all(interior(volumetric_ocean.salinity) .== 35)
    @test keys(parent_ocean_sources(volumetric_ocean)) == (:u, :v, :η, :T, :S)

    @test size(ocean_temperature(volumetric_ocean)) == (8, 8, 4, 1)
    @test size(ocean_surface_temperature(surface_ocean)) == (8, 8, 1, 1)
end

@testset "nested_ocean_model: boundary conditions land where the nest needs them on $(arch)" for arch in test_architectures
    model = nested_ocean_model(nesting_test_parent(arch), nesting_test_child_grid(arch))
    child = model.child

    free_surface = child.free_surface
    U_bcs = free_surface.barotropic_velocities.U.boundary_conditions
    V_bcs = free_surface.barotropic_velocities.V.boundary_conditions

    # Flather on the barotropic transport, on the sides each component is normal to
    for side in (:west, :east)
        @test getproperty(U_bcs, side).classification isa NormalFlow
        @test getproperty(U_bcs, side).classification.scheme isa GravityWaveRadiation
    end

    for side in (:south, :north)
        @test getproperty(V_bcs, side).classification isa NormalFlow
        @test getproperty(V_bcs, side).classification.scheme isa GravityWaveRadiation
    end

    # Chapman on the free surface at the Flather sides
    η_bcs = free_surface.displacement.boundary_conditions
    for side in lateral_sides
        @test getproperty(η_bcs, side).classification isa Value
        @test getproperty(η_bcs, side).classification.scheme isa SurfaceWaveRadiation
    end

    # Tracers are reservoirs (Value); momentum radiates obliquely, NormalFlow on its own side and Value on the
    # tangential ones
    for side in lateral_sides
        for name in (:T, :S)
            bc = getproperty(child.tracers[name].boundary_conditions, side)
            @test bc.classification isa Value
            @test bc.classification.scheme isa TracerReservoir
        end

        u_bc = getproperty(child.velocities.u.boundary_conditions, side)
        v_bc = getproperty(child.velocities.v.boundary_conditions, side)
        @test u_bc.classification isa (side in (:west, :east) ? NormalFlow : Value)
        @test v_bc.classification isa (side in (:south, :north) ? NormalFlow : Value)
        @test u_bc.classification.scheme isa ObliqueRadiation
        @test v_bc.classification.scheme isa ObliqueRadiation
    end

    # The coupled interface reaches the child through the wrapper
    @test prognostic_model(model) === child
end

@testset "nested_ocean_model: a quiescent parent leaves the child unchanged on $(arch)" for arch in test_architectures
    T₀, S₀ = 15.0, 35.0
    model = nested_ocean_model(nesting_test_parent(arch; u = 0, T = T₀, S = S₀), nesting_test_child_grid(arch))
    set!(model, T = T₀, S = S₀, u = 0, v = 0)
    update_state!(model)

    for n in 1:20
        time_step!(model, 60.0)
    end

    T = Array(interior(model.child.tracers.T))
    S = Array(interior(model.child.tracers.S))

    # Radiating boundaries on all four sides must not manufacture a gradient out of a uniform state
    @test maximum(abs, T .- T₀) < 1e-10
    @test maximum(abs, S .- S₀) < 1e-10
    @test model.parent.clock.time == model.clock.time == 20 * 60.0
end

@testset "nested_ocean_model: Flather holds the volume budget on $(arch)" for arch in test_architectures
    u₀ = 0.1
    model = nested_ocean_model(nesting_test_parent(arch; u = u₀), nesting_test_child_grid(arch); coriolis = nothing)
    set!(model, T = 15, S = 35, u = u₀, v = 0)
    update_state!(model)

    grid = on_architecture(CPU(), model.child.grid)
    Nx, Ny, _ = size(grid)
    cell_area = [Azᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny]
    displacement = model.child.free_surface.displacement
    total_volume() = sum(Array(interior(displacement))[:, :, 1] .* cell_area)

    initial_volume = total_volume()
    for n in 1:200
        time_step!(model, 60.0)
    end

    # A steady throughflow leaves the mean sea level alone to within a micrometer over ~3 hours
    mean_sea_level_drift = (total_volume() - initial_volume) / sum(cell_area)
    @test abs(mean_sea_level_drift) < 1e-6
end

@testset "Flather takes the parent's own depth mean, and the normal velocity integrates to it, on $(arch)" for arch in test_architectures
    # A sheared parent current, u = 0.1 + 0.01k on its six levels, over a 500 m seafloor
    parent_ocean = nesting_test_parent(arch)
    for k in 1:6
        interior(parent_ocean.velocities.u)[:, :, k, :] .= 0.1 + 0.01k
    end
    fill_prescribed_halo_regions!(parent_ocean)

    parent_bottom_height = Field{Center, Center, Nothing}(parent_ocean.grid)
    set!(parent_bottom_height, -500)

    grid = nesting_test_child_grid(arch)
    model = nested_ocean_model(parent_ocean, grid; parent_bottom_height)
    child = model.child
    clock = child.clock
    Nz = size(grid, 3)
    Δz = 1000 / Nz

    @allowscalar begin
        u_condition = child.velocities.u.boundary_conditions.west.condition
        @test u_condition isa ConsistentBoundaryVelocity
        U_condition = child.free_surface.barotropic_velocities.U.boundary_conditions.west.condition

        # The parent's levels with centers above -500 m are k = 4, 5, 6: ū = 0.15, over the child's 1000 m column
        Uᵉˣᵗ, _ = getbc(on_architecture(CPU(), U_condition), 10, 1, on_architecture(CPU(), grid), clock)
        @test Uᵉˣᵗ ≈ 0.15 * 1000

        transport = sum(getbc(on_architecture(CPU(), u_condition), 10, k, on_architecture(CPU(), grid), clock) * Δz for k in 1:Nz)
        @test transport ≈ Uᵉˣᵗ
    end
end

@testset "nested_ocean_model blends the child bathymetry toward the parent's seafloor on $(arch)" for arch in test_architectures
    parent_ocean = nesting_test_parent(arch)

    # A flat 1000 m seafloor, with land in the southwest corner
    underlying_grid = nesting_test_child_grid(arch)
    coastal_bottom(λ, φ) = (λ < -0.8) & (φ < 29.5) ? 0 : -1000
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(coastal_bottom))

    parent_bottom_height = Field{Center, Center, Nothing}(parent_ocean.grid)
    set!(parent_bottom_height, -500)

    model = nested_ocean_model(parent_ocean, grid; parent_bottom_height, open_sides = (:west, :east, :north))
    @test model.child.grid === grid

    # The parent's seafloor in the boundary cells of the open sides, the child's beyond the blend, on the
    # closed south side, and on land
    bottom_height = Array(interior(bottom_height_field(grid)))
    @test bottom_height[1, 10, 1] ≈ -500
    @test bottom_height[20, 10, 1] ≈ -500
    @test bottom_height[10, 20, 1] ≈ -500
    @test bottom_height[10, 10, 1] ≈ -1000
    @test bottom_height[10, 1, 1] ≈ -1000
    @test bottom_height[1, 1, 1] == 0

    # The child steps from the parent's time
    @test_throws ArgumentError nested_ocean_model(parent_ocean, underlying_grid; clock = Clock(time = 60.0))
end

@testset "Boundary relaxation: cos² masks, and velocities relax only their baroclinic part" begin
    grid = nesting_test_child_grid(CPU())
    width = 4
    masks = relaxation_masks(grid, lateral_sides, width)
    μ = Array(interior(masks.c))[:, :, 1]
    m = cos(π / 4width)^2   # half a cell from a boundary face

    @test μ[1, 10] ≈ m
    @test μ[1, 1] ≈ 1 - (1 - m)^2      # two sides combine near a corner
    @test μ[10, 10] == 0

    # A band stops at the first land cell in from its side
    coastal_bottom(λ, φ) = (λ < -0.8) & (φ < 29.5) ? 0 : -1000
    coastal = ImmersedBoundaryGrid(grid, GridFittedBottom(coastal_bottom))
    μ = Array(interior(relaxation_masks(coastal, (:west,), width).c))[:, :, 1]
    @test μ[1, 3] == 0 && μ[3, 3] == 0   # behind the land in rows j ≤ 5
    @test μ[1, 10] ≈ m

    # A sheared child current under a uniform parent one: the relaxation removes the shear, not the transport
    model = nested_ocean_model(nesting_test_parent(CPU(); u = 0.1), grid; coriolis = nothing)
    set!(model, u = (λ, φ, z) -> 0.1 + 1e-4 * (z + 500), v = 0, T = 15, S = 35)
    child = model.child
    forcing = child.forcing.u
    model_fields = merge(child.velocities, child.tracers)
    column = [forcing(2, 10, k, grid, child.clock, model_fields) for k in 1:size(grid, 3)]
    @test maximum(abs, column) > 0
    @test abs(sum(column)) < 1e-12 * maximum(abs, column)   # uniform levels: the depth integral is the sum
end

@testset "regrid_bathymetry from the missing values of an ocean dataset on $(arch)" for arch in test_architectures
    grid = coastal_grid(arch).underlying_grid
    metadatum = Metadatum(:temperature; dataset = SyntheticOcean(), date = first(synthetic_dates),
                          region = BoundingBox(grid; padding = 10))

    bottom_height = Array(interior(regrid_bathymetry(grid, metadatum; major_basins = Inf, cache = false)))

    # The western columns lie on the synthetic continent (λ ≤ 20°); the eastern ones are open ocean
    @test all(bottom_height[1, :, 1] .== 0)
    @test all(bottom_height[end, :, 1] .< -1000)
end

@testset "A daily-mean parent brackets the run with its samples" begin
    # GLORYSDaily samples sit at noon, so a run starting at midnight needs the day before
    dates = (DateTime(2018, 1, 29), DateTime(2018, 2, 5))
    @test bracketing_dates(GLORYSDaily(), :temperature, dates) == DateTime(2018, 1, 28):Day(1):DateTime(2018, 2, 5)
    @test bracketing_dates(GLORYSDaily(), :temperature, [first(dates)]) == [DateTime(2018, 1, 28), DateTime(2018, 1, 29)]
end

@testset "bottom_height_from_mask: the seafloor a dataset marks with missing values" begin
    # z runs bottom-to-top: k = 1 is the deepest cell, whose bottom face sits at -100
    grid = LatitudeLongitudeGrid(size = (3, 1, 4), longitude = (-1, 1), latitude = (-1, 1),
                                 z = (-100, 0), topology = (Bounded, Bounded, Bounded))

    mask = Field{Center, Center, Center}(grid, Bool)

    # Column 1 wet throughout, column 2 wet only in the top two cells, column 3 dry throughout
    interior(mask)[1, 1, :] .= false
    interior(mask)[2, 1, :] .= (true, true, false, false)
    interior(mask)[3, 1, :] .= true

    bottom_height = Array(interior(bottom_height_from_mask(mask)))

    @test bottom_height[1, 1, 1] == -100
    @test bottom_height[2, 1, 1] == -50
    @test bottom_height[3, 1, 1] == 0
end
