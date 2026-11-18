include("runtests_setup.jl")

using NumericalEarth.Bathymetry: bottom_height_from_mask
using NumericalEarth.DataWrangling: bracketing_dates
using NumericalEarth.NestedModels: open_boundary_sides, parent_ocean_sources, relaxation_masks, sample_parent!,
                                   BoundaryTimeSeries, BarotropicBoundaryTransport, ConsistentBoundaryVelocity,
                                   OceanBoundaryExchanger, ScaledParentSeries
using NumericalEarth.Oceans: fill_prescribed_halo_regions!, forcing_barotropic_potential
using Oceananigans.Grids: TwoSided, halo_size, topology
using NumericalEarth.EarthSystemModels: BoundaryPrescribedComponent, ocean_surface_temperature,
                                        ocean_temperature, prognostic_model
using Oceananigans.BoundaryConditions: fill_halo_regions!, getbc, GravityWaveRadiation, NormalFlow, ObliqueRadiation,
                                       SurfaceWaveRadiation, TracerReservoir, Value
using Oceananigans.Grids: MutableVerticalDiscretization, inactive_cell
using Oceananigans.ImmersedBoundaries: bottom_height_field, has_active_cells_map
using Oceananigans.Operators: Azᶜᶜᶜ, Δzᶠᶜᶜ
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.OutputReaders: InMemory, OnDisk
using Oceananigans.Units: days, hours, minutes

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
