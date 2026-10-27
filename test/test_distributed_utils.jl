include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.DataWrangling.ORCA: ORCAOne
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: all_reduce, concatenate_local_sizes, local_size, reconstruct_global_grid
using Oceananigans.ImmersedBoundaries: bottom_height_field
using Oceananigans.Operators: Azᶜᶜᶜ
using NumericalEarth.Lands: build_flux_routing
using NumericalEarth.NestedModels: default_velocity_scheme, open_boundary_sides, relaxation_masks
using NumericalEarth.Oceans: river_mouth_vertical_diffusivity

@testset "Distributed ECCO download" begin
    dates = DateTimeProlepticGregorian(1992, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 4, 1)
    metadata = Metadata(:u_velocity; dataset=ECCO4Monthly(), dates)
    download(metadata)

    @root for metadatum in metadata
        @test isfile(metadata_path(metadatum))
    end
end

@testset "Distributed Bathymetry interpolation" begin
    metadata = Metadatum(:bottom_height; dataset = SyntheticBathymetry())

    global_grid = LatitudeLongitudeGrid(CPU();
                                        size = (40, 40, 1),
                                        longitude = (0, 100),
                                        latitude = (0, 20),
                                        z = (0, 1))

    interpolation_passes = 4
    global_height = regrid_bathymetry(global_grid, metadata; interpolation_passes, cache = false)

    arch_x  = Distributed(CPU(), partition=Partition(4, 1))
    arch_y  = Distributed(CPU(), partition=Partition(1, 4))
    arch_xy = Distributed(CPU(), partition=Partition(2, 2))

    for arch in (arch_x, arch_y, arch_xy)
        local_grid = LatitudeLongitudeGrid(arch;
                                           size = (40, 40, 1),
                                           longitude = (0, 100),
                                           latitude = (0, 20),
                                           z = (0, 1))

        local_height = regrid_bathymetry(local_grid, metadata; interpolation_passes, cache = false)

        Nx, Ny, _ = size(local_grid)
        rx, ry, _ = arch.local_index
        irange = (rx - 1) * Nx + 1 : rx * Nx
        jrange = (ry - 1) * Ny + 1 : ry * Ny

        begin
            @test interior(global_height, irange, jrange, 1) == interior(local_height, :, :, 1)
        end
    end
end

# The window of the serial grid held by this rank
function rank_window(arch, global_size, local_size_)
    sizes = local_size(arch, global_size)
    rx, ry, _ = arch.local_index
    istart = 1 + sum(concatenate_local_sizes(sizes, arch, 1)[1:rx-1])
    jstart = 1 + sum(concatenate_local_sizes(sizes, arch, 2)[1:ry-1])
    return istart:istart+local_size_[1]-1, jstart:jstart+local_size_[2]-1
end

# Each rank's metrics are its window of the serial grid
@testset "Distributed ORCAGrid" begin
    orca(arch) = ORCAGrid(arch; dataset = ORCAOne(), Nz = 5, z = (-5000, 0), with_bathymetry = false)

    global_grid = orca(CPU())

    for partition in (Partition(1, 4), Partition(2, 2))
        arch = Distributed(CPU(); partition)
        local_grid = orca(arch)
        irange, jrange = rank_window(arch, size(global_grid), size(local_grid))

        for metric in (:λᶜᶜᵃ, :φᶜᶜᵃ, :Δxᶜᶜᵃ, :Δyᶜᶜᵃ, :Azᶜᶜᵃ)
            local_metric  = getproperty(local_grid, metric)[1:length(irange), 1:length(jrange)]
            global_metric = getproperty(global_grid, metric)[irange, jrange]
            @test local_metric == global_metric
        end
    end
end

# Each rank's bottom height, basins removed, is its window of the serial one
@testset "Distributed ORCAGrid bathymetry" begin
    orca(arch) = ORCAGrid(arch; dataset = ORCAOne(), Nz = 5, z = (-5000, 0), major_basins = 1)

    global_grid = orca(CPU())

    for partition in (Partition(1, 4), Partition(2, 2))
        arch = Distributed(CPU(); partition)
        local_grid = orca(arch)
        irange, jrange = rank_window(arch, size(global_grid), size(local_grid))

        local_bottom  = local_grid.immersed_boundary.bottom_height[1:length(irange), 1:length(jrange), 1]
        global_bottom = global_grid.immersed_boundary.bottom_height[irange, jrange, 1]
        @test local_bottom == global_bottom
    end
end

# Every mouth is routed by exactly one rank, so the ranks together deposit the serial total
@testset "Distributed river routing" begin
    function coastal_grid(arch)
        underlying = LatitudeLongitudeGrid(arch; size = (20, 20, 1), longitude = (-10, 10), latitude = (-10, 10),
                                           z = (-10, 0), halo = (4, 4, 4))
        return ImmersedBoundaryGrid(underlying, GridFittedBottom((λ, φ) -> ifelse(λ < 0, -10, 10)))
    end

    # A JRA55-like per-area flux on the coastal land column of a finer source grid
    source_grid = LatitudeLongitudeGrid(CPU(); size = (40, 40), longitude = (-10, 10), latitude = (-10, 10),
                                        topology = (Bounded, Bounded, Flat))
    flux_time_series = FieldTimeSeries{Center, Center, Nothing}(source_grid, [0.0])
    coast_i = findfirst(>(0), Array(λnodes(source_grid, Center(), Center(), Center())))
    interior(flux_time_series[1])[coast_i, :, 1] .= 1:40
    source_flux = Array(interior(flux_time_series[1]))[:, :, 1]

    function deposited_mass_rate(routing, grid)
        ti, tj, offsets = Array(routing.target_i), Array(routing.target_j), Array(routing.offsets)
        oi, oj, weight = Array(routing.contribution_outlet_i), Array(routing.contribution_outlet_j), Array(routing.contribution_weight)
        return sum((weight[k] * source_flux[oi[k], oj[k]] * Azᶜᶜᶜ(ti[c], tj[c], 1, grid)
                    for c in eachindex(ti) for k in offsets[c]:offsets[c+1]-1); init = 0.0)
    end

    serial_grid = coastal_grid(CPU())
    serial_mass_rate = deposited_mass_rate(build_flux_routing(serial_grid, flux_time_series), serial_grid)
    @test serial_mass_rate ≈ sum(source_flux[coast_i, j] * Azᶜᶜᶜ(coast_i, j, 1, source_grid) for j in 1:40)

    for partition in (Partition(1, 4), Partition(2, 2))
        arch = Distributed(CPU(); partition)
        local_grid = coastal_grid(arch)
        routing = build_flux_routing(local_grid, flux_time_series)
        @test all_reduce(+, deposited_mass_rate(routing, local_grid), arch) ≈ serial_mass_rate

        river_mixing = river_mouth_vertical_diffusivity(local_grid, (; rivers = routing))
        @test count(>(0), interior(river_mixing.κ.parameters)) == length(routing.target_i)
    end
end

@testset "Distributed nested ocean model" begin
    function nested_parent()
        grid = LatitudeLongitudeGrid(CPU(); size = (12, 12, 6), longitude = (-2, 2), latitude = (28, 32),
                                     z = (-1000, 0), topology = (Bounded, Bounded, Bounded))
        parent_ocean = PrescribedOcean(grid, [0.0, 864000.0])
        set!(parent_ocean; T = (λ, φ, z, t) -> 15 + λ + (φ - 30) / 2 + z / 1000,
                           S = (λ, φ, z, t) -> 35 + φ / 100,
                           u = (λ, φ, z, t) -> 0.02φ - 0.5, v = (λ, φ, z, t) -> 0.05λ, η = 0)
        return parent_ocean
    end

    function nested_child(arch)
        underlying_grid = LatitudeLongitudeGrid(arch; size = (32, 32, 6), halo = (7, 7, 7), longitude = (-1, 1),
                                                latitude = (29, 31), z = (-1000, 0),
                                                topology = (Bounded, Bounded, Bounded))
        bottom = PartialCellBottom((λ, φ) -> -900 + 100 * sin(3λ) * cos(3φ); minimum_fractional_cell_height = 0.2)
        grid = ImmersedBoundaryGrid(underlying_grid, bottom; active_cells_map = true)
        parent_bottom_height = Field{Center, Center, Nothing}(underlying_grid)
        set!(parent_bottom_height, -700)
        # ObliqueRadiation's tangential stencil differs across rank seams over an immersed bottom in Oceananigans itself
        nested = nested_ocean_model(nested_parent(), grid; parent_bottom_height, relaxation_width = 4,
                                    bathymetry_blend_width = 3,
                                    normal_velocity_scheme = NormalRadiation(inflow_timescale = 3 * 86400, outflow_timescale = 360 * 86400),
                                    tangential_velocity_scheme = default_velocity_scheme(),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps = 6))
        for _ in 1:5
            time_step!(nested, 60)
        end
        return nested.child
    end

    serial = nested_child(CPU())
    serial_masks = relaxation_masks(serial.grid, open_boundary_sides(serial.grid), 4)

    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        arch = Distributed(CPU(); partition)
        child = nested_child(arch)
        grid = child.grid
        Nx, Ny, _ = size(grid)
        irange, jrange = rank_window(arch, size(serial.grid), size(grid))
        window(field, serial_field) = (Array(interior(serial_field))[irange, jrange, :],
                                       Array(interior(field))[1:Nx, 1:Ny, :])

        Rx, Ry, _ = arch.ranks
        rx, ry, _ = arch.local_index
        walls = (west = rx == 1, east = rx == Rx, south = ry == 1, north = ry == Ry)
        @test Set(open_boundary_sides(grid)) == Set(side for side in keys(walls) if walls[side])

        @test ==(window(bottom_height_field(grid), bottom_height_field(serial.grid))...)

        masks = relaxation_masks(grid, open_boundary_sides(grid), 4)
        for name in (:c, :u, :v)
            @test ==(window(masks[name], serial_masks[name])...)
        end

        fields = (u = child.velocities.u, v = child.velocities.v, T = child.tracers.T, S = child.tracers.S,
                  η = child.free_surface.displacement)
        serial_fields = (u = serial.velocities.u, v = serial.velocities.v, T = serial.tracers.T, S = serial.tracers.S,
                         η = serial.free_surface.displacement)
        for name in keys(fields)
            expected, local_values = window(fields[name], serial_fields[name])
            @test maximum(abs, local_values .- expected) < 1e-5
        end
    end
end

MPI.Finalize()
