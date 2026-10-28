include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.DataWrangling.ORCA: ORCAOne
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: all_reduce, concatenate_local_sizes, local_size, reconstruct_global_grid, synchronize_communication!,
                                           cpu_architecture, reconstruct_global_field
using Oceananigans.Operators: Azᶜᶜᶜ
using NumericalEarth.Lands: build_flux_routing
using NumericalEarth.Oceans: river_mouth_vertical_diffusivity
using SeawaterPolynomials.TEOS10: θ_from_Θ
using NumericalEarth.SeaIces: sea_ice_dynamics
using ClimaSeaIce.SeaIceDynamics: SplitExplicitSolver
using Oceananigans.BoundaryConditions: regularize_boundary_condition, LeftBoundary
using Oceananigans.Grids: topology

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

# After a coupled step, the exchanger holds the ocean's surface potential temperature over the whole interface range,
# the rows of halo next to each rank edge included
@testset "Distributed exchanger ocean state" begin
    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        arch = Distributed(CPU(); partition)
        grid = LatitudeLongitudeGrid(arch; size = (40, 40, 4), longitude = (0, 20), latitude = (10, 30), z = (-100, 0),
                                     halo = (7, 7, 7))

        ocean = ocean_simulation(grid; closure = nothing, Δt = 60)
        set!(ocean.model, T = (λ, φ, z) -> 20 + 5 * cosd(20λ) * cosd(20φ), S = 35)

        atmosphere = PrescribedAtmosphere(grid, [0.0, 86400.0])
        for n in 1:2
            set!(atmosphere.velocities.u[n], 10)
            set!(atmosphere.temperature[n], 290)
        end

        model = OceanOnlyModel(ocean; atmosphere, radiation = PrescribedRadiation(grid))
        time_step!(model, 60)

        T, S = ocean.model.tracers.T, ocean.model.tracers.S
        synchronize_communication!(T)
        synchronize_communication!(S)

        Tᵉˣ = model.interfaces.exchanger.ocean.state.T
        Nx, Ny, Nz = size(grid)
        @test all(Tᵉˣ[i, j, 1] == θ_from_Θ(max(0, S[i, j, Nz]), T[i, j, Nz]) for i in 0:Nx+1, j in 0:Ny+1)
    end
end

function ocean_sea_ice_model(grid; dynamics = false)
    ocean = ocean_simulation(grid; closure = nothing, Δt = 60, free_surface = SplitExplicitFreeSurface(grid; substeps = 20))
    set!(ocean.model, T = (λ, φ, z) -> -2.5 + 0.1λ, S = 34, u = 0.1)

    ice_dynamics = dynamics ? sea_ice_dynamics(grid, ocean; solver = SplitExplicitSolver(grid; substeps = 8)) : nothing
    sea_ice = sea_ice_simulation(grid, ocean; dynamics = ice_dynamics)
    patch(λ, φ) = exp(-(λ - 10)^2 - (φ - 55)^2)
    set!(sea_ice.model, h = patch, ℵ = (λ, φ) -> min(1, 1.2patch(λ, φ)))

    atmosphere_grid = LatitudeLongitudeGrid(size = (24, 14), longitude = (-1, 23), latitude = (49, 63),
                                            topology = (Bounded, Bounded, Flat))
    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0, 86400.0])
    for n in 1:2
        set!(atmosphere.velocities.u[n], 10)
        set!(atmosphere.velocities.v[n], -5)
        set!(atmosphere.temperature[n], 253)
    end

    return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation = PrescribedRadiation(atmosphere_grid))
end

ocean_sea_ice_grid(arch) = LatitudeLongitudeGrid(arch; size = (80, 80, 4), longitude = (0, 20), latitude = (50, 60),
                                                 z = (-100, 0), halo = (7, 7, 7))

# The part of a serial field held by the rank that holds `field`
rank_window(serial, field) = Oceananigans.DistributedComputations.partition(interior(serial),
                                                                            cpu_architecture(field.grid.architecture),
                                                                            size(interior(field)))

# A supercooled ocean under an ice patch that crosses the rank edges
@testset "Distributed ocean-sea ice initial fluxes" begin
    serial = ocean_sea_ice_model(ocean_sea_ice_grid(CPU()))
    τˢ = serial.interfaces.net_fluxes.ocean

    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        model = ocean_sea_ice_model(ocean_sea_ice_grid(Distributed(CPU(); partition)))
        τ = model.interfaces.net_fluxes.ocean
        @test interior(τ.u) == rank_window(τˢ.u, τ.u)
        @test interior(τ.v) == rank_window(τˢ.v, τ.v)
    end
end

# The ice-ocean drag coefficients on the velocity faces off the walls, with elasto-visco-plastic sea-ice dynamics
@testset "Distributed ice-ocean drag" begin
    serial = ocean_sea_ice_model(ocean_sea_ice_grid(CPU()); dynamics = true)
    λˣˢ = interior(serial.interfaces.sea_ice_ocean_interface.fluxes.x_momentum_coefficient)
    λʸˢ = interior(serial.interfaces.sea_ice_ocean_interface.fluxes.y_momentum_coefficient)

    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        model = ocean_sea_ice_model(ocean_sea_ice_grid(Distributed(CPU(); partition)); dynamics = true)
        fluxes = model.interfaces.sea_ice_ocean_interface.fluxes
        λˣ = interior(reconstruct_global_field(fluxes.x_momentum_coefficient))
        λʸ = interior(reconstruct_global_field(fluxes.y_momentum_coefficient))
        @test λˣ[2:end-1, :, :] == λˣˢ[2:end-1, :, :]
        @test λʸ[:, 2:end-1, :] == λʸˢ[:, 2:end-1, :]
    end
end

# Every rank regularizes the west boundary condition, but only the westmost ranks keep it, so a source
# spanning only the child's west edge must not be checked against the other ranks' local edges.
@testset "Distributed interpolated boundary source bracket" begin
    loc = (Center(), Center(), Center())

    for (longitude, valid) in (((-101, -99), true), ((-99, -97), false))
        source_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 4), longitude, latitude = (29, 41), z = (0, 1),
                                            topology = (Bounded, Bounded, Bounded))
        source = NumericalEarth.NestedModels.Interpolated(CenterField(source_grid))

        for partition in (Partition(4, 1), Partition(2, 2))
            grid = LatitudeLongitudeGrid(Distributed(CPU(); partition); size = (20, 20, 4), longitude = (-100, -90),
                                         latitude = (30, 40), z = (0, 1), topology = (Bounded, Bounded, Bounded))
            west_rank = topology(grid, 1) ∈ (Bounded, RightConnected)

            if valid || !west_rank
                @test regularize_boundary_condition(source, grid, loc, 1, LeftBoundary) isa NumericalEarth.NestedModels.Interpolated
            else
                @test_throws ArgumentError regularize_boundary_condition(source, grid, loc, 1, LeftBoundary)
            end
        end
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

MPI.Finalize()
