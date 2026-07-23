include("runtests_setup.jl")

using Oceananigans.Grids: Center, Face, λnodes, φnodes
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Units: Time
using NumericalEarth.Lands: RiverRouting, build_river_routing, coastal_outlet_indices, positive_outlet_indices, source_cell_areas
using NumericalEarth.EarthSystemModels: interpolate_state!

# A target ocean grid whose western half (longitude < 0) is ocean and whose
# eastern half is land, so the coastline runs down longitude = 0.
function half_land_ocean_grid(arch)
    underlying = LatitudeLongitudeGrid(arch;
                                       size = (20, 20, 1),
                                       longitude = (-10, 10),
                                       latitude = (-10, 10),
                                       z = (-100, 0),
                                       halo = (4, 4, 4))

    bottom_height(λ, φ) = ifelse(λ < 0, -100, 10) # ocean west, land east
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_height))
end

# A native forcing grid (GloFAS-like) covering the same region, with ocean cells
# set to NaN. A single river mouth carries discharge `Q₀` just east of the coast.
function synthetic_discharge_field(arch, Q₀)
    grid = LatitudeLongitudeGrid(arch;
                                 size = (40, 40),
                                 longitude = (-10, 10),
                                 latitude = (-10, 10),
                                 topology = (Bounded, Bounded, Flat),
                                 halo = (3, 3))

    discharge = Field{Center, Center, Nothing}(grid)
    λc = Array(λnodes(grid, Center(), Center(), Center()))

    data = zeros(Float64, size(grid)...)             # finite over land
    for i in axes(data, 1), j in axes(data, 2)
        if λc[i] < 0
            data[i, j, 1] = NaN                       # ocean
        end
    end

    # One river mouth: the easternmost land column nearest the coast, mid-domain.
    coast_i = findfirst(>(0), λc)                      # first land column east of coast
    data[coast_i, 20, 1] = Q₀

    set!(discharge, data)
    return discharge
end

@testset "River routing conservation [$arch]" for arch in test_architectures
    Q₀ = 1234.0          # m³ s⁻¹
    ρ = 1000.0           # kg m⁻³

    discharge = synthetic_discharge_field(arch, Q₀)
    target_grid = half_land_ocean_grid(arch)

    outlet_i, outlet_j, outlet_λ, outlet_φ = coastal_outlet_indices(discharge)
    @test length(outlet_i) > 0

    # Volumetric discharge (m³ s⁻¹): the mass-rate scale is the freshwater density.
    outlet_weight = fill(ρ, length(outlet_i))
    routing = build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight; maximum_search_radius = 5)

    ti  = Array(routing.target_i)
    tj  = Array(routing.target_j)
    off = Array(routing.offsets)
    coi = Array(routing.contribution_outlet_i)
    coj = Array(routing.contribution_outlet_j)
    cw  = Array(routing.contribution_weight)

    # Scalar metric/mask queries run on a CPU copy of the grid (GPU-safe).
    cpu_grid = on_architecture(CPU(), target_grid)
    kᴺ = size(cpu_grid, 3)

    # Every destination must be an active (wet) ocean cell.
    for c in eachindex(ti)
        @test !inactive_node(ti[c], tj[c], kᴺ, cpu_grid, Center(), Center(), Center())
    end

    # Reconstruct the scattered freshwater mass flux and integrate it over the
    # ocean grid. It must equal ρ × total discharge (volume conservation).
    discharge_cpu = Array(interior(discharge))[:, :, 1]
    Nx, Ny, _ = size(cpu_grid)
    flux = zeros(Float64, Nx, Ny)
    for c in eachindex(ti)
        for k in off[c]:(off[c+1] - 1)
            Q = discharge_cpu[coi[k], coj[k]]
            isnan(Q) && continue
            flux[ti[c], tj[c]] += cw[k] * Q
        end
    end

    integrated_mass_flux = 0.0
    for i in 1:Nx, j in 1:Ny
        integrated_mass_flux += flux[i, j] * Azᶜᶜᶜ(i, j, kᴺ, cpu_grid)
    end

    total_discharge = sum(q for q in discharge_cpu if !isnan(q))
    @test integrated_mass_flux ≈ ρ * total_discharge rtol = 1e-5
    @test total_discharge ≈ Q₀
end

@testset "Routed PrescribedLand interpolate_state! [$arch]" for arch in test_architectures
    Q₀ = 555.0
    ρ = 1000.0

    snapshot = synthetic_discharge_field(arch, Q₀)
    target_grid = half_land_ocean_grid(arch)

    # A two-snapshot FieldTimeSeries holding the same discharge at both times.
    native_grid = snapshot.grid
    times = [0.0, 86400.0]
    discharge = FieldTimeSeries{Center, Center, Nothing}(native_grid, times)
    parent(discharge[1]) .= parent(snapshot)
    parent(discharge[2]) .= parent(snapshot)

    outlets = coastal_outlet_indices(snapshot)
    outlet_weight = fill(ρ, length(outlets[1]))
    routing = build_river_routing(target_grid, outlets..., outlet_weight;
                                  maximum_search_radius = 5)

    land = PrescribedLand((; rivers = discharge); river_routing = (; rivers = routing))

    exchanger = (; state = (; freshwater_flux = Field{Center, Center, Nothing}(target_grid)))
    coupled_model = (; clock = Clock(time = 0.0))

    interpolate_state!(exchanger, target_grid, land, coupled_model)

    flux = Array(interior(exchanger.state.freshwater_flux))[:, :, 1]
    cpu_grid = on_architecture(CPU(), target_grid)
    Nx, Ny, _ = size(cpu_grid)
    kᴺ = size(cpu_grid, 3)

    integrated_mass_flux = 0.0
    for i in 1:Nx, j in 1:Ny
        integrated_mass_flux += flux[i, j] * Azᶜᶜᶜ(i, j, kᴺ, cpu_grid)
    end

    @test integrated_mass_flux ≈ ρ * Q₀ rtol = 1e-5
end

# JRA55-do provides runoff as a per-area mass flux (kg m⁻² s⁻¹) at coastal cells with
# no NaN ocean mask: every positive cell is a mouth, weighted by its own source-cell area.
function synthetic_flux_field(arch, F₀)
    grid = LatitudeLongitudeGrid(arch;
                                 size = (40, 40),
                                 longitude = (-10, 10),
                                 latitude = (-10, 10),
                                 topology = (Bounded, Bounded, Flat),
                                 halo = (3, 3))

    flux = Field{Center, Center, Nothing}(grid)
    λc = Array(λnodes(grid, Center(), Center(), Center()))
    coast_i = findfirst(>(0), λc)                    # a cell just east of the coast
    data = zeros(Float64, size(grid)...)
    data[coast_i, 20, 1] = F₀
    set!(flux, data)
    return flux
end

@testset "River routing conservation, per-area flux (JRA55) [$arch]" for arch in test_architectures
    F₀ = 2.5e-4          # kg m⁻² s⁻¹
    flux_field = synthetic_flux_field(arch, F₀)
    target_grid = half_land_ocean_grid(arch)

    oi, oj, oλ, oφ = positive_outlet_indices(flux_field)
    @test length(oi) == 1

    A_source = source_cell_areas(flux_field.grid, oi, oj)
    routing = build_river_routing(target_grid, oi, oj, oλ, oφ, A_source; maximum_search_radius = 5)

    ti  = Array(routing.target_i)
    tj  = Array(routing.target_j)
    off = Array(routing.offsets)
    coi = Array(routing.contribution_outlet_i)
    coj = Array(routing.contribution_outlet_j)
    cw  = Array(routing.contribution_weight)

    cpu_grid = on_architecture(CPU(), target_grid)
    Nx, Ny, _ = size(cpu_grid)
    kᴺ = size(cpu_grid, 3)

    # Every destination must be an active ocean cell (runoff never lands on dry cells).
    for c in eachindex(ti)
        @test !inactive_node(ti[c], tj[c], kᴺ, cpu_grid, Center(), Center(), Center())
    end

    flux_source = Array(interior(flux_field))[:, :, 1]
    flux = zeros(Float64, Nx, Ny)
    for c in eachindex(ti), k in off[c]:(off[c+1] - 1)
        flux[ti[c], tj[c]] += cw[k] * flux_source[coi[k], coj[k]]
    end

    integrated_mass_flux = 0.0
    for i in 1:Nx, j in 1:Ny
        integrated_mass_flux += flux[i, j] * Azᶜᶜᶜ(i, j, kᴺ, cpu_grid)
    end

    # Delivered mass rate must equal the source flux integrated over its own cell.
    @test integrated_mass_flux ≈ F₀ * A_source[1] rtol = 1e-5
end

function tripolar_land_ocean_grid(arch)
    underlying = TripolarGrid(arch; size = (40, 40, 1), z = (-100, 0), halo = (4, 4, 4))
    return ImmersedBoundaryGrid(underlying, GridFittedBottom((λ, φ) -> ifelse(φ > 60, 10, -100)))
end

@testset "River routing conservation on a tripolar grid [$arch]" for arch in test_architectures
    F₀ = 3.0e-4
    flux_field = synthetic_flux_field(arch, F₀)
    target_grid = tripolar_land_ocean_grid(arch)

    oi, oj, oλ, oφ = positive_outlet_indices(flux_field)
    A_source = source_cell_areas(flux_field.grid, oi, oj)
    routing = build_river_routing(target_grid, oi, oj, oλ, oφ, A_source; maximum_search_radius = 5)

    ti  = Array(routing.target_i)
    tj  = Array(routing.target_j)
    off = Array(routing.offsets)
    coi = Array(routing.contribution_outlet_i)
    coj = Array(routing.contribution_outlet_j)
    cw  = Array(routing.contribution_weight)

    cpu_grid = on_architecture(CPU(), target_grid)
    Nx, Ny, _ = size(cpu_grid)
    kᴺ = size(cpu_grid, 3)

    @test length(ti) > 0
    for c in eachindex(ti)
        @test !inactive_node(ti[c], tj[c], kᴺ, cpu_grid, Center(), Center(), Center())
    end

    flux_source = Array(interior(flux_field))[:, :, 1]
    flux = zeros(Float64, Nx, Ny)
    for c in eachindex(ti), k in off[c]:(off[c+1] - 1)
        flux[ti[c], tj[c]] += cw[k] * flux_source[coi[k], coj[k]]
    end

    integrated_mass_flux = 0.0
    for i in 1:Nx, j in 1:Ny
        integrated_mass_flux += flux[i, j] * Azᶜᶜᶜ(i, j, kᴺ, cpu_grid)
    end

    @test integrated_mass_flux ≈ F₀ * A_source[1] rtol = 1e-4
end
