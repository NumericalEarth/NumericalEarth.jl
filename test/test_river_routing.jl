include("runtests_setup.jl")

using Oceananigans.Grids: Center, Face, Flat, λnodes, φnodes
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Units: Time
using NumericalEarth.Lands: RiverRouting, build_river_routing, coastal_outlet_indices, routable_grid
using NumericalEarth.EarthSystemModels: interpolate_state!
using NumericalEarth.Oceans: river_mouth_vertical_diffusivity
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity

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

function integrated_mass_flux(flux, cpu_grid)
    Nx, Ny, _ = size(cpu_grid)
    kᴺ = size(cpu_grid, 3)
    total = 0.0
    for i in 1:Nx, j in 1:Ny
        total += flux[i, j] * Azᶜᶜᶜ(i, j, kᴺ, cpu_grid)
    end
    return total
end

function routing_for(discharge, target_grid, ρ; kw...)
    outlets = coastal_outlet_indices(discharge)
    outlet_weight = fill(ρ, length(outlets[1]))
    return build_river_routing(target_grid, outlets..., outlet_weight; maximum_search_radius = 5, kw...)
end

# Rebuild the flux a `RiverRouting` deposits, on the CPU.
function scattered_flux(routing, discharge, cpu_grid)
    ti  = Array(routing.target_i)
    tj  = Array(routing.target_j)
    off = Array(routing.offsets)
    coi = Array(routing.contribution_outlet_i)
    coj = Array(routing.contribution_outlet_j)
    cw  = Array(routing.contribution_weight)

    Nx, Ny, _ = size(cpu_grid)
    flux = zeros(Float64, Nx, Ny)
    for c in eachindex(ti), k in off[c]:(off[c+1] - 1)
        Q = discharge[coi[k], coj[k]]
        isnan(Q) && continue
        flux[ti[c], tj[c]] += cw[k] * Q
    end

    return flux, ti, tj
end

@testset "River routing conservation and spreading [$arch]" for arch in test_architectures
    Q₀ = 1234.0          # m³ s⁻¹
    ρ = 1000.0           # kg m⁻³

    discharge = synthetic_discharge_field(arch, Q₀)
    target_grid = half_land_ocean_grid(arch)
    cpu_grid = on_architecture(CPU(), target_grid)   # scalar metric/mask queries are GPU-unsafe
    kᴺ = size(cpu_grid, 3)
    discharge_cpu = Array(interior(discharge))[:, :, 1]

    concentrated = routing_for(discharge, target_grid, ρ; spread_radius = nothing, maximum_spread_cells = 1)
    spread       = routing_for(discharge, target_grid, ρ; spread_radius = 1.2)
    capped       = routing_for(discharge, target_grid, ρ; spread_radius = 1.2, maximum_spread_cells = 3)

    flux, ti, tj = scattered_flux(spread, discharge_cpu, cpu_grid)
    @test length(ti) > 0
    for c in eachindex(ti)
        @test !inactive_node(ti[c], tj[c], kᴺ, cpu_grid, Center(), Center(), Center())
    end

    total_discharge = sum(q for q in discharge_cpu if !isnan(q))
    @test total_discharge ≈ Q₀

    # Spreading only redistributes: every footprint deposits ρ × total discharge.
    for routing in (concentrated, spread, capped)
        deposited, _, _ = scattered_flux(routing, discharge_cpu, cpu_grid)
        @test integrated_mass_flux(deposited, cpu_grid) ≈ ρ * total_discharge rtol = 1e-5
    end

    wet_cells_reached(routing) = count(>(0), scattered_flux(routing, discharge_cpu, cpu_grid)[1])
    @test wet_cells_reached(concentrated) == 1
    @test wet_cells_reached(capped) == 3
    @test wet_cells_reached(spread) > wet_cells_reached(capped)

    peak(routing) = maximum(scattered_flux(routing, discharge_cpu, cpu_grid)[1])
    @test peak(spread) < peak(capped) < peak(concentrated)
end

@testset "Routed PrescribedLand interpolate_state! [$arch]" for arch in test_architectures
    Q₀ = 555.0
    Q₁ = 111.0
    ρ = 1000.0

    river_snapshot = synthetic_discharge_field(arch, Q₀)
    iceberg_snapshot = synthetic_discharge_field(arch, Q₁)
    target_grid = half_land_ocean_grid(arch)

    function constant_time_series(snapshot)
        fts = FieldTimeSeries{Center, Center, Nothing}(snapshot.grid, [0.0, 86400.0])
        parent(fts[1]) .= parent(snapshot)
        parent(fts[2]) .= parent(snapshot)
        return fts
    end

    routing = routing_for(river_snapshot, target_grid, ρ)
    freshwater_flux = (rivers = constant_time_series(river_snapshot),
                       icebergs = constant_time_series(iceberg_snapshot))
    land = PrescribedLand(freshwater_flux; river_routing = (rivers = routing, icebergs = routing))

    exchanger = (; state = (; freshwater_flux = Field{Center, Center, Nothing}(target_grid)))
    interpolate_state!(exchanger, target_grid, land, (; clock = Clock(time = 0.0)))

    flux = Array(interior(exchanger.state.freshwater_flux))[:, :, 1]

    # Both components accumulate into the same freshwater flux.
    @test integrated_mass_flux(flux, on_architecture(CPU(), target_grid)) ≈ ρ * (Q₀ + Q₁) rtol = 1e-5
end

@testset "River mouth vertical mixing [$arch]" for arch in test_architectures
    ρ = 1000.0
    κʳ = 0.25
    mixing_depth = 25

    discharge = synthetic_discharge_field(arch, 1234.0)

    underlying = LatitudeLongitudeGrid(arch;
                                       size = (20, 20, 4),
                                       longitude = (-10, 10),
                                       latitude = (-10, 10),
                                       z = (-100, 0),
                                       halo = (7, 7, 4))

    bottom_height(λ, φ) = ifelse(λ < 0, -100, 10)
    target_grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_height))
    river_routing = (; rivers = routing_for(discharge, target_grid, ρ))

    river_mixing = river_mouth_vertical_diffusivity(target_grid, river_routing;
                                                    river_mouth_diffusivity = κʳ,
                                                    river_mouth_mixing_depth = mixing_depth)

    @test river_mixing isa VerticalScalarDiffusivity

    mask = Array(interior(river_mixing.κ.parameters))
    zᶜ = Array(znodes(target_grid, Center()))
    shallow = findall(z -> z > -mixing_depth, zᶜ)
    deep = findall(z -> z <= -mixing_depth, zᶜ)

    # The extra diffusivity is confined to the mixing depth and to the cells receiving discharge.
    @test all(iszero, mask[:, :, deep])
    @test maximum(mask) == κʳ
    @test count(>(0), mask[:, :, first(shallow)]) == length(Array(river_routing.rivers.target_i))

    ocean = ocean_simulation(target_grid; closure = nothing, river_routing,
                             river_mouth_diffusivity = κʳ, river_mouth_mixing_depth = mixing_depth)

    @test ocean.model.closure isa VerticalScalarDiffusivity
    @test isnothing(ocean_simulation(target_grid; closure = nothing).model.closure)
end

@testset "Single-column grids carry no routing" begin
    column = LatitudeLongitudeGrid(CPU(); size = 1, latitude = 10, longitude = 10,
                                   z = (-1, 0), topology = (Flat, Flat, Bounded))

    @test !routable_grid(column)
    @test routable_grid(half_land_ocean_grid(CPU()))
end
