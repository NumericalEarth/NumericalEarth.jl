include("runtests_setup.jl")

using Oceananigans.Grids: Center, Face, λnodes, φnodes
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Units: Time
using NumericalEarth.Lands: RiverRouting, build_river_routing, coastal_outlet_indices
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

@testset "River routing conservation [$arch]" for arch in test_architectures
    Q₀ = 1234.0          # m³ s⁻¹
    ρ = 1000.0           # kg m⁻³

    discharge = synthetic_discharge_field(arch, Q₀)
    target_grid = half_land_ocean_grid(arch)

    outlet_i, outlet_j, outlet_λ, outlet_φ = coastal_outlet_indices(discharge)
    @test length(outlet_i) > 0

    outlet_weight = fill(ρ, length(outlet_i))
    routing = build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                                  maximum_search_radius = 5, spread_radius = 1.2)

    # Scalar metric/mask queries run on a CPU copy of the grid (GPU-safe).
    cpu_grid = on_architecture(CPU(), target_grid)
    kᴺ = size(cpu_grid, 3)

    discharge_cpu = Array(interior(discharge))[:, :, 1]
    flux, ti, tj = scattered_flux(routing, discharge_cpu, cpu_grid)

    # Every destination must be an active (wet) ocean cell.
    for c in eachindex(ti)
        @test !inactive_node(ti[c], tj[c], kᴺ, cpu_grid, Center(), Center(), Center())
    end

    # The deposited mass flux integrates to ρ × total discharge (volume conservation).
    total_discharge = sum(q for q in discharge_cpu if !isnan(q))
    @test integrated_mass_flux(flux, cpu_grid) ≈ ρ * total_discharge rtol = 1e-5
    @test total_discharge ≈ Q₀
end

@testset "River spreading footprint [$arch]" for arch in test_architectures
    Q₀ = 1234.0
    ρ = 1000.0

    discharge = synthetic_discharge_field(arch, Q₀)
    target_grid = half_land_ocean_grid(arch)
    cpu_grid = on_architecture(CPU(), target_grid)
    discharge_cpu = Array(interior(discharge))[:, :, 1]

    outlets = coastal_outlet_indices(discharge)
    outlet_weight = fill(ρ, length(outlets[1]))

    build(; kw...) = build_river_routing(target_grid, outlets..., outlet_weight;
                                         maximum_search_radius = 5, kw...)

    # A single landing cell when the footprint is one cell wide, many when it is a degree wide.
    concentrated = build(spread_radius = nothing, maximum_spread_cells = 1)
    spread       = build(spread_radius = 1.2)
    capped       = build(spread_radius = 1.2, maximum_spread_cells = 3)

    wet_cells_reached(routing) = count(>(0), scattered_flux(routing, discharge_cpu, cpu_grid)[1])

    @test wet_cells_reached(concentrated) == 1
    @test wet_cells_reached(spread) > wet_cells_reached(capped)
    @test wet_cells_reached(capped) == 3

    # Spreading only redistributes: all three conserve the same total mass.
    for routing in (concentrated, spread, capped)
        flux, _, _ = scattered_flux(routing, discharge_cpu, cpu_grid)
        @test integrated_mass_flux(flux, cpu_grid) ≈ ρ * Q₀ rtol = 1e-5
    end

    # The peak flux is diluted by the number of cells receiving the discharge.
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

    # Two-snapshot FieldTimeSeries holding the same discharge at both times.
    function constant_time_series(snapshot)
        times = [0.0, 86400.0]
        fts = FieldTimeSeries{Center, Center, Nothing}(snapshot.grid, times)
        parent(fts[1]) .= parent(snapshot)
        parent(fts[2]) .= parent(snapshot)
        return fts
    end

    outlets = coastal_outlet_indices(river_snapshot)
    outlet_weight = fill(ρ, length(outlets[1]))
    routing = build_river_routing(target_grid, outlets..., outlet_weight; maximum_search_radius = 5)

    freshwater_flux = (rivers = constant_time_series(river_snapshot),
                       icebergs = constant_time_series(iceberg_snapshot))

    land = PrescribedLand(freshwater_flux; river_routing = (rivers = routing, icebergs = routing))

    exchanger = (; state = (; freshwater_flux = Field{Center, Center, Nothing}(target_grid)))
    coupled_model = (; clock = Clock(time = 0.0))

    interpolate_state!(exchanger, target_grid, land, coupled_model)

    flux = Array(interior(exchanger.state.freshwater_flux))[:, :, 1]
    cpu_grid = on_architecture(CPU(), target_grid)

    # Both components accumulate into the same freshwater flux.
    @test integrated_mass_flux(flux, cpu_grid) ≈ ρ * (Q₀ + Q₁) rtol = 1e-5
end

@testset "River mouth vertical mixing [$arch]" for arch in test_architectures
    Q₀ = 1234.0
    ρ = 1000.0
    κʳ = 0.25
    mixing_depth = 25

    discharge = synthetic_discharge_field(arch, Q₀)

    underlying = LatitudeLongitudeGrid(arch;
                                       size = (20, 20, 4),
                                       longitude = (-10, 10),
                                       latitude = (-10, 10),
                                       z = (-100, 0),
                                       halo = (7, 7, 4))

    bottom_height(λ, φ) = ifelse(λ < 0, -100, 10)
    target_grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_height))

    outlets = coastal_outlet_indices(discharge)
    outlet_weight = fill(ρ, length(outlets[1]))
    routing = build_river_routing(target_grid, outlets..., outlet_weight; maximum_search_radius = 5)
    river_routing = (; rivers = routing)

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
    @test count(>(0), mask[:, :, first(shallow)]) == length(Array(routing.target_i))

    # `ocean_simulation` appends the extra diffusivity to the closure it is given.
    ocean = ocean_simulation(target_grid; closure = nothing, river_routing,
                             river_mouth_diffusivity = κʳ, river_mouth_mixing_depth = mixing_depth)

    @test ocean.model.closure isa VerticalScalarDiffusivity
    @test isnothing(ocean_simulation(target_grid; closure = nothing).model.closure)
end
