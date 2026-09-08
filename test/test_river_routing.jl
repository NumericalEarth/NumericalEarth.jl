include("runtests_setup.jl")

using Oceananigans.Grids: Center, Face, λnodes, φnodes
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Units: Time
using NumericalEarth.Lands: RiverRouting, build_river_routing, coastal_outlet_indices
using NumericalEarth.EarthSystemModels: interpolate_state!
using NumericalEarth.Lands: ever_positive_mask, outlet_indices_from_mask, source_cell_areas,
                            river_mouth_vertical_diffusivity, spread_target_cells

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

    routing = build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ;
                                  freshwater_density = ρ, maximum_search_radius = 5)

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
    routing = build_river_routing(target_grid, outlets...;
                                  freshwater_density = ρ, maximum_search_radius = 5)

    land = PrescribedLand((; rivers = discharge); river_routing = routing)

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

@testset "JRA55-style spreading and mixing [$arch]" for arch in test_architectures
    # Two columns have different depths, and the mouth straddles the date line.
    underlying = LatitudeLongitudeGrid(arch;
                                       size=(4, 4, 4), longitude=(178, 182), latitude=(-2, 2),
                                       z=[-60, -20, -10, -5, 0], halo=(3, 3, 3))
    bottom(λ, φ) = ifelse(λ < 180, -20, -60)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
    source_grid = LatitudeLongitudeGrid(arch;
                                       size=(2, 2), longitude=(-181, -179), latitude=(-1, 1),
                                       topology=(Bounded, Bounded, Flat))
    times = [0.0, 86400.0]
    rivers = FieldTimeSeries{Center, Center, Nothing}(source_grid, times)
    icebergs = FieldTimeSeries{Center, Center, Nothing}(source_grid, times)
    for n in 1:2
        set!(rivers[n], n == 1 ? 0.0 : 2.0)
        set!(icebergs[n], 3.0)
    end

    # A mouth that is dry initially must still be routed when it discharges later.
    mask = ever_positive_mask(rivers, 2)
    @test all(mask)
    oi, oj, λ, φ = outlet_indices_from_mask(mask, source_grid)
    areas = source_cell_areas(source_grid, oi, oj)
    routing = build_river_routing(grid, oi, oj, λ, φ, areas;
                                  maximum_search_radius=5, spread_radius=1.2)
    @test length(routing.target_i) > 1

    # Separate river and iceberg maps must add their mass fluxes, including at shared cells.
    land = PrescribedLand((; rivers, icebergs); river_routing=(; rivers=routing, icebergs=routing))
    exchanger = (; state=(; freshwater_flux=Field{Center, Center, Nothing}(grid)))
    coupled = (; clock=Clock(time=43200.0))
    interpolate_state!(exchanger, grid, land, coupled)
    flux = exchanger.state.freshwater_flux
    @test sum(Field(Integral(flux))) ≈ 4 * sum(areas) rtol=1e-5
    first_flux = Array(interior(flux))
    interpolate_state!(exchanger, grid, land, coupled)
    @test Array(interior(flux)) ≈ first_flux

    # Each source conserves its mass and uses the 50 m depth cap in its shares.
    cpu_grid = on_architecture(CPU(), grid)
    ti, tj = Array(routing.target_i), Array(routing.target_j)
    offsets = Array(routing.offsets)
    weights = Array(routing.contribution_weight)
    coi, coj = Array(routing.contribution_outlet_i), Array(routing.contribution_outlet_j)
    for n in eachindex(oi)
        shares = Float64[]
        depths = Float64[]
        for c in eachindex(ti), k in offsets[c]:(offsets[c+1]-1)
            (coi[k], coj[k]) == (oi[n], oj[n]) || continue
            area = Azᶜᶜᶜ(ti[c], tj[c], 4, cpu_grid)
            push!(shares, weights[k] * area)
            push!(depths, ti[c] <= 2 ? 20.0 : 50.0)
        end
        @test sum(shares) ≈ areas[n]
        @test shares ./ sum(shares) ≈ depths ./ sum(depths)
    end

    # Mixing acts at centers above -10 m, including iceberg targets, without doubling overlaps.
    closure = river_mouth_vertical_diffusivity(grid, land.river_routing; κ=0.1, mixing_depth=10)
    mixing = Array(interior(closure.κ.parameters))
    expected = zeros(size(grid)...)
    for c in eachindex(ti)
        expected[ti[c], tj[c], 3:4] .= 0.1
    end
    @test mixing ≈ expected
    @test maximum(mixing) ≈ 0.1

    # Distance wrapping must route mouths across ±180°, and unreachable mouths stay empty.
    @test spread_target_cells([1], [1], [179.5], [0.0], -180.5, 0.0, 1.0, 1.2, nothing) == [(1, 1)]
    @test isempty(spread_target_cells([1], [1], [0.0], [0.0], 10.0, 0.0, 1.0, 1.2, nothing))
end

@testset "Routing on a tripolar grid [$arch]" for arch in test_architectures
    grid = TripolarGrid(arch; size=(32, 16, 4), z=(-20, 0), halo=(3, 3, 3))
    λ = Array(λnodes(grid, Center(), Center(), Center()))
    φ = Array(φnodes(grid, Center(), Center(), Center()))
    routing = build_river_routing(grid, [1], [1], [λ[10, 8]], [φ[10, 8]], [123.0];
                                  maximum_search_radius=5, spread_radius=1.2)
    @test (10, 8) in zip(Array(routing.target_i), Array(routing.target_j))
    cpu_grid = on_architecture(CPU(), grid)
    ti, tj = Array(routing.target_i), Array(routing.target_j)
    weights = Array(routing.contribution_weight)
    @test sum(weights[c] * Azᶜᶜᶜ(ti[c], tj[c], 4, cpu_grid) for c in eachindex(ti)) ≈ 123
    closure = river_mouth_vertical_diffusivity(grid, (; rivers=routing, icebergs=routing))
    @test maximum(Array(interior(closure.κ.parameters))) ≈ 0.1
end
