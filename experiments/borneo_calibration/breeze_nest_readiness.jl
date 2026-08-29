# How far the nested Breeze pipeline goes under Reactant: an ERA5 pressure-level parent held in
# memory, a small terrain-following child over Central Borneo with Davies relaxation, built and
# stepped eagerly, then on a `ReactantState` grid, then coupled to the two-tile canopy land, then
# differentiated through one coupled step. Each stage reports its outcome; failures do not stop
# the following stages.
#
#   ARCH=cpu NEST_N=16 julia --project=docs breeze_nest_readiness.jl

include(joinpath(@__DIR__, "borneo_model.jl"))
using Breeze
using CopernicusClimateDataStore
using CloudMicrophysics
using ArchGDAL
using Oceananigans.Architectures: ReactantState, architecture
using Reactant
using Enzyme
using Printf
import Dates: DateTime

FT = Float64
Δt = 2.0
backend = get(ENV, "ARCH", "cpu")
Reactant.set_default_backend(backend)
N = parse(Int, get(ENV, "NEST_N", "16"))
stages = split(get(ENV, "NEST_STAGES", "N0,N0t,N1,N2,N3,N4"), ",")

function stage(f, name)
    first(split(name)) in stages || return nothing
    wall = @elapsed result = try
        f()
    catch err
        io = IOBuffer(); showerror(io, err); message = first(split(String(take!(io)), '\n'))
        @error "[$name] FAILED: $(message)" exception = (err, catch_backtrace())
        nothing
    end
    isnothing(result) || @info @sprintf("[%s] ok in %.0f s", name, wall)
    return result
end

# A 0.5° box inside the calibration domain, N × N cells (≈ 3.5 km at N = 16), 6 hours of parent data.
latitude  = (1.0, 1.5)
longitude = (113.75, 114.25)
dates = (DateTime(2020, 4, 1), DateTime(2020, 4, 1, 6))

z = ReferenceToStretchedDiscretization(extent = 10000.0, bias = :left, bias_edge = 0.0,
                                       constant_spacing = 200.0, constant_spacing_extent = 200.0,
                                       maximum_spacing = 1000.0, stretching = LinearStretching(0.2))

terrain_following = get(ENV, "NEST_TERRAIN_FOLLOWING", "1") == "1"
child_grid(arch) = LatitudeLongitudeGrid(arch, FT; longitude, latitude, z = terrain_following ? TerrainFollowingVerticalDiscretization(z) : z,
                                         size = (N, N, length(z)), halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))
nest_land_grid(arch) = LatitudeLongitudeGrid(arch, FT; longitude, latitude, size = (N, N), halo = (5, 5),
                                             topology = (Bounded, Bounded, Flat))

# The parent window is fully resident (`parent_time_indices_in_memory = nothing`): a disk-backed
# moving window cannot be traced. The balancer is Breeze's eager adiabatic spin-up; it runs on
# the CPU build only.
build_nest(arch; terrain = nothing, balancer = true) =
    nested_atmosphere_model(child_grid(arch), ERA5HourlyPressureLevels(); dates,
                            parent_time_indices_in_memory = nothing,
                            terrain, terrain_smoothing_passes = 2,
                            relaxation_rate = 1/300, relaxation_width = 3,
                            momentum_advection = WENO(order = 5),
                            balancer)

parameters = (; porosity = 0.45, residual_liquid_fraction = 0.06, inverse_air_entry_head = 2.0,
                pore_size_uniformity = 1.4, matching_point_conductivity = 5e-6, pore_connectivity_exponent = 0.5,
                leaf_area_index = 4.0, tile_lai = 4.5, vegetation_fraction = 0.9, canopy_height = 25.0,
                vegetated_roughness_length = 1.0, bare_roughness_length = 0.05,
                vegetated_scalar_roughness_length = 0.1, bare_scalar_roughness_length = 0.005,
                albedo = 0.13, emissivity = 0.97, dry_heat_capacity = 840 * 0.15 * 1200.0,
                deep_temperature = 298.0, infiltration_capacity = 2e-3, scalar_porosity = 0.45)

function constant_radiation(land_grid)
    radiation = PrescribedRadiation(land_grid, 0:3600:7200; ocean_surface = nothing, sea_ice_surface = nothing,
                                    land_surface = SurfaceRadiationProperties(0.2, 0.95))
    for n in 1:3
        parent(radiation.downwelling_shortwave[n]) .= 600
        parent(radiation.downwelling_longwave[n]) .= 350
    end
    update_state!(radiation)
    return radiation
end

function coupled_nest(nest)
    grid = nest.child.grid
    land_grid = nest_land_grid(architecture(grid))
    land = borneo_land(land_grid, FT, parameters; slab_depth = (h = surface_field(land_grid); parent(h) .= 0.5; h))
    atmosphere = Simulation(nest; Δt)
    interface = borneo_interface(land_grid, FT, atmosphere, land, parameters; inner_iterations = 4, similarity_iterations = 4)
    return AtmosphereLandModel(atmosphere, land; atmosphere_land_interface = interface, radiation = constant_radiation(land_grid),
                               clock = Clock(grid))
end

function initialize_land!(model)
    parent(model.land.temperature) .= 298
    parent(model.land.water_storage) .= 1000 * 0.3 * 0.5
    parent(model.land.saturation) .= 0.6
    for field in values(model.land.prognostic)
        parent(field) .= 0
    end
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= 298
        parent(tile.temperature.state.specific_humidity) .= 0.016
    end
    update_state!(model)
    return nothing
end

step!(model, Δt) = (time_step!(model, Δt); nothing)

function land_temperature_loss(model, h, Δt)
    parent(model.land.hydrology.soil.soil.slab_depth) .= parent(h)
    initialize_land!(model)
    time_step!(model, Δt)
    return sum(parent(model.land.temperature))
end

function grad_land_temperature(model, dmodel, h, dh, Δt)
    parent(dh) .= 0
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), land_temperature_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(h, dh), Enzyme.Const(Δt))
    return dh, L
end

# ## Stage N0: eager CPU nest (fetches the ERA5 pressure-level parent), flat and with ETOPO terrain

stage("N0 eager nest, flat") do
    nest = build_nest(CPU())
    time_step!(nest, Δt)
    @info @sprintf("[N0] child %s; max |u| %.2f m s⁻¹", summary(nest.child.grid), maximum(abs, nest.child.velocities.u))
    nest
end

stage("N0t eager nest with ETOPO terrain") do
    nest = build_nest(CPU(); terrain = ETOPO2022())
    time_step!(nest, Δt)
    nest
end

# ## Stage N1: the nest on ReactantState, then one compiled step

nest_r = stage("N1 nest construction on ReactantState") do
    build_nest(ReactantState(); balancer = false)
end

isnothing(nest_r) || stage("N2 compiled nest step") do
    compiled = Reactant.@compile raise=true raise_first=true sync=true step!(nest_r, Δt)
    compiled(nest_r, Δt)
    nest_r
end

# ## Stage N3: nest + two-tile canopy land, construction and compiled coupled step

coupled_r = isnothing(nest_r) ? nothing : stage("N3 coupled nest + canopy construction") do
    model = coupled_nest(nest_r)
    Oceananigans.initialize!(model)
    model
end

isnothing(coupled_r) || stage("N3c compiled coupled step") do
    compiled = Reactant.@compile raise=true raise_first=true sync=true step!(coupled_r, Δt)
    compiled(coupled_r, Δt)
    coupled_r
end

# ## Stage N4: reverse pass through one coupled step

isnothing(coupled_r) || stage("N4 reverse pass through one coupled nest step") do
    h = surface_field(coupled_r.land.grid); parent(h) .= 0.5
    dh = Enzyme.make_zero(h)
    dmodel = Enzyme.make_zero(coupled_r)
    compiled = Reactant.@compile raise=true raise_first=true sync=true grad_land_temperature(coupled_r, dmodel, h, dh, Δt)
    out = compiled(coupled_r, dmodel, h, dh, Δt)
    @info @sprintf("[N4] dL/dh[1:2] = %s, L = %.4f", string(Array(parent(out[1]))[1:2]), Reactant.to_number(out[2]))
    out
end
