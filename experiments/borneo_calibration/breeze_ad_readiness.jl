# How far the Breeze-coupled land goes under Reactant: a small doubly periodic Breeze
# atmosphere on a `ReactantState` grid, alone, coupled to the bare variably saturated slab,
# coupled to the two-tile canopy, and finally differentiated through one coupled step.
# Each stage reports its outcome; a failure does not stop the next stage.

include(joinpath(@__DIR__, "borneo_model.jl"))
using Breeze
using Oceananigans.Architectures: ReactantState, architecture
using Reactant
using Enzyme
using Printf

FT = Float64
Δt = 2.0
backend = get(ENV, "ARCH", "cpu")
Reactant.set_default_backend(backend)

function stage(f, name)
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

breeze_size = parse(Int, get(ENV, "BREEZE_N", "8"))
atmosphere_grid(arch) = RectilinearGrid(arch, FT; size = (breeze_size, breeze_size), halo = (5, 5), x = (-1kilometer, 1kilometer), z = (0, 1kilometer),
                                        topology = (Periodic, Flat, Bounded))
land_grid_of(grid) = RectilinearGrid(architecture(grid), FT; size = grid.Nx, halo = grid.Hx, x = (-1kilometer, 1kilometer),
                                     topology = (Periodic, Flat, Flat))

function breeze_atmosphere(grid)
    atmosphere = atmosphere_simulation(grid; potential_temperature = 295.0)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 2)
    return atmosphere
end

parameters = (; porosity = 0.45, residual_liquid_fraction = 0.06, inverse_air_entry_head = 2.0,
                pore_size_uniformity = 1.4, matching_point_conductivity = 5e-6, pore_connectivity_exponent = 0.5,
                leaf_area_index = 4.0, tile_lai = 4.5, vegetation_fraction = 0.9, canopy_height = 25.0,
                vegetated_roughness_length = 1.0, bare_roughness_length = 0.05,
                vegetated_scalar_roughness_length = 0.1, bare_scalar_roughness_length = 0.005,
                albedo = 0.13, emissivity = 0.97, dry_heat_capacity = 840 * 0.15 * 1200.0,
                deep_temperature = 295.0, infiltration_capacity = 2e-3, scalar_porosity = 0.45)

# A timed series (a single-slice series scalar-indexes under Reactant) holding constant fluxes.
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

function bare_coupled_model(grid)
    atmosphere = breeze_atmosphere(grid)
    land_grid = land_grid_of(grid)
    land = borneo_land(land_grid, FT, parameters; slab_depth = (h = surface_field(land_grid); parent(h) .= 0.5; h))
    humidity = DryLayerHumidity(FT; dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 1.0),
                                vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3, molecular_diffusivity = 2.4e-5),
                                thermal_exchange_depth = 0.05, porosity = 0.45)
    fluxes = SimilarityTheoryFluxes(FT; momentum_roughness_length = 0.1, temperature_roughness_length = 0.01, water_vapor_roughness_length = 0.01,
                                    stability_functions = atmosphere_land_stability_functions(FT), solver_stop_criteria = FixedIterations(4))
    interface = atmosphere_land_interface(land_grid, atmosphere, land; specific_humidity = humidity, fluxes)
    radiation = constant_radiation(land_grid)
    model = AtmosphereLandModel(atmosphere, land; atmosphere_land_interface = interface, radiation, clock = Clock(grid))
    return model
end

function canopy_coupled_model(grid)
    atmosphere = breeze_atmosphere(grid)
    land_grid = land_grid_of(grid)
    land = borneo_land(land_grid, FT, parameters; slab_depth = (h = surface_field(land_grid); parent(h) .= 0.5; h))
    interface = borneo_interface(land_grid, FT, atmosphere, land, parameters; inner_iterations = 4, similarity_iterations = 4)
    radiation = constant_radiation(land_grid)
    return AtmosphereLandModel(atmosphere, land; atmosphere_land_interface = interface, radiation, clock = Clock(grid))
end

function initialize_land!(model)
    parent(model.land.temperature) .= 295
    parent(model.land.water_storage) .= 1000 * 0.3 * 0.5
    parent(model.land.saturation) .= 0.6
    for field in values(model.land.prognostic)
        parent(field) .= 0
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

# ## Stage A: Breeze alone, eager CPU then compiled on ReactantState

skip_breeze_alone = get(ENV, "SKIP_BREEZE_ALONE", "0") == "1"
only_reverse = get(ENV, "ONLY_REVERSE", "0") == "1"      # build the coupled models and run stage D only

skip_breeze_alone || stage("A0 Breeze eager CPU step") do
    atmosphere = breeze_atmosphere(atmosphere_grid(CPU()))
    time_step!(atmosphere.model, Δt)
    atmosphere
end

skip_breeze_alone || stage("A1 Breeze compiled step on ReactantState") do
    atmosphere = breeze_atmosphere(atmosphere_grid(ReactantState()))
    compiled = Reactant.@compile raise=true raise_first=true sync=true step!(atmosphere.model, Δt)
    compiled(atmosphere.model, Δt)
    atmosphere
end

# ## Stage A2: reverse pass through one Breeze step alone (no land, no exchanger)

function atmosphere_energy_loss(model, Δt)
    time_step!(model, Δt)
    return sum(parent(model.velocities.u))
end

function grad_atmosphere_energy(model, dmodel, Δt)
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), atmosphere_energy_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Const(Δt))
    return L
end

get(ENV, "BREEZE_ALONE_REVERSE", "0") == "1" && stage("A2 Breeze alone: reverse pass through one step") do
    atmosphere = breeze_atmosphere(atmosphere_grid(ReactantState()))
    dmodel = Enzyme.make_zero(atmosphere.model)
    compiled = Reactant.@compile raise=true raise_first=true sync=true grad_atmosphere_energy(atmosphere.model, dmodel, Δt)
    L = compiled(atmosphere.model, dmodel, Δt)
    @info @sprintf("[A2] L = %.4f", Reactant.to_number(L))
    atmosphere
end

# ## Stage B: Breeze + bare slab, eager then compiled

only_reverse || stage("B0 coupled bare slab eager CPU step") do
    model = bare_coupled_model(atmosphere_grid(CPU()))
    initialize_land!(model)
    time_step!(model, Δt)
    model
end

model_b = stage("B1 coupled bare slab construction on ReactantState") do
    model = bare_coupled_model(atmosphere_grid(ReactantState()))
    Oceananigans.initialize!(model)
    model
end

isnothing(model_b) || only_reverse || stage("B2 coupled bare slab compiled step") do
    compiled = Reactant.@compile raise=true raise_first=true sync=true step!(model_b, Δt)
    compiled(model_b, Δt)
    model_b
end

# ## Stage C: Breeze + two-tile canopy

model_c = stage("C1 coupled canopy construction on ReactantState") do
    model = canopy_coupled_model(atmosphere_grid(ReactantState()))
    Oceananigans.initialize!(model)
    model
end

isnothing(model_c) || only_reverse || stage("C2 coupled canopy compiled step") do
    compiled = Reactant.@compile raise=true raise_first=true sync=true step!(model_c, Δt)
    compiled(model_c, Δt)
    model_c
end

# ## Stage D: one-step gradient of the land temperature w.r.t. the slab depth through Breeze

for (label, model) in (("bare slab", model_b), ("canopy", model_c))
    isnothing(model) && continue
    stage("D $label: reverse pass through one coupled Breeze step") do
        h = surface_field(model.land.grid); parent(h) .= 0.5
        dh = Enzyme.make_zero(h)
        dmodel = Enzyme.make_zero(model)
        compiled = Reactant.@compile raise=true raise_first=true sync=true grad_land_temperature(model, dmodel, h, dh, Δt)
        out = compiled(model, dmodel, h, dh, Δt)
        @info @sprintf("[D %s] dL/dh = %s, L = %.4f", label, string(Array(parent(out[1]))[1:2]), Reactant.to_number(out[2]))
        out
    end
end
