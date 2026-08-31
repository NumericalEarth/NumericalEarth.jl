# Map setup shared by `map_calibration.jl` and `map_fd_sweep.jl`: fields, targets, state
# initialization and the eager forward run over the box.

include(joinpath(@__DIR__, "borneo_config.jl"))
include(joinpath(@__DIR__, "borneo_model.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Statistics: mean, median, cor
using Printf

FT = Float64
Δt = 10minutes
Nsteps = parse(Int, get(ENV, "NSTEPS", "900"))       # 150 h = 6.25 days; a perfect square for checkpointing
run_hours = Nsteps * Δt / 3600
h₀ = 0.28
lapse_rate = 6.5e-3
inner_iterations = parse(Int, get(ENV, "INNER_ITERATIONS", "6"))         # canopy Newton iterations per step
similarity_iterations = parse(Int, get(ENV, "SIMILARITY_ITERATIONS", "4"))  # Monin–Obukhov iterates per step
backend = get(ENV, "ARCH", "cpu")
tag = "map_calibration_r$(refinement)_$(backend)"

static    = load_static()
forcing   = load_cache("forcing")
era5_land = load_cache("era5_land")
cpu_grid  = land_grid()

# The hourly ERA5-Land target interpolated to every model step, in the parent layout the
# loss reads (halos zero).
Hx, Hy = cpu_grid.Hx, cpu_grid.Hy
θ_target = zeros(FT, Nsteps, Nx + 2Hx, Ny + 2Hy, 1)
for n in 1:Nsteps
    t = n * Δt / 3600
    k = clamp(floor(Int, t) + 1, 1, length(era5_land.times) - 1)
    a = t - (k - 1)
    θ_target[n, 1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] .= (1 - a) .* era5_land_soil_water(era5_land, k) .+ a .* era5_land_soil_water(era5_land, k + 1)
end
θ_target_end = θ_target[end, 1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1]
weight = FT.(.!static.water)
θ₀ = FT.(static.initial_soil_water)
T₀ = FT.(forcing.skin_temperature)
correction = AltitudeCorrection(forcing.land_elevation, forcing.era5_elevation; lapse_rate)

# ## State initialization shared by every run (parent-level, so it traces on any backend)

function initialize_map!(model, h, θ₀, T₀, q₀)
    hydrology = model.land.hydrology.soil.soil
    parent(hydrology.slab_depth) .= parent(h)
    parent(model.land.water_storage) .= 1000 .* parent(θ₀) .* parent(h)
    parent(model.land.temperature) .= parent(T₀)
    ν, θʳ = parent(hydrology.porosity), parent(hydrology.residual_liquid_fraction)
    parent(model.land.saturation) .= clamp.((parent(θ₀) .- θʳ) ./ (ν .- θʳ), 0, 1)
    parent(model.land.prognostic.canopy_water_storage) .= 0
    parent(model.land.prognostic.surface_water_storage) .= 0
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= parent(T₀)
        parent(tile.temperature.state.specific_humidity) .= parent(q₀)
    end
    update_state!(model)   # fluxes consistent with the reset state
    return nothing
end

soil_water(model, h) = parent(model.land.water_storage) ./ (1000 .* parent(h))
cell_loss(model, h, w, θᵗ) = parent(w) .* (soil_water(model, h) .- θᵗ).^2

# Fields on `grid` for the state and target arrays; `h` carries h₀ into the halos too, so
# the parent-level θ = M / (ρ h) stays finite where the loss weight is zero.
function map_fields(grid, depth)
    h = surface_property(grid, fill(FT(1), Nx, Ny)); parent(h) .= 0; set!(h, depth); parent(h) .= ifelse.(parent(h) .== 0, h₀, parent(h))
    return (; h, θ₀ = surface_property(grid, θ₀), T₀ = surface_property(grid, T₀),
              q₀ = surface_property(grid, forcing.q[1]), w = surface_property(grid, weight))
end

# ## Eager CPU forward, recording hourly snapshots

snapshot_names = (:θ, :T, :LST, :𝒮, :Wᶜ, :LE, :LEᶜ, :LEᵍ, :H, :rain, :E)

function forward_map(depth; record = false, modify! = nothing)
    fields = map_fields(cpu_grid, depth)
    s = surface_parameters(static, cpu_grid, FT)
    model = borneo_coupled_model(cpu_grid, FT, forcing, s; slab_depth = surface_field(cpu_grid),
                                 exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                 inner_iterations, similarity_iterations)
    isnothing(modify!) || modify!(model)
    initialize_map!(model, fields.h, fields.θ₀, fields.T₀, fields.q₀)
    interface = model.interfaces.atmosphere_land_interface
    land = model.land
    steps_per_hour = round(Int, 3600 / Δt)
    snapshots = record ? NamedTuple{snapshot_names}(ntuple(_ -> zeros(Float32, Nsteps ÷ steps_per_hour + 1, Nx, Ny), length(snapshot_names))) : nothing
    take!(k) = (snapshots.θ[k, :, :]   .= interior(land.water_storage, :, :, 1) ./ (1000 .* interior(fields.h, :, :, 1));
                snapshots.T[k, :, :]   .= interior(land.temperature, :, :, 1);
                snapshots.LST[k, :, :] .= interior(interface.temperature.effective, :, :, 1);
                snapshots.𝒮[k, :, :]   .= interior(land.saturation, :, :, 1);
                snapshots.Wᶜ[k, :, :]  .= interior(land.prognostic.canopy_water_storage, :, :, 1);
                snapshots.LE[k, :, :]  .= interior(interface.fluxes.latent_heat, :, :, 1);
                snapshots.LEᶜ[k, :, :] .= interior(interface.temperature.canopy_latent_heat, :, :, 1);
                snapshots.LEᵍ[k, :, :] .= interior(interface.temperature.soil_latent_heat, :, :, 1);
                snapshots.H[k, :, :]   .= interior(interface.fluxes.sensible_heat, :, :, 1);
                snapshots.rain[k, :, :] .= interior(model.interfaces.exchanger.atmosphere.state.Jʳⁿ, :, :, 1);
                snapshots.E[k, :, :]   .= interior(land.fluxes.vapor_flux, :, :, 1))
    record && take!(1)
    losses = zeros(FT, size(parent(fields.h)))
    wall = time_ns()
    for n in 1:Nsteps
        time_step!(model, Δt)
        losses .+= cell_loss(model, fields.h, fields.w, view(θ_target, n, :, :, :))
        record && n % steps_per_hour == 0 && take!(n ÷ steps_per_hour + 1)
    end
    @info @sprintf("eager forward (%d × %d, %d steps) in %.1f s", Nx, Ny, Nsteps, 1e-9 * (time_ns() - wall))
    losses = losses[1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] ./ Nsteps
    θ_end = interior(land.water_storage, :, :, 1) ./ (1000 .* interior(fields.h, :, :, 1))
    return (; losses, θ_end = Array(θ_end), snapshots)
end

land = weight .> 0
rms(losses) = sqrt(sum(losses) / count(land))
