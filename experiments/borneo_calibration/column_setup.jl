# Column setup shared by `column_calibration.jl` and `column_fd_check.jl`: the cell, its
# parameters and forcing, the ERA5-Land target, the state initialization and the eager
# forward run.

include(joinpath(@__DIR__, "borneo_config.jl"))
include(joinpath(@__DIR__, "borneo_model.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Statistics: mean
using Printf

FT = Float64
Δt = 10minutes
Nsteps = 30^2                          # 150 h = 6.25 days; a perfect square for checkpointing
run_hours = Nsteps * Δt / 3600
h₀ = 0.28                              # the ERA5-Land 0–28 cm column the target is built from
lapse_rate = 6.5e-3
inner_iterations = parse(Int, get(ENV, "INNER_ITERATIONS", "6"))         # canopy Newton iterations per step
similarity_iterations = parse(Int, get(ENV, "SIMILARITY_ITERATIONS", "4"))  # Monin–Obukhov iterates per step

static    = load_static()
forcing   = load_forcing = load_cache("forcing")
era5_land = load_cache("era5_land")
cpu_grid  = land_grid()

# ## The column: a densely vegetated land cell near the domain center

function choose_cell(static)
    candidates = findall(.!static.water .& (static.vegetation_fraction .> 0.7) .& (static.leaf_area_index .> 2))
    center = (Nx + 1) / 2, (Ny + 1) / 2
    return Tuple(candidates[argmin([hypot(c[1] - center[1], c[2] - center[2]) for c in candidates])])
end
cell = haskey(ENV, "CELL") ? Tuple(parse.(Int, split(ENV["CELL"], ","))) : choose_cell(static)
i, j = cell
λ, φ = static.longitude[i], static.latitude[j]
@info @sprintf("column (%d, %d) at %.2f°E %.2f°N: %s, LAI %.2f, canopy %.1f m, f_veg %.2f, ℓᵐ %.2f m, ν %.3f, θʳ %.3f, θ₀ %.3f, z %.0f m (ERA5 %.0f m)",
               i, j, λ, φ, static.canopy_class[i, j], static.leaf_area_index[i, j], static.canopy_height[i, j],
               static.vegetation_fraction[i, j], static.vegetated_roughness_length[i, j],
               static.porosity[i, j], static.residual_liquid_fraction[i, j], static.initial_soil_water[i, j],
               forcing.land_elevation[i, j], forcing.era5_elevation[i, j])

parameters = surface_parameters(static, nothing, FT, cell)
column = column_forcing(forcing, cpu_grid, cell)

# Lift the ERA5 near-surface state from ERA5's terrain to the ETOPO surface: a lapse-rate
# temperature shift and the matching hydrostatic pressure change.
Δz = forcing.land_elevation[i, j] - forcing.era5_elevation[i, j]
column = merge(column, (; T = column.T .- lapse_rate * Δz,
                          p = column.p .* exp.(-9.81 * Δz ./ (287 .* column.T))))

θ_obs = [era5_land_soil_water(era5_land, n)[i, j] for n in eachindex(era5_land.times)]
θ_obs_layer_1 = era5_land.layer_1[:, i, j]
θ₀ = FT(static.initial_soil_water[i, j])
T₀ = FT(forcing.skin_temperature[i, j])
q₀ = FT(column.q[1])

# The hourly ERA5-Land target interpolated to every model step.
step_times = (1:Nsteps) .* Δt
θ_target = [begin k = clamp(floor(Int, t / 3600) + 1, 1, length(θ_obs) - 1); a = t / 3600 - (k - 1)
                  FT((1 - a) * θ_obs[k] + a * θ_obs[k + 1]) end for t in step_times]

# ## State initialization shared by every run

function initialize_column!(model, h, θ₀, T₀, q₀)
    hydrology = model.land.hydrology.soil.soil
    parent(hydrology.slab_depth) .= parent(h)
    parent(model.land.water_storage) .= 1000 .* θ₀ .* parent(h)
    parent(model.land.temperature) .= T₀
    ν, θʳ = hydrology.porosity, hydrology.residual_liquid_fraction
    parent(model.land.saturation) .= clamp((θ₀ - θʳ) / (ν - θʳ), 0, 1)
    parent(model.land.prognostic.canopy_water_storage) .= 0
    parent(model.land.prognostic.surface_water_storage) .= 0
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= T₀
        parent(tile.temperature.state.specific_humidity) .= q₀
    end
    update_state!(model)   # fluxes consistent with the reset state
    return nothing
end

soil_water(model, h) = interior(model.land.water_storage) ./ (1000 .* interior(h))
scalar(f) = first(interior(f))

# ## Eager forward run, recording the column's evolution

series_names = (:t, :T, :θ, :𝒮, :Wᶜ, :Wᵖ, :LE, :LEᶜ, :LEᵍ, :LEʷ, :H, :Tᵃᶜ, :Tˡᵉᵃᶠ, :Tᵍ, :E, :P, :R, :D, :u★, :Tᵃ, :rain, :sw)

function forward_column(depth)
    grid = RectilinearGrid(CPU(), FT; size = (), topology = (Flat, Flat, Flat))
    h = surface_field(grid); parent(h) .= depth
    model = borneo_coupled_model(grid, FT, column, parameters; slab_depth = surface_field(grid),
                                 surface_layer_height, boundary_layer_height, inner_iterations, similarity_iterations)
    initialize_column!(model, h, θ₀, T₀, q₀)
    interface = model.interfaces.atmosphere_land_interface
    land = model.land
    atmosphere_state = model.interfaces.exchanger.atmosphere.state
    series = NamedTuple{series_names}(ntuple(_ -> zeros(Nsteps), length(series_names)))
    for n in 1:Nsteps
        time_step!(model, Δt)
        series.t[n]  = model.clock.time / 3600
        series.T[n]  = scalar(land.temperature)
        series.θ[n]  = first(soil_water(model, h))
        series.𝒮[n]  = scalar(land.saturation)
        series.Wᶜ[n] = scalar(land.prognostic.canopy_water_storage)
        series.Wᵖ[n] = scalar(land.prognostic.surface_water_storage)
        series.LE[n]  = scalar(interface.fluxes.latent_heat)
        series.LEᶜ[n] = scalar(interface.temperature.canopy_latent_heat)
        series.LEᵍ[n] = scalar(interface.temperature.soil_latent_heat)
        series.LEʷ[n] = scalar(interface.temperature.canopy_wet_latent_heat)
        series.H[n]   = scalar(interface.fluxes.sensible_heat)
        series.Tᵃᶜ[n]   = scalar(interface.temperature.interface)
        series.Tˡᵉᵃᶠ[n] = scalar(interface.temperature.canopy)
        series.Tᵍ[n]    = scalar(interface.temperature.soil_skin)
        series.E[n]  = scalar(land.fluxes.vapor_flux)
        series.P[n]  = scalar(land.fluxes.liquid_precipitation_flux)
        series.R[n]  = scalar(land.diagnostics.surface_water_runoff)
        series.D[n]  = scalar(land.diagnostics.deep_liquid_flux)
        series.u★[n] = scalar(interface.fluxes.friction_velocity)
        series.Tᵃ[n]   = scalar(atmosphere_state.T)
        series.rain[n] = scalar(atmosphere_state.Jʳⁿ)
        series.sw[n]   = scalar(model.interfaces.exchanger.radiation.state.ℐꜜˢʷ)
    end
    return series
end

