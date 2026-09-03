# Map setup shared by `map_calibration.jl` and `map_fd_sweep.jl`: fields, targets, state
# initialization and the eager forward run over the box.

include(joinpath(@__DIR__, "borneo_config.jl"))
include(joinpath(@__DIR__, "borneo_model.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Statistics: mean, median, cor, std
using Oceananigans.OutputReaders: FieldTimeSeries
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
deep_flux = get(ENV, "DEEP_FLUX", "free")                       # "free" drainage or "darcy" exchange
exchange_length = parse(Float64, get(ENV, "EXCHANGE_LENGTH", "0.36"))   # m, slab bottom to the deep reservoir
tag_suffix = get(ENV, "TAG_SUFFIX", "")
tag = "map_calibration_r$(refinement)_$(backend)$(tag_suffix)"

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
θᵈ_target = zeros(FT, Nsteps, Nx + 2Hx, Ny + 2Hy, 1)      # ERA5-Land 28–100 cm, the deep store's target when DEEP_LOSS=1
for n in 1:Nsteps
    t = n * Δt / 3600
    k = clamp(floor(Int, t) + 1, 1, length(era5_land.times) - 1)
    a = t - (k - 1)
    θᵈ_target[n, 1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] .= (1 - a) .* era5_land.layer_3[k, :, :] .+ a .* era5_land.layer_3[k + 1, :, :]
end
weight = FT.(.!static.water)
θ₀ = FT.(static.initial_soil_water)
T₀ = FT.(forcing.skin_temperature)
correction = AltitudeCorrection(forcing.land_elevation, forcing.era5_elevation; lapse_rate)

# ## The bottom boundary: free drainage, or Darcy exchange with a deep reservoir held at the
# head of ERA5-Land's 28–100 cm layer through each cell's own van Genuchten curve

function deep_head(θ, α, n, ν, θʳ)
    𝒮 = clamp((θ - θʳ) / (ν - θʳ), 1e-6, 1)
    m = 1 - 1 / n
    return 𝒮 ≥ 1 ? 0.0 : -(𝒮^(-1 / m) - 1)^(1 / n) / α
end

deep_head_table() = deep_head.(era5_land.layer_3, reshape(static.inverse_air_entry_head, 1, Nx, Ny),
                               reshape(static.pore_size_uniformity, 1, Nx, Ny), reshape(static.porosity, 1, Nx, Ny),
                               reshape(static.residual_liquid_fraction, 1, Nx, Ny))

function running_time_mean(table, window)
    Nt = size(table, 1)
    return [mean(view(table, max(1, k - window ÷ 2):min(Nt, k + window ÷ 2), i, j)) for k in 1:Nt, i in 1:Nx, j in 1:Ny]
end

function deep_pressure_head_series(grid, table)
    Πᵈ = FieldTimeSeries{Center, Center, Nothing}(grid, era5_land.times)
    slice = zeros(FT, Nx + 2Hx, Ny + 2Hy, 1)
    for k in eachindex(era5_land.times)
        slice[1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] .= table[k, :, :]
        parent(Πᵈ[k]) .= slice
    end
    return Πᵈ
end

exchange_field = get(ENV, "EXCHANGE_FIELD", "0") == "1"     # carry ℓ as a per-cell Field (calibratable)
deep_head_mode = get(ENV, "DEEP_HEAD", "series")             # ERA5-Land 28–100 cm through the retention curve as a "series",
deep_head_smooth_days = parse(Float64, get(ENV, "DEEP_HEAD_SMOOTH_DAYS", "2"))   # its running mean over this window ("smooth"),
deep_head_value = parse(Float64, get(ENV, "DEEP_HEAD_VALUE", "-1.0"))           # per-cell Fields of its time "mean" or "initial" value,
                                                                                # or a uniform "constant" head (m)
function deep_pressure_head_on(grid)
    deep_head_mode == "constant" && return surface_property(grid, fill(FT(deep_head_value), Nx, Ny))
    table = deep_head_table()
    deep_head_mode == "series" && return deep_pressure_head_series(grid, table)
    if deep_head_mode == "smooth"
        window = round(Int, deep_head_smooth_days * 86400 / (era5_land.times[2] - era5_land.times[1]))
        return deep_pressure_head_series(grid, running_time_mean(table, window))
    end
    head = deep_head_mode == "mean" ? dropdims(mean(table; dims = 1); dims = 1) : table[1, :, :]
    return surface_property(grid, FT.(head))
end
exchange_length_on(grid) = exchange_field ? surface_property(grid, fill(FT(exchange_length), Nx, Ny)) : exchange_length

# ## Or a prognostic deep store under the slab (DEEP_STORE=1): the Darcy exchange then talks to a
# reservoir of thickness hᵈ whose head follows its own water content, initialized from ERA5-Land's
# 28–100 cm layer at t = 0 and never read again

deep_store = get(ENV, "DEEP_STORE", "0") == "1"
deep_store_thickness = parse(Float64, get(ENV, "DEEP_STORE_THICKNESS", "0.72"))       # m, ERA5-Land layer 3
deep_store_drainage = get(ENV, "DEEP_STORE_DRAINAGE", "free")                        # "free" (K(𝒮ᵈ)) or "none"
deep_initial_soil_water = FT.(era5_land.layer_3[1, :, :])

deep_loss = get(ENV, "DEEP_LOSS", "0") == "1"                                  # add the store's mismatch to ERA5-Land 28–100 cm
deep_loss_weight = parse(Float64, get(ENV, "DEEP_LOSS_WEIGHT", "1"))            # relative to the surface term

deep_store_options(grid) = (; thickness = surface_property(grid, fill(FT(deep_store_thickness), Nx + 2Hx, Ny + 2Hy, 1)),
                              drainage = deep_store_drainage == "free" ? FreeDrainageFlux(FT) : NoDeepLiquidFlux(),
                              conductivity = surface_property(grid, FT.(static.matching_point_conductivity)))
hydrology_options(grid) = deep_flux == "darcy" ?
    (; deep_liquid_flux = DarcyDeepLiquidFlux(FT; exchange_length = exchange_length_on(grid)),
       deep_pressure_head = deep_store ? surface_property(grid, zeros(FT, Nx, Ny)) : deep_pressure_head_on(grid),
       deep_store = deep_store ? deep_store_options(grid) : nothing) : (;)

# ## Reaching the closures inside InterceptingHydrology(SurfaceWaterStore([DeepWaterStore(]soil[)]))

unwrap(h, T) = h isa T ? h : unwrap(h.soil, T)
soil_hydrology(model) = unwrap(model.land.hydrology, VariablySaturatedHydrology)
deep_water_store(model) = unwrap(model.land.hydrology, DeepWaterStore)

# ## State initialization shared by every run (parent-level, so it traces on any backend)

function initialize_map!(model, h, θ₀, T₀, q₀, θᵈ₀ = nothing)
    hydrology = soil_hydrology(model)
    parent(hydrology.slab_depth) .= parent(h)
    parent(model.land.water_storage) .= 1000 .* parent(θ₀) .* parent(h)
    parent(model.land.temperature) .= parent(T₀)
    ν, θʳ = parent(hydrology.porosity), parent(hydrology.residual_liquid_fraction)
    parent(model.land.saturation) .= clamp.((parent(θ₀) .- θʳ) ./ (ν .- θʳ), 0, 1)
    parent(model.land.prognostic.canopy_water_storage) .= 0
    parent(model.land.prognostic.surface_water_storage) .= 0
    if haskey(model.land.prognostic, :deep_water_storage)
        isnothing(θᵈ₀) && error("the deep store needs its initial water content")
        parent(model.land.prognostic.deep_water_storage) .= 1000 .* parent(deep_water_store(model).thickness) .* parent(θᵈ₀)
    end
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= parent(T₀)
        parent(tile.temperature.state.specific_humidity) .= parent(q₀)
    end
    update_state!(model)   # fluxes consistent with the reset state
    return nothing
end

soil_water(model, h) = parent(model.land.water_storage) ./ (1000 .* parent(h))
cell_loss(model, h, w, θᵗ) = parent(w) .* (soil_water(model, h) .- θᵗ).^2
deep_soil_water(model) = parent(model.land.prognostic.deep_water_storage) ./ (1000 .* parent(deep_water_store(model).thickness))
deep_cell_loss(model, w, θᵈᵗ) = parent(w) .* (deep_soil_water(model) .- θᵈᵗ).^2

# Fields on `grid` for the state and target arrays; `h` carries h₀ into the halos too, so
# the parent-level θ = M / (ρ h) stays finite where the loss weight is zero.
function map_fields(grid, depth)
    h = surface_property(grid, fill(FT(1), Nx, Ny)); parent(h) .= 0; set!(h, depth); parent(h) .= ifelse.(parent(h) .== 0, h₀, parent(h))
    return (; h, θ₀ = surface_property(grid, θ₀), T₀ = surface_property(grid, T₀),
              q₀ = surface_property(grid, forcing.q[1]), w = surface_property(grid, weight),
              θᵈ₀ = surface_property(grid, deep_initial_soil_water))
end

# ## Eager CPU forward, recording hourly snapshots

snapshot_names = (:θ, :θᵈ, :T, :LST, :𝒮, :Wᶜ, :LE, :LEᶜ, :LEᵍ, :H, :rain, :E)

function forward_map(depth; record = false, modify! = nothing, hydrology...)
    fields = map_fields(cpu_grid, depth)
    s = surface_parameters(static, cpu_grid, FT)
    model = borneo_coupled_model(cpu_grid, FT, forcing, s; slab_depth = surface_field(cpu_grid),
                                 exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                 inner_iterations, similarity_iterations, merge(hydrology_options(cpu_grid), hydrology)...)
    isnothing(modify!) || modify!(model)
    initialize_map!(model, fields.h, fields.θ₀, fields.T₀, fields.q₀, fields.θᵈ₀)
    interface = model.interfaces.atmosphere_land_interface
    land = model.land
    steps_per_hour = round(Int, 3600 / Δt)
    snapshots = record ? NamedTuple{snapshot_names}(ntuple(_ -> zeros(Float32, Nsteps ÷ steps_per_hour + 1, Nx, Ny), length(snapshot_names))) : nothing
    deep_water(k) = haskey(land.prognostic, :deep_water_storage) ?
        (snapshots.θᵈ[k, :, :] .= interior(land.prognostic.deep_water_storage, :, :, 1) ./ (1000 .* interior(deep_water_store(model).thickness, :, :, 1))) : nothing
    take!(k) = (deep_water(k);
                snapshots.θ[k, :, :]   .= interior(land.water_storage, :, :, 1) ./ (1000 .* interior(fields.h, :, :, 1));
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
    Σθ, Σθ² = zero(losses), zero(losses)
    wall = time_ns()
    for n in 1:Nsteps
        time_step!(model, Δt)
        losses .+= cell_loss(model, fields.h, fields.w, view(θ_target, n, :, :, :))
        deep_loss && (losses .+= deep_loss_weight .* deep_cell_loss(model, fields.w, view(θᵈ_target, n, :, :, :)))
        θ = soil_water(model, fields.h)
        Σθ .+= θ
        Σθ² .+= θ .^ 2
        record && n % steps_per_hour == 0 && take!(n ÷ steps_per_hour + 1)
    end
    @info @sprintf("eager forward (%d × %d, %d steps) in %.1f s", Nx, Ny, Nsteps, 1e-9 * (time_ns() - wall))
    inner(a) = a[1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] ./ Nsteps
    losses, θ_mean, θ² = inner(losses), inner(Σθ), inner(Σθ²)
    θ_end = interior(land.water_storage, :, :, 1) ./ (1000 .* interior(fields.h, :, :, 1))
    return (; losses, θ_end = Array(θ_end), snapshots, θ_mean, θ_variance = θ² .- θ_mean .^ 2)
end

land = weight .> 0
rms(losses) = sqrt(sum(losses) / count(land))

# ## Writing calibrated per-cell fields into a model (eager or traced) through their parents

conductivity_field(model) = soil_hydrology(model).hydraulic_conductivity.matching_point_conductivity
cpu_scratch = surface_field(cpu_grid)
function set_cells!(field, values, fill_value)
    set!(cpu_scratch, values)
    parent(cpu_scratch) .= ifelse.(parent(cpu_scratch) .== 0, fill_value, parent(cpu_scratch))
    parent(field) .= parent(cpu_scratch)
    return field
end
with_conductivity(q) = model -> (set_cells!(conductivity_field(model), exp10.(q), exp10(median(q[land]))); nothing)

# ## Hourly ERA5-Land series and per-window tracking scores of a recorded run

hourly_observations() = permutedims(cat([era5_land_soil_water(era5_land, m) for m in 1:(Nsteps ÷ round(Int, 3600 / Δt) + 1)]...; dims = 3), (3, 1, 2))

function window_scores(θ_model, θ_obs, hours)
    r = Float64[]; σratio = Float64[]; mse = Float64[]
    for c in findall(land)
        m, o = θ_model[hours, c], θ_obs[hours, c]
        push!(mse, sum(abs2, m .- o) / length(hours))
        push!(σratio, std(m) / std(o))
        std(m) > 0 && push!(r, cor(m, o))
    end
    return (; rms = sqrt(mean(mse)), r = median(r), σ = median(σratio))
end

# ## Applying a saved calibration to a model: whichever fields the file carries
#
# `q` is log₁₀K₀; `log_exchange_length` the Darcy exchange length (needs EXCHANGE_FIELD=1 so the
# model carries it as a Field); `log_n_minus_1` the retention exponent, with the air-entry
# parameter α slaved so the curve keeps its pedotransfer saturation at the matching head ψ★ = 1 m.

exchange_length_field(model) = soil_hydrology(model).deep_liquid_flux.exchange_length
deep_pressure_head_field(model) = soil_hydrology(model).deep_pressure_head
thickness_field(model) = deep_water_store(model).thickness
deep_conductivity_field(model) = deep_water_store(model).hydraulic_conductivity.matching_point_conductivity
air_entry_field(model) = soil_hydrology(model).retention_curve.inverse_air_entry_head
pore_size_uniformity_fields(model) = (soil_hydrology(model).retention_curve.pore_size_uniformity,
                                      soil_hydrology(model).hydraulic_conductivity.pore_size_uniformity)

matching_head = 1.0
van_genuchten_saturation(α, n, ψ) = (1 + (α * ψ)^n)^(-(1 - 1 / n))
matched_air_entry(n, 𝒮★) = (𝒮★^(-1 / (1 - 1 / n)) - 1)^(1 / n) / matching_head
saturation_at_matching_head = van_genuchten_saturation.(static.inverse_air_entry_head, static.pore_size_uniformity, matching_head)

function with_calibration(cal)
    return function (model)
        haskey(cal, "q") && set_cells!(conductivity_field(model), exp10.(cal["q"]), exp10(median(cal["q"][land])))
        haskey(cal, "log_exchange_length") &&
            set_cells!(exchange_length_field(model), exp.(cal["log_exchange_length"]), exchange_length)
        haskey(cal, "log_deep_suction") &&
            set_cells!(deep_pressure_head_field(model), -exp.(cal["log_deep_suction"]), -exp(median(cal["log_deep_suction"][land])))
        haskey(cal, "log_thickness") &&
            set_cells!(thickness_field(model), exp.(cal["log_thickness"]), deep_store_thickness)
        q_deep = get(cal, "q_deep", haskey(cal, "log_thickness") ? cal["q"] : nothing)   # stores calibrated before K₀ᵈ existed drained at the slab's K₀
        isnothing(q_deep) || set_cells!(deep_conductivity_field(model), exp10.(q_deep), exp10(median(q_deep[land])))
        if haskey(cal, "log_n_minus_1")
            n = 1 .+ exp.(cal["log_n_minus_1"])
            foreach(f -> set_cells!(f, n, median(n[land])), pore_size_uniformity_fields(model))
            α = matched_air_entry.(n, saturation_at_matching_head)
            set_cells!(air_entry_field(model), α, median(α[land]))
        end
        return nothing
    end
end
