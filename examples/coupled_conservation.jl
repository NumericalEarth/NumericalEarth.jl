# # Coupled conservation on a z-star grid
#
# In this example, we run a minimal-physics `OceanSeaIceModel` through a freeze-then-melt cycle on a
# free-surface-following (z-star) grid, and verify that three coupled budgets close to machine precision:
# volume, salt, and energy. The setup is a small doubly periodic ocean with thermodynamics-only sea ice,
# a snow layer on top of the ice, and a uniform prescribed atmosphere. We drive two phases: a cold phase with
# light snowfall that grows the ice, and a warm phase with rainfall that melts it.
#
# On a z-star grid the freshwater flux ``J^w`` forces the free surface, so the ocean genuinely gains and loses
# volume. Each budget is therefore a statement about quantities we can measure directly from the model state:
#
# ```math
# ΔV = \int \dot{M} / ρ^{oc} \, \mathrm{d}t, \qquad Δ𝒮 = 0, \qquad ΔE = \int (𝒬 + 𝒬^H) \, \mathrm{d}t
# ```
#
# Each budget is stored in two places, the ocean and the sea ice, which exchange it internally while the
# atmosphere supplies the rest. The stored volume ``V = V^{oc} + M^{si} / ρ^{oc}`` — the ocean water plus the
# volume that the ice and snow mass would occupy as ocean water — changes by the atmospheric freshwater input
# ``\dot{M}`` (precipitation minus evaporation). The total salt ``𝒮 = 𝒮^{oc} + 𝒮^{si}`` does not change at all:
# the atmosphere delivers only freshwater, so the ocean and the ice can only pass salt between them as the ice
# freezes and melts. Rain and meltwater dilute the ocean by growing its volume; that volume carries ``S^N J^w``
# back in and cancels the virtual salt flux, so they change salinity without creating or destroying salt.
# Finally, the stored energy ``E = ℋ^{oc} + E^{si}`` changes by the atmospheric heat flux ``𝒬`` together with
# the enthalpy ``𝒬^H`` that the freshwater carries across the surface with its volume. Closure to machine
# precision requires that every internal flux cancels exactly between the components.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, ClimaSeaIce, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Operators: Azᶜᶜᶜ, volume

using ClimaSeaIce

using NumericalEarth
using NumericalEarth.Diagnostics: atmosphere_ocean_heat_flux, frazil_heat_flux
using NumericalEarth.EarthSystemModels: update_state!

using CairoMakie
using Printf

# ## Constant latent heat for diagnostic closure
#
# The slab mass balance in ClimaSeaIce uses a temperature-dependent latent heat,
# `ℒ(T) = ℒ₀ + (ρˡ cˡ / ρⁱ − cⁱ)(T − T₀)`, evaluated at the bottom interface temperature when the ice freezes
# and at 0 °C when its top melts. A single state-based stored energy `Eˢⁱ = − ℵ ρⁱ ℒ h Az` cannot match both:
# the resulting 4.7 kJ kg⁻¹ gap leaves a ~1% residual that scales with the top-melt mass. To isolate the
# coupler's bookkeeping from this intrinsic mismatch, we override `latent_heat` with the constant
# `reference_latent_heat`. The override is a diagnostic choice confined to this example.

@inline ClimaSeaIce.SeaIceThermodynamics.latent_heat(pt::ClimaSeaIce.SeaIceThermodynamics.PhaseTransitions, T) =
    pt.reference_latent_heat

# ## Grid, ocean, sea ice, atmosphere, and radiation
#
# The ocean is 100 m deep with 10 levels on a doubly periodic `RectilinearGrid` whose vertical coordinate is a
# `MutableVerticalDiscretization`, so the levels stretch and compress with the free surface. A `SplitExplicitFreeSurface`
# lets the surface respond to the freshwater flux, and tracer advection carries the grid-motion term, which
# cancels against the surface-value exchange in the tracer boundary conditions. The domain is horizontally
# resolved because a single z-star column is degenerate. The ocean starts just above freezing at `S = 34`, with
# neither momentum advection nor Coriolis, and with `CATKEVerticalDiffusivity` providing vertical mixing.

arch = CPU()
Lx = Ly = 100kilometers

grid = RectilinearGrid(arch;
                       size     = (4, 4, 10),
                       halo     = (4, 4, 4),
                       x        = (0, Lx), y = (0, Ly),
                       z        = MutableVerticalDiscretization((-100, 0)),
                       topology = (Periodic, Periodic, Bounded))

ocean = ocean_simulation(grid;
                         momentum_advection = nothing,
                         free_surface = SplitExplicitFreeSurface(substeps=30),
                         coriolis = nothing,
                         radiative_forcing = nothing,
                         closure = CATKEVerticalDiffusivity(),
                         bottom_drag_coefficient = 0)

S₀ = 34.0
T₀ = -1.5
set!(ocean.model, T = T₀, S = S₀)

# The sea ice includes only thermodynamics and is initialized with `h = 1 m`, `ℵ = 1`, and a `0.1 m` snow layer.
# The ice keeps its default salinity: the budgets below track stored energy, mass, and salt, and they close
# whatever salt the ice carries, because the coupling routes every internal exchange through a single stream.

sea_ice = sea_ice_simulation(grid, ocean;
                             dynamics  = nothing,
                             advection = nothing)

set!(sea_ice.model, h = 1, ℵ = 1, hs = 0.1)

# The atmosphere and radiation are prescribed and spatially uniform, on a single-level grid that spans the same
# horizontal domain. At the start of each phase, we overwrite their variables in place.

atmosphere_grid = RectilinearGrid(arch;
                                  size     = (4, 4, 1),
                                  x        = (0, Lx), y = (0, Ly), z = (-1, 0),
                                  topology = (Periodic, Periodic, Bounded))

times = [0.0, 1e9]
atmosphere = PrescribedAtmosphere(atmosphere_grid, times)
radiation = PrescribedRadiation(atmosphere_grid, times)

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

# ## Helpers
#
# `set_forcing!` fills every `FieldTimeSeries` of the prescribed atmosphere and radiation with the constants
# of a `forcing` phase, so that the forcing is uniform in space and time.

function set_forcing!(atmosphere, radiation, forcing)
    fill!(parent(atmosphere.temperature),             forcing.T)
    fill!(parent(atmosphere.specific_humidity),       forcing.q)
    fill!(parent(atmosphere.velocities.u),            forcing.u)
    fill!(parent(atmosphere.velocities.v),            forcing.v)
    fill!(parent(atmosphere.pressure),                forcing.p)
    fill!(parent(atmosphere.precipitation_flux.rain), forcing.Jʳⁿ)
    fill!(parent(atmosphere.precipitation_flux.snow), forcing.Jˢⁿ)
    fill!(parent(radiation.downwelling_shortwave),    forcing.ℐꜜˢʷ)
    fill!(parent(radiation.downwelling_longwave),     forcing.ℐꜜˡʷ)
    return nothing
end

# We read the physical constants directly from the coupled model, so that the budget diagnostics use the same
# constants as the model itself. `Az` is the horizontal cell area and `A` the area of the whole domain.

ρⁱ  = sea_ice.model.sea_ice_density[1, 1, 1]
ρˢⁿ = sea_ice.model.snow_density[1, 1, 1]
Sˢⁱ = sea_ice.model.tracers.S[1, 1, 1]
ℒ₀  = sea_ice.model.phase_transitions.reference_latent_heat
ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
cᵒᶜ = coupled_model.interfaces.ocean_properties.heat_capacity

Az = Azᶜᶜᶜ(1, 1, 1, grid)
A  = Az * size(grid, 1) * size(grid, 2)

# The volume integrals of the ocean temperature and salinity, and the cell volume, are built once and
# re-evaluated at every step. They keep a reference to the live fields and grid, so each evaluation sees the
# current state, including the z-star cell volumes as the levels move.

∫T = Field(Integral(ocean.model.tracers.T))
∫S = Field(Integral(ocean.model.tracers.S))
cell_volume = KernelFunctionOperation{Center, Center, Center}(volume, grid, Center(), Center(), Center())

# `∫dA` integrates a surface field or operation over the domain. The fluxes are horizontally uniform here,
# but integrating keeps the diagnostics honest on a horizontally resolved domain.

∫dA(field) = sum(field) * Az

# `coupled_state` returns a snapshot of the quantities that the budgets track: the ice and snow geometry, the
# latent energy and mass stored in the ice and snow, the salt held by the ice, and the ocean volume, heat content,
# and salt content. We express the ice salt content in the ocean's `psu m³`, so that `𝒮ˢⁱ = ρⁱ h ℵ Az Sˢⁱ / ρᵒᶜ`
# and the ocean's `∫S dV` are directly comparable. Snow is fresh, so it stores no salt.

function coupled_state(coupled_model)
    h  = coupled_model.sea_ice.model.ice_thickness
    ℵ  = coupled_model.sea_ice.model.ice_concentration
    hs = coupled_model.sea_ice.model.snow_thickness

    Mˢⁱ = ∫dA(ℵ * (ρⁱ * h + ρˢⁿ * hs))
    Eˢⁱ = - ℒ₀ * Mˢⁱ
    𝒮ˢⁱ = ∫dA(ρⁱ * h * ℵ) * Sˢⁱ / ρᵒᶜ
    ℋᵒᶜ = ρᵒᶜ * cᵒᶜ * compute!(∫T)[1, 1, 1]
    𝒮ᵒᶜ = compute!(∫S)[1, 1, 1]
    Vᵒᶜ = sum(cell_volume)

    return (; h = h[1, 1, 1], ℵ = ℵ[1, 1, 1], hs = hs[1, 1, 1], Eˢⁱ, Mˢⁱ, 𝒮ˢⁱ, ℋᵒᶜ, 𝒮ᵒᶜ, Vᵒᶜ)
end

# `net_top_heat_flux` returns the atmospheric energy input to the coupled ice–ocean system in watts:
# `𝒬 = − ∫ (𝒬ᵃⁱ + 𝒬ᵃᵒ) dA − ℒ₀ Jˢⁿ A`, where `𝒬ᵃⁱ` is the sea-ice top heat flux, `𝒬ᵃᵒ` is the
# atmosphere–ocean heat flux over the open-water fraction, and the last term is the latent energy deficit of
# snow, which falls already frozen. `atmosphere_ocean_heat_flux` excludes the frazil and ice–ocean interface
# contributions, so `𝒬ᵃᵒ` holds no ice–ocean exchange.

function net_top_heat_flux(coupled_model, forcing)
    𝒬ᵃⁱ = ∫dA(coupled_model.interfaces.net_fluxes.sea_ice.top.heat)
    𝒬ᵃᵒ = ∫dA(atmosphere_ocean_heat_flux(coupled_model))
    return - (𝒬ᵃⁱ + 𝒬ᵃᵒ) - ℒ₀ * forcing.Jˢⁿ * A
end

# `net_freshwater_flux` returns the atmospheric freshwater input to the coupled system in kg s⁻¹:
# rain and snow fall at rates `Jʳⁿ` and `Jˢⁿ` (kg m⁻² s⁻¹), while evaporation removes water over the
# open-water fraction `1 - ℵ` at the rate of the atmosphere–ocean water vapor flux `Jᵛ`. The snowfall
# intercepted by the ice cancels between the ice gain and the ocean loss, so it does not appear here.

function net_freshwater_flux(coupled_model, forcing)
    ℵ  = coupled_model.sea_ice.model.ice_concentration
    Jᵛ = coupled_model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor
    return (forcing.Jʳⁿ + forcing.Jˢⁿ) * A - ∫dA((1 - ℵ) * Jᵛ)
end

# `flux_state` reads the internal exchanges that the budgets need: the ocean freshwater volume flux `Jʷ`
# (m³ s⁻¹, positive adds volume) that forces the free surface; the salt flux `Jˢ` carried by the sea ice
# (positive extracts salt from the ocean); the enthalpy `𝒬ᴴ` that the atmospheric freshwater brings in with
# its volume; and the rate `∂ₜM` at which sea-ice thermodynamics change the ice and snow mass.

function flux_state(coupled_model)
    ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean
    mass_fluxes = coupled_model.sea_ice.model.mass_fluxes
    Jʷ  = ∫dA(ocean_fluxes.η)
    Jˢ  = ∫dA(ocean_fluxes.S)
    𝒬ᴴ  = ρᵒᶜ * cᵒᶜ * ∫dA(ocean_fluxes.freshwater_heat_content)
    ∂ₜM = ∫dA(mass_fluxes.thermodynamics.ice) + ∫dA(mass_fluxes.thermodynamics.snow) +
          ∫dA(mass_fluxes.intercepted_snowfall)
    return (; Jʷ, Jˢ, 𝒬ᴴ, ∂ₜM)
end

# ## Running the freeze–melt cycle
#
# We run two 40-day phases with `Δt = 20 min`: a cold phase with light snowfall that grows the ice, then a warm
# phase with rain and strong radiation that melts it back. A single `Simulation` spans the full cycle, and two
# callbacks do the bookkeeping: `budget_callback` records the state at every step, and `phase_switch_callback`
# switches the atmosphere from freeze to melt at the end of the first phase.

Δt = 20minutes
phase_duration = 40days
simulation = Simulation(coupled_model; Δt, stop_time = 2phase_duration)

freeze_phase = (T    = 253.15,
                q    = 1.0e-4,
                u    = 2.0,
                v    = 0.0,
                p    = 101325.0,
                ℐꜜˢʷ = 50.0,
                ℐꜜˡʷ = 180.0,
                Jʳⁿ  = 0.0,
                Jˢⁿ  = 1.0e-5)

melt_phase   = (T    = 278.15,
                q    = 5.0e-3,
                u    = 2.0,
                v    = 0.0,
                p    = 101325.0,
                ℐꜜˢʷ = 250.0,
                ℐꜜˡʷ = 320.0,
                Jʳⁿ  = 5.0e-6,
                Jˢⁿ  = 0.0)

# We keep a history of the budget quantities at every time step.

history = (t     = Float64[],
           phase = Int[],
           h     = Float64[],
           ℵ     = Float64[],
           hs    = Float64[],
           Eˢⁱ   = Float64[],
           Mˢⁱ   = Float64[],
           𝒮ˢⁱ   = Float64[],
           ℋᵒᶜ   = Float64[],
           𝒮ᵒᶜ   = Float64[],
           Vᵒᶜ   = Float64[],
           𝒬     = Float64[],
           𝒬ᴴ    = Float64[],
           𝒬ᶠʳᶻ  = Float64[],
           Ṁ     = Float64[],
           Jʷ    = Float64[],
           Jˢ    = Float64[],
           ∂ₜM   = Float64[])

function record!(history, coupled_model, phase, forcing)
    state  = coupled_state(coupled_model)
    fluxes = flux_state(coupled_model)
    push!(history.t,     coupled_model.clock.time)
    push!(history.phase, phase)
    push!(history.h,     state.h)
    push!(history.ℵ,     state.ℵ)
    push!(history.hs,    state.hs)
    push!(history.Eˢⁱ,   state.Eˢⁱ)
    push!(history.Mˢⁱ,   state.Mˢⁱ)
    push!(history.𝒮ˢⁱ,   state.𝒮ˢⁱ)
    push!(history.ℋᵒᶜ,   state.ℋᵒᶜ)
    push!(history.𝒮ᵒᶜ,   state.𝒮ᵒᶜ)
    push!(history.Vᵒᶜ,   state.Vᵒᶜ)
    push!(history.𝒬,     net_top_heat_flux(coupled_model, forcing))
    push!(history.𝒬ᴴ,    fluxes.𝒬ᴴ)
    push!(history.𝒬ᶠʳᶻ,  ∫dA(frazil_heat_flux(coupled_model)))
    push!(history.Ṁ,     net_freshwater_flux(coupled_model, forcing))
    push!(history.Jʷ,    fluxes.Jʷ)
    push!(history.Jˢ,    fluxes.Jˢ)
    push!(history.∂ₜM,   fluxes.∂ₜM)
    return nothing
end

# `budget_callback` records the history under the current phase, which `phase_switch_callback` updates at the
# phase boundary.

current_phase = Ref((phase = 1, forcing = freeze_phase))

function budget_callback(simulation)
    (; phase, forcing) = current_phase[]
    record!(history, simulation.model, phase, forcing)
    return nothing
end

# At the end of the first phase the atmosphere switches from freeze to melt. The ocean sits at its freezing
# point, so `update_state!` would zero the pending frazil heat flux `𝒬ᶠʳᶻ` and strand the latent energy that the
# last freeze step already deposited into the ocean. We therefore preserve `𝒬ᶠʳᶻ` across the update and add it
# back into the sea-ice bottom heat flux `𝒬ⁱᵒ` that the slab reads. The callback also overwrites the flux entries
# just recorded with the melt-phase starting values, which drive the next step under rectangle-at-start
# integration.
#
# Oceananigans fires every scheduled callback once at initialization, so we skip the call at `t = 0`.

function phase_switch_callback(simulation)
    model = simulation.model
    model.clock.time < phase_duration && return nothing

    set_forcing!(atmosphere, radiation, melt_phase)

    𝒬ᶠʳᶻ = model.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
    𝒬ⁱᵒ  = model.interfaces.net_fluxes.sea_ice.bottom.heat
    pending_frazil_heat = similar(𝒬ᶠʳᶻ)
    set!(pending_frazil_heat, 𝒬ᶠʳᶻ)
    update_state!(model)
    set!(𝒬ᶠʳᶻ, pending_frazil_heat)
    set!(𝒬ⁱᵒ, 𝒬ⁱᵒ + pending_frazil_heat)

    fluxes = flux_state(model)

    current_phase[]  = (phase = 2, forcing = melt_phase)
    history.𝒬[end]  = net_top_heat_flux(model, melt_phase)
    history.Ṁ[end]  = net_freshwater_flux(model, melt_phase)
    history.𝒬ᴴ[end] = fluxes.𝒬ᴴ
    history.Jʷ[end] = fluxes.Jʷ
    history.Jˢ[end] = fluxes.Jˢ
    return nothing
end

add_callback!(simulation, budget_callback,       IterationInterval(1))
add_callback!(simulation, phase_switch_callback, SpecifiedTimes([phase_duration]))

# We start from the freeze-phase atmosphere. At initialization `run!` calls `update_state!` and then fires
# `budget_callback` once, which records the `t = 0` entry of the history.

set_forcing!(atmosphere, radiation, freeze_phase)
run!(simulation)

# ## Budget analysis
#
# Every cumulative integral uses rectangle-at-start integration, which is what the coupler actually applies:
# it assembles the fluxes at the end of step `n` and holds them fixed while the ocean takes step `n + 1`.
# `δt[n]` is the length of the step that follows record `n`.

t = history.t
elapsed_days = t ./ day

accumulate_rate(rate) = [0; cumsum(rate[1:end-1] .* diff(t))]

δt = [diff(t); last(diff(t))]
nothing #hide

# ### Volume
#
# The free surface is forced by `Jʷ`, so the ocean volume grows by exactly the freshwater it takes in. Both sides
# of this budget live entirely in the ocean, so it closes without any bookkeeping lag.

ΔVᵒᶜ = history.Vᵒᶜ .- history.Vᵒᶜ[1]
∫Jʷ  = accumulate_rate(history.Jʷ)
nothing #hide

# The coupled budget adds the ice and snow, expressed as the volume `Mˢⁱ / ρᵒᶜ` that their mass would occupy
# as ocean water, so that both stores share the ocean's units. The coupler assembles the ocean freshwater flux
# at the end of step `n` from the sea-ice mass change of that step, but the ocean receives it only during step
# `n + 1`. We account for this one-step lag by rolling the ice and snow mass back by `∂ₜM(n) δt(n)`, which keeps
# the two sides in step.

Vˢⁱ = history.Mˢⁱ ./ ρᵒᶜ
Ṽˢⁱ = (history.Mˢⁱ .- history.∂ₜM .* δt) ./ ρᵒᶜ
ΔV  = (history.Vᵒᶜ .+ Ṽˢⁱ) .- (history.Vᵒᶜ[1] + Ṽˢⁱ[1])
∫Ṁ  = accumulate_rate(history.Ṁ) ./ ρᵒᶜ
nothing #hide

# ### Salt
#
# Nothing puts salt into the coupled system, because the atmosphere delivers only freshwater. The ocean and the
# ice merely pass salt back and forth (freezing locks some away in the ice and melting returns it), so the total
# `𝒮 = 𝒮ᵒᶜ + 𝒮ˢⁱ` must not change at all. Rain and meltwater dilute the ocean by growing its volume: the volume
# they add carries `Sᴺ Jʷ` back in and cancels the virtual salt flux at every Runge–Kutta stage, so they change
# the ocean's salinity without creating or destroying salt.
#
# The ice salt content has already changed during the last step, while the ocean receives the matching flux only
# during the next one. The ice salt therefore carries the same one-step lag as the ice mass, and we roll it back
# by `Jˢ(n) δt(n)`.

𝒮̃ˢⁱ = history.𝒮ˢⁱ .- history.Jˢ .* δt
𝒮   = history.𝒮ᵒᶜ .+ 𝒮̃ˢⁱ
Δ𝒮  = 𝒮 .- 𝒮[1]
∫Jˢ = zero(t)   ## the atmosphere supplies no salt
nothing #hide

# ### Energy
#
# The coupler deposits the frazil heat at the end of step `n`, warming the ocean and writing `𝒬ᶠʳᶻ`, but the
# corresponding ice growth happens only during step `n + 1`. At a diagnostic snapshot, the ocean therefore shows
# the warming while the ice has not yet grown. We account for this pending growth by adding `𝒬ᶠʳᶻ(n) δt(n)` to
# `Eˢⁱ(n)`, so that the bookkeeping lag does not pollute the closure. The freshwater carries its own enthalpy
# `𝒬ᴴ` across the surface with its volume, so it is an energy input alongside the surface heat flux.

Ẽˢⁱ = history.Eˢⁱ .+ history.𝒬ᶠʳᶻ .* δt
ΔE  = (Ẽˢⁱ .+ history.ℋᵒᶜ) .- (Ẽˢⁱ[1] + history.ℋᵒᶜ[1])
∫𝒬  = accumulate_rate(history.𝒬) .+ accumulate_rate(history.𝒬ᴴ)
nothing #hide

# ## Visualizing the budgets
#
# Each column shows one budget. The top row plots the ocean and the ice and snow stores on the same axes as
# anomalies from their initial values (the raw values would bury the ice, which holds a thousand times less salt
# than the ocean), so the internal exchange shows up as two curves that mirror each other. The middle row is the
# closure itself: the change in storage against the flux that drove it. The bottom row shows the relative
# residual on a logarithmic scale.

set_theme!(Theme(fontsize=14, linewidth=2))

function budget_column!(fig, column, name, unit, stores, Δ, ∫F;
                        scale = maximum(abs, Δ), flux_label = "∫ atmospheric flux dt", legend_position = :lt)
    residual = Δ .- ∫F

    ax_stores = Axis(fig[1, column], title = "$name ($unit)", ylabel = "Store anomaly ($unit)")
    for ((label, data), color) in zip(stores, (:royalblue, :orange))
        lines!(ax_stores, elapsed_days, data .- data[1]; label, color)
    end
    axislegend(ax_stores, position = :lt, framevisible = false)

    ## the two curves overlap, so sparse markers keep the flux visible under the line
    ax_closure = Axis(fig[2, column], ylabel = "Cumulative ($unit)")
    marked = 1:(length(elapsed_days) ÷ 25):length(elapsed_days)
    lines!(ax_closure, elapsed_days, Δ, label = "Δ total", color = :black)
    scatter!(ax_closure, elapsed_days[marked], ∫F[marked], label = flux_label, color = :crimson, markersize = 10)
    axislegend(ax_closure, position = legend_position, framevisible = false)

    ax_residual = Axis(fig[3, column], ylabel = "log₁₀|relative residual|", xlabel = "Time (days)")
    ε = log10.(abs.(residual ./ max(scale, 1)))
    finite = isfinite.(ε)   ## a residual of exactly zero takes the log to -Inf
    lines!(ax_residual, elapsed_days[finite], ε[finite], color = :seagreen)

    for ax in (ax_stores, ax_closure, ax_residual)
        vlines!(ax, [phase_duration / day], color = :gray, linestyle = :dot, linewidth = 1)
    end

    return nothing
end

# The ocean takes in the freshwater that the ice gives up, and gives it back as the ice grows; the salt that the
# ice locks away is the salt that the ocean loses, and the total never changes; the ocean warms as the ice melts.
# In each case the two stores mirror each other, and what is left over is exactly what the atmosphere delivered.

fig = Figure(size=(1500, 780))

budget_column!(fig, 1, "Volume", "m³",
               ["Ocean" => history.Vᵒᶜ, "Ice + snow" => Vˢⁱ], ΔV, ∫Ṁ;
               flux_label = "∫ atmospheric freshwater dt")

budget_column!(fig, 2, "Salt", "psu m³",
               ["Ocean" => history.𝒮ᵒᶜ, "Ice" => history.𝒮ˢⁱ], Δ𝒮, ∫Jˢ;
               scale = maximum(abs, history.𝒮ᵒᶜ .- history.𝒮ᵒᶜ[1]),
               flux_label = "zero (the atmosphere supplies no salt)",
               legend_position = :rb)

budget_column!(fig, 3, "Energy", "J",
               ["Ocean" => history.ℋᵒᶜ, "Ice + snow" => history.Eˢⁱ], ΔE, ∫𝒬;
               flux_label = "∫ atmospheric heat dt")

save("coupled_conservation.png", fig)
nothing #hide

# ![](coupled_conservation.png)

# ## Summary by phase
#
# The first row checks the ocean alone, against the freshwater crossing its surface; the others check the
# coupled budgets.

last_freeze_record = findlast(==(1), history.phase)

function report(name, unit, Δ, ∫F; scale = maximum(abs, Δ))
    s = max(scale, 1)
    freeze_change = Δ[last_freeze_record] - Δ[1]
    melt_change   = Δ[end] - Δ[last_freeze_record]
    freeze_input  = ∫F[last_freeze_record]
    melt_input    = ∫F[end] - ∫F[last_freeze_record]
    @printf("  %-12s freeze: Δ = %+.3e %-6s ∫ dt = %+.3e %-6s residual = %+.2e (%.1e relative)\n",
            name, freeze_change, unit, freeze_input, unit, freeze_change - freeze_input,
            abs(freeze_change - freeze_input) / s)
    @printf("  %-12s melt  : Δ = %+.3e %-6s ∫ dt = %+.3e %-6s residual = %+.2e (%.1e relative)\n",
            name, melt_change, unit, melt_input, unit, melt_change - melt_input,
            abs(melt_change - melt_input) / s)
    @printf("  %-12s full-cycle relative residual: %.1e\n", name, abs(Δ[end] - ∫F[end]) / s)
    return nothing
end

report("ocean volume", "m³",     ΔVᵒᶜ, ∫Jʷ)
report("volume",       "m³",     ΔV,   ∫Ṁ)
report("salt",         "psu m³", Δ𝒮,   ∫Jˢ; scale = maximum(abs, history.𝒮ᵒᶜ .- history.𝒮ᵒᶜ[1]))
report("energy",       "J",      ΔE,   ∫𝒬)
nothing #hide
