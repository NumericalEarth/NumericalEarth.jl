# # Coupled energy and freshwater conservation
#
# In this example, we run a minimal-physics `OceanSeaIceModel` through a freeze-then-melt cycle and verify
# that the coupled energy budget and the coupled freshwater budget both close. The setup is a single ocean
# column with thermodynamics-only sea ice, a snow layer on top of the ice, and a uniform prescribed atmosphere.
# We drive two phases: a cold phase with light snowfall that grows the ice, and a warm phase with rainfall
# that melts it.
#
# The two invariants we check at the end of the run are
#
# ```math
# ΔE = \int 𝒬 \, \mathrm{d}t, \qquad ΔM = \int \dot{M} \, \mathrm{d}t
# ```
#
# where ``E = ℋᵒᶜ + E_{is}`` is the total stored energy — ``ℋᵒᶜ`` the ocean heat content and ``E_{is}`` the
# combined ice and snow stored latent energy — and ``𝒬`` is the atmospheric heat flux into the coupled system.
# Likewise ``M = M^{fw} + M_{is}`` is the total stored freshwater — ``M^{fw}`` the ocean freshwater content and
# ``M_{is}`` the ice and snow mass — and ``\dot{M}`` is the atmospheric freshwater flux (precipitation minus
# evaporation) into the coupled system. Closure to machine precision requires that every internal flux cancels
# exactly between the components.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, ClimaSeaIce, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators: Azᶜᶜᶜ

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: latent_heat

using NumericalEarth
using NumericalEarth.Diagnostics: atmosphere_ocean_heat_flux, frazil_heat_flux
using NumericalEarth.EarthSystemModels: update_state!

using CairoMakie
using Printf

# ## Constant latent heat for diagnostic closure
#
# `ClimaSeaIce`'s slab mass balance uses a temperature-dependent latent heat, `ℒ(T) = ℒ₀ + (ρℓ * cℓ / ρi − ci)(T − T₀)`,
# with `ℰu = ρi * ℒ(T_u)` at the top interface and `ℰb = ρi * ℒ(T_b)` at the bottom. A single state-based
# `Eᵢₛ = − ℵ * ρi * ℒ * h * Az` cannot simultaneously match freeze at `T_b` and top-melt at 0 ᵒC: the 4.7 kJ/kg gap accounts
# for a ~1% residual scaling with top-melt mass. To isolate coupler-side bookkeeping from this intrinsic `ℒ(T)` mismatch we
# locally override `latent_heat` to the constant `pt.reference_latent_heat`. This is a diagnostic choice for the present
# example and does not modify upstream.

@inline ClimaSeaIce.SeaIceThermodynamics.latent_heat(pt::ClimaSeaIce.SeaIceThermodynamics.PhaseTransitions, T) =
    pt.reference_latent_heat

# ## Grid, ocean, sea ice, atmosphere, and radiation
#
# We build a single ocean column, 100 m deep, with 10 vertical levels on a `(Flat, Flat, Bounded)` `RectilinearGrid`.
# The ocean is initialized just above freezing at `S = 34`, without advection nor Coriolis and with
# `CATKEVerticalDiffusivity` providing vertical mixing.

arch = CPU()
grid = RectilinearGrid(arch; size = 10, z = (-100, 0), topology = (Flat, Flat, Bounded))

ocean = ocean_simulation(grid;
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         coriolis = nothing,
                         radiative_forcing = nothing,
                         closure = CATKEVerticalDiffusivity(),
                         bottom_drag_coefficient = 0)

Sᵢ = 34.0
Tᵢ = -1.5
set!(ocean.model, T = Tᵢ, S = Sᵢ)

# Sea ice only includes thermodynamics and is initialized with `h = 1 m`, `ℵ = 1`, and a `0.1 m` snow layer.
# We take fresh ice (`ice_salinity = 0`) so that the ice mass is entirely freshwater and the freshwater budget
# below tracks a single unambiguous stored quantity.

sea_ice = sea_ice_simulation(grid, ocean;
                             dynamics     = nothing,
                             advection    = nothing,
                             ice_salinity = 0)

set!(sea_ice.model, h = 1, ℵ = 1, hs = 0.1)

# The atmosphere and radiation are prescribed and spatially uniform and they both live on a scalar grid.
# We overwrite the `parent` array of their variables in place at the start of each phase.

surface_grid = slice(grid, :, :, 1)
times = [0.0, 1e9]
atmosphere = PrescribedAtmosphere(surface_grid, times)
radiation = PrescribedRadiation(surface_grid, times)

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

# ## Helpers
#
# `set_forcing!` fills every `FieldTimeSeries` of the prescribed atmosphere and
# prescribed radiation with scalar constants so the forcing is
# spatio-temporally uniform.

function set_forcing!(atmosphere, radiation, T, q, u, v, p, ℐꜜˢʷ, ℐꜜˡʷ, Jᶜ, Jˢⁿ)
    fill!(parent(atmosphere.temperature),          T   )
    fill!(parent(atmosphere.specific_humidity),    q   )
    fill!(parent(atmosphere.velocities.u),         u   )
    fill!(parent(atmosphere.velocities.v),         v   )
    fill!(parent(atmosphere.pressure),             p   )
    fill!(parent(atmosphere.precipitation_flux.rain), Jᶜ  )
    fill!(parent(atmosphere.precipitation_flux.snow), Jˢⁿ )
    fill!(parent(radiation.downwelling_shortwave), ℐꜜˢʷ)
    fill!(parent(radiation.downwelling_longwave),  ℐꜜˡʷ)
    return nothing
end

# Physical constants are read directly from the coupled model so the budget diagnostics stay
# consistent with the constants the model itself is using. `Az` is the horizontal cell area
# (unity for `(Flat, Flat, Bounded)`).

ρi  = coupled_model.sea_ice.model.sea_ice_density[1, 1, 1]
ρs  = coupled_model.sea_ice.model.snow_density[1, 1, 1]
ℒ₀  = ClimaSeaIce.SeaIceThermodynamics.latent_heat(coupled_model.sea_ice.model.phase_transitions, 0.0)
ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
cᵒᶜ = coupled_model.interfaces.ocean_properties.heat_capacity

Az = Azᶜᶜᶜ(1, 1, 1, grid)

# Volume integral of the ocean temperature, built once and re-used every step via `compute!`. The underlying `Integral` operation
# keeps a reference to the live `T` field, so each `compute!` re-evaluates over the current state.

∫T = Field(Integral(ocean.model.tracers.T))

# `column_state` returns a snapshot of the diagnostic quantities we track: ice and snow geometry, the ice+snow
# stored latent energy, the ice+snow mass, and the ocean heat content.

function column_state(coupled_model)
    h  = first(coupled_model.sea_ice.model.ice_thickness)
    ℵ  = first(coupled_model.sea_ice.model.ice_concentration)
    hs = first(coupled_model.sea_ice.model.snow_thickness)

    Eᵢₛ = -ℵ * (ρi * ℒ₀ * h + ρs * ℒ₀ * hs) * Az
    Mᵢₛ =  ℵ * (ρi *      h + ρs *      hs) * Az
    ℋᵒᶜ = ρᵒᶜ * cᵒᶜ * first(compute!(∫T))

    return (; h, ℵ, hs, Eᵢₛ, Mᵢₛ, ℋᵒᶜ)
end

# `net_top_heat_flux` returns the atmospheric energy input to the coupled (ice + ocean) system in Watts:
# `Q_atm = − (ΣQt + ΣQao) * Az`, where `ΣQt` is the sea-ice top heat flux per cell and `ΣQao` is the per-cell
# atmosphere-to-ocean flux over the open-water fraction. The ocean-side piece comes from `atmosphere_ocean_heat_flux`,
# which subtracts the frazil and interface contributions internally so it never picks up a spurious ocean / ice exchange term.

function net_top_heat_flux(coupled_model)
    ΣQt  = first(coupled_model.interfaces.net_fluxes.sea_ice.top.heat)
    ΣQao = first(atmosphere_ocean_heat_flux(coupled_model))
    return - (ΣQt + ΣQao) * Az
end

# `net_freshwater_flux` returns the atmospheric freshwater input to the coupled system in kg s⁻¹:
# rain and snow fall in at rates `Jᶜ` and `Jˢⁿ` (kg m⁻² s⁻¹), while evaporation removes water over the
# open-water fraction `(1 - ℵ)` at the rate given by the atmosphere-ocean water-vapor flux `Jᵛ`. The snowfall
# intercepted by the ice cancels between the ice gain and the ocean loss, so it does not appear here.

function net_freshwater_flux(coupled_model, Jᶜ, Jˢⁿ)
    ℵ  = first(coupled_model.sea_ice.model.ice_concentration)
    Jᵛ = first(coupled_model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor)
    return (Jᶜ + Jˢⁿ - (1 - ℵ) * Jᵛ) * Az
end

# `freshwater_state` reads the two internal exchanges the freshwater budget needs. `Jʷ` is the
# ocean freshwater volume flux (m s⁻¹, positive adds volume) that the coupler assembles at the surface, and
# `∂ₜM` is the rate at which sea-ice thermodynamics change the ice+snow mass. On this fixed-volume grid `Jʷ`
# is diagnostic, but integrating it recovers the ocean freshwater content exchanged with the ice and atmosphere.

function freshwater_state(coupled_model)
    Jʷ = first(coupled_model.interfaces.net_fluxes.ocean.η)
    mass_fluxes = coupled_model.sea_ice.model.thermodynamic_mass_fluxes
    ∂ₜM = first(mass_fluxes.ice) + first(mass_fluxes.snow) + first(mass_fluxes.intercepted_snowfall)
    return (; Jʷ, ∂ₜM)
end

# ## Running the freeze-melt cycle
#
# We run two 40-day phases at `Δt = 20 min`: a cold phase with light snowfall that grows the ice, then a warm phase with
# rain and strong radiation that melts it back. The run is driven by a single `Simulation` spanning the full cycle.
# Two callbacks do the bookkeeping: `budget_callback` records state at every step, and `phase_switch_callback` swaps the
# atmosphere from freeze to melt at `t = Δτ`.

Δt = 20minutes
Δτ = 40days
simulation = Simulation(coupled_model; Δt, stop_time = 2Δτ)

freeze_phase = (T    = 253.15,
                q    = 1.0e-4,
                u    = 2.0,
                v    = 0.0,
                p    = 101325.0,
                ℐꜜˢʷ = 50.0,
                ℐꜜˡʷ = 180.0,
                Jᶜ   = 0.0,
                Jˢⁿ  = 1.0e-5)

melt_phase   = (T    = 278.15,
                q    = 5.0e-3,
                u    = 2.0,
                v    = 0.0,
                p    = 101325.0,
                ℐꜜˢʷ = 250.0,
                ℐꜜˡʷ = 320.0,
                Jᶜ   = 5.0e-6,
                Jˢⁿ  = 0.0)

# We keep a history of the budget-relevant quantities at every time step.

history = (t     = Float64[],
           phase = Int[],
           h     = Float64[],
           ℵ     = Float64[],
           hs    = Float64[],
           Eᵢₛ   = Float64[],
           Mᵢₛ   = Float64[],
           ℋᵒᶜ   = Float64[],
           𝒬     = Float64[],
           𝒬ᶠʳᶻ  = Float64[],
           Ṁ     = Float64[],
           Jʷ    = Float64[],
           ∂ₜM   = Float64[])

function record!(history, coupled_model, phase_id, 𝒬, Ṁ)
    st = column_state(coupled_model)
    fw = freshwater_state(coupled_model)
    𝒬f = first(frazil_heat_flux(coupled_model))
    push!(history.t,     coupled_model.clock.time)
    push!(history.phase, phase_id)
    push!(history.h,     st.h)
    push!(history.ℵ,     st.ℵ)
    push!(history.hs,    st.hs)
    push!(history.Eᵢₛ,   st.Eᵢₛ)
    push!(history.Mᵢₛ,   st.Mᵢₛ)
    push!(history.ℋᵒᶜ,   st.ℋᵒᶜ)
    push!(history.𝒬,     𝒬)
    push!(history.𝒬ᶠʳᶻ,  𝒬f)
    push!(history.Ṁ,     Ṁ)
    push!(history.Jʷ,    fw.Jʷ)
    push!(history.∂ₜM,   fw.∂ₜM)
    return nothing
end

# The `budget_callback` reads the current `phase_ctx` — a small mutable box holding the current phase id, the
# snowfall enthalpy `𝒬ᵖ` to add to the net atmospheric heat flux, and the phase's rain and snowfall rates that
# set the atmospheric freshwater input. `phase_switch_callback` is the one that updates this context at the
# phase boundary.

phase_ctx = Ref((; phase_id = 1,
                   𝒬ᵖ  = - freeze_phase.Jˢⁿ * ℒ₀ * Az,
                   Jᶜ  = freeze_phase.Jᶜ,
                   Jˢⁿ = freeze_phase.Jˢⁿ))

function budget_callback(simulation)
    ctx = phase_ctx[]
    𝒬   = net_top_heat_flux(simulation.model)  + ctx.𝒬ᵖ
    Ṁ   = net_freshwater_flux(simulation.model, ctx.Jᶜ, ctx.Jˢⁿ)
    record!(history, simulation.model, ctx.phase_id, 𝒬, Ṁ)
    return nothing
end

# At `t = Δτ` the atmosphere swaps from freeze to melt. `update_state!` would zero the pending frazil flux
# (the ocean is at `Tₘ`, no supercooling), stranding the latent energy already deposited into the ocean by
# the last freeze step's frazil mutation. We preserve `𝒬ᶠʳᶻ` across the refresh and add it back into the
# sea-ice bottom heat flux that the slab will read. The callback also overwrites the just-recorded `Q` entry
# with the melt-phase starting flux, which is the flux that will drive the next step under rectangle-at-start
# integration.
#
# Oceananigans fires every scheduled callback once at initialization to sync its schedule, so we guard against
# the `t = 0` fire — we only want to switch at the actual phase boundary.

function phase_switch_callback(simulation)
    simulation.model.clock.time < Δτ && return nothing

    set_forcing!(atmosphere, radiation, melt_phase.T, melt_phase.q, melt_phase.u, melt_phase.v,
                 melt_phase.p, melt_phase.ℐꜜˢʷ, melt_phase.ℐꜜˡʷ, melt_phase.Jᶜ, melt_phase.Jˢⁿ)

    𝒬ᵖ   = - melt_phase.Jˢⁿ * ℒ₀ * Az
    𝒬ᶠʳᶻ = simulation.model.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
    ΣQb  = simulation.model.interfaces.net_fluxes.sea_ice.bottom.heat
    𝒬⁻   = first(𝒬ᶠʳᶻ)
    update_state!(simulation.model)
    interior(𝒬ᶠʳᶻ, 1, 1, 1)  .= 𝒬⁻
    interior(ΣQb,  1, 1, 1) .+= 𝒬⁻

    fw = freshwater_state(simulation.model)

    phase_ctx[]     = (; phase_id = 2, 𝒬ᵖ, Jᶜ = melt_phase.Jᶜ, Jˢⁿ = melt_phase.Jˢⁿ)
    history.𝒬[end]  = net_top_heat_flux(simulation.model)  + 𝒬ᵖ
    history.Ṁ[end]  = net_freshwater_flux(simulation.model, melt_phase.Jᶜ, melt_phase.Jˢⁿ)
    history.Jʷ[end] = fw.Jʷ
    return nothing
end

add_callback!(simulation, budget_callback,       IterationInterval(1))
add_callback!(simulation, phase_switch_callback, SpecifiedTimes([Δτ]))

# Initialize the coupler for the freeze atmosphere. Oceananigans' `run!` will re-run `update_state!` at init
# and then fire `budget_callback` once, which serves as the `t = 0` seed entry for the history.

set_forcing!(atmosphere, radiation, freeze_phase.T, freeze_phase.q, freeze_phase.u, freeze_phase.v,
             freeze_phase.p, freeze_phase.ℐꜜˢʷ, freeze_phase.ℐꜜˡʷ, freeze_phase.Jᶜ, freeze_phase.Jˢⁿ)

@info "Running $(simulation.stop_time / days)-day coupled freeze/melt cycle…"
run!(simulation)

# ## Budget analysis
#
# Cumulative atmospheric input uses rectangle-at-start integration to match what the coupler actually applied
# during each step.

t = history.t
τ = t ./ day

∫𝒬 = similar(t)
∫𝒬[1] = 0.0
for n in 2:length(t)
    ∫𝒬[n] = ∫𝒬[n-1] + history.𝒬[n-1] * (t[n] - t[n-1])
end

# The frazil mass gain is deposited by `compute_sea_ice_ocean_fluxes!` at the end of step `n`
# (mutating ocean `T` and writing `𝒬ᶠʳᶻ`) but the corresponding ice mass gain is consumed only during
# step `n + 1`. At a diagnostic snapshot the ocean shows the warming while the ice has not yet grown.
# We anticipate this one-step pending quantity by adding `𝒬ᶠʳᶻ(n) * Δt⁺ * Az`, where `Δt⁺ = t[n+1] - t[n]`,
# to `Eᵢₛ[n]` so the energy budget closure is not polluted by bookkeeping lag.

Δt⁺ = similar(t)
for n in 1:(length(t) - 1)
    Δt⁺[n] = t[n+1] - t[n]
end
Δt⁺[end] = Δt⁺[end-1]
δE = history.𝒬ᶠʳᶻ .* Δt⁺ .* Az

Ẽᵢₛ = history.Eᵢₛ .+ δE
ΔE  = (Ẽᵢₛ .+ history.ℋᵒᶜ) .- (Ẽᵢₛ[1] + history.ℋᵒᶜ[1])
R   = ΔE .- ∫𝒬
nothing #hide

# The freshwater budget mirrors the energy budget. The ocean freshwater content is recovered by integrating the
# surface freshwater volume flux, `Mᶠʷ(n) = Σ ρᵒᶜ * Az * Jʷ * Δt`, and the ice+snow mass `Mᵢₛ` is read directly
# from the state. The same one-step bookkeeping lag applies: the coupler assembles the ocean freshwater flux at
# the end of step `n` from the sea-ice mass change of that step, but the ocean receives it only during step
# `n + 1`. We anticipate it by rolling the ice+snow mass back by `∂ₜM(n) * Δt⁺ * Az` so the two sides stay in
# step.

∫Ṁ = similar(t)
∫Ṁ[1] = 0.0
for n in 2:length(t)
    ∫Ṁ[n] = ∫Ṁ[n-1] + history.Ṁ[n-1] * (t[n] - t[n-1])
end

Mᶠʷ = similar(t)
Mᶠʷ[1] = 0.0
for n in 2:length(t)
    Mᶠʷ[n] = Mᶠʷ[n-1] + ρᵒᶜ * Az * history.Jʷ[n-1] * (t[n] - t[n-1])
end

δM  = history.∂ₜM .* Δt⁺ .* Az
M̃ᵢₛ = history.Mᵢₛ .- δM
ΔM  = (M̃ᵢₛ .+ Mᶠʷ) .- (M̃ᵢₛ[1] + Mᶠʷ[1])
Rₘ  = ΔM .- ∫Ṁ
nothing #hide

# ## Visualizing the budgets
#
# Each figure shows the two stored components, the cumulative match between the stored change and the integrated
# atmospheric flux, and the residual on both absolute and relative log scales. The first figure is the energy
# budget, the second the freshwater budget.

set_theme!(Theme(fontsize=16, linewidth=2))

fig = Figure(size=(1100, 900))
ax1 = Axis(fig[1, 1], ylabel = "Ocean heat content (J)",  title = "Ocean heat content")
ax2 = Axis(fig[2, 1], ylabel = "Ice + snow stored E (J)", title = "Ice + snow stored latent energy")
ax3 = Axis(fig[3, 1], ylabel = "Cumulative (J)",          title = "ΔE vs. ∫𝒬 dt")
ax4 = Axis(fig[4, 1], ylabel = "Residual (J)",            title = "Energy residual = ΔE − ∫𝒬 dt")
ax5 = Axis(fig[5, 1], ylabel = "log₁₀|rel residual|",     title = "Relative energy residual", xlabel = "Time (days)")

lines!(ax1, τ, history.ℋᵒᶜ, color = :royalblue)

lines!(ax2, τ, history.Eᵢₛ, color = :orange)

lines!(ax3, τ, ΔE, label = "ΔE", color = :black)
lines!(ax3, τ, ∫𝒬, label = "∫𝒬 dt", color = :crimson, linestyle = :dash)
axislegend(ax3, position = :lt)

lines!(ax4, τ, R, color = :seagreen)
hlines!(ax4, [0], color = :gray, linestyle = :dot)

ε = log10.(abs.(R ./ max(maximum(abs.(ΔE)), 1)))
lines!(ax5, τ[2:end], ε[2:end], color = :seagreen)

for ax in (ax1, ax2, ax3, ax4, ax5)
    vlines!(ax, [Δτ / day], color = :gray, linestyle = :dot, linewidth = 1)
end

save("coupled_conservation_energy.png", fig)
nothing #hide

# ![](coupled_conservation_energy.png)

# The freshwater budget uses the same layout: ocean freshwater content, ice+snow mass, the cumulative match
# between `ΔM` and `∫Ṁ dt`, and the residual on absolute and relative log scales.

figₘ = Figure(size=(1100, 900))
axₘ1 = Axis(figₘ[1, 1], ylabel = "Ocean freshwater (kg)", title = "Ocean freshwater content")
axₘ2 = Axis(figₘ[2, 1], ylabel = "Ice + snow mass (kg)",  title = "Ice + snow mass")
axₘ3 = Axis(figₘ[3, 1], ylabel = "Cumulative (kg)",       title = "ΔM vs. ∫Ṁ dt")
axₘ4 = Axis(figₘ[4, 1], ylabel = "Residual (kg)",         title = "Freshwater residual = ΔM − ∫Ṁ dt")
axₘ5 = Axis(figₘ[5, 1], ylabel = "log₁₀|rel residual|",   title = "Relative freshwater residual", xlabel = "Time (days)")

lines!(axₘ1, τ, Mᶠʷ, color = :royalblue)

lines!(axₘ2, τ, history.Mᵢₛ, color = :orange)

lines!(axₘ3, τ, ΔM, label = "ΔM", color = :black)
lines!(axₘ3, τ, ∫Ṁ, label = "∫Ṁ dt", color = :crimson, linestyle = :dash)
axislegend(axₘ3, position = :lt)

lines!(axₘ4, τ, Rₘ, color = :seagreen)
hlines!(axₘ4, [0], color = :gray, linestyle = :dot)

εₘ = log10.(abs.(Rₘ ./ max(maximum(abs.(ΔM)), 1)))
lines!(axₘ5, τ[2:end], εₘ[2:end], color = :seagreen)

for ax in (axₘ1, axₘ2, axₘ3, axₘ4, axₘ5)
    vlines!(ax, [Δτ / day], color = :gray, linestyle = :dot, linewidth = 1)
end

save("coupled_conservation_freshwater.png", figₘ)
nothing #hide

# ![](coupled_conservation_freshwater.png)

# ## Per-phase summary

nᶠ = findlast(p -> p == 1, history.phase)

ΔEᶠ = Ẽᵢₛ[nᶠ]  + history.ℋᵒᶜ[nᶠ]  - Ẽᵢₛ[1]  - history.ℋᵒᶜ[1]
ΔEᵐ = Ẽᵢₛ[end] + history.ℋᵒᶜ[end] - Ẽᵢₛ[nᶠ] - history.ℋᵒᶜ[nᶠ]
∫𝒬ᶠ = ∫𝒬[nᶠ]
∫𝒬ᵐ = ∫𝒬[end] - ∫𝒬[nᶠ]

ΔMᶠ = M̃ᵢₛ[nᶠ]  + Mᶠʷ[nᶠ]  - M̃ᵢₛ[1]  - Mᶠʷ[1]
ΔMᵐ = M̃ᵢₛ[end] + Mᶠʷ[end] - M̃ᵢₛ[nᶠ] - Mᶠʷ[nᶠ]
∫Ṁᶠ = ∫Ṁ[nᶠ]
∫Ṁᵐ = ∫Ṁ[end] - ∫Ṁ[nᶠ]

@printf("  energy     freeze: ΔE = %+.3e J   ∫𝒬 dt = %+.3e J   residual = %+.2e (%.1e rel)\n",
        ΔEᶠ, ∫𝒬ᶠ, ΔEᶠ - ∫𝒬ᶠ, abs(ΔEᶠ - ∫𝒬ᶠ) / max(abs(ΔEᶠ), 1))
@printf("  energy     melt  : ΔE = %+.3e J   ∫𝒬 dt = %+.3e J   residual = %+.2e (%.1e rel)\n",
        ΔEᵐ, ∫𝒬ᵐ, ΔEᵐ - ∫𝒬ᵐ, abs(ΔEᵐ - ∫𝒬ᵐ) / max(abs(ΔEᵐ), 1))
@printf("  freshwater freeze: ΔM = %+.3e kg  ∫Ṁ dt = %+.3e kg  residual = %+.2e (%.1e rel)\n",
        ΔMᶠ, ∫Ṁᶠ, ΔMᶠ - ∫Ṁᶠ, abs(ΔMᶠ - ∫Ṁᶠ) / max(abs(ΔMᶠ), 1))
@printf("  freshwater melt  : ΔM = %+.3e kg  ∫Ṁ dt = %+.3e kg  residual = %+.2e (%.1e rel)\n",
        ΔMᵐ, ∫Ṁᵐ, ΔMᵐ - ∫Ṁᵐ, abs(ΔMᵐ - ∫Ṁᵐ) / max(abs(ΔMᵐ), 1))
@printf("  full-cycle relative residual: energy %.1e   freshwater %.1e\n",
        abs(R[end])  / max(maximum(abs.(ΔE)), 1),
        abs(Rₘ[end]) / max(maximum(abs.(ΔM)), 1))
nothing #hide
