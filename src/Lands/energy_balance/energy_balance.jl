#####
##### `AbstractEnergyBalance` — slab-land energy-balance closure interface.
#####
##### A concrete `AbstractEnergyBalance` is responsible for advancing the
##### skin temperature (and any companion thermal state) of a `SlabLand`.
##### It declares which prognostic state variables and which incoming flux
##### accumulators it owns through `prognostic_variables` and
##### `flux_variables`. The container queries those at construction time
##### to size the shared `state` and `fluxes` `NamedTuple`s.
#####
##### Atmosphere-facing contract: `surface_temperature(energy, state)` is
##### the field the atmosphere reads as the surface BC. Simple slabs return
##### `state.T`; force-restore returns the skin temperature regardless of
##### deep-soil companion state; data-driven and prescribed variants return
##### whatever they hold or compute.
#####

abstract type AbstractEnergyBalance end

#####
##### Interface — per-closure overrides
#####

"""
    flux_variables(energy::AbstractEnergyBalance) -> Tuple{Vararg{Symbol}}

Names of flux/forcing accumulator fields this closure consumes from
`SlabLand.fluxes`. The coupler writes into these every time step.
"""
flux_variables(::AbstractEnergyBalance) = ()

"""
    initial_flux(energy::AbstractEnergyBalance, name::Symbol, grid)

Build the initial flux `Field` for flux variable `name`. Defaults to a
zeroed `CenterField`.
"""
initial_flux(::AbstractEnergyBalance, ::Symbol, grid) = CenterField(grid)

"""
    diagnostic_variables(energy::AbstractEnergyBalance) -> Tuple{Vararg{Symbol}}

Names of closure-owned diagnostic fields published into `SlabLand.diagnostics`.
Default: no diagnostics.
"""
diagnostic_variables(::AbstractEnergyBalance) = ()

"""
    initial_diagnostic(energy::AbstractEnergyBalance, name::Symbol, grid)

Build the initial diagnostic `Field` for diagnostic variable `name`. Defaults
to a zeroed `CenterField`.
"""
initial_diagnostic(::AbstractEnergyBalance, ::Symbol, grid) = CenterField(grid)

"""
    time_step!(energy, land, Δt[, time])

Advance the energy-balance state by `Δt`. The closure reads the land's
prognostic fields (`land.temperature`, `land.water_storage`) and the net
energy flux (`land.fluxes.net_energy_flux`) as needed. The default is a no-op
(used by closures with no prognostic state, e.g. prescribed surface temperature).
"""
time_step!(::AbstractEnergyBalance, land, Δt, time) = nothing
time_step!(energy::AbstractEnergyBalance, land, Δt) = time_step!(energy, land, Δt, zero(Δt))

"""
    update_diagnostics!(energy, land)

Refresh any cached diagnostics owned by the energy closure. Called by the
container's `update_state!` at the end of each step. Default is no-op.
"""
update_diagnostics!(::AbstractEnergyBalance, land) = nothing

#####
##### Atmosphere-facing accessor
#####

"""
    surface_temperature(energy::AbstractEnergyBalance, land) -> AbstractField

Return the field of surface temperatures the atmosphere reads as a BC.
"""
surface_temperature(::AbstractEnergyBalance, land) = land.temperature
