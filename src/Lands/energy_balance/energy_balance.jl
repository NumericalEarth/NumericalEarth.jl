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
    prognostic_variables(energy::AbstractEnergyBalance) -> Tuple{Vararg{Symbol}}

Names of prognostic state variables this closure adds to `SlabLand.state`.
"""
prognostic_variables(::AbstractEnergyBalance) = ()

"""
    flux_variables(energy::AbstractEnergyBalance) -> Tuple{Vararg{Symbol}}

Names of flux/forcing accumulator fields this closure consumes from
`SlabLand.fluxes`. The coupler writes into these every time step.
"""
flux_variables(::AbstractEnergyBalance) = ()

"""
    initial_state(energy::AbstractEnergyBalance, name::Symbol, grid)

Build the initial `Field` for prognostic variable `name`. Override on a
closure to inject a non-zero default (e.g. fill with a reference T).
Defaults to a freshly allocated `CenterField`.
"""
initial_state(::AbstractEnergyBalance, ::Symbol, grid) = CenterField(grid)

"""
    initial_flux(energy::AbstractEnergyBalance, name::Symbol, grid)

Build the initial flux `Field` for flux variable `name`. Defaults to a
zeroed `CenterField`.
"""
initial_flux(::AbstractEnergyBalance, ::Symbol, grid) = CenterField(grid)

"""
step!(energy, state, fluxes, surface, grid, Δt)

Advance the energy-balance state by `Δt`. The closure may read other
closures' state through the `state` `NamedTuple` and shared surface
properties through `surface`, but it does not call other closures
directly. The default is a no-op (used by closures with no prognostic
state, e.g. prescribed surface temperature).
"""
step!(::AbstractEnergyBalance, state, fluxes, surface, grid, Δt) = nothing

"""
    update_diagnostics!(energy, state, fluxes, surface, grid)

Refresh any cached diagnostics owned by the energy closure. Called by the
container's `update_state!` at the end of each step. Default is no-op.
"""
update_diagnostics!(::AbstractEnergyBalance, state, fluxes, surface, grid) = nothing

#####
##### Atmosphere-facing accessor
#####

"""
    surface_temperature(energy::AbstractEnergyBalance, state) -> AbstractField

Return the field of skin temperatures the atmosphere reads as a BC.
"""
surface_temperature(::AbstractEnergyBalance, state) = state.T
