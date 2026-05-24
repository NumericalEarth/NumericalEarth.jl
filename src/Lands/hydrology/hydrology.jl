#####
##### `AbstractHydrology` — slab-land hydrology closure interface.
#####
##### A concrete `AbstractHydrology` closes the surface water budget.
##### Like `AbstractEnergyBalance`, it declares its prognostic state and
##### incoming flux accumulators through `prognostic_variables` and
##### `flux_variables`. Beyond that, it must provide `wetness(...)`, the
##### moisture-availability factor `β ∈ [0, 1]` that the atmosphere's
##### latent-heat BC reads. β is the only consumer-facing diagnostic.
#####
##### Cross-axis state mutations (e.g. snow melt cooling the slab T) are
##### permitted: the closure receives the full `state` NamedTuple and can
##### write into other axes' fields when it represents a conservative
##### exchange of mass or energy. The "no closure-to-closure direct reads"
##### invariant of the design doc is preserved because the channel is the
##### shared `state`, not a direct reference to the energy closure.
#####

abstract type AbstractHydrology end

#####
##### Interface — per-closure overrides
#####

prognostic_variables(::AbstractHydrology) = ()
flux_variables(::AbstractHydrology)       = ()

initial_state(::AbstractHydrology, ::Symbol, grid)      = CenterField(grid)
initial_flux(::AbstractHydrology, ::Symbol, grid)       = CenterField(grid)

step!(::AbstractHydrology, state, fluxes, surface, grid, Δt) = nothing
update_diagnostics!(::AbstractHydrology, state, fluxes, surface, grid) = nothing

#####
##### Atmosphere-facing accessor
#####

"""
    wetness(hydrology::AbstractHydrology, state) -> AbstractField

Return the moisture-availability factor `β ∈ [0, 1]` the atmosphere's
latent-heat BC reads. May be a `Field`, `ZeroField`, any `AbstractField`,
or a `Number`.
"""
function wetness end
