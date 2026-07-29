#####
##### `AbstractHydrology` — slab-land hydrology closure interface.
#####
##### A concrete `AbstractHydrology` closes the surface water budget.
##### Like `AbstractEnergyBalance`, it declares its prognostic state and
##### incoming flux accumulators through `prognostic_variables` and
##### `flux_variables`. Beyond that, it must provide `saturation(...)`, the
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

flux_variables(::AbstractHydrology) = ()

initial_flux(::AbstractHydrology, ::Symbol, grid) = CenterField(grid)

"""
    prognostic_variables(hydrology::AbstractHydrology) -> Tuple{Vararg{Symbol}}

Names of *extra* prognostic fields this closure owns beyond the container's
hardcoded `temperature`/`water_storage` (e.g. a canopy water store). The container
sizes the `SlabLand.prognostic` `NamedTuple` from these and checkpoints them
generically. Default: none.
"""
prognostic_variables(::AbstractHydrology) = ()

"""
    initial_prognostic(hydrology::AbstractHydrology, name::Symbol, grid)

Build the initial prognostic `Field` for prognostic variable `name`. Defaults to
a zeroed `CenterField`.
"""
initial_prognostic(::AbstractHydrology, ::Symbol, grid) = CenterField(grid)

"""
    diagnostic_variables(hydrology::AbstractHydrology) -> Tuple{Vararg{Symbol}}

Names of closure-owned diagnostic fields published into `SlabLand.diagnostics`.
Default: no diagnostics.
"""
diagnostic_variables(::AbstractHydrology) = ()

"""
    initial_diagnostic(hydrology::AbstractHydrology, name::Symbol, grid)

Build the initial diagnostic `Field` for diagnostic variable `name`. Defaults
to a zeroed `CenterField`.
"""
initial_diagnostic(::AbstractHydrology, ::Symbol, grid) = CenterField(grid)

time_step!(::AbstractHydrology, land, Δt) = nothing
time_step!(hydrology::AbstractHydrology, land, Δt, time) = time_step!(hydrology, land, Δt)
update_diagnostics!(::AbstractHydrology, land) = nothing

#####
##### Atmosphere-facing accessor
#####

"""
    saturation(hydrology::AbstractHydrology, land) -> AbstractField

Return the moisture-availability factor `β ∈ [0, 1]` the atmosphere's
latent-heat BC reads. May be a `Field`, `ZeroField`, any `AbstractField`,
or a `Number`.
"""
function saturation end
