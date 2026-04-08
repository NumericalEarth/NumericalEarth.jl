using Oceananigans.Operators: Δzᶜᶜᶜ

using Adapt

"""
    FluxAndRestoring(flux_field, restoring)

A boundary-condition condition (intended to be wrapped in a discrete-form
`FluxBoundaryCondition`) that combines two contributions at a tracer's top
boundary:

1. `flux_field`: a 2D `Field{Center, Center, Nothing}` that an external flux
   solver (e.g. the OMIP coupled atmosphere/sea-ice solver) writes into each
   step. This is read at `(i, j, 1)`.

2. `restoring`: a callable with signature `(i, j, k, grid, clock, fields)` that
   returns a tendency in the top cell — typically a `DatasetRestoring`,
   evaluating to `r * μ * (ψ_dataset - ψ)`. The tendency is converted to a
   surface flux by multiplying by `-Δz` at the top cell, consistent with the
   Oceananigans top-flux sign convention (top cell tendency contribution is
   `-J / Δz`).

This lets the coupled flux solver and a dataset restoring share the same top
boundary condition without one clobbering the other.
"""
struct FluxAndRestoring{F, R} <: Function
    flux_field :: F
    restoring  :: R
end

Adapt.adapt_structure(to, fr::FluxAndRestoring) =
    FluxAndRestoring(adapt(to, fr.flux_field),
                     adapt(to, fr.restoring))

@inline function (fr::FluxAndRestoring)(i, j, grid, clock, fields)
    Nz = grid.Nz
    @inbounds J = fr.flux_field[i, j, 1]

    # Restoring accessed as a tendency forcing (compatible with DatasetRestoring)
    G = fr.restoring(i, j, Nz, grid, clock, fields)

    # Top BC convention: tendency contribution = -J / Δz, so to inject
    # `G` in the top cell the flux is `-G * Δz`.
    Δz = Δzᶜᶜᶜ(i, j, Nz, grid)
    return J - G * Δz
end
