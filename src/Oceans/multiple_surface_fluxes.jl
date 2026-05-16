using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Operators: Δzᶜᶜᶜ

using Adapt

"""
    MultipleFluxes(flux_field, additional_fluxes)

A boundary-condition callable (intended to be wrapped in a discrete-form
`FluxBoundaryCondition`) that combines two contributions at a field's top
boundary:

1. `flux_field`: a 2D `Field` that an external flux solver (e.g. the OMIP
   coupled atmosphere/sea-ice solver) writes into each step.

2. `additional_fluxes`: any object compatible with `getbc` that returns a flux in the top cell

This lets the coupled flux solver and an additional flux share the same
top boundary condition without one clobbering the other.
"""
struct MultipleFluxes{F, A} <: Function
    flux_field        :: F
    additional_fluxes :: A
end

Adapt.adapt_structure(to, mf::MultipleFluxes) = MultipleFluxes(adapt(to, mf.flux_field), adapt(to, mf.additional_fluxes))

@inline function (mf::MultipleFluxes)(i, j, grid, clock, fields)
    @inbounds J = mf.flux_field[i, j, 1]
    G = getbc(mf.additional_fluxes, i, j, grid, clock, fields)
    return J + G
end
