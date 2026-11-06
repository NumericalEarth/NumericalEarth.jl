#####
##### Ice shelf tracer flux forcing spread over the boundary layer
#####

using Oceananigans.Forcings: Forcing

# Spread the flux over the sensed boundary layer, weighted by cell thickness
@inline function ice_shelf_tracer_flux_forcing(i, j, k, grid, clock, model_fields, p)
    kt = @inbounds p.k_draft[i, j, 1]
    kt_safe = max(kt, 1)

    FT = eltype(grid)
    active = (kt > 0) & (k <= kt)

    total_weight = zero(FT)
    own_weight   = zero(FT)
    remaining    = p.boundary_layer_thickness

    if active
        for kk in kt_safe:-1:1
            immersed_cell(i, j, kk, grid) && break
            Δz = Δzᶜᶜᶜ(i, j, kk, grid)
            w = min(Δz, remaining)
            total_weight += w
            own_weight = ifelse(kk == k, w, own_weight)
            remaining -= w
            remaining > zero(FT) || break
        end
    end

    J      = @inbounds -p.flux[i, j, 1]
    Δz_own = Δzᶜᶜᶜ(i, j, k, grid)

    return ifelse(own_weight > zero(FT), J * own_weight / total_weight / Δz_own, zero(grid))
end

"""
$(TYPEDSIGNATURES)

Return a `NamedTuple` `(T = Forcing(...), S = Forcing(...))` that injects the
ice shelf melt/heat/salt fluxes computed by [`compute_ice_shelf_fluxes!`](@ref)
as an explicit tracer tendency (Losch, 2008). The tendency is spread, weighted by
cell thickness, across the same boundary layer (thickness
`interface.properties.boundary_layer_thickness`) that the ocean state is averaged
over (see [`boundary_layer_average`](@ref)), rather than concentrated in the topmost
active cell. This avoids an amplified tendency in thin `PartialCellCavity` cells and
a mismatch between the sensing and forcing layers in shallow water columns. Merge into a model's own `forcing`
`NamedTuple` before passing to the model constructor:

```jldoctest
using Oceananigans
using NumericalEarth

underlying_grid = RectilinearGrid(size = (2, 1, 4), x = (0, 2), y = (0, 1), z = (-1, 0),
                                  topology = (Bounded, Periodic, Bounded))
bottom(x, y)  = -1
ceiling(x, y) = x < 1 ? -0.5 : 0.0
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(bottom, ceiling))

interface = IceShelfOceanInterface(grid)
forcing = ice_shelf_tracer_forcing(interface)

keys(forcing)

# output
(:T, :S)
```
"""
function ice_shelf_tracer_forcing(interface::IceShelfOceanInterface)
    H_TBL = interface.properties.boundary_layer_thickness
    T_parameters = (k_draft = interface.k_draft, flux = interface.fluxes.temperature, boundary_layer_thickness = H_TBL)
    S_parameters = (k_draft = interface.k_draft, flux = interface.fluxes.salt, boundary_layer_thickness = H_TBL)

    return (T = Forcing(ice_shelf_tracer_flux_forcing; discrete_form = true, parameters = T_parameters),
            S = Forcing(ice_shelf_tracer_flux_forcing; discrete_form = true, parameters = S_parameters))
end
