"""
    ComponentExchanger(component, exchange_grid)

Hold a regridder, a buffer of `state` fields, and an optional `correction` used
to bring data from a component (radiation, atmosphere, land, ocean, sea ice)
onto a shared `exchange_grid`, where atmosphere--ocean and atmosphere--sea-ice
fluxes are computed.

The optional `correction` is an in-place post-regrid hook applied to `state`
after each `interpolate_state!` (e.g. [`ElevationCorrection`](@ref) on the
atmosphere). When `correction === nothing`, the per-step `correct_state!` sweep
is a no-op for this component.
"""
struct ComponentExchanger{S, EX, C}
    state      :: S
    regridder  :: EX
    correction :: C
end

# Two-arg convenience: defaults the correction to nothing so existing per-component
# `ComponentExchanger(state, regridder)` returns work unchanged.
ComponentExchanger(state, regridder) = ComponentExchanger(state, regridder, nothing)

"""
    StateExchanger(grid, radiation, atmosphere, land, ocean, sea_ice;
                   atmosphere_correction = nothing)

Container for one `ComponentExchanger` per component. The `grid` is the shared
exchange grid onto which each component's state is regridded each time step.
Per-component post-regrid corrections live on each `ComponentExchanger` and run
as a sweep in phase 1.5 of the time step (see `correct_state!`).
"""
struct StateExchanger{G, R, A, L, O, S}
    grid       :: G
    radiation  :: R
    atmosphere :: A
    land       :: L
    ocean      :: O
    sea_ice    :: S

    function StateExchanger(grid, radiation, atmosphere, land, ocean, sea_ice;
                            atmosphere_correction = nothing)
        radiation_exchanger  = ComponentExchanger(radiation, grid)
        atmosphere_exchanger = ComponentExchanger(atmosphere, grid;
                                                  correction = atmosphere_correction)
        land_exchanger       = ComponentExchanger(land, grid)
        ocean_exchanger      = ComponentExchanger(ocean, grid)
        sea_ice_exchanger    = ComponentExchanger(sea_ice, grid)

        G = typeof(grid)
        R = typeof(radiation_exchanger)
        A = typeof(atmosphere_exchanger)
        L = typeof(land_exchanger)
        O = typeof(ocean_exchanger)
        S = typeof(sea_ice_exchanger)

        return new{G, R, A, L, O, S}(grid,
                                     radiation_exchanger,
                                     atmosphere_exchanger,
                                     land_exchanger,
                                     ocean_exchanger,
                                     sea_ice_exchanger)
    end
end

# For ``nothing'' components, we don't need an exchanger
ComponentExchanger(::Nothing, grid; kw...) = nothing

function Oceananigans.initialize!(exchanger::StateExchanger, model)
    initialize!(exchanger.radiation,  exchanger.grid, model.radiation)
    initialize!(exchanger.atmosphere, exchanger.grid, model.atmosphere)
    initialize!(exchanger.land,       exchanger.grid, model.land)
    initialize!(exchanger.ocean,      exchanger.grid, model.ocean)
    initialize!(exchanger.sea_ice,    exchanger.grid, model.sea_ice)
    return nothing
end

# fallback
Oceananigans.initialize!(::Nothing, grid, component) = nothing
Oceananigans.initialize!(exchanger::ComponentExchanger, grid, component) = nothing
