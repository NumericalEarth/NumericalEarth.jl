"""
    ComponentExchanger(component, exchange_grid)

Holds a regridder and a buffer of `state` fields used to bring data from a
component (atmosphere, land, ocean, sea ice) onto a shared `exchange_grid`,
where atmosphere--ocean and atmosphere--sea-ice fluxes are computed.
"""
struct ComponentExchanger{S, EX}
    state :: S
    regridder :: EX
end

"""
    StateExchanger(grid, atmosphere, land, ocean, sea_ice)

Container for one `ComponentExchanger` per component. The `grid` is the shared
exchange grid onto which each component's state is regridded each time step.
"""
struct StateExchanger{G, A, L, O, S}
    grid :: G
    atmosphere :: A
    land :: L
    ocean :: O
    sea_ice :: S

    function StateExchanger(grid, atmosphere, land, ocean, sea_ice)
        atmosphere_exchanger = ComponentExchanger(atmosphere, grid)
        land_exchanger       = ComponentExchanger(land, grid)
        ocean_exchanger      = ComponentExchanger(ocean, grid)
        sea_ice_exchanger    = ComponentExchanger(sea_ice, grid)

        G = typeof(grid)
        A = typeof(atmosphere_exchanger)
        L = typeof(land_exchanger)
        O = typeof(ocean_exchanger)
        S = typeof(sea_ice_exchanger)

        return new{G, A, L, O, S}(grid,
                                  atmosphere_exchanger,
                                  land_exchanger,
                                  ocean_exchanger,
                                  sea_ice_exchanger)
    end
end

# For ``nothing'' components, we don't need an exchanger
ComponentExchanger(::Nothing, grid) = nothing

function initialize!(exchanger::StateExchanger, model)
    initialize!(exchanger.atmosphere, exchanger.grid, model.atmosphere)
    initialize!(exchanger.land,       exchanger.grid, model.land)
    initialize!(exchanger.ocean,      exchanger.grid, model.ocean)
    initialize!(exchanger.sea_ice,    exchanger.grid, model.sea_ice)
    return nothing
end

# fallback
initialize!(::Nothing, grid, component) = nothing
initialize!(exchanger::ComponentExchanger, grid, component) = nothing
