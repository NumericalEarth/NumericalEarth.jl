using Oceananigans.Grids: nodes

same_nodes(::Nothing, ::Nothing) = true
same_nodes(nodes₁, nodes₂) = size(nodes₁) == size(nodes₂) && nodes₁ ≈ nodes₂

function same_horizontal_grid(grid₁, grid₂)
    topology(grid₁)[1:2] == topology(grid₂)[1:2] || return false
    x₁, y₁, _ = nodes(on_architecture(CPU(), grid₁), (Face(), Face(), Center()))
    x₂, y₂, _ = nodes(on_architecture(CPU(), grid₂), (Face(), Face(), Center()))
    return same_nodes(x₁, x₂) && same_nodes(y₁, y₂)
end

function validate_exchange_grid(component_name, component_grid, exchange_grid)
    same_horizontal_grid(component_grid, exchange_grid) ||
        throw(ArgumentError("the $component_name must share the horizontal grid of the exchange grid"))
    return nothing
end

"""
    ComponentExchanger(component, exchange_grid)

Hold a regridder, a buffer of `state` fields, and an optional `correction` used
to bring data from a component (radiation, atmosphere, land, ocean, sea ice)
onto a shared `exchange_grid`, where atmosphere--ocean and atmosphere--sea-ice
fluxes are computed.

The optional `correction` is an in-place post-regrid hook applied to `state`
after each `interpolate_state!` (e.g. [`AltitudeCorrection`](@ref) on the
atmosphere). When `correction === nothing`, the per-step `correct_state!` sweep
is a no-op for this component.
"""
struct ComponentExchanger{S, EX, C}
    state      :: S
    regridder  :: EX
    correction :: C
end

ComponentExchanger(state::NamedTuple, regridder) = ComponentExchanger(state, regridder, nothing)

function ComponentExchanger(component, exchange_grid; correction = nothing, kw...)
    model = component_model(component)
    validate_exchange_grid(nameof(typeof(model)), Oceananigans.Grids.grid(model), exchange_grid)
    correction = materialize_correction(correction, exchange_grid, model)
    return ComponentExchanger(surface_state(model; kw...), nothing, correction)
end

function EarthSystemModels.interpolate_state!(exchanger, exchange_grid, component, coupled_model)
    isnothing(exchanger) && return nothing
    update_surface_state!(exchanger.state, component_model(component))
    return nothing
end

"""
    StateExchanger(grid, radiation, atmosphere, land, ocean, sea_ice;
                   atmosphere_correction = nothing,
                   land_accessors = (;))

Container for one `ComponentExchanger` per component. The `grid` is the shared
exchange grid onto which each component's state is regridded each time step.
The land exchanger publishes the land state returned by the `land_accessors`.
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
                            atmosphere_correction = nothing,
                            land_accessors = (;))
        radiation_exchanger  = ComponentExchanger(radiation, grid)
        atmosphere_exchanger = ComponentExchanger(atmosphere, grid;
                                                  correction = atmosphere_correction)
        land_exchanger       = ComponentExchanger(land, grid; state_accessors = land_accessors)
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
