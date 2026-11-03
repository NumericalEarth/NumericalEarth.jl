using Oceananigans.Grids: AbstractGrid, nodes
using Oceananigans.ImmersedBoundaries: underlying_grid
using Oceananigans.Operators: extrinsic_vector, intrinsic_vector
using Oceananigans.Utils: launch!, unparameterized_type

same_nodes(::Nothing, ::Nothing) = true
same_nodes(nodes₁, nodes₂) = size(nodes₁) == size(nodes₂) && nodes₁ ≈ nodes₂

same_horizontal_grid(grid₁, grid₂) = false

function same_horizontal_grid(grid₁::AbstractGrid, grid₂::AbstractGrid)
    underlying_grid(grid₂) isa unparameterized_type(typeof(underlying_grid(grid₁))) || return false
    topology(grid₁)[1:2] == topology(grid₂)[1:2] || return false
    x₁, y₁, _ = nodes(on_architecture(CPU(), grid₁), (Face(), Face(), Center()))
    x₂, y₂, _ = nodes(on_architecture(CPU(), grid₂), (Face(), Face(), Center()))
    return same_nodes(x₁, x₂) && same_nodes(y₁, y₂)
end

struct ConservativeExchange{F, T}
    from_component :: F
    to_component :: T
end

conservative_exchange(component_grid, exchange_grid) =
    throw(ArgumentError("coupling a component on a different horizontal grid requires `using ConservativeRegridding`"))

exchange_regridder(component_grid, exchange_grid) =
    same_horizontal_grid(component_grid, exchange_grid) ? nothing : conservative_exchange(component_grid, exchange_grid)

function regrid_surface_field! end

@kernel function _rotate_to_exchange_frame!(u, v, grid)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    uₑ, vₑ = intrinsic_vector(i, j, kᴺ, grid, u, v)
    @inbounds u[i, j, 1] = uₑ
    @inbounds v[i, j, 1] = vₑ
end

@kernel function _rotate_to_geographic_frame!(u, v, grid)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    uₑ, vₑ = extrinsic_vector(i, j, kᴺ, grid, u, v)
    @inbounds u[i, j, 1] = uₑ
    @inbounds v[i, j, 1] = vₑ
end

function rotate_to_exchange_frame!(u, v, grid)
    launch!(architecture(grid), grid, :xy, _rotate_to_exchange_frame!, u, v, grid)
    return nothing
end

function rotate_to_geographic_frame!(u, v, grid)
    launch!(architecture(grid), grid, :xy, _rotate_to_geographic_frame!, u, v, grid)
    return nothing
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
    surface_fields = surface_state(model; kw...)
    correction = materialize_correction(correction, exchange_grid, model)

    exchange = exchange_regridder(on_architecture(CPU(), Oceananigans.Grids.grid(model)), exchange_grid)
    isnothing(exchange) && return ComponentExchanger(surface_fields, nothing, correction)

    velocity_bcs = vector_component_boundary_conditions(exchange_grid, (Center(), Center(), nothing))
    state = merge(map(field -> Field{Center, Center, Nothing}(exchange_grid), surface_fields),
                  (; u = Field{Center, Center, Nothing}(exchange_grid; boundary_conditions = velocity_bcs),
                     v = Field{Center, Center, Nothing}(exchange_grid; boundary_conditions = velocity_bcs)))

    return ComponentExchanger(state, (; exchange, surface_state = surface_fields), correction)
end

regrid_surface_state!(::Nothing, state, exchange_grid) = nothing

function regrid_surface_state!(regridder, state, exchange_grid)
    from_component = regridder.exchange.from_component

    for name in keys(regridder.surface_state)
        regrid_surface_field!(state[name], from_component, regridder.surface_state[name])
    end

    rotate_to_exchange_frame!(state.u, state.v, exchange_grid)
    fill_halo_regions!((state.u, state.v))

    for name in keys(regridder.surface_state)
        name in (:u, :v) || fill_halo_regions!(state[name])
    end

    return nothing
end

function EarthSystemModels.interpolate_state!(exchanger, exchange_grid, component, coupled_model)
    isnothing(exchanger) && return nothing
    surface_fields = isnothing(exchanger.regridder) ? exchanger.state : exchanger.regridder.surface_state
    update_surface_state!(surface_fields, component_model(component))
    regrid_surface_state!(exchanger.regridder, exchanger.state, exchange_grid)
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

        all(on_exchange_grid, (land_exchanger, ocean_exchanger, sea_ice_exchanger)) ||
            throw(ArgumentError("land, ocean and sea ice must share the horizontal grid of the exchange grid"))

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

on_exchange_grid(::Nothing) = true
on_exchange_grid(exchanger) = isnothing(exchanger.regridder)

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
