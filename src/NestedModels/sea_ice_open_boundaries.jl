#####
##### Open lateral boundary conditions for the sea ice over a nested ocean, read from the parent's sea ice
#####

using ..DataWrangling: Metadata, bracketing_dates
using ..SeaIces: sea_ice_simulation
using Oceananigans.OutputReaders: Clamp, FieldTimeSeries

@inline zero_nan(x) = ifelse(isnan(x), zero(x), x)

"""
    SeaIceBoundarySeries{Dim}(series, index, along)

Boundary-condition value read from the parent's sea ice `series` (a `FieldTimeSeries` on the child grid) at the
boundary row (`Dim = 1`, at `i = index`) or column (`Dim = 2`, at `j = index`) of one open side. Open water,
missing in the parent, is no ice. The along-boundary index is clamped to `along`, as for a [`BoundaryTimeSeries`](@ref).
"""
struct SeaIceBoundarySeries{Dim, S}
    series :: S
    index  :: Int
    along  :: NTuple{2, Int}
end

SeaIceBoundarySeries{Dim}(series::S, index, along) where {Dim, S} = SeaIceBoundarySeries{Dim, S}(series, index, along)

Adapt.adapt_structure(to, b::SeaIceBoundarySeries{Dim}) where Dim =
    SeaIceBoundarySeries{Dim}(adapt(to, b.series), b.index, b.along)

Oceananigans.Architectures.on_architecture(to, b::SeaIceBoundarySeries{Dim}) where Dim =
    SeaIceBoundarySeries{Dim}(on_architecture(to, b.series), b.index, b.along)

@inline BoundaryConditions.getbc(b::SeaIceBoundarySeries{1}, j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) =
    @inbounds zero_nan(b.series[b.index, clamp(j, b.along...), 1, boundary_time(clock)])

@inline BoundaryConditions.getbc(b::SeaIceBoundarySeries{2}, i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) =
    @inbounds zero_nan(b.series[clamp(i, b.along...), b.index, 1, boundary_time(clock)])

"""
    SeaIceBoundaryVelocity{Dim}(ice_velocity, concentration, ocean_velocity, cell, face, surface, inward, along)

The boundary-normal sea ice velocity on the boundary `face` of one open side. The exterior velocity is the parent's
`ice_velocity` at the boundary-adjacent `cell`, or, where the parent's ice is free drifting with no velocity of its
own, the nested ocean's boundary-normal `ocean_velocity` band at level `surface`. With the interior velocity `uᴵ` on
the first interior face and `inward` the sign of the inward normal:

- exterior inflow: `uᴵ`, so the boundary row does not diverge,
- exterior outflow: `uᴵ`, without its inward part,
- no parent ice (`concentration` zero or missing): `uᴵ`.
"""
struct SeaIceBoundaryVelocity{Dim, U, A, O}
    ice_velocity   :: U
    concentration  :: A
    ocean_velocity :: O
    cell           :: Int
    face           :: Int
    surface        :: Int
    inward         :: Int
    along          :: NTuple{2, Int}
end

SeaIceBoundaryVelocity{Dim}(ice_velocity::U, concentration::A, ocean_velocity::O, cell, face, surface, inward, along) where {Dim, U, A, O} =
    SeaIceBoundaryVelocity{Dim, U, A, O}(ice_velocity, concentration, ocean_velocity, cell, face, surface, inward, along)

Adapt.adapt_structure(to, b::SeaIceBoundaryVelocity{Dim}) where Dim =
    SeaIceBoundaryVelocity{Dim}(adapt(to, b.ice_velocity), adapt(to, b.concentration), adapt(to, b.ocean_velocity),
                                b.cell, b.face, b.surface, b.inward, b.along)

Oceananigans.Architectures.on_architecture(to, b::SeaIceBoundaryVelocity{Dim}) where Dim =
    SeaIceBoundaryVelocity{Dim}(on_architecture(to, b.ice_velocity), on_architecture(to, b.concentration),
                                on_architecture(to, b.ocean_velocity), b.cell, b.face, b.surface, b.inward, b.along)

@inline function boundary_ice_velocity(uᵢ, ℵ, uₒ, uᴵ, inward)
    uᵉˣᵗ = ifelse(isnan(uᵢ) | iszero(uᵢ), uₒ, uᵢ)
    uᵒᵘᵗ = ifelse(inward * uᴵ > 0, zero(uᴵ), uᴵ)
    uᵇ   = ifelse(inward * uᵉˣᵗ > 0, uᴵ, uᵒᵘᵗ)
    return ifelse(isnan(ℵ) | iszero(ℵ), uᴵ, uᵇ)
end

# Halo fills without the model fields (e.g. during `set!`) see no interior velocity.
@inline interior_ice_velocity(::Val{1}, fields, i, j, grid) = @inbounds fields.u[i, j, 1]
@inline interior_ice_velocity(::Val{2}, fields, i, j, grid) = @inbounds fields.v[i, j, 1]
@inline interior_ice_velocity(::Val{1}, ::Nothing, i, j, grid) = zero(grid)
@inline interior_ice_velocity(::Val{2}, ::Nothing, i, j, grid) = zero(grid)

@inline function BoundaryConditions.getbc(b::SeaIceBoundaryVelocity{1}, j::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, fields=nothing, args...)
    t = boundary_time(clock)
    jᵇ = clamp(j, b.along...)
    uᵢ = @inbounds b.ice_velocity[b.cell, jᵇ, 1, t]
    ℵ  = @inbounds b.concentration[b.cell, jᵇ, 1, t]
    uₒ = band_value(b.ocean_velocity, b.face, jᵇ, b.surface, t)
    uᴵ = interior_ice_velocity(Val(1), fields, b.face + b.inward, j, grid)
    return boundary_ice_velocity(uᵢ, ℵ, uₒ, uᴵ, b.inward)
end

@inline function BoundaryConditions.getbc(b::SeaIceBoundaryVelocity{2}, i::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, fields=nothing, args...)
    t = boundary_time(clock)
    iᵇ = clamp(i, b.along...)
    vᵢ = @inbounds b.ice_velocity[iᵇ, b.cell, 1, t]
    ℵ  = @inbounds b.concentration[iᵇ, b.cell, 1, t]
    vₒ = band_value(b.ocean_velocity, iᵇ, b.face, b.surface, t)
    vᴵ = interior_ice_velocity(Val(2), fields, i, b.face + b.inward, grid)
    return boundary_ice_velocity(vᵢ, ℵ, vₒ, vᴵ, b.inward)
end

"""
$(TYPEDSIGNATURES)

Return the parent's sea ice thickness `h`, concentration `ℵ` and velocities `u` and `v` in `parent_dataset`
(e.g. `GLORYSDaily()`) over `dates`, as `FieldTimeSeries` on `grid` holding `time_indices_in_memory` samples.
Time `0` is the start of `first(dates)`, as for the parent of [`nested_ocean_model`](@ref).
"""
function parent_sea_ice(grid::AbstractGrid, parent_dataset; dates,
                        dir = default_download_directory(parent_dataset),
                        parent_padding = default_horizontal_padding(parent_dataset),
                        time_indices_in_memory = 2)

    region = BoundingBox(grid; padding = parent_padding)

    function series(name)
        metadata = Metadata(name; dataset = parent_dataset, dates = bracketing_dates(parent_dataset, name, dates), region, dir)
        return FieldTimeSeries(metadata, grid; time_indices_in_memory, time_indexing = Clamp(), start_time = first(dates))
    end

    return (h = series(:sea_ice_thickness),
            ℵ = series(:sea_ice_concentration),
            u = series(:sea_ice_u_velocity),
            v = series(:sea_ice_v_velocity))
end

"""
$(TYPEDSIGNATURES)

Build the open lateral boundary conditions of the sea ice over the nested `ocean` (a `Simulation` of a
[`nested_ocean_model`](@ref)) from `parent_dataset`'s sea ice over `dates` (see [`parent_sea_ice`](@ref)).
The remaining keyword arguments go to `nested_sea_ice_boundary_conditions(ocean, parent_sea_ice; tracer_scheme)`.
"""
function nested_sea_ice_boundary_conditions(ocean::Simulation, parent_dataset; dates,
                                            dir = default_download_directory(parent_dataset),
                                            parent_padding = default_horizontal_padding(parent_dataset),
                                            time_indices_in_memory = 2,
                                            tracer_scheme = default_tracer_scheme())

    grid = ocean.model.child.grid
    parent_ice = parent_sea_ice(grid, parent_dataset; dates, dir, parent_padding, time_indices_in_memory)

    return nested_sea_ice_boundary_conditions(ocean, parent_ice; tracer_scheme)
end

"""
$(TYPEDSIGNATURES)

Build the open lateral boundary conditions of the sea ice over the nested `ocean` (a `Simulation` of a
[`nested_ocean_model`](@ref)) from `parent_ice`, a `NamedTuple` of `FieldTimeSeries` on the child grid holding the
parent's sea ice thickness `h`, concentration `ℵ` and velocities `u` and `v`. The open sides are the ocean's.
On each:

- `h` and `ℵ`: `ValueBoundaryCondition` toward the parent with `tracer_scheme`,
- the snow thickness `hs`: `ValueBoundaryCondition(0)` with `tracer_scheme`, so that inflowing ice carries no snow,
- the boundary-normal velocity: `NormalFlowBoundaryCondition` that lets the ice leave but not be pulled inward
  ([`SeaIceBoundaryVelocity`](@ref)).

The tangential velocity keeps the condition of `sea_ice_simulation`'s `lateral_boundary_condition`.
Returns a `NamedTuple` of `FieldBoundaryConditions` for `h`, `ℵ`, `hs`, `u` and `v`, to pass as the `boundary_conditions`
of `sea_ice_simulation`.
"""
function nested_sea_ice_boundary_conditions(ocean::Simulation, parent_ice::NamedTuple;
                                            tracer_scheme = default_tracer_scheme())

    nested = ocean.model
    grid = nested.child.grid
    bands = nested.exchanger.bands
    surface = size(grid, 3)

    along_dimension(side) = 3 - boundary_dimension(side)
    along_bounds(data, ℓ, side) = parent_index_bounds(ℓ, topology(grid, along_dimension(side)),
                                                      size(grid, along_dimension(side)), axes(data, along_dimension(side)))

    function boundary_series(series, side)
        dim = boundary_dimension(side)
        index = boundary_index(Center(), side, size(grid, dim))
        return SeaIceBoundarySeries{dim}(series, index, along_bounds(series.data, Center(), side))
    end

    function boundary_velocity(side)
        dim = boundary_dimension(side)
        N = size(grid, dim)
        name = normal_velocity_name(side)
        band = bands[side][name]
        face = boundary_index(Face(), side, N)
        cell = boundary_index(Center(), side, N)
        inward = side ∈ (:west, :south) ? 1 : -1
        along = along_bounds(band.previous.data, Center(), side)
        return SeaIceBoundaryVelocity{dim}(parent_ice[name], parent_ice.ℵ, band, cell, face, surface, inward, along)
    end

    h_bcs = Pair{Symbol, Any}[]
    ℵ_bcs = Pair{Symbol, Any}[]
    hs_bcs = Pair{Symbol, Any}[]
    u_bcs = Pair{Symbol, Any}[]
    v_bcs = Pair{Symbol, Any}[]

    for side in keys(bands)
        push!(h_bcs, side => ValueBoundaryCondition(boundary_series(parent_ice.h, side); scheme = tracer_scheme))
        push!(ℵ_bcs, side => ValueBoundaryCondition(boundary_series(parent_ice.ℵ, side); scheme = tracer_scheme))
        push!(hs_bcs, side => ValueBoundaryCondition(zero(eltype(grid)); scheme = tracer_scheme))
        normal_bcs = boundary_dimension(side) == 1 ? u_bcs : v_bcs
        push!(normal_bcs, side => NormalFlowBoundaryCondition(boundary_velocity(side)))
    end

    return (h = FieldBoundaryConditions(grid, (Center(), Center(), nothing); h_bcs...),
            ℵ = FieldBoundaryConditions(grid, (Center(), Center(), nothing); ℵ_bcs...),
            hs = FieldBoundaryConditions(grid, (Center(), Center(), nothing); hs_bcs...),
            u = FieldBoundaryConditions(grid, (Face(), Center(), nothing); u_bcs...),
            v = FieldBoundaryConditions(grid, (Center(), Face(), nothing); v_bcs...))
end

"""
$(TYPEDSIGNATURES)

Set the thickness and concentration of the sea ice in `sea_ice` (a `Simulation`) to those of `parent_ice`
(see [`parent_sea_ice`](@ref)) at the sea ice's time, with open water as no ice.
"""
function set_parent_sea_ice!(sea_ice::Simulation, parent_ice::NamedTuple)
    model = sea_ice.model
    t = Time(model.clock.time)
    set!(model, h = parent_ice.h[t], ℵ = parent_ice.ℵ[t])

    for φ in (model.ice_thickness, model.ice_concentration)
        parent(φ) .= zero_nan.(parent(φ))
    end

    return sea_ice
end

"""
$(TYPEDSIGNATURES)

Return a sea-ice `Simulation` over the nested `ocean` (a `Simulation` of a [`nested_ocean_model`](@ref)), the nested
counterpart of [`sea_ice_simulation`](@ref), with the parent's sea ice read from `parent_dataset` over `dates`
(see [`parent_sea_ice`](@ref)). The remaining keyword arguments go to
`nested_sea_ice_simulation(ocean, parent_ice; tracer_scheme, boundary_conditions, kw...)`.
"""
function nested_sea_ice_simulation(ocean::Simulation, parent_dataset; dates,
                                   dir = default_download_directory(parent_dataset),
                                   parent_padding = default_horizontal_padding(parent_dataset),
                                   time_indices_in_memory = 2,
                                   kw...)

    grid = ocean.model.child.grid
    parent_ice = parent_sea_ice(grid, parent_dataset; dates, dir, parent_padding, time_indices_in_memory)

    return nested_sea_ice_simulation(ocean, parent_ice; kw...)
end

"""
$(TYPEDSIGNATURES)

Return a sea-ice `Simulation` on the child grid of the nested `ocean`, with the parent's sea ice `parent_ice`
(as for [`nested_sea_ice_boundary_conditions`](@ref)). The sea ice crosses the ocean's open sides through
`nested_sea_ice_boundary_conditions(ocean, parent_ice; tracer_scheme)` and starts from the parent's thickness and
concentration at the sea ice's time. `boundary_conditions` are merged with the nested ones, and the remaining keyword
arguments go to [`sea_ice_simulation`](@ref).
"""
function nested_sea_ice_simulation(ocean::Simulation, parent_ice::NamedTuple;
                                   tracer_scheme = default_tracer_scheme(),
                                   boundary_conditions = NamedTuple(),
                                   kw...)

    grid = ocean.model.child.grid
    nested_bcs = nested_sea_ice_boundary_conditions(ocean, parent_ice; tracer_scheme)
    boundary_conditions = merge_field_boundary_conditions(NamedTuple(boundary_conditions), nested_bcs)

    sea_ice = sea_ice_simulation(grid, ocean; boundary_conditions, kw...)
    set_parent_sea_ice!(sea_ice, parent_ice)

    return sea_ice
end
