#####
##### Open lateral boundary conditions for the sea ice over a nested ocean, read from the parent's sea ice
#####

using ..DataWrangling: Metadata, bracketing_dates
using ..SeaIces: sea_ice_simulation
using Oceananigans.OutputReaders: Clamp, FieldTimeSeries

@inline zero_nan(x) = ifelse(isnan(x), zero(x), x)
@inline normal_component(::Val{Dim}, vector) where Dim = vector[Dim]

# The parent's sea ice on the boundary row or column of one open side, with open water as no ice
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

@inline function BoundaryConditions.getbc(b::SeaIceBoundarySeries{Dim}, s::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, args...) where Dim
    i, j = horizontal_indices(Val(Dim), b.index, clamp(s, b.along...))
    return @inbounds zero_nan(b.series[i, j, 1, boundary_time(clock)])
end

# The boundary-normal ice velocity, which lets the ice leave through the side but not be pulled in
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

# Halo fills without the model fields, as during `set!`, see no interior velocity
@inline interior_ice_velocity(dim, fields, i, j, grid) = @inbounds normal_component(dim, (fields.u, fields.v))[i, j, 1]
@inline interior_ice_velocity(dim, ::Nothing, i, j, grid) = zero(grid)

@inline function BoundaryConditions.getbc(b::SeaIceBoundaryVelocity{Dim}, s::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, fields=nothing, args...) where Dim
    dim = Val(Dim)
    t = boundary_time(clock)
    sᵇ = clamp(s, b.along...)
    i, j = horizontal_indices(dim, b.cell, sᵇ)
    uₑ = @inbounds b.ice_velocity[1][i, j, 1, t]
    vₑ = @inbounds b.ice_velocity[2][i, j, 1, t]
    uᵢ = normal_component(dim, intrinsic_vector(i, j, 1, grid, uₑ, vₑ))
    ℵ  = @inbounds b.concentration[i, j, 1, t]
    uₒ = band_value_at(dim, b.ocean_velocity, b.face, sᵇ, b.surface, t)
    uᴵ = interior_ice_velocity(dim, fields, horizontal_indices(dim, b.face + b.inward, s)..., grid)
    return boundary_ice_velocity(uᵢ, ℵ, uₒ, uᴵ, b.inward)
end

"""
$(TYPEDSIGNATURES)

Return the parent's sea ice thickness `h`, concentration `ℵ` and velocities `u` and `v` in `parent_dataset` (e.g.
`GLORYSDaily()`) over `dates`, as `FieldTimeSeries` on `grid` holding `time_indices_in_memory` samples. Time `0` is
the start of `first(dates)`.
"""
function parent_sea_ice(grid::AbstractGrid, parent_dataset; dates,
                        dir = default_download_directory(parent_dataset),
                        parent_padding = default_horizontal_padding(parent_dataset),
                        time_indices_in_memory = 2)

    region = BoundingBox(underlying_grid(grid); padding = parent_padding)

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

Return the open lateral boundary conditions of the sea ice over the nested `ocean` (a `Simulation` of a
[`nested_ocean_model`](@ref)) from the parent's sea ice `parent_ice` (see [`parent_sea_ice`](@ref)): `h` and `ℵ` relax
toward the parent with `tracer_scheme`, inflowing ice carries no snow, and the boundary-normal velocity lets the ice
leave but not be pulled inward.
"""
function nested_sea_ice_boundary_conditions(ocean::Simulation, parent_ice::NamedTuple;
                                            tracer_scheme = default_tracer_scheme())

    nested = ocean.model
    grid = nested.child.grid
    bands = nested.exchanger.bands
    surface = size(grid, 3)

    function boundary_series(series, side)
        dim = boundary_dimension(side)
        index = boundary_index(Center(), side, size(grid, dim))
        return SeaIceBoundarySeries{dim}(series, index, along_bounds(grid, series.data, tracer_location, side))
    end

    function boundary_velocity(side)
        dim = boundary_dimension(side)
        N = size(grid, dim)
        name = normal_velocity_name(side)
        band = bands[side][name]
        face = boundary_index(Face(), side, N)
        cell = boundary_index(Center(), side, N)
        inward = side ∈ (:west, :south) ? 1 : -1
        along = along_bounds(grid, band.previous.data, tracer_location, side)
        return SeaIceBoundaryVelocity{dim}((parent_ice.u, parent_ice.v), parent_ice.ℵ, band, cell, face, surface, inward, along)
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

Set the thickness and concentration of `sea_ice` (a `Simulation`) to the parent's `parent_ice` at its time, with open
water as no ice, `0 ≤ ℵ ≤ 1`, `h ≥ 0`, and either both or neither in each cell.
"""
function set_parent_sea_ice!(sea_ice::Simulation, parent_ice::NamedTuple)
    model = sea_ice.model
    t = Time(model.clock.time)
    set!(model, h = parent_ice.h[t], ℵ = parent_ice.ℵ[t])

    # `h` and `ℵ` are interpolated separately, so a cell can hold one without the other
    h = parent(model.ice_thickness)
    ℵ = parent(model.ice_concentration)
    ℵ .= clamp.(zero_nan.(ℵ), 0, 1)
    h .= ifelse.(ℵ .> 0, max.(zero_nan.(h), 0), 0)
    ℵ .= ifelse.(h .> 0, ℵ, 0)

    return sea_ice
end

"""
$(TYPEDSIGNATURES)

Return a sea ice `Simulation` over the nested `ocean`, the nested counterpart of [`sea_ice_simulation`](@ref), with the
parent's sea ice read from `parent_dataset` over `dates`. The remaining keyword arguments `kw` go to
`nested_sea_ice_simulation(ocean, parent_ice; kw...)`.
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

Return a sea ice `Simulation` on the child grid of the nested `ocean`, crossing its open sides with
[`nested_sea_ice_boundary_conditions`](@ref) and starting from `parent_ice`. A caller's `boundary_conditions` are merged
with the nested ones, and the remaining keyword arguments `kw` go to [`sea_ice_simulation`](@ref).
"""
function nested_sea_ice_simulation(ocean::Simulation, parent_ice::NamedTuple;
                                   tracer_scheme = default_tracer_scheme(),
                                   boundary_conditions = NamedTuple(),
                                   kw...)

    grid = ocean.model.child.grid
    nested_bcs = nested_sea_ice_boundary_conditions(ocean, parent_ice; tracer_scheme)
    boundary_conditions = merge_by_name(merge_boundary_conditions, NamedTuple(boundary_conditions), nested_bcs)

    sea_ice = sea_ice_simulation(grid, ocean; boundary_conditions, kw...)
    set_parent_sea_ice!(sea_ice, parent_ice)

    return sea_ice
end
