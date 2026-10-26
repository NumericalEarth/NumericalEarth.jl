#####
##### Open lateral boundary conditions for a regional ocean, read from its boundary bands
#####

using Oceananigans.BoundaryConditions: BoundaryConditions, GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition,
                                       ValueBoundaryCondition
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ

"""
    BoundaryTimeSeries{Dim}(series, index, along)

Boundary-condition value read from a [`BoundaryBand`](@ref) at the boundary row (`Dim = 1`, at
`i = index`) or column (`Dim = 2`, at `j = index`) of one open side, interpolated in time only. The
along-boundary index is clamped to `1:along`, the interior points along the boundary: halo fills, such as
the barotropic ones on a grid with extended halos, reach beyond the series' own halos.
"""
struct BoundaryTimeSeries{Dim, S}
    series :: S
    index  :: Int
    along  :: Int
end

BoundaryTimeSeries{Dim}(series::S, index, along) where {Dim, S} = BoundaryTimeSeries{Dim, S}(series, index, along)

Adapt.adapt_structure(to, b::BoundaryTimeSeries{Dim}) where Dim =
    BoundaryTimeSeries{Dim}(adapt(to, b.series), b.index, b.along)

"""
    BarotropicBoundaryTransport{Dim}(velocity, face, cell, along)

The exterior barotropic transport `Uᵉˣᵗ = Σₖ uₖ Δzₖ` of a `GravityWaveRadiationBoundaryCondition` (Flather)
on one open side: the boundary-normal `velocity` series integrated over the child's wet column (wet at the
boundary-adjacent `cell`) at the boundary `face`, with the live `Δz`, so it follows a moving vertical
coordinate and integrates exactly the profile the baroclinic condition imposes. The series is read at the
along-boundary index clamped to `1:along`, as for a [`BoundaryTimeSeries`](@ref).
"""
struct BarotropicBoundaryTransport{Dim, U}
    velocity :: U
    face     :: Int
    cell     :: Int
    along    :: Int
end

BarotropicBoundaryTransport{Dim}(velocity::U, face, cell, along) where {Dim, U} =
    BarotropicBoundaryTransport{Dim, U}(velocity, face, cell, along)

Adapt.adapt_structure(to, U::BarotropicBoundaryTransport{Dim}) where Dim =
    BarotropicBoundaryTransport{Dim}(adapt(to, U.velocity), U.face, U.cell, U.along)

# Halo fills without a clock (e.g. during `set!`) evaluate the boundary at t = 0.
@inline boundary_time(clock) = Time(clock.time)
@inline boundary_time(::Nothing) = Time(0)

@inline BoundaryConditions.getbc(b::BoundaryTimeSeries{1}, j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) =
    band_value(b.series, b.index, clamp(j, 1, b.along), k, boundary_time(clock))

@inline BoundaryConditions.getbc(b::BoundaryTimeSeries{2}, i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) =
    band_value(b.series, clamp(i, 1, b.along), b.index, k, boundary_time(clock))

@inline function BoundaryConditions.getbc(U::BarotropicBoundaryTransport{1}, j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...)
    t = boundary_time(clock)
    i = U.face
    jᵇ = clamp(j, 1, U.along)
    Uᵉˣᵗ = zero(grid)
    for k′ in 1:size(grid, 3)
        u = band_value(U.velocity, i, jᵇ, k′, t)
        Uᵉˣᵗ += ifelse(inactive_cell(U.cell, j, k′, grid), zero(grid), u * Δzᶠᶜᶜ(i, j, k′, grid))
    end
    return Uᵉˣᵗ
end

@inline function BoundaryConditions.getbc(V::BarotropicBoundaryTransport{2}, i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...)
    t = boundary_time(clock)
    j = V.face
    iᵇ = clamp(i, 1, V.along)
    Vᵉˣᵗ = zero(grid)
    for k′ in 1:size(grid, 3)
        v = band_value(V.velocity, iᵇ, j, k′, t)
        Vᵉˣᵗ += ifelse(inactive_cell(i, V.cell, k′, grid), zero(grid), v * Δzᶜᶠᶜ(i, j, k′, grid))
    end
    return Vᵉˣᵗ
end

const velocity_locations = (u = (Face(), Center(), Center()), v = (Center(), Face(), Center()))
const tracer_location = (Center(), Center(), Center())
const free_surface_location = (Center(), Center(), nothing)

boundary_location(name) = name ∈ keys(velocity_locations) ? velocity_locations[name] :
                          name === :η ? free_surface_location : tracer_location

"""
    ocean_open_boundary_conditions(grid, bands; normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

Build the open lateral boundary conditions of a regional ocean on `grid` from its boundary `bands`, a
`NamedTuple` keyed by open side of `NamedTuple`s of [`BoundaryBand`](@ref)s holding the parent's velocities
`u` and `v`, free surface `η`, and tracers (every other entry). Each open side gets five conditions:

- the boundary-normal velocity: `NormalFlowBoundaryCondition` with `normal_velocity_scheme`,
- the boundary-tangential velocity: `ValueBoundaryCondition` with `tangential_velocity_scheme`
  (the default, a no-flux wall, is wrong on an open side),
- the barotropic transport: `GravityWaveRadiationBoundaryCondition` (Flather) toward the parent's
  transport through the child's wet column ([`BarotropicBoundaryTransport`](@ref)) and sea surface height,
- the free surface: `SurfaceWaveRadiationBoundaryCondition` (Chapman),
- each tracer: `ValueBoundaryCondition` with `tracer_scheme`.

Returns a `NamedTuple` of `FieldBoundaryConditions` for `u`, `v`, `U`, `V`, `η`, and the tracers.
"""
function ocean_open_boundary_conditions(grid, bands; normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

    sides = keys(bands)
    tracer_names = isempty(sides) ? () : filter(name -> name ∉ (:u, :v, :η), keys(first(bands)))

    along_dimension(side) = 3 - boundary_dimension(side)
    along_size(name, side) = last_index(boundary_location(name)[along_dimension(side)], size(grid, along_dimension(side)))

    function boundary_value(band, name, side)
        dim = boundary_dimension(side)
        index = boundary_index(boundary_location(name)[dim], side, size(grid, dim))
        return BoundaryTimeSeries{dim}(band, index, along_size(name, side))
    end

    u_bcs = Pair{Symbol, Any}[]
    v_bcs = Pair{Symbol, Any}[]
    U_bcs = Pair{Symbol, Any}[]
    V_bcs = Pair{Symbol, Any}[]
    η_bcs = Pair{Symbol, Any}[]
    tracer_bcs = Dict(name => Pair{Symbol, Any}[] for name in tracer_names)

    for side in sides
        side_bands = bands[side]
        dim = boundary_dimension(side)
        N = size(grid, dim)
        normal, tangential = dim == 1 ? (:u, :v) : (:v, :u)
        normal_bcs, tangential_bcs, transport_bcs = dim == 1 ? (u_bcs, v_bcs, U_bcs) : (v_bcs, u_bcs, V_bcs)

        push!(normal_bcs, side => NormalFlowBoundaryCondition(boundary_value(side_bands[normal], normal, side);
                                                              scheme = normal_velocity_scheme))
        push!(tangential_bcs, side => ValueBoundaryCondition(boundary_value(side_bands[tangential], tangential, side);
                                                             scheme = tangential_velocity_scheme))

        # Flather is the only route by which the parent's η reaches the child: Chapman on η only radiates.
        face = boundary_index(Face(), side, N)
        cell = boundary_index(Center(), side, N)
        transport = BarotropicBoundaryTransport{dim}(side_bands[normal], face, cell, along_size(normal, side))
        push!(transport_bcs, side => GravityWaveRadiationBoundaryCondition((transport, boundary_value(side_bands.η, :η, side))))
        push!(η_bcs, side => SurfaceWaveRadiationBoundaryCondition())

        for name in tracer_names
            push!(tracer_bcs[name], side => ValueBoundaryCondition(boundary_value(side_bands[name], name, side); scheme = tracer_scheme))
        end
    end

    velocity_bcs = (u = FieldBoundaryConditions(; u_bcs...),
                    v = FieldBoundaryConditions(; v_bcs...),
                    U = FieldBoundaryConditions(grid, (Face(), Center(), nothing); U_bcs...),
                    V = FieldBoundaryConditions(grid, (Center(), Face(), nothing); V_bcs...),
                    η = FieldBoundaryConditions(grid, (Center(), Center(), Face()); η_bcs...))

    tracer_conditions = NamedTuple(name => FieldBoundaryConditions(; tracer_bcs[name]...) for name in tracer_names)

    return merge(velocity_bcs, tracer_conditions)
end
