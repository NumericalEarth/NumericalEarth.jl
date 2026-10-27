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
along-boundary index is clamped to `along = (first, last)`, the points along the boundary that hold the parent: halo
fills, such as the barotropic ones on a grid with extended halos, reach beyond the series' own halos.
"""
struct BoundaryTimeSeries{Dim, S}
    series :: S
    index  :: Int
    along  :: NTuple{2, Int}
end

BoundaryTimeSeries{Dim}(series::S, index, along) where {Dim, S} = BoundaryTimeSeries{Dim, S}(series, index, along)

Adapt.adapt_structure(to, b::BoundaryTimeSeries{Dim}) where Dim =
    BoundaryTimeSeries{Dim}(adapt(to, b.series), b.index, b.along)

Oceananigans.Architectures.on_architecture(to, b::BoundaryTimeSeries{Dim}) where Dim =
    BoundaryTimeSeries{Dim}(on_architecture(to, b.series), b.index, b.along)

"""
    BarotropicBoundaryTransport{Dim}(velocity, depth_mean, face, cell, along)

The exterior barotropic transport `Uᵉˣᵗ` of a `GravityWaveRadiationBoundaryCondition` (Flather) on one open
side, at the boundary `face` over the child's wet column (wet at the boundary-adjacent `cell`), with the live `Δz`.
With the parent's `depth_mean` velocity (a one-cell [`BoundaryBand`](@ref)), `Uᵉˣᵗ = ū H`, the parent's depth mean
over its own column times the child's wet depth `H`, as in the reference Mid-Atlantic Bight configuration; without
it (`nothing`), `Uᵉˣᵗ = Σₖ uₖ Δzₖ`, the boundary-normal `velocity` series summed over the child's wet column.
The series are read at the along-boundary index clamped to `along`, as for a [`BoundaryTimeSeries`](@ref).
"""
struct BarotropicBoundaryTransport{Dim, U, M}
    velocity   :: U
    depth_mean :: M
    face       :: Int
    cell       :: Int
    along      :: NTuple{2, Int}
end

BarotropicBoundaryTransport{Dim}(velocity::U, depth_mean::M, face, cell, along) where {Dim, U, M} =
    BarotropicBoundaryTransport{Dim, U, M}(velocity, depth_mean, face, cell, along)

Adapt.adapt_structure(to, U::BarotropicBoundaryTransport{Dim}) where Dim =
    BarotropicBoundaryTransport{Dim}(adapt(to, U.velocity), adapt(to, U.depth_mean), U.face, U.cell, U.along)

Oceananigans.Architectures.on_architecture(to, U::BarotropicBoundaryTransport{Dim}) where Dim =
    BarotropicBoundaryTransport{Dim}(on_architecture(to, U.velocity), on_architecture(to, U.depth_mean), U.face, U.cell, U.along)

"""
    ConsistentBoundaryVelocity{Dim}(series, depth_mean, index, cell, along)

The boundary-normal velocity of one open side read from a [`BoundaryBand`](@ref) `series`, plus the depth-uniform
velocity that makes its integral over the child's wet column equal the Flather exterior transport
`ū H` of a [`BarotropicBoundaryTransport`](@ref) with the same `depth_mean`, as the reference Mid-Atlantic Bight
configuration does (its "consistent normal flow"): without it, the difference between the two appears in the last
interior column as a spurious vertical velocity.
"""
struct ConsistentBoundaryVelocity{Dim, S, M}
    series     :: S
    depth_mean :: M
    index      :: Int
    cell       :: Int
    along      :: NTuple{2, Int}
end

ConsistentBoundaryVelocity{Dim}(series::S, depth_mean::M, index, cell, along) where {Dim, S, M} =
    ConsistentBoundaryVelocity{Dim, S, M}(series, depth_mean, index, cell, along)

Adapt.adapt_structure(to, b::ConsistentBoundaryVelocity{Dim}) where Dim =
    ConsistentBoundaryVelocity{Dim}(adapt(to, b.series), adapt(to, b.depth_mean), b.index, b.cell, b.along)

Oceananigans.Architectures.on_architecture(to, b::ConsistentBoundaryVelocity{Dim}) where Dim =
    ConsistentBoundaryVelocity{Dim}(on_architecture(to, b.series), on_architecture(to, b.depth_mean), b.index, b.cell, b.along)

# Halo fills without a clock (e.g. during `set!`) evaluate the boundary at t = 0.
@inline boundary_time(clock) = Time(clock.time)
@inline boundary_time(::Nothing) = Time(0)

@inline BoundaryConditions.getbc(b::BoundaryTimeSeries{1}, j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) =
    band_value(b.series, b.index, clamp(j, b.along...), k, boundary_time(clock))

@inline BoundaryConditions.getbc(b::BoundaryTimeSeries{2}, i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) =
    band_value(b.series, clamp(i, b.along...), b.index, k, boundary_time(clock))

# The wet depth of the child's column at boundary face `i` (`Dim = 1`) or `j` (`Dim = 2`), and the integral over it
# of the boundary-normal velocity `series`
@inline function wet_column_integral(::Val{1}, series, face, cell, s, sᵇ, grid, t)
    H = zero(grid)
    U = zero(grid)
    for k in 1:size(grid, 3)
        Δ = ifelse(inactive_cell(cell, s, k, grid), zero(grid), Δzᶠᶜᶜ(face, s, k, grid))
        H += Δ
        U += Δ * band_value(series, face, sᵇ, k, t)
    end
    return U, H
end

@inline function wet_column_integral(::Val{2}, series, face, cell, s, sᵇ, grid, t)
    H = zero(grid)
    U = zero(grid)
    for k in 1:size(grid, 3)
        Δ = ifelse(inactive_cell(s, cell, k, grid), zero(grid), Δzᶜᶠᶜ(s, face, k, grid))
        H += Δ
        U += Δ * band_value(series, sᵇ, face, k, t)
    end
    return U, H
end

@inline band_value_at(::Val{1}, band, face, sᵇ, k, t) = band_value(band, face, sᵇ, k, t)
@inline band_value_at(::Val{2}, band, face, sᵇ, k, t) = band_value(band, sᵇ, face, k, t)

# The exterior transport: the parent's depth mean times the child's wet depth, or the series summed over it.
@inline exterior_transport(dim, ::Nothing, U, H, face, sᵇ, t) = U
@inline exterior_transport(dim, depth_mean, U, H, face, sᵇ, t) = band_value_at(dim, depth_mean, face, sᵇ, 1, t) * H

@inline function BoundaryConditions.getbc(U::BarotropicBoundaryTransport{Dim}, s::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, args...) where Dim
    t = boundary_time(clock)
    sᵇ = clamp(s, U.along...)
    Σ, H = wet_column_integral(Val(Dim), U.velocity, U.face, U.cell, s, sᵇ, grid, t)
    return exterior_transport(Val(Dim), U.depth_mean, Σ, H, U.face, sᵇ, t)
end

@inline function BoundaryConditions.getbc(b::ConsistentBoundaryVelocity{Dim}, s::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, args...) where Dim
    t = boundary_time(clock)
    sᵇ = clamp(s, b.along...)
    u = band_value_at(Val(Dim), b.series, b.index, sᵇ, k, t)
    Σ, H = wet_column_integral(Val(Dim), b.series, b.index, b.cell, s, sᵇ, grid, t)
    ū = band_value_at(Val(Dim), b.depth_mean, b.index, sᵇ, 1, t)
    correction = (ū * H - Σ) / max(H, eps(H))
    return u + ifelse(H > 0, correction, zero(correction))
end

const velocity_locations = (u = (Face(), Center(), Center()), v = (Center(), Face(), Center()))
const tracer_location = (Center(), Center(), Center())
const free_surface_location = (Center(), Center(), nothing)

boundary_location(name) = name ∈ keys(velocity_locations) ? velocity_locations[name] :
                          name === :η ? free_surface_location : tracer_location

"""
    ocean_open_boundary_conditions(grid, bands; depth_means, normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

Build the open lateral boundary conditions of a regional ocean on `grid` from its boundary `bands`, a
`NamedTuple` keyed by open side of `NamedTuple`s of [`BoundaryBand`](@ref)s holding the parent's velocities
`u` and `v`, free surface `η`, and tracers (every other entry). Each open side gets five conditions:

- the boundary-normal velocity: `NormalFlowBoundaryCondition` with `normal_velocity_scheme`, made consistent with
  the Flather exterior transport when the side has a parent `depth_means` band ([`ConsistentBoundaryVelocity`](@ref)),
- the boundary-tangential velocity: `ValueBoundaryCondition` with `tangential_velocity_scheme`
  (the default, a no-flux wall, is wrong on an open side),
- the barotropic transport: `GravityWaveRadiationBoundaryCondition` (Flather) toward the parent's
  transport through the child's wet column ([`BarotropicBoundaryTransport`](@ref)) and sea surface height,
- the free surface: `SurfaceWaveRadiationBoundaryCondition` (Chapman),
- each tracer: `ValueBoundaryCondition` with `tracer_scheme`.

Returns a `NamedTuple` of `FieldBoundaryConditions` for `u`, `v`, `U`, `V`, `η`, and the tracers.
"""
function ocean_open_boundary_conditions(grid, bands;
                                        depth_means = map(Returns(nothing), bands),
                                        normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

    sides = keys(bands)
    tracer_names = isempty(sides) ? () : filter(name -> name ∉ (:u, :v, :η), keys(first(bands)))

    along_dimension(side) = 3 - boundary_dimension(side)
    function along_bounds(band, name, side)
        dim = along_dimension(side)
        ℓ = boundary_location(name)[dim]
        return parent_index_bounds(ℓ, topology(grid, dim), size(grid, dim), axes(band.previous.data, dim))
    end

    function boundary_value(band, name, side)
        dim = boundary_dimension(side)
        index = boundary_index(boundary_location(name)[dim], side, size(grid, dim))
        return BoundaryTimeSeries{dim}(band, index, along_bounds(band, name, side))
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

        face = boundary_index(Face(), side, N)
        cell = boundary_index(Center(), side, N)
        depth_mean = depth_means[side]

        normal_value = isnothing(depth_mean) ? boundary_value(side_bands[normal], normal, side) :
                       ConsistentBoundaryVelocity{dim}(side_bands[normal], depth_mean, face, cell, along_bounds(side_bands[normal], normal, side))

        push!(normal_bcs, side => NormalFlowBoundaryCondition(normal_value; scheme = normal_velocity_scheme))
        push!(tangential_bcs, side => ValueBoundaryCondition(boundary_value(side_bands[tangential], tangential, side);
                                                             scheme = tangential_velocity_scheme))

        # Flather is the only route by which the parent's η reaches the child: Chapman on η only radiates.
        transport = BarotropicBoundaryTransport{dim}(side_bands[normal], depth_mean, face, cell, along_bounds(side_bands[normal], normal, side))
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
