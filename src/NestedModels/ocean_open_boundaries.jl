#####
##### Open lateral boundary conditions for a regional ocean, read from its boundary bands
#####

using Oceananigans.BoundaryConditions: BoundaryConditions, GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition,
                                       ValueBoundaryCondition
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ

# The band's boundary row or column, clamped along the boundary to `along` for halo fills that reach past the band
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

# Flather's exterior transport: `ū H` with the parent's depth mean `ū`, or `Σₖ uₖ Δzₖ` over the child's wet column
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

# The boundary-normal velocity plus the depth-uniform correction that integrates it to `ū H`
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

@inline band_value_at(dim, band, normal, along, k, t) = band_value(band, horizontal_indices(dim, normal, along)..., k, t)

@inline BoundaryConditions.getbc(b::BoundaryTimeSeries{Dim}, s::Integer, k::Integer, grid::AbstractGrid,
                                 clock=nothing, args...) where Dim =
    band_value_at(Val(Dim), b.series, b.index, clamp(s, b.along...), k, boundary_time(clock))

@inline boundary_face_thickness(::Val{1}, face, s, k, grid) = Δzᶠᶜᶜ(face, s, k, grid)
@inline boundary_face_thickness(::Val{2}, face, s, k, grid) = Δzᶜᶠᶜ(s, face, k, grid)

# The wet depth of the child's column at boundary `face`, and the integral over it of the boundary-normal velocity `series`
@inline function wet_column_integral(dim, series, face, cell, s, sᵇ, grid, t)
    H = zero(grid)
    U = zero(grid)
    for k in 1:size(grid, 3)
        inactive = inactive_cell(horizontal_indices(dim, cell, s)..., k, grid)
        Δ = ifelse(inactive, zero(grid), boundary_face_thickness(dim, face, s, k, grid))
        H += Δ
        U += Δ * band_value_at(dim, series, face, sᵇ, k, t)
    end
    return U, H
end

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

# The indices along `side` of `data`, at location `loc` on `grid`, that hold the parent
function along_bounds(grid, data, loc, side)
    dim = 3 - boundary_dimension(side)
    return parent_index_bounds(loc[dim], topology(grid, dim), size(grid, dim), axes(data, dim))
end

side_scheme(scheme, side) = scheme

side_scheme(schemes::NamedTuple, side) = schemes[side]

function ocean_open_boundary_conditions(grid, bands;
                                        depth_means,
                                        normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

    sides = keys(bands)
    tracer_names = isempty(sides) ? () : filter(name -> name ∉ (:u, :v, :η), keys(first(bands)))

    function boundary_value(band, name, side)
        dim = boundary_dimension(side)
        loc = boundary_location(name)
        index = boundary_index(loc[dim], side, size(grid, dim))
        return BoundaryTimeSeries{dim}(band, index, along_bounds(grid, band.previous.data, loc, side))
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
        along = along_bounds(grid, side_bands[normal].previous.data, boundary_location(normal), side)

        normal_value = isnothing(depth_mean) ? boundary_value(side_bands[normal], normal, side) :
                       ConsistentBoundaryVelocity{dim}(side_bands[normal], depth_mean, face, cell, along)

        push!(normal_bcs, side => NormalFlowBoundaryCondition(normal_value; scheme = side_scheme(normal_velocity_scheme, side)))
        push!(tangential_bcs, side => ValueBoundaryCondition(boundary_value(side_bands[tangential], tangential, side);
                                                             scheme = side_scheme(tangential_velocity_scheme, side)))

        # Flather is the only route by which the parent's η reaches the child: Chapman on η only radiates.
        transport = BarotropicBoundaryTransport{dim}(side_bands[normal], depth_mean, face, cell, along)
        push!(transport_bcs, side => GravityWaveRadiationBoundaryCondition((transport, boundary_value(side_bands.η, :η, side))))
        push!(η_bcs, side => SurfaceWaveRadiationBoundaryCondition())

        for name in tracer_names
            push!(tracer_bcs[name], side => ValueBoundaryCondition(boundary_value(side_bands[name], name, side); scheme = side_scheme(tracer_scheme, side)))
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
