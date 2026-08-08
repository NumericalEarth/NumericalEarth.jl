# The geometry is evaluated by LibGEOS on the CPU in Float64. The resulting
# sparse matrices can be reused for every transport sample on the same grids.

const FPivotTripolarGrid = TripolarGrid{FT, TX, RightFaceFolded} where {FT, TX}

@inline face_linear_index(i, j, Nx) = i + (j - 1) * Nx

@inline function unwrap_longitude_pair(longitude₁::Real, longitude₂::Real)
    λ₁ = Float64(longitude₁)
    λ₂ = Float64(longitude₂)

    while λ₂ - λ₁ > 180
        λ₂ -= 360
    end

    while λ₂ - λ₁ < -180
        λ₂ += 360
    end

    return λ₁, λ₂
end

@inline function midpoint_longitude_latitude(point₁::Tuple{<:Real, <:Real},
                                              point₂::Tuple{<:Real, <:Real})
    λ₁, λ₂ = unwrap_longitude_pair(point₁[1], point₂[1])
    φ₁ = Float64(point₁[2])
    φ₂ = Float64(point₂[2])
    return ((λ₁ + λ₂) / 2, (φ₁ + φ₂) / 2)
end

function continuous_longitude_ring(points::NTuple{4, Tuple{<:Real, <:Real}})
    unwrapped = Vector{Tuple{Float64, Float64}}(undef, 4)
    previous_longitude = Float64(points[1][1])
    unwrapped[1] = (previous_longitude, Float64(points[1][2]))

    for n in 2:4
        longitude = Float64(points[n][1])
        latitude = Float64(points[n][2])

        while longitude - previous_longitude > 180
            longitude -= 360
        end

        while longitude - previous_longitude < -180
            longitude += 360
        end

        unwrapped[n] = (longitude, latitude)
        previous_longitude = longitude
    end

    return unwrapped
end

@inline geometry_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid
@inline geometry_grid(grid) = grid

@inline function center_horizontal_size(grid)
    underlying = geometry_grid(grid)
    Nx, Ny, _ = size(underlying)
    return underlying isa FPivotTripolarGrid ? (Nx, Ny - 1) : (Nx, Ny)
end

@inline function face_basis_u(grid, i, j)
    underlying = geometry_grid(grid)
    underlying isa LatitudeLongitudeGrid && return 1.0, 0.0
    Nx, Ny = center_horizontal_size(grid)
    ii = clamp(i, 1, Nx)
    jj = clamp(j, 1, Ny)
    east, north = extrinsic_vector(ii, jj, 1, underlying, 1.0, 0.0)
    return Float64(east), Float64(north)
end

@inline function face_basis_v(grid, i, j)
    underlying = geometry_grid(grid)
    underlying isa LatitudeLongitudeGrid && return 0.0, 1.0
    Nx, Ny = center_horizontal_size(grid)
    ii = clamp(i, 1, Nx)
    jj = clamp(j, 1, Ny)
    east, north = extrinsic_vector(ii, jj, 1, underlying, 0.0, 1.0)
    return Float64(east), Float64(north)
end

@inline function face_node_longitude_latitude(grid, i, j)
    underlying = geometry_grid(grid)
    longitude = ξnode(i, j, 1, underlying, Face(), Face(), nothing)
    latitude = ηnode(i, j, 1, underlying, Face(), Face(), nothing)
    return Float64(longitude), Float64(latitude)
end

@inline function source_u_segment_endpoints(grid, i, j)
    point₁ = face_node_longitude_latitude(grid, i, j)
    point₂ = face_node_longitude_latitude(grid, i, j + 1)
    λ₁, λ₂ = unwrap_longitude_pair(point₁[1], point₂[1])
    return λ₁, point₁[2], λ₂, point₂[2]
end

@inline function source_v_segment_endpoints(grid, i, j)
    point₁ = face_node_longitude_latitude(grid, i, j)
    point₂ = face_node_longitude_latitude(grid, i + 1, j)
    λ₁, λ₂ = unwrap_longitude_pair(point₁[1], point₂[1])
    return λ₁, point₁[2], λ₂, point₂[2]
end

function destination_u_control_polygon_points(grid, i, j)
    southwest = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i - 1, j),
                                            face_node_longitude_latitude(grid, i, j))
    southeast = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i, j),
                                            face_node_longitude_latitude(grid, i + 1, j))
    northeast = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i, j + 1),
                                            face_node_longitude_latitude(grid, i + 1, j + 1))
    northwest = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i - 1, j + 1),
                                            face_node_longitude_latitude(grid, i, j + 1))
    return continuous_longitude_ring((southwest, southeast, northeast, northwest))
end

function destination_v_control_polygon_points(grid, i, j)
    southwest = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i, j - 1),
                                            face_node_longitude_latitude(grid, i, j))
    southeast = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i + 1, j - 1),
                                            face_node_longitude_latitude(grid, i + 1, j))
    northeast = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i + 1, j),
                                            face_node_longitude_latitude(grid, i + 1, j + 1))
    northwest = midpoint_longitude_latitude(face_node_longitude_latitude(grid, i, j),
                                            face_node_longitude_latitude(grid, i, j + 1))
    return continuous_longitude_ring((southwest, southeast, northeast, northwest))
end

function polygon_wkt(points::AbstractVector{<:Tuple{<:Real, <:Real}}, shift::Real)
    io = IOBuffer()
    print(io, "POLYGON((")

    for point in points
        print(io, point[1] + shift, " ", point[2], ",")
    end

    print(io, points[1][1] + shift, " ", points[1][2], "))")
    return String(take!(io))
end

@inline function polyline_length(coordinates)
    length(coordinates) < 2 && return 0.0
    length_sum = 0.0

    @inbounds for n in 1:length(coordinates)-1
        x₁ = Float64(coordinates[n][1])
        y₁ = Float64(coordinates[n][2])
        x₂ = Float64(coordinates[n + 1][1])
        y₂ = Float64(coordinates[n + 1][2])
        length_sum += hypot(x₂ - x₁, y₂ - y₁)
    end

    return length_sum
end

overlap_line_length(geometry) = overlap_line_length(geometry, GeoInterface.geomtrait(geometry))
overlap_line_length(geometry, ::GeoInterface.LineStringTrait) =
    polyline_length(GeoInterface.coordinates(geometry))

function overlap_line_length(geometry, ::GeoInterface.MultiLineStringTrait)
    length_sum = 0.0
    for coordinates in GeoInterface.coordinates(geometry)
        length_sum += polyline_length(coordinates)
    end
    return length_sum
end

function overlap_line_length(geometry, ::GeoInterface.GeometryCollectionTrait)
    length_sum = 0.0
    for n in 1:GeoInterface.ngeom(geometry)
        length_sum += overlap_line_length(GeoInterface.getgeom(geometry, n))
    end
    return length_sum
end

overlap_line_length(_geometry, _trait) = 0.0

struct SourceSegmentMetadata{T}
    source_kind :: UInt8
    source_face_index :: Int
    east_component :: T
    north_component :: T
    segment_length :: T
end

struct VelocityTransportRegridder{Muu, Muv, Mvu, Mvv}
    Wuu :: Muu
    Wuv :: Muv
    Wvu :: Mvu
    Wvv :: Mvv
    source_u_size :: NTuple{2, Int}
    source_v_size :: NTuple{2, Int}
    destination_u_size :: NTuple{2, Int}
    destination_v_size :: NTuple{2, Int}
end

function build_source_segments(source_u, source_v)
    grid = source_u.grid
    Nxᵤ, Nyᵤ, _ = size(source_u)
    Nxᵥ, Nyᵥ, _ = size(source_v)
    geometries = LibGEOS.LineString[]
    metadata = SourceSegmentMetadata{Float64}[]
    lookup = IdDict{LibGEOS.LineString, Int}()

    for j in 1:Nyᵤ, i in 1:Nxᵤ
        λ₁, φ₁, λ₂, φ₂ = source_u_segment_endpoints(grid, i, j)
        segment_length = hypot(λ₂ - λ₁, φ₂ - φ₁)

        if isfinite(segment_length) && segment_length > eps(Float64)
            segment = LibGEOS.LineString([[λ₁, φ₁], [λ₂, φ₂]])
            east, north = face_basis_u(grid, i, j)
            index = face_linear_index(i, j, Nxᵤ)
            push!(geometries, segment)
            push!(metadata, SourceSegmentMetadata(0x01, index, east, north, segment_length))
            lookup[segment] = length(geometries)
        end
    end

    for j in 1:Nyᵥ, i in 1:Nxᵥ
        λ₁, φ₁, λ₂, φ₂ = source_v_segment_endpoints(grid, i, j)
        segment_length = hypot(λ₂ - λ₁, φ₂ - φ₁)

        if isfinite(segment_length) && segment_length > eps(Float64)
            segment = LibGEOS.LineString([[λ₁, φ₁], [λ₂, φ₂]])
            east, north = face_basis_v(grid, i, j)
            index = face_linear_index(i, j, Nxᵥ)
            push!(geometries, segment)
            push!(metadata, SourceSegmentMetadata(0x02, index, east, north, segment_length))
            lookup[segment] = length(geometries)
        end
    end

    return geometries, metadata, lookup
end

function accumulate_face_weights!(row_u, column_u, value_u,
                                  row_v, column_v, value_v,
                                  destination_index,
                                  destination_east,
                                  destination_north,
                                  points,
                                  tree,
                                  source_lookup,
                                  source_metadata)
    for shift in (-360.0, 0.0, 360.0)
        polygon = LibGEOS.readgeom(polygon_wkt(points, shift))
        candidates = LibGEOS.query(tree, polygon)

        for segment in candidates
            metadata = source_metadata[source_lookup[segment]]
            overlap = overlap_line_length(LibGEOS.intersection(segment, polygon))
            overlap <= 0 && continue
            fraction = overlap / max(metadata.segment_length, eps(Float64))
            projection = metadata.east_component * destination_east +
                         metadata.north_component * destination_north
            weight = fraction * projection

            if metadata.source_kind == 0x01
                push!(row_u, destination_index)
                push!(column_u, metadata.source_face_index)
                push!(value_u, weight)
            else
                push!(row_v, destination_index)
                push!(column_v, metadata.source_face_index)
                push!(value_v, weight)
            end
        end
    end

    return nothing
end

function reconcile_column_targets(weights, targets)
    column_sum = vec(sum(weights; dims=1))
    scales = zeros(Float64, length(targets))

    @inbounds for column in eachindex(scales)
        denominator = column_sum[column]
        target = Float64(targets[column])
        scales[column] = abs(denominator) <= eps(Float64) ? 0.0 : target / denominator
    end

    return weights * Diagonal(scales)
end

function Diagnostics.velocity_transport_regridder(destination_grid,
                                                    source_grid)
    cpu_source_grid = on_architecture(CPU(), source_grid)
    cpu_destination_grid = on_architecture(CPU(), destination_grid)
    destination_underlying = geometry_grid(cpu_destination_grid)
    destination_underlying isa LatitudeLongitudeGrid ||
        throw(ArgumentError("the destination grid must be a LatitudeLongitudeGrid or wrap one"))

    source_u = XFaceField(cpu_source_grid)
    source_v = YFaceField(cpu_source_grid)
    destination_u = XFaceField(cpu_destination_grid)
    destination_v = YFaceField(cpu_destination_grid)

    Nxˢᵤ, Nyˢᵤ, _ = size(source_u)
    Nxˢᵥ, Nyˢᵥ, _ = size(source_v)
    Nxᵈᵤ, Nyᵈᵤ, _ = size(destination_u)
    Nxᵈᵥ, Nyᵈᵥ, _ = size(destination_v)

    nˢᵤ = Nxˢᵤ * Nyˢᵤ
    nˢᵥ = Nxˢᵥ * Nyˢᵥ
    nᵈᵤ = Nxᵈᵤ * Nyᵈᵤ
    nᵈᵥ = Nxᵈᵥ * Nyᵈᵥ

    segments, source_metadata, source_lookup = build_source_segments(source_u, source_v)
    tree = LibGEOS.STRtree(segments)

    row_uu = Int[]
    column_uu = Int[]
    value_uu = Float64[]
    row_uv = Int[]
    column_uv = Int[]
    value_uv = Float64[]
    row_vu = Int[]
    column_vu = Int[]
    value_vu = Float64[]
    row_vv = Int[]
    column_vv = Int[]
    value_vv = Float64[]

    source_u_east = zeros(Float64, nˢᵤ)
    source_u_north = zeros(Float64, nˢᵤ)
    source_v_east = zeros(Float64, nˢᵥ)
    source_v_north = zeros(Float64, nˢᵥ)

    for metadata in source_metadata
        if metadata.source_kind == 0x01
            source_u_east[metadata.source_face_index] = metadata.east_component
            source_u_north[metadata.source_face_index] = metadata.north_component
        else
            source_v_east[metadata.source_face_index] = metadata.east_component
            source_v_north[metadata.source_face_index] = metadata.north_component
        end
    end

    for j in 1:Nyᵈᵤ, i in 1:Nxᵈᵤ
        destination_index = face_linear_index(i, j, Nxᵈᵤ)
        east, north = face_basis_u(cpu_destination_grid, i, j)
        points = destination_u_control_polygon_points(cpu_destination_grid, i, j)
        accumulate_face_weights!(row_uu, column_uu, value_uu,
                                 row_uv, column_uv, value_uv,
                                 destination_index, east, north, points,
                                 tree, source_lookup, source_metadata)
    end

    for j in 1:Nyᵈᵥ, i in 1:Nxᵈᵥ
        destination_index = face_linear_index(i, j, Nxᵈᵥ)
        east, north = face_basis_v(cpu_destination_grid, i, j)
        points = destination_v_control_polygon_points(cpu_destination_grid, i, j)
        accumulate_face_weights!(row_vu, column_vu, value_vu,
                                 row_vv, column_vv, value_vv,
                                 destination_index, east, north, points,
                                 tree, source_lookup, source_metadata)
    end

    Wuu = sparse(row_uu, column_uu, value_uu, nᵈᵤ, nˢᵤ)
    Wuv = sparse(row_uv, column_uv, value_uv, nᵈᵤ, nˢᵥ)
    Wvu = sparse(row_vu, column_vu, value_vu, nᵈᵥ, nˢᵤ)
    Wvv = sparse(row_vv, column_vv, value_vv, nᵈᵥ, nˢᵥ)

    Wuu = reconcile_column_targets(Wuu, source_u_east)
    Wuv = reconcile_column_targets(Wuv, source_v_east)
    Wvu = reconcile_column_targets(Wvu, source_u_north)
    Wvv = reconcile_column_targets(Wvv, source_v_north)

    return VelocityTransportRegridder(Wuu, Wuv, Wvu, Wvv,
                                      (Nxˢᵤ, Nyˢᵤ), (Nxˢᵥ, Nyˢᵥ),
                                      (Nxᵈᵤ, Nyᵈᵤ), (Nxᵈᵥ, Nyᵈᵥ))
end
