using CubedSphere.SphericalGeometry: lat_lon_to_cartesian, cartesian_to_lat_lon,
                                     spherical_area_quadrilateral
using Distances: haversine
using Oceananigans.BoundaryConditions: fill_halo_regions!, BoundaryCondition, Zipper, FPivot,
                                       NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!
using Oceananigans.Grids: RightCenterFolded, RightFaceFolded, generate_coordinate, longitude_in_same_window,
                          peripheral_node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, distribute_tripolar_grid
using Printf: @sprintf

using ..DataWrangling: dataset_variable_name, default_download_directory
using ..DataWrangling.ORCA: ORCAOne, default_south_rows_to_remove, periodic_overlap, north_fold_pivot

# Build an Oceananigans OrthogonalSphericalShellGrid with topology (Periodic, fold, Bounded) from a NEMO eORCA
# mesh_mask file, where the fold is `RightFaceFolded` for an F-pivot mesh (eORCA1) and `RightCenterFolded` with a
# `TPivot` for a T-pivot mesh (eORCA025, eORCA12).
#
# NEMO C-grid: T is the cell center, U the east face of T, V the north face of T, F the northeast corner.
# eORCA quirks handled before constructing the grid:
#
#   - Duplicated east-edge periodic columns (`periodic_overlap`, `shift_face_x`, `chop`).
#   - Optional southern land padding rows (`south_rows_to_remove`, `chop`).
#   - The last row, a mirror copy of the rows below the fold (`chop`).
#
# NEMO → Oceananigans index mapping:
#
#   Center-y[:, 1:Ny] ← NEMO[:, 1:Ny]
#   Face-y  [:, 2:]   ← NEMO V/F[:, 1:]     (NEMO V is the north face of T[:, j] = south face of T[:, j+1])
#
# Face-y row j=1 has no NEMO counterpart; `halo_filled_data` copies the southernmost data row into it
# (cf. Oceananigans #5565 — copy the j-row instead of using a LatitudeLongitudeGrid). `fill_halo_regions!`
# then propagates it south using PeriodicBC east/west, NoFluxBC south, and the zipper north.
#
# `read_orca_staggered_mesh` supports two read paths: a full staggered NEMO mesh used directly, or T/F
# coordinates only, with U/V coordinates and all `e1`/`e2`/`Az` metrics reconstructed from spherical
# midpoints, haversine distances, and spherical quadrilateral areas. Both paths return arrays of size
# `(Nx, Ny)` shifted in x but not in y, so the mapping above applies once to either. The reconstruction
# resolves spacings down to the coordinate precision of the file: near the two tripolar north poles the
# true spacing falls below Float32 resolution, so a file storing Float32 `glamt`/`gphit` without `e1`/`e2`
# yields coincident points and zero spacings there. Bathymetry (when
# `with_bathymetry = true`): NEMO stores positive depth, so we negate it and map land (depth ≤ 0 or
# missing) to +100 so `GridFittedBottom` masks it. `major_basins` optionally drops smaller disconnected
# basins via `remove_minor_basins!`.

"""
    read_2d_nemo_variable(ds, name)

Read a 2D variable from a NEMO NetCDF dataset, handling varying
dimension layouts: `(x, y)`, `(x, y, z)`, or `(x, y, z, t)`.
"""
function read_2d_nemo_variable(ds, name)
    var = ds[name]
    data = Array(var)

    if ndims(data) < 2
        throw(ArgumentError("Variable $name could not be reduced to 2D. Size after slicing: $(size(data))"))
    end

    if ndims(data) > 2
        sizes = collect(size(data))
        keep = sort(sortperm(sizes; rev = true)[1:2])
        indices = ntuple(d -> (d in keep ? Colon() : 1), ndims(data))
        data = @view data[indices...]
    end

    if ndims(data) != 2
        throw(ArgumentError("Variable $name could not be reduced to 2D. Size after slicing: $(size(data))"))
    end

    return Array(data)
end

has_all_variables(ds, names) = all(name -> name in keys(ds), names)

function orient_xy(data, Nx, Ny; name = "variable")
    sx, sy = size(data)
    if (sx, sy) == (Nx, Ny)
        return data
    elseif (sx, sy) == (Ny, Nx)
        return permutedims(data, (2, 1))
    else
        throw(ArgumentError("Cannot orient $name with size $(size(data)) to (Nx, Ny)=($Nx, $Ny)."))
    end
end

@inline function midpoint_longitude(λ₁, λ₂)
    Δλ = longitude_in_same_window(λ₂, λ₁) - λ₁
    return longitude_in_same_window(λ₁ + Δλ / 2, 0)
end

@inline function spherical_midpoint(λ₁, φ₁, λ₂, φ₂)
    x₁, y₁, z₁ = lat_lon_to_cartesian(φ₁, λ₁; radius = 1, check_latitude_bounds = false)
    x₂, y₂, z₂ = lat_lon_to_cartesian(φ₂, λ₂; radius = 1, check_latitude_bounds = false)
    x = x₁ + x₂
    y = y₁ + y₂
    z = z₁ + z₂
    n = sqrt(x^2 + y^2 + z^2)

    if n < 1e-12
        λm = midpoint_longitude(λ₁, λ₂)
        φm = (φ₁ + φ₂) / 2
        return λm, φm
    end

    x /= n
    y /= n
    z /= n

    φm, λm = cartesian_to_lat_lon(x, y, z)
    return longitude_in_same_window(λm, 0), φm
end

@inline function spherical_quadrilateral_area_unit(λ₁, φ₁, λ₂, φ₂, λ₃, φ₃, λ₄, φ₄)
    a = lat_lon_to_cartesian(φ₁, λ₁; radius = 1, check_latitude_bounds = false)
    b = lat_lon_to_cartesian(φ₂, λ₂; radius = 1, check_latitude_bounds = false)
    c = lat_lon_to_cartesian(φ₃, λ₃; radius = 1, check_latitude_bounds = false)
    d = lat_lon_to_cartesian(φ₄, λ₄; radius = 1, check_latitude_bounds = false)
    return spherical_area_quadrilateral(a, b, c, d; radius = 1)
end

# The duplicated east-edge columns (`periodic_overlap`) alias the west edge: column `i` and column
# `Nx - overlap + i` hold the same point. Wrapping to `1`/`Nx` would therefore land on a copy of the
# starting column and yield a zero spacing, so the wrap skips the duplicates.
@inline east_idx(i, Nx, overlap) = ifelse(i == Nx, overlap + 1, i + 1)
@inline west_idx(i, Nx, overlap) = ifelse(i == 1, Nx - overlap, i - 1)

# `λFF`/`φFF` are shifted in x only, matching the staggered read path: `FF[i, j]` is the north-west
# corner of `T[i, j]` and `CF[i, j]` its north face, both in NEMO's y-indexing. `halo_filled_data`
# applies the single +1 y-shift to the Face-y quantities once the mesh is assembled.
@kernel function _reconstruct_λFC_φFC_λCF_φCF!(λFC, φFC, λCF, φCF, λCC, φCC, λFF, φFF, Nx, Ny, overlap)
    i, j = @index(Global, NTuple)
    iE = east_idx(i, Nx, overlap)
    iW = west_idx(i, Nx, overlap)
    λm₁, φm₁ = spherical_midpoint(λCC[iW, j], φCC[iW, j], λCC[i, j], φCC[i, j])
    λFC[i, j] = λm₁
    φFC[i, j] = φm₁
    λm₂, φm₂ = spherical_midpoint(λFF[i, j], φFF[i, j], λFF[iE, j], φFF[iE, j])
    λCF[i, j] = λm₂
    φCF[i, j] = φm₂
end

# Every metric is a distance between points written by `_reconstruct_λFC_φFC_λCF_φCF!` in a prior launch,
# so no thread reads a value another thread produces in this launch.
@kernel function _reconstruct_e1_e2_metrics!(e1u, e1v, e1f, e1t, e2u, e2v, e2f, e2t, λCC, φCC, λFF, φFF, λFC, φFC, λCF, φCF, radius, Nx, Ny, overlap)
    i, j = @index(Global, NTuple)
    iE = east_idx(i, Nx, overlap)
    iW = west_idx(i, Nx, overlap)

    e1t[i, j] = haversine((λFC[i, j],  φFC[i, j]),  (λFC[iE, j], φFC[iE, j]), radius)
    e1u[i, j] = haversine((λCC[iW, j], φCC[iW, j]), (λCC[i, j],  φCC[i, j]),  radius)
    e1v[i, j] = haversine((λFF[i, j],  φFF[i, j]),  (λFF[iE, j], φFF[iE, j]), radius)
    e1f[i, j] = haversine((λCF[iW, j], φCF[iW, j]), (λCF[i, j],  φCF[i, j]),  radius)

    if Ny == 1
        e2t[i, j] = e1t[i, j]
        e2u[i, j] = e1u[i, j]
        e2v[i, j] = e1v[i, j]
        e2f[i, j] = e1f[i, j]
    else
        # South of the first V row only the half-cell T→V is available, so double it.
        if j > 1
            e2t[i, j] = haversine((λCF[i, j-1], φCF[i, j-1]), (λCF[i, j], φCF[i, j]), radius)
            e2u[i, j] = haversine((λFF[i, j-1], φFF[i, j-1]), (λFF[i, j], φFF[i, j]), radius)
        else
            e2t[i, 1] = 2 * haversine((λCC[i, 1], φCC[i, 1]), (λCF[i, 1], φCF[i, 1]), radius)
            e2u[i, 1] = 2 * haversine((λFC[i, 1], φFC[i, 1]), (λFF[i, 1], φFF[i, 1]), radius)
        end

        # The north row reuses the last interior difference rather than copying a neighbor's output.
        if j < Ny
            e2v[i, j] = haversine((λCC[i, j], φCC[i, j]), (λCC[i, j+1], φCC[i, j+1]), radius)
            e2f[i, j] = haversine((λFC[i, j], φFC[i, j]), (λFC[i, j+1], φFC[i, j+1]), radius)
        else
            e2v[i, Ny] = haversine((λCC[i, Ny-1], φCC[i, Ny-1]), (λCC[i, Ny], φCC[i, Ny]), radius)
            e2f[i, Ny] = haversine((λFC[i, Ny-1], φFC[i, Ny-1]), (λFC[i, Ny], φFC[i, Ny]), radius)
        end
    end
end

@kernel function _reconstruct_Az_interior!(AzCC, AzFF, λCC, φCC, λFF, φFF, radius, Nx, Ny, overlap)
    i, j = @index(Global, NTuple)
    iE = east_idx(i, Nx, overlap)
    iW = west_idx(i, Nx, overlap)
    if j > 1
        A = spherical_quadrilateral_area_unit(λFF[i, j-1],  φFF[i, j-1],
                                              λFF[iE, j-1], φFF[iE, j-1],
                                              λFF[iE, j],   φFF[iE, j],
                                              λFF[i, j],    φFF[i, j])
        AzCC[i, j] = A * radius^2
    end
    if j < Ny
        A = spherical_quadrilateral_area_unit(λCC[iW, j],   φCC[iW, j],
                                              λCC[i, j],    φCC[i, j],
                                              λCC[i, j+1],  φCC[i, j+1],
                                              λCC[iW, j+1], φCC[iW, j+1])
        AzFF[i, j] = A * radius^2
    end
end

@kernel function _fill_AzCC_boundaries!(AzCC, AzFF, Ny)
    i = @index(Global, Linear)
    AzCC[i, 1] = AzCC[i, 2]
    AzFF[i, Ny] = AzFF[i, Ny-1]
end

function reconstruct_orca_mesh_from_CC_FF_points(λCC, φCC, λFF, φFF, overlap; radius)
    size(λCC) == size(φCC) || throw(ArgumentError("glamt and gphit size mismatch: $(size(λCC)) vs $(size(φCC))."))
    size(λFF) == size(φFF) || throw(ArgumentError("glamf and gphif size mismatch: $(size(λFF)) vs $(size(φFF))."))
    size(λCC) == size(λFF) || throw(ArgumentError("T-point and F-point grids must have matching size, got $(size(λCC)) and $(size(λFF))."))

    Nx, Ny = size(λCC)
    AFT = promote_type(eltype(λCC), eltype(φCC), eltype(λFF), eltype(φFF), typeof(radius))

    λFFₒ = shift_face_x(λFF, overlap)
    φFFₒ = shift_face_x(φFF, overlap)

    λFC  = similar(λCC, AFT)
    φFC  = similar(φCC, AFT)
    λCF  = similar(λCC, AFT)
    φCF  = similar(φCC, AFT)
    dev  = Oceananigans.Architectures.device(architecture(λFC))

    _reconstruct_λFC_φFC_λCF_φCF!(dev, (16, 16), (Nx, Ny))(λFC, φFC, λCF, φCF, λCC, φCC, λFFₒ, φFFₒ, Nx, Ny, overlap)

    e1u = similar(λCC, AFT)
    e2u = similar(λCC, AFT)
    e1v = similar(λCC, AFT)
    e2v = similar(λCC, AFT)
    e1f = similar(λCC, AFT)
    e2f = similar(λCC, AFT)
    e1t = similar(λCC, AFT)
    e2t = similar(λCC, AFT)

    _reconstruct_e1_e2_metrics!(dev, (16, 16), (Nx, Ny))(e1u, e1v, e1f, e1t, e2u, e2v, e2f, e2t, λCC, φCC, λFFₒ, φFFₒ, λFC, φFC, λCF, φCF, radius, Nx, Ny, overlap)

    AzCC = similar(λCC, AFT)
    AzFC = e1u .* e2u
    AzCF = e1v .* e2v
    AzFF = similar(λCC, AFT)

    if Ny > 1
        _reconstruct_Az_interior!(dev, (16, 16), (Nx, Ny))(AzCC, AzFF, λCC, φCC, λFFₒ, φFFₒ, radius, Nx, Ny, overlap)
        _fill_AzCC_boundaries!(dev, 16, Nx)(AzCC, AzFF, Ny)
    else
        AzCC .= e1t .* e2t
        AzFF .= AzCC
    end

    return (; λCC, λFC, λCF, λFF = λFFₒ, φCC, φFC, φCF, φFF = φFFₒ,
              e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
              AzCC, AzFC, AzCF, AzFF)
end

"""
    read_orca_staggered_mesh(ds, overlap; radius, straits = ())

Read ORCA horizontal coordinates and metrics.

Supports:
- full NEMO staggered mesh variables (`glamt/gphit/e1u/...`), and
- approximate reconstruction from T/F coordinates only (`glamt/gphit/glamf/gphif`)
  using Tripolar-style spherical metric assumptions.

The returned mesh also carries `straits`, the channel cells of the mesh's narrowed straits in NEMO's
indexing (see [`narrow_strait_cells!`](@ref)), with any `straits` overrides applied.
"""
function read_orca_staggered_mesh(ds, overlap; radius = Oceananigans.defaults.planet_radius, straits = ())
    metrics = ("glamt", "glamu", "glamv", "glamf",
               "gphit", "gphiu", "gphiv", "gphif",
               "e1t", "e1u", "e1v", "e1f",
               "e2t", "e2u", "e2v", "e2f")

    λCC = read_2d_nemo_variable(ds, "glamt")
    Nx, Ny = size(λCC)

    orcaread(data, name) = orient_xy(read_2d_nemo_variable(data, name), Nx, Ny; name)
    shift_x(data) = shift_face_x(data, overlap)

    # Face-y: no pre-shift here; halo_filled_data does the +1 y-shift after chop.
    if has_all_variables(ds, metrics)
        λCC, λFC, λCF, λFF = orcaread(ds, "glamt"), shift_x(orcaread(ds, "glamu")), orcaread(ds, "glamv"), shift_x(orcaread(ds, "glamf"))
        φCC, φFC, φCF, φFF = orcaread(ds, "gphit"), shift_x(orcaread(ds, "gphiu")), orcaread(ds, "gphiv"), shift_x(orcaread(ds, "gphif"))
        e1t, e1u, e1v, e1f = orcaread(ds, "e1t"),   orcaread(ds, "e1u"),   orcaread(ds, "e1v"),   orcaread(ds, "e1f")
        e2t, e2u, e2v, e2f = orcaread(ds, "e2t"),   orcaread(ds, "e2u"),   orcaread(ds, "e2v"),   orcaread(ds, "e2f")

        if "e1e2t" in keys(ds)
            AzCC, AzFC = orcaread(ds, "e1e2t"), orcaread(ds, "e1e2u")
            AzCF, AzFF = orcaread(ds, "e1e2v"), orcaread(ds, "e1e2f")
        else
            AzCC, AzFC, AzCF, AzFF = e1t .* e2t, e1u .* e2u, e1v .* e2v, e1f .* e2f
        end

        # In NEMO's own indexing, before the u and F points are shifted to Oceananigans' convention
        channels = narrow_strait_cells!(e1t, e2t, AzCC, AzFC, AzCF, e1u, e2u, e1v, e2v,
                                        orcaread(ds, "glamf"), orcaread(ds, "gphif"), λCC, φCC; radius, straits)

        e1u, e1f, e2u, e2f = shift_x(e1u), shift_x(e1f), shift_x(e2u), shift_x(e2f)
        AzFC, AzFF = shift_x(AzFC), shift_x(AzFF)

        return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
                  e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
                  AzCC, AzFC, AzCF, AzFF, straits = channels)
    end

    # a reconstructed mesh has no stored metrics, so there are no narrowed straits to set
    isempty(straits) || throw(ArgumentError("`straits` needs a mesh with NEMO's stored metrics (e1u, e2u, e1v, ...)"))

    coords = ("glamt", "gphit", "glamf", "gphif")
    if has_all_variables(ds, coords)
        λCC = orcaread(ds, "glamt")
        λFF = orcaread(ds, "glamf")
        φCC = orcaread(ds, "gphit")
        φFF = orcaread(ds, "gphif")
        return merge(reconstruct_orca_mesh_from_CC_FF_points(λCC, φCC, λFF, φFF, overlap; radius), (; straits = ()))
    end

    throw(ArgumentError("Unsupported ORCA mesh: needs staggered variables $(metrics) or T/F variables $(coords)"))
end

"""
    narrow_strait_cells!(e1t, e2t, AzCC, AzFC, AzCF, e1u, e2u, e1v, e2v, λFF, φFF, λCC, φCC;
                         radius, threshold = 0.9, straits = ())

Make the cell geometry at NEMO's narrowed straits consistent with their narrowed faces, after setting the
widths given in `straits`, and return the straits' channel cells.

NEMO represents a strait narrower than a grid cell by cutting the width of its velocity faces, `e1v` for a
north-south strait and `e2u` for an east-west one, and leaves the tracer cell between them and the
velocity cells' areas at full size (`e1e2u`, `e1e2v` hold the original areas). The face then carries the
strait's transport, but the tracer cell holds a 110 km-wide column of water behind a 13 km (say) face,
and the velocity cell's area no longer matches its face. Here the geometry is made consistent:

* a tracer cell between two narrowed faces (the strait itself) takes their width, so its volume and
  surface area are those of the channel;
* a velocity cell at a narrowed face takes the area `e1v e2v` (`e1u e2u`) of its narrowed face times its
  length, the area for which the pressure work on it equals the work done by the transport through the
  face, so the discrete kinetic energy budget closes.

The face widths themselves are left narrowed: they set the transport through the strait. A velocity cell
between the channel and the open sea is then handed half of the open column's vertical transport over a
small area, so its vertical advection runs at a Courant number several times the columns' own; the
transport-based CFL of `AdaptiveVerticallyImplicitDiscretization` treats that part implicitly.

A face counts as narrowed when its width falls short of `threshold` times the distance between its two
corners while the tracer cells either side do not, which leaves out the degenerate padding rows south of
Antarctica, whose metrics are all small. All arrays are in NEMO's indexing: u(i, j) lies between T(i, j)
and T(i+1, j) and is bounded by F(i, j-1) and F(i, j); v(i, j) lies between T(i, j) and T(i, j+1) and is
bounded by F(i-1, j) and F(i, j).

Each entry of `straits` is a `(; longitude, latitude, width, depth)`, matched to the channel cell nearest
`(longitude, latitude)`, which must lie within one grid cell of it. `width` (m) replaces both faces' width
before the geometry is made consistent; a width at or above `threshold` times the faces' corner distance
removes the narrowing, and the faces then take their full corner distance. `depth` is not used here but
returned with the channel, for the bathymetry. Either may be left out to keep the mesh's value.

Returns a vector of `(; i, j, longitude, latitude, width, depth)`, one per channel cell, with `depth`
`nothing` unless `straits` sets it.
"""
function narrow_strait_cells!(e1t, e2t, AzCC, AzFC, AzCF, e1u, e2u, e1v, e2v, λFF, φFF, λCC, φCC;
                              radius, threshold = 0.9, straits = ())
    Nx, Ny = size(e1t)

    narrowed_v = falses(Nx, Ny)
    narrowed_u = falses(Nx, Ny)
    v_corners  = zeros(Nx, Ny)
    u_corners  = zeros(Nx, Ny)

    for j in 2:Ny-1, i in 2:Nx-1
        v_corners[i, j]  = haversine((λFF[i-1, j], φFF[i-1, j]), (λFF[i, j], φFF[i, j]), radius)
        narrowed_v[i, j] = (e1v[i, j] < threshold * v_corners[i, j]) &
                           (e1t[i, j] > threshold * v_corners[i, j]) & (e1t[i, j+1] > threshold * v_corners[i, j])

        u_corners[i, j]  = haversine((λFF[i, j-1], φFF[i, j-1]), (λFF[i, j], φFF[i, j]), radius)
        narrowed_u[i, j] = (e2u[i, j] < threshold * u_corners[i, j]) &
                           (e2t[i, j] > threshold * u_corners[i, j]) & (e2t[i+1, j] > threshold * u_corners[i, j])
    end

    # A channel is the tracer cell between two narrowed faces: the strait itself
    channels = NamedTuple{(:i, :j, :axis), Tuple{Int, Int, Symbol}}[]
    for j in 2:Ny-1, i in 2:Nx-1
        narrowed_v[i, j-1] & narrowed_v[i, j] && push!(channels, (; i, j, axis = :v))
        narrowed_u[i-1, j] & narrowed_u[i, j] && push!(channels, (; i, j, axis = :u))
    end

    channel_faces(c) = c.axis === :v ? ((c.i, c.j-1), (c.i, c.j)) : ((c.i-1, c.j), (c.i, c.j))
    face_widths(c)   = c.axis === :v ? (e1v, v_corners, narrowed_v) : (e2u, u_corners, narrowed_u)

    # Faces whose width `straits` sets, narrowed or not, so their velocity cells' areas follow
    reset_v = copy(narrowed_v)
    reset_u = copy(narrowed_u)
    depths  = Dict{Int, Float64}()
    matched = Dict{Int, Any}()

    for (name, strait) in pairs(straits)
        n = nearest_channel(channels, strait, λCC, φCC, e1t, e2t, radius, name)
        haskey(matched, n) && throw(ArgumentError("straits $(matched[n]) and $name both match the channel cell at NEMO ($(channels[n].i), $(channels[n].j))"))
        matched[n] = name
        c = channels[n]

        width = get(strait, :width, nothing)
        if !isnothing(width)
            widths, corners, narrowed = face_widths(c)
            reset = c.axis === :v ? reset_v : reset_u
            for (i, j) in channel_faces(c)
                narrowed[i, j] = width < threshold * corners[i, j]
                widths[i, j]   = narrowed[i, j] ? width : corners[i, j]
                reset[i, j]    = true
            end
        end

        depth = get(strait, :depth, nothing)
        isnothing(depth) || (depths[n] = depth)
    end

    result = NamedTuple{(:i, :j, :longitude, :latitude, :width, :depth),
                        Tuple{Int, Int, Float64, Float64, Float64, Union{Nothing, Float64}}}[]
    descriptions = String[]

    for (n, c) in enumerate(channels)
        widths, _, narrowed = face_widths(c)
        (f₁, f₂) = channel_faces(c)
        width = (widths[f₁...] + widths[f₂...]) / 2

        if narrowed[f₁...] & narrowed[f₂...]
            if c.axis === :v
                AzCC[c.i, c.j] *= width / e1t[c.i, c.j]
                e1t[c.i, c.j]   = width
            else
                AzCC[c.i, c.j] *= width / e2t[c.i, c.j]
                e2t[c.i, c.j]   = width
            end
        end

        depth = get(depths, n, nothing)
        push!(result, (; i = c.i, j = c.j, longitude = Float64(λCC[c.i, c.j]), latitude = Float64(φCC[c.i, c.j]),
                         width = Float64(width), depth))
        push!(descriptions, @sprintf("(%.1fᵒ, %.1fᵒ) %.1f km", λCC[c.i, c.j], φCC[c.i, c.j], width / 1e3) *
                            (isnothing(depth) ? "" : @sprintf(" %.0f m deep", depth)))
    end

    AzCF[reset_v] .= e1v[reset_v] .* e2v[reset_v]
    AzFC[reset_u] .= e1u[reset_u] .* e2u[reset_u]

    nfaces = count(narrowed_u) + count(narrowed_v)
    isempty(channels) || @info "Narrowed straits: $nfaces faces, channel cells at (λ, φ, width) $(join(descriptions, ", "))"

    return result
end

# The channel nearest the strait's (longitude, latitude), which must lie within one (unnarrowed) cell of it
function nearest_channel(channels, strait, λCC, φCC, e1t, e2t, radius, name)
    location = (strait.longitude, strait.latitude)
    distances = [haversine(location, (λCC[c.i, c.j], φCC[c.i, c.j]), radius) for c in channels]
    isempty(distances) && throw(ArgumentError("strait $name at $location: the mesh has no narrowed straits"))
    d, n = findmin(distances)
    c = channels[n]
    d <= max(e1t[c.i, c.j], e2t[c.i, c.j]) ||
        throw(ArgumentError(@sprintf("strait %s at (%.2fᵒ, %.2fᵒ): the nearest narrowed strait is %.0f km away at (%.2fᵒ, %.2fᵒ)",
                                     name, location..., d / 1e3, λCC[c.i, c.j], φCC[c.i, c.j])))
    return n
end

function shift_face_x(data, overlap)
    Nx = size(data, 1)
    No = Nx - overlap
    return data[vcat(No, 1:Nx-1), :]
end

function halo_filled_data(data, helper_grid, bcs, LX, LY)
    TX, TY, _ = topology(helper_grid)
    Nx, Ny, _ = size(helper_grid)
    Ni = Base.length(LX(), TX(), Nx)
    Nj = Base.length(LY(), TY(), Ny)

    field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    if LY === Face
        field.data[1:Ni, 2:Nj, 1] .= data[1:Ni, 1:Nj-1]
        field.data[1:Ni, 1, 1]    .= data[1:Ni, 1]
    else
        field.data[1:Ni, 1:Nj, 1] .= data[1:Ni, 1:Nj]
    end
    fill_halo_regions!(field)

    return deepcopy(dropdims(field.data, dims = 3))
end

function halo_fill_stagger(CC, FC, CF, FF, helper_grid, bcs)
    return (
        halo_filled_data(CC, helper_grid, bcs, Center, Center),
        halo_filled_data(FC, helper_grid, bcs, Face,   Center),
        halo_filled_data(CF, helper_grid, bcs, Center, Face),
        halo_filled_data(FF, helper_grid, bcs, Face,   Face),
    )
end

# Rank 0 reads the global bottom height and removes the minor basins; the host array is shared with every rank
function global_orca_bottom_height(read_global_bottom_height, grid, arch, FT, major_basins)

    grid isa DistributedGrid || return remove_minor_orca_basins(read_global_bottom_height(), major_basins)

    bottom_height = if arch.local_rank == 0
        convert(Matrix{FT}, remove_minor_orca_basins(read_global_bottom_height(), major_basins))
    else
        Matrix{FT}(undef, 0, 0)
    end

    # the other ranks learn the global shape before the reduction of the field itself
    dimensions = arch.local_rank == 0 ? collect(size(bottom_height)) : zeros(Int, 2)
    Nx, Ny = all_reduce(+, dimensions, arch)

    if arch.local_rank != 0
        bottom_height = zeros(FT, Nx, Ny)
    end

    DistributedComputations.barrier(arch.communicator)

    return all_reduce(+, bottom_height, arch)
end

# The basin labeling depends only on the x-topology and the shape, so a plain grid of the array's size stands in
function remove_minor_orca_basins(bottom_height, major_basins)
    major_basins < Inf || return bottom_height

    Nx, Ny = size(bottom_height)

    labeling_grid = RectilinearGrid(CPU(); size = (Nx, Ny), topology = (Periodic, Bounded, Flat),
                                    x = (0, 1), y = (0, 1))

    bottom_field = Field{Center, Center, Nothing}(labeling_grid)
    set!(bottom_field, bottom_height)
    remove_minor_basins!(bottom_field, major_basins)

    return Array(bottom_field.data[1:Nx, 1:Ny, 1])
end

"""
    ORCAGrid(arch = CPU(), FT::DataType = Float64;
             dataset,
             halo = (4, 4, 4),
             z = (-6000, 0),
             Nz = 50,
             radius = Oceananigans.defaults.planet_radius,
             with_bathymetry = true,
             active_cells_map = true,
             major_basins = Inf,
             south_rows_to_remove = default_south_rows_to_remove(dataset),
             straits = (),
             dir = default_download_directory(dataset))

Construct an `OrthogonalSphericalShellGrid` using coordinate and metric data from a NEMO eORCA `mesh_mask` file.
The topology follows the north fold of the mesh: `(Periodic, RightFaceFolded, Bounded)` for the F-pivot eORCA1,
`(Periodic, RightCenterFolded, Bounded)` with a `TPivot` fold for the T-pivot eORCA025 and eORCA12.

The `dataset` keyword argument specifies which ORCA configuration to use (e.g., `ORCAOne()`, `ORCAQuarter()`, or `ORCATwelfth()`).
The mesh mask and bathymetry files are downloaded automatically via the
`DataWrangling.ORCA` metadata interface.

The horizontal grid (including coordinates, scale factors, and areas) is loaded
directly from the `mesh_mask` NetCDF file. If all staggered NEMO fields are present
(`T`, `U`, `V`, `F` points), they are used directly. If only `T` and `F`
coordinates are available (`glamt/gphit/glamf/gphif`), staggered coordinates and
metrics are reconstructed approximately using Tripolar-style spherical assumptions.
The duplicated columns eORCA carries at its east edge for cyclic exchange are dropped, so `Nx` is the number
of distinct columns: 360 for eORCA1, 1440 for eORCA025 and 4320 for eORCA12.

When `with_bathymetry = true` (the default), the bathymetry is also downloaded
and the grid is returned as an `ImmersedBoundaryGrid` whose bottom is built by `immersed_bottom`.

Positional Arguments
====================

- `arch`: The architecture (e.g., `CPU()` or `GPU()`). Default: `CPU()`.
- `FT`: Floating point type. Default: `Float64`.

Keyword Arguments
=================

- `dataset`: The ORCA dataset to use. Default: `ORCAOne()`. `ORCAQuarter()` (eORCA025, quarter-degree) and `ORCATwelfth()`
             (eORCA12, twelfth-degree) are also supported (eORCA1 data from Zenodo; <https://doi.org/10.5281/zenodo.4436658>).
- `halo`: Halo size tuple `(Hx, Hy, Hz)`. Default: `(4, 4, 4)`.
- `z`: Vertical coordinate specification. Can be a 2-tuple `(z_bottom, z_top)`, an array of z-interfaces,
       or, e.g., an `ExponentialDiscretization`. Default: `(-6000, 0)`.
- `Nz`: Number of vertical levels (only used when `z` is a 2-tuple). Default: `50`.
- `radius`: Planet radius. Default: `Oceananigans.defaults.planet_radius`.
- `with_bathymetry`: If `true`, download the bathymetry and return an `ImmersedBoundaryGrid`. Default: `true`.
- `immersed_bottom`: Constructor of the immersed bottom, called with the bottom height field, e.g.
                     `GridFittedBottom` (full cells), `PartialCellBottom` or `ShavedCellBottom`. Default: `GridFittedBottom`.
- `active_cells_map`: If `true` and `with_bathymetry = true`, build an active cells map
                      for efficient kernel execution over wet cells only. Default: `true`.
- `major_basins`: Number of independent connected ocean basins to retain via
                  [`remove_minor_basins!`](@ref). Basins are removed from smallest to largest;
                  `major_basins = 1` keeps only the largest. Default: `Inf` (keep all basins).
- `minimum_depth`: Minimum depth [m] of the wet columns, defined as a positive value; shallower wet columns are
                   deepened to it. Default: 0.
- `south_rows_to_remove`: Number of southern rows to remove from the eORCA grid.  The "extended" eORCA grid
                          contains degenerate padding rows near Antarctica that are entirely land.
                          Removing them reduces memory usage and computation.
- `straits`: Widths and depths for the mesh's narrowed straits, overriding the mesh and bathymetry files.
             A collection (e.g. a `NamedTuple` of named entries) of `(; longitude, latitude, width, depth)`,
             each matched to the channel cell of the narrowed strait nearest `(longitude, latitude)`, which
             must lie within one grid cell of it. `width` (m) sets both faces of the strait; a width of a
             whole cell or more removes the narrowing. `depth` (m, positive) sets the channel cell's bottom.
             Either may be left out to keep the files' value. Default: `()`, the files' geometry.
             See [`narrow_strait_cells!`](@ref).
- `dir`: Directory to store and look up ORCA files (`mesh_mask` and bathymetry).
         Defaults to the dataset scratch cache via `default_download_directory(dataset)`.
- `subcell_slope_dataset`: Dataset supplying the bathymetric slope within each cell, for an
                           `immersed_bottom` that reconstructs its bottom from cell corners, such as
                           `ShavedCellBottom`. The ORCA bathymetry is defined at cell centers, which
                           fixes the depth of a cell but says nothing about how the bottom tilts across
                           it; a finer dataset, e.g. `ETOPO2022()`, is regridded onto the corners and
                           then corrected so that the mean of each cell's four corners reproduces the
                           ORCA depth exactly. Depths, sills and the land mask therefore remain ORCA's,
                           and only the tilt comes from the finer dataset. Default: `nothing`, which
                           reconstructs the corners from the ORCA centers alone.
"""
function ORCAGrid(arch = CPU(), FT::DataType = Float64;
                  dataset = ORCAOne(),
                  halo = (4, 4, 4),
                  z = (-6000, 0),
                  Nz = 50,
                  radius = Oceananigans.defaults.planet_radius,
                  with_bathymetry = true,
                  immersed_bottom = GridFittedBottom,
                  active_cells_map = true,
                  major_basins = Inf,
                  minimum_depth = 0,
                  south_rows_to_remove = default_south_rows_to_remove(dataset),
                  straits = (),
                  dir = default_download_directory(dataset),
                  subcell_slope_dataset = nothing)

    mesh_meta = Metadatum(:mesh_mask; dataset, dir)
    mesh_mask_path = download(mesh_meta)

    ds = Dataset(mesh_mask_path)
    overlap = periodic_overlap(dataset)
    mesh = read_orca_staggered_mesh(ds, overlap; radius, straits)
    close(ds)

    λCC,  λFC,  λCF,  λFF  = mesh.λCC,  mesh.λFC,  mesh.λCF,  mesh.λFF
    φCC,  φFC,  φCF,  φFF  = mesh.φCC,  mesh.φFC,  mesh.φCF,  mesh.φFF
    e1t,  e1u,  e1v,  e1f  = mesh.e1t,  mesh.e1u,  mesh.e1v,  mesh.e1f
    e2t,  e2u,  e2v,  e2f  = mesh.e2t,  mesh.e2u,  mesh.e2v,  mesh.e2f
    AzCC, AzFC, AzCF, AzFF = mesh.AzCC, mesh.AzFC, mesh.AzCF, mesh.AzFF

    pole_idx = argmin(φFF[:, end])
    north_poles_latitude = φFF[pole_idx, end]
    first_pole_longitude = Float64(λFF[pole_idx, end])

    pivot = north_fold_pivot(dataset)
    fold_topology = pivot === FPivot ? RightFaceFolded : RightCenterFolded

    # The first kept column `ic + 1` aligns NEMO's fold mirror (i ↔ jpi + 1 - i for an F pivot,
    # i ↔ jpi + 2 - i for a T pivot) with the Oceananigans one.
    ic = pivot === FPivot ? 1 : 2
    jr = south_rows_to_remove
    chop(data) = data[ic+1:ic+size(data, 1)-overlap, jr+1:end-1]

    λCC, λFC, λCF, λFF     = chop(λCC),  chop(λFC),  chop(λCF),  chop(λFF)
    φCC, φFC, φCF, φFF     = chop(φCC),  chop(φFC),  chop(φCF),  chop(φFF)
    e1t, e1u, e1v, e1f     = chop(e1t),  chop(e1u),  chop(e1v),  chop(e1f)
    e2t, e2u, e2v, e2f     = chop(e2t),  chop(e2u),  chop(e2v),  chop(e2f)
    AzCC, AzFC, AzCF, AzFF = chop(AzCC), chop(AzFC), chop(AzCF), chop(AzFF)

    Nx, Ny = size(λCC)

    southernmost_latitude = Float64(minimum(φCC))

    Hx, Hy, Hz = halo

    topo = (Periodic, fold_topology, Bounded)
    Lz, z_coord = generate_coordinate(FT, topo, (Nx, Ny, Nz), halo, z, :z, 3, CPU())

    helper_grid = RectilinearGrid(; size = (Nx, Ny), halo = (Hx, Hy),
                                    x = (0, 1), y = (0, 1),
                                    topology = (Periodic, fold_topology, Flat))

    bcs = FieldBoundaryConditions(north  = BoundaryCondition(Zipper{pivot}(), 1),
                                  south  = NoFluxBoundaryCondition(),
                                  west   = Oceananigans.PeriodicBoundaryCondition(),
                                  east   = Oceananigans.PeriodicBoundaryCondition(),
                                  top    = nothing,
                                  bottom = nothing)

    λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ     = halo_fill_stagger(λCC,  λFC,  λCF,  λFF,  helper_grid, bcs)
    φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ     = halo_fill_stagger(φCC,  φFC,  φCF,  φFF,  helper_grid, bcs)
    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ = halo_fill_stagger(e1t,  e1u,  e1v,  e1f,  helper_grid, bcs)
    Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ = halo_fill_stagger(e2t,  e2u,  e2v,  e2f,  helper_grid, bcs)
    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = halo_fill_stagger(AzCC, AzFC, AzCF, AzFF, helper_grid, bcs)

    to_host(data) = map(FT, data)

    # The north fold spans all of x, so the mesh is assembled globally and then cut to this rank's slice
    global_grid = OrthogonalSphericalShellGrid{Periodic, fold_topology, Bounded}(
        CPU(),
        Nx, Ny, Nz,
        Hx, Hy, Hz,
        convert(FT, Lz),
        to_host(λᶜᶜᵃ), to_host(λᶠᶜᵃ), to_host(λᶜᶠᵃ), to_host(λᶠᶠᵃ),
        to_host(φᶜᶜᵃ), to_host(φᶠᶜᵃ), to_host(φᶜᶠᵃ), to_host(φᶠᶠᵃ),
        z_coord,
        to_host(Δxᶜᶜᵃ), to_host(Δxᶠᶜᵃ), to_host(Δxᶜᶠᵃ), to_host(Δxᶠᶠᵃ),
        to_host(Δyᶜᶜᵃ), to_host(Δyᶠᶜᵃ), to_host(Δyᶜᶠᵃ), to_host(Δyᶠᶠᵃ),
        to_host(Azᶜᶜᵃ), to_host(Azᶠᶜᵃ), to_host(Azᶜᶠᵃ), to_host(Azᶠᶠᵃ),
        convert(FT, radius),
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude, fold_topology, pivot)
    )

    underlying_grid = distribute_tripolar_grid(arch, global_grid)

    if !with_bathymetry
        any(!isnothing(strait.depth) for strait in mesh.straits) &&
            throw(ArgumentError("a strait `depth` needs `with_bathymetry = true`"))
        return underlying_grid
    end

    bathy_meta = Metadatum(:bottom_height; dataset, dir)
    # `download` is `@root` internally, so every rank must call it.
    bathymetry_path = download(bathy_meta)

    # Deferred: on a distributed grid only rank 0 reads the dataset.
    read_global_bottom_height = function ()
        bathy_ds   = Dataset(bathymetry_path)
        bathy_name = dataset_variable_name(bathy_meta)
        bathy_data = read_2d_nemo_variable(bathy_ds, bathy_name)
        close(bathy_ds)

        bathy_data = orient_xy(bathy_data, size(bathy_data)...; name = string(bathy_name))

        bathy_data = chop(bathy_data)

        bottom_height  = FT.(coalesce.(bathy_data, FT(0)))
        bottom_height .= ifelse.(isfinite.(bottom_height) .& (bottom_height .> 0), .-bottom_height, FT(100))

        # the channel cells are in NEMO's indexing, which `chop` shifts by `ic` and `jr`
        for strait in mesh.straits
            isnothing(strait.depth) || (bottom_height[strait.i - ic, strait.j - jr] = -strait.depth)
        end

        return bottom_height
    end

    bottom_field = Field{Center, Center, Nothing}(underlying_grid)
    set!(bottom_field, global_orca_bottom_height(read_global_bottom_height, underlying_grid, arch, FT, major_basins))

    if minimum_depth > 0
        launch!(arch, underlying_grid, :xy, _enforce_minimum_depth!, bottom_field, convert(FT, minimum_depth))
    end

    if !isnothing(subcell_slope_dataset)
        corner_field = subcell_slope_corner_bottom_height(underlying_grid, bottom_field, subcell_slope_dataset,
                                                         immersed_bottom)
        return ImmersedBoundaryGrid(underlying_grid, immersed_bottom(corner_field); active_cells_map)
    end

    return ImmersedBoundaryGrid(underlying_grid, immersed_bottom(bottom_field); active_cells_map)
end

"""
    subcell_slope_corner_bottom_height(grid, center_height, dataset; iterations = 50)

Bottom height at cell corners whose four-corner mean reproduces `center_height`, with the structure
within each cell taken from `dataset`.

The corner heights are regridded from `dataset` and then relaxed: each sweep measures how far the mean
of a cell's four corners sits from `center_height` and spreads that mismatch back over the corners. The
fixed point leaves every cell's mean depth equal to `center_height`, so depths, sills and the land mask
stay those of `center_height`, while the tilt across each cell comes from the finer dataset.

A checkerboard in `center_height` lies in the null space of the corner-to-center mean and cannot be
represented by any bilinear surface; the residual of the relaxation is reported and is confined to it.
"""
function subcell_slope_corner_bottom_height(grid, center_height, dataset, immersed_bottom;
                                            iterations = 200, mask_sweeps = 12)
    arch = architecture(grid)
    corner_field = regrid_bathymetry(grid; dataset, location = (Face, Face), major_basins = Inf)
    mismatch = Field{Center, Center, Nothing}(grid)

    # the land mask belongs to `center_height`: a corner with no wet neighbouring cell is land
    launch!(arch, grid, :xy, _mask_dry_corners!, corner_field, center_height, grid)
    fill_halo_regions!(corner_field)

    # A seamount the finer dataset resolves but the ORCA cell cannot would raise a face into a sill that blocks the
    # flow, so each sweep also returns a corner to the depth range of the cells it joins: slopes survive, spurious
    # peaks cannot. The two constraints conflict at a local extremum, where the surface stays smoothed.
    for _ in 1:iterations
        launch!(arch, grid, :xy, _corner_mean_mismatch!, mismatch, corner_field, center_height, grid)
        fill_halo_regions!(mismatch)
        launch!(arch, grid, :xy, _spread_mismatch_to_corners!, corner_field, mismatch, center_height, grid)
        fill_halo_regions!(corner_field)
        launch!(arch, grid, :xy, _bound_corners_by_neighbouring_cells!, corner_field, center_height, grid)
        fill_halo_regions!(corner_field)
    end

    # A cell whose corner mean lands on the wrong side of sea level would move the coastline, so any cell still
    # misclassified after the relaxation falls back to the corners reconstructed from the ORCA centers alone.
    fallback = Field{Face, Face, Nothing}(grid)
    launch!(arch, grid, :xy, _corners_from_centers!, fallback, center_height, grid)
    fill_halo_regions!(fallback)

    # Reverting a corner perturbs its neighbouring cells, which can misclassify those in turn, so a corner that
    # reverts stays reverted: the set only grows, the sweep terminates, and at worst it is the plain reconstruction.
    misclassified = Field{Center, Center, Nothing}(grid)
    reverted = Field{Face, Face, Nothing}(grid)
    for _ in 1:iterations
        launch!(arch, grid, :xy, _flag_misclassified_cells!, misclassified, corner_field, center_height, grid)
        sum(interior(misclassified)) == 0 && break
        fill_halo_regions!(misclassified)
        launch!(arch, grid, :xy, _restore_mask_at_corners!, corner_field, fallback, misclassified, reverted, grid)
        fill_halo_regions!(corner_field)
        fill_halo_regions!(reverted)
    end

    # The sign of the corner mean is only a proxy for the mask: a column's wet cells are decided by the immersed
    # boundary itself, after its own ϵ-limiting. Match that mask to the one reconstructed from the ORCA centers, so
    # that taking the slopes from a finer dataset leaves the coastline untouched.
    reference_mask = wet_column_mask(grid, fallback, immersed_bottom)
    for _ in 1:mask_sweeps
        differing = wet_column_mask(grid, corner_field, immersed_bottom) .!= reference_mask
        sum(differing) == 0 && break
        set!(misclassified, reshape(differing, size(differing)..., 1))
        fill_halo_regions!(misclassified)
        launch!(arch, grid, :xy, _restore_mask_at_corners!, corner_field, fallback, misclassified, reverted, grid)
        fill_halo_regions!(corner_field)
        fill_halo_regions!(reverted)
    end
    remaining = sum(wet_column_mask(grid, corner_field, immersed_bottom) .!= reference_mask)

    launch!(arch, grid, :xy, _corner_mean_mismatch!, mismatch, corner_field, center_height, grid)
    wet_mismatch = abs.(vec(Array(interior(mismatch))))
    @info string("Sub-cell slopes from ", summary(dataset), ": cell-mean depth reproduced to ",
                 prettysummary(median(wet_mismatch)), " m (median), ",
                 prettysummary(maximum(wet_mismatch)), " m (max); ",
                 remaining, " columns differ from the mask of the ORCA-center reconstruction")

    return corner_field
end

# Which columns hold at least one wet cell once `immersed_bottom` has been materialized on `bottom_height`.
function wet_column_mask(grid, bottom_height, immersed_bottom)
    ibg = ImmersedBoundaryGrid(grid, immersed_bottom(bottom_height); active_cells_map = false)
    Nx, Ny, Nz = size(grid)
    mask = falses(Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        for k in 1:Nz   # `break` would leave a fused i, j, k loop entirely, so the column scan is its own loop
            if !peripheral_node(i, j, k, ibg, Center(), Center(), Center())
                mask[i, j] = true
                break
            end
        end
    end
    return mask
end

# The reconstruction `ShavedCellBottom` performs internally: the mean of the wet neighbouring cells, land otherwise.
@kernel function _corners_from_centers!(corner_field, center_height, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Σ = zero(eltype(corner_field))
        n = 0
        for (i′, j′) in ((i-1, j-1), (i, j-1), (i-1, j), (i, j))
            wet = !dry_cell(center_height[i′, j′, 1])
            Σ += ifelse(wet, center_height[i′, j′, 1], zero(Σ))
            n += ifelse(wet, 1, 0)
        end
        corner_field[i, j, 1] = ifelse(n > 0, Σ / max(n, 1), one(Σ) * 100)
    end
end

@kernel function _bound_corners_by_neighbouring_cells!(corner_field, center_height, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        shallowest = -Inf * one(eltype(corner_field))
        deepest = Inf * one(eltype(corner_field))
        for (i′, j′) in ((i-1, j-1), (i, j-1), (i-1, j), (i, j))
            if !dry_cell(center_height[i′, j′, 1])
                shallowest = max(shallowest, center_height[i′, j′, 1])
                deepest = min(deepest, center_height[i′, j′, 1])
            end
        end
        bounded = isfinite(shallowest) & isfinite(deepest)
        corner_field[i, j, 1] = ifelse(bounded, clamp(corner_field[i, j, 1], deepest, shallowest),
                                                corner_field[i, j, 1])
    end
end

@kernel function _flag_misclassified_cells!(misclassified, corner_field, center_height, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        corner_mean = (corner_field[i, j, 1] + corner_field[i+1, j, 1] +
                       corner_field[i, j+1, 1] + corner_field[i+1, j+1, 1]) / 4
        misclassified[i, j, 1] = ifelse(dry_cell(center_height[i, j, 1]) == dry_cell(corner_mean), 0, 1)
    end
end

@kernel function _restore_mask_at_corners!(corner_field, fallback, misclassified, reverted, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        touched = (misclassified[i-1, j-1, 1] + misclassified[i, j-1, 1] +
                   misclassified[i-1, j, 1] + misclassified[i, j, 1]) > 0
        keep = touched | (reverted[i, j, 1] > 0)
        reverted[i, j, 1] = ifelse(keep, 1, 0)
        corner_field[i, j, 1] = ifelse(keep, fallback[i, j, 1], corner_field[i, j, 1])
    end
end

@inline dry_cell(height) = height ≥ 0

# A corner keeps the bathymetry of the finer dataset only where the ORCA mask says there is ocean.
@kernel function _mask_dry_corners!(corner_field, center_height, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        wet = !dry_cell(center_height[i-1, j-1, 1]) | !dry_cell(center_height[i, j-1, 1]) |
              !dry_cell(center_height[i-1, j, 1])   | !dry_cell(center_height[i, j, 1])
        corner_field[i, j, 1] = ifelse(wet, corner_field[i, j, 1], one(eltype(corner_field)) * 100)
    end
end

# Only wet cells constrain the surface; a dry cell reports no mismatch so its corners are left at land.
@kernel function _corner_mean_mismatch!(mismatch, corner_field, center_height, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        corner_mean = (corner_field[i, j, 1] + corner_field[i+1, j, 1] +
                       corner_field[i, j+1, 1] + corner_field[i+1, j+1, 1]) / 4
        wet = !dry_cell(center_height[i, j, 1])
        mismatch[i, j, 1] = ifelse(wet, center_height[i, j, 1] - corner_mean, zero(corner_mean))
    end
end

@kernel function _spread_mismatch_to_corners!(corner_field, mismatch, center_height, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        wet = !dry_cell(center_height[i-1, j-1, 1]) | !dry_cell(center_height[i, j-1, 1]) |
              !dry_cell(center_height[i-1, j, 1])   | !dry_cell(center_height[i, j, 1])
        correction = (mismatch[i-1, j-1, 1] + mismatch[i, j-1, 1] +
                      mismatch[i-1, j, 1] + mismatch[i, j, 1]) / 4
        corner_field[i, j, 1] += ifelse(wet, correction, zero(correction))
    end
end
