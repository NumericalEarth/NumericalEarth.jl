using CubedSphere.SphericalGeometry: lat_lon_to_cartesian, cartesian_to_lat_lon,
                                     spherical_area_quadrilateral
using Distances: haversine
using Oceananigans.BoundaryConditions: fill_halo_regions!, FPivotZipperBoundaryCondition,
                                       NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!, convert_to_0_360
using Oceananigans.DistributedComputations: Distributed, concatenate_local_sizes,
                                            insert_connected_topology, local_size, ranks
using Oceananigans.Grids: FullyConnected, RightFaceFolded, generate_coordinate, halo_size
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Grids: peripheral_node
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, partition_tripolar_metric, receiving_rank

using ..DataWrangling: dataset_variable_name, default_download_directory
using ..DataWrangling.ORCA: ORCAOne, default_south_rows_to_remove, periodic_overlap

# Build an Oceananigans OrthogonalSphericalShellGrid with topology (Periodic, RightFaceFolded, Bounded) from
# a NEMO eORCA mesh_mask file.
#
# NEMO C-grid: T is the cell center, U the east face of T, V the north face of T, F the northeast corner.
# eORCA quirks handled before constructing the grid:
#
<<<<<<< HEAD
#   - Duplicated east-edge periodic columns (`periodic_overlap_index`, `shift_face_x`, `chop`).
=======
#   - Duplicated east-edge periodic columns (`periodic_overlap`, `shift_face_x`, `chop`).
>>>>>>> origin/jsw/bgc-coupling
#   - Optional southern land padding rows (`south_rows_to_remove`, `chop`).
#
# NEMO → Oceananigans index mapping:
#
#   Center-y[:, 1:Ny]   ← NEMO[:, 1:Ny]
#   Face-y  [:, 2:Ny+1] ← NEMO V/F[:, 1:Ny]   (NEMO V is the north face of T[:, j] = south face of T[:, j+1])
#
# Face-y row j=1 has no NEMO counterpart; `halo_filled_data` mirrors the southernmost data row into it
# (cf. Oceananigans #5565 — copy the j-row instead of using a LatitudeLongitudeGrid). `fill_halo_regions!`
# then propagates it south using PeriodicBC east/west, NoFluxBC south, and FPivotZipperBC north.
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

# The duplicated east-edge columns (`periodic_overlap_index`) alias the west edge: column `i` and column
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
    read_orca_staggered_mesh(ds, overlap)

Read ORCA horizontal coordinates and metrics.

Supports:
- full NEMO staggered mesh variables (`glamt/gphit/e1u/...`), and
- approximate reconstruction from T/F coordinates only (`glamt/gphit/glamf/gphif`)
  using Tripolar-style spherical metric assumptions.
"""
function read_orca_staggered_mesh(ds, overlap; radius = Oceananigans.defaults.planet_radius)
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
        e1t, e1u, e1v, e1f = orcaread(ds, "e1t"),   shift_x(orcaread(ds, "e1u")),   orcaread(ds, "e1v"),   shift_x(orcaread(ds, "e1f"))
        e2t, e2u, e2v, e2f = orcaread(ds, "e2t"),   shift_x(orcaread(ds, "e2u")),   orcaread(ds, "e2v"),   shift_x(orcaread(ds, "e2f"))

        if "e1e2t" in keys(ds)
            AzCC, AzFC = orcaread(ds, "e1e2t"), shift_x(orcaread(ds, "e1e2u"))
            AzCF, AzFF = orcaread(ds, "e1e2v"), shift_x(orcaread(ds, "e1e2f"))
        else
            AzCC, AzFC, AzCF, AzFF = e1t .* e2t, e1u .* e2u, e1v .* e2v, e1f .* e2f
        end

        return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
                  e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
                  AzCC, AzFC, AzCF, AzFF)
    end

    coords = ("glamt", "gphit", "glamf", "gphif")
    if has_all_variables(ds, coords)
        λCC = orcaread(ds, "glamt")
        λFF = orcaread(ds, "glamf")
        φCC = orcaread(ds, "gphit")
        φFF = orcaread(ds, "gphif")
        return reconstruct_orca_mesh_from_CC_FF_points(λCC, φCC, λFF, φFF, overlap; radius)
    end

    throw(ArgumentError("Unsupported ORCA mesh: needs staggered variables $(metrics) or T/F variables $(coords)"))
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
    Nj_data = size(data, 2)

    field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    if Nj_data == Nj
        field.data[1:Ni, 1:Nj, 1] .= data[1:Ni, 1:Nj]
    elseif LY === Face && Nj_data == Nj - 1
        field.data[1:Ni, 2:Nj, 1] .= data[1:Ni, 1:Nj-1]
        field.data[1:Ni, 1, 1]    .= data[1:Ni, 1]
    else
        throw(DimensionMismatch("data has $Nj_data rows but $LY field expects $Nj rows"))
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

# Bottom height for the whole globe, as a host array, with the minor basins already removed.
#
# `remove_minor_basins!` labels basins with a flood fill, so it has to see the whole globe at
# once. A field built on a distributed `grid` holds only this rank's slice: labeling that cuts
# every basin at the rank boundaries, ranks the pieces by *local* area, and — with
# `major_basins = 1`, which `build_grid` passes — leaves every rank keeping a different piece of
# a different basin. So the distributed path follows `regrid_bathymetry(::DistributedGrid, ...)`:
# rank 0 alone reads the dataset and removes the minor basins, `all_reduce` shares the global
# array, and the caller's `set!` partitions it.
#
# The result stays on the host deliberately: a distributed `set!` auto-partitions a global-size
# `Array`, but not a `CuArray`.
function global_orca_bottom_height(read_global_bottom_height, grid, arch, FT, major_basins)

    grid isa DistributedGrid ||
        return remove_minor_orca_basins(read_global_bottom_height(), major_basins)

    # Every rank must contribute the same element type to a shared reduction: mismatched MPI
    # datatypes, or mismatched counts, corrupt the collective ("Message truncated").
    bottom_height = if arch.local_rank == 0
        convert(Matrix{FT}, remove_minor_orca_basins(read_global_bottom_height(), major_basins))
    else
        Matrix{FT}(undef, 0, 0)
    end

    # The receiving ranks have not opened the dataset, so they cannot size their buffer from it, and
    # they cannot size it from the grid either: `size(grid)` is this rank's slice, and recovering the
    # global shape from it would bake in the assumption that the model grid and the dataset array
    # agree — which holds for eORCA today only because `south_rows_to_remove` is applied to both.
    # Share the shape first, in its own two-element reduction, and only then the field itself. Getting
    # this wrong does not raise: a count mismatch corrupts the collective ("Message truncated").
    dimensions = arch.local_rank == 0 ? collect(size(bottom_height)) : zeros(Int, 2)
    Nx, Ny = all_reduce(+, dimensions, arch)

    if arch.local_rank != 0
        bottom_height = zeros(FT, Nx, Ny)
    end

    DistributedComputations.barrier(arch.communicator)

    return all_reduce(+, bottom_height, arch)
end

# `remove_minor_basins!` wants a `Field`, but the flood fill behind it reads only the grid's
# x-topology and its `(Nx, Ny)` — never the metrics. So stage the global array through a
# throwaway host grid of the array's own size, rather than reconstructing the eORCA mesh, which
# at 1/12° would mean rebuilding twenty 4320×3606 coordinate arrays to label a land mask.
# `Periodic` in x is the topology every `ORCAGrid` is built with, and `Bounded` in y gives the
# same `Base.length(Center(), TY(), Ny)` as the grid's own `RightFaceFolded`, so the labeling is
# identical to running it on the grid itself.
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

# On a serial architecture there is nothing to distribute: the grid read from the `mesh_mask` file
# already spans the whole globe, and only has to be moved off the host.
distribute_orca_grid(global_grid, arch) = on_architecture(arch, global_grid)

# Cut the global eORCA grid down to this rank's slice.
#
# The eORCA mesh has to be assembled globally — `halo_fill_stagger` closes the northern fold, which
# pairs points on opposite sides of the tripolar seam and so spans all of x — but every rank must end
# up holding only its own piece. Without this, each rank builds and integrates the entire globe: at
# 1/12° that is 4322 × 3146 × 100 cells and ~11 GB per Float64 field per rank.
#
# This is the same partitioning `TripolarGrid(arch::Distributed, ...)` performs, and it reuses that
# implementation's `partition_tripolar_metric`. The two differ only in where the global grid comes
# from: a generated tripolar mesh there, an eORCA `mesh_mask` file here. The supported partitionings
# are therefore identical — y-only, or an x-y pencil with an even x partition.
#
# TODO: this belongs in Oceananigans as `distribute_tripolar_grid(arch, global_grid)`. The body of
# `TripolarGrid(arch::Distributed, ...)` in `OrthogonalSphericalShellGrids/distributed_tripolar_grid.jl`
# after its `global_grid = TripolarGrid(CPU(), FT; ...)` line is exactly this function, and `ORCAGrid`
# is its second caller. Until it is extracted, the partition validation, the `i`/`j` range arithmetic,
# the `insert_connected_topology` call and the north-corner connectivity fix below have to be kept in
# step with that file by hand — the two are deliberately written to stay diffable against each other.
function distribute_orca_grid(global_grid, arch::Distributed)

    workers = ranks(arch.partition)
    px = ifelse(isnothing(arch.partition.x), 1, arch.partition.x)
    py = ifelse(isnothing(arch.partition.y), 1, arch.partition.y)

    if isodd(px) && px != 1
        throw(ArgumentError("The x partition $(px) is not supported by ORCAGrid: the fold pairs each " *
                            "northern rank with the rank mirrored across the pole, so the x partition " *
                            "must be 1 or an even number."))
    end

    if px != 1 && py == 1
        throw(ArgumentError("An x-only partitioning is not supported by ORCAGrid. " *
                            "Use a y partitioning or an x-y pencil partitioning."))
    end

    Nx, Ny, Nz = global_size = size(global_grid)
    Hx, Hy, Hz = halo_size(global_grid)

    lsize   = local_size(arch, global_size)
    nxlocal = concatenate_local_sizes(lsize, arch, 1)
    nylocal = concatenate_local_sizes(lsize, arch, 2)

    xrank = ifelse(isnothing(arch.partition.x), 0, arch.local_index[1] - 1)
    yrank = ifelse(isnothing(arch.partition.y), 0, arch.local_index[2] - 1)

    # The j-range
    jstart = 1 + sum(nylocal[1:yrank])
    jend   = yrank == workers[2] - 1 ? Ny : sum(nylocal[1:yrank+1])
    jrange = jstart-Hy:jend+Hy

    # The i-range
    istart = 1 + sum(nxlocal[1:xrank])
    iend   = xrank == workers[1] - 1 ? Nx : sum(nxlocal[1:xrank+1])
    irange = istart-Hx:iend+Hx

    slice(metric_name) = on_architecture(arch, partition_tripolar_metric(global_grid, metric_name, irange, jrange))

    LX = workers[1] == 1 ? Periodic : FullyConnected

    # 1-based indices for insert_connected_topology
    Rx, Ry = workers[1], workers[2]
    rx, ry = xrank + 1, yrank + 1
    LY = insert_connected_topology(topology(global_grid, 2), Ry, ry, Rx, rx)
    nx = nxlocal[rx]
    ny = nylocal[ry]

    # Fix corner halos passing in case workers[1] != 1
    if workers[1] != 1
        northwest_idx_x = ranks(arch)[1] - arch.local_index[1] + 2
        northeast_idx_x = ranks(arch)[1] - arch.local_index[1]

        if northwest_idx_x > workers[1]
            northwest_idx_x = arch.local_index[1]
        end

        if northeast_idx_x < 1
            northeast_idx_x = arch.local_index[1]
        end

        # Make sure the northwest and northeast connectivities are correct
        northwest_recv_rank = receiving_rank(arch; receive_idx_x = northwest_idx_x)
        northeast_recv_rank = receiving_rank(arch; receive_idx_x = northeast_idx_x)
        north_recv_rank     = receiving_rank(arch)

        if yrank == workers[2] - 1
            arch.connectivity.northwest = northwest_recv_rank
            arch.connectivity.northeast = northeast_recv_rank
            arch.connectivity.north     = north_recv_rank
        end
    end

    FT = eltype(global_grid)

    return OrthogonalSphericalShellGrid{LX, LY, Bounded}(
        arch,
        nx, ny, Nz,
        Hx, Hy, Hz,
        convert(FT, global_grid.Lz),
        slice(:λᶜᶜᵃ), slice(:λᶠᶜᵃ), slice(:λᶜᶠᵃ), slice(:λᶠᶠᵃ),
        slice(:φᶜᶜᵃ), slice(:φᶠᶜᵃ), slice(:φᶜᶠᵃ), slice(:φᶠᶠᵃ),
        on_architecture(arch, global_grid.z),
        slice(:Δxᶜᶜᵃ), slice(:Δxᶠᶜᵃ), slice(:Δxᶜᶠᵃ), slice(:Δxᶠᶠᵃ),
        slice(:Δyᶜᶜᵃ), slice(:Δyᶠᶜᵃ), slice(:Δyᶜᶠᵃ), slice(:Δyᶠᶠᵃ),
        slice(:Azᶜᶜᵃ), slice(:Azᶠᶜᵃ), slice(:Azᶜᶠᵃ), slice(:Azᶠᶠᵃ),
        convert(FT, global_grid.radius),
        global_grid.conformal_mapping
    )
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
             dir = default_download_directory(dataset))

Construct an `OrthogonalSphericalShellGrid` with `(Periodic, RightFaceFolded, Bounded)`
topology using coordinate and metric data from a NEMO eORCA `mesh_mask` file.

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
- `south_rows_to_remove`: Number of southern rows to remove from the eORCA grid.  The "extended" eORCA grid
                          contains degenerate padding rows near Antarctica that are entirely land.
                          Removing them reduces memory usage and computation.
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
                  south_rows_to_remove = default_south_rows_to_remove(dataset),
                  dir = default_download_directory(dataset),
                  subcell_slope_dataset = nothing)

    mesh_meta = Metadatum(:mesh_mask; dataset, dir)
    mesh_mask_path = download(mesh_meta)

    ds = Dataset(mesh_mask_path)
    overlap = periodic_overlap(dataset)
    mesh = read_orca_staggered_mesh(ds, overlap; radius)
    close(ds)

    λCC,  λFC,  λCF,  λFF  = mesh.λCC,  mesh.λFC,  mesh.λCF,  mesh.λFF
    φCC,  φFC,  φCF,  φFF  = mesh.φCC,  mesh.φFC,  mesh.φCF,  mesh.φFF
    e1t,  e1u,  e1v,  e1f  = mesh.e1t,  mesh.e1u,  mesh.e1v,  mesh.e1f
    e2t,  e2u,  e2v,  e2f  = mesh.e2t,  mesh.e2u,  mesh.e2v,  mesh.e2f
    AzCC, AzFC, AzCF, AzFF = mesh.AzCC, mesh.AzFC, mesh.AzCF, mesh.AzFF

    pole_idx = argmin(φFF[:, end])
    north_poles_latitude = φFF[pole_idx, end]
    first_pole_longitude = Float64(λFF[pole_idx, end])

    # eORCA repeats its first `ir` columns at the east edge, and a Periodic topology carries distinct cells only
    ir = periodic_overlap_index(λCC)
    jr = south_rows_to_remove
    chop(data) = data[1:end-ir, jr+1:end]

    λCC, λFC, λCF, λFF     = chop(λCC),  chop(λFC),  chop(λCF),  chop(λFF)
    φCC, φFC, φCF, φFF     = chop(φCC),  chop(φFC),  chop(φCF),  chop(φFF)
    e1t, e1u, e1v, e1f     = chop(e1t),  chop(e1u),  chop(e1v),  chop(e1f)
    e2t, e2u, e2v, e2f     = chop(e2t),  chop(e2u),  chop(e2v),  chop(e2f)
    AzCC, AzFC, AzCF, AzFF = chop(AzCC), chop(AzFC), chop(AzCF), chop(AzFF)

    Nx, Ny = size(λCC)

    southernmost_latitude = Float64(minimum(φCC))

    Hx, Hy, Hz = halo

    topo = (Periodic, RightFaceFolded, Bounded)
    Lz, z_coord = generate_coordinate(FT, topo, (Nx, Ny, Nz), halo, z, :z, 3, CPU())

    helper_grid = RectilinearGrid(; size = (Nx, Ny), halo = (Hx, Hy),
                                    x = (0, 1), y = (0, 1),
                                    topology = (Periodic, RightFaceFolded, Flat))

    bcs = FieldBoundaryConditions(north  = FPivotZipperBoundaryCondition(),
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

    # `halo_fill_stagger` above closed the northern fold, which pairs points on opposite sides of the
    # tripolar seam and so spans all of x: the mesh can only be assembled for the whole globe. Build
    # that global grid on the host, then let `distribute_orca_grid` cut it down to this rank's slice.
    global_grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
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
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude, RightFaceFolded)
    )

    underlying_grid = distribute_orca_grid(global_grid, arch)

    with_bathymetry || return underlying_grid

    bathy_meta = Metadatum(:bottom_height; dataset, dir)
    # `download` uses `@root` internally, so every rank must call it.
    bathymetry_path = download(bathy_meta)

    # Deferred, because on a distributed grid only rank 0 reads the dataset. NEMO stores
    # positive depth: negate it, and map land (missing, or depth ≤ 0) to +100 so
    # `GridFittedBottom` masks it. The result spans the whole globe on whichever rank runs it.
    read_global_bottom_height = function ()
        bathy_ds   = Dataset(bathymetry_path)
        bathy_name = dataset_variable_name(bathy_meta)
        bathy_data = read_2d_nemo_variable(bathy_ds, bathy_name)
        close(bathy_ds)

        bathy_data = orient_xy(bathy_data, size(bathy_data)...; name = string(bathy_name))

        bathy_data = chop(bathy_data)

        bottom_height  = FT.(coalesce.(bathy_data, FT(0)))
        bottom_height .= ifelse.(isfinite.(bottom_height) .& (bottom_height .> 0), .-bottom_height, FT(100))

        return bottom_height
    end

    bottom_field = Field{Center, Center, Nothing}(underlying_grid)
    set!(bottom_field, global_orca_bottom_height(read_global_bottom_height, underlying_grid,
                                                 arch, FT, major_basins))

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
