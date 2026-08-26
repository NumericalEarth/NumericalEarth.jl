using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: KernelParameters

#####
##### Reading one horizontal window of a dataset file
#####

"""
    windowed_retrieval(dataset)

Whether `dataset` reads one horizontal window of its file through `retrieve_window`, which lets
`Field(metadatum, grid)` regrid it in tiles. A dataset that overrides `interpolate_physical!`
regrids by its own scheme and leaves this `false`. Default `false`.
"""
windowed_retrieval(dataset) = false

"""
    retrieve_window(metadata, longitude_indices, latitude_indices)

Return `(data, λ, φ)` for one horizontal window of `metadata`'s file: the data, oriented as
`retrieve_data` orients it and always three-dimensional, together with the cell centers the
window spans. Serves the hyperslab of a plain lon/lat NetCDF, north-first files included;
datasets stored otherwise extend this.
"""
function retrieve_window(metadata::Metadatum, longitude_indices, latitude_indices)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    reversed = reversed_latitude_axis(metadata.dataset)

    return Dataset(path) do ds
        λ = ds[longitude_name(metadata)][:]
        φ = ds[latitude_name(metadata)][:]

        # A north-first file stores ascending row j at row length(φ) - j + 1.
        rows = latitude_indices
        if reversed
            rows = (length(φ) - last(rows) + 1):(length(φ) - first(rows) + 1)
        end

        data = if is_three_dimensional(metadata)
            ds[name][longitude_indices, rows, :, 1]
        else
            ds[name][longitude_indices, rows, 1]
        end

        data = ndims(data) == 2 ? reshape(data, size(data, 1), size(data, 2), 1) : data

        if is_three_dimensional(metadata) && reversed_vertical_axis(metadata.dataset)
            data = reverse(data, dims = 3)
        end

        if reversed
            data = reverse(data, dims = 2)
            reverse!(φ)
        end

        return (data, λ[longitude_indices], φ[latitude_indices])
    end
end

#####
##### Tiled regridding — bounding what is resident by the tile, not by the window
#####

# Source data one tile may hold.
const default_tile_bytes = 64 * 1024^2

# Index of the file cell holding native cell 1; zero when file and native grid start together.
function native_file_offset(coordinates, native_centers)
    i₁, _ = compute_bounding_indices((native_centers[1], native_centers[end]), coordinates)
    return i₁ - 1
end

# Cells whose centers bracket `[lo, hi]`, widened by `margin` so bilinear interpolation of any
# point inside stays within them.
function bracketing_indices(coordinates, lo, hi; margin = 1)
    n = length(coordinates)
    n < 2 && return 1:n
    i₁ = clamp(searchsortedlast(coordinates, lo) - margin, 1, n)
    i₂ = clamp(searchsortedfirst(coordinates, hi) + margin, 1, n)
    i₂ = max(i₂, min(i₁ + 1, n))
    return i₁:i₂
end

# Tile `k` of `n` covering `N` target cells.
function tile_indices(N, n, k)
    first_index = 1 + ((k - 1) * N) ÷ n
    last_index = (k * N) ÷ n
    return first_index:last_index
end

# Tiles per side, chosen so one tile's source stays under `tile_bytes`.
function tile_count(native, FT, tile_bytes)
    total = sizeof(FT) * prod(size(native))
    return max(1, ceil(Int, sqrt(total / tile_bytes)))
end

# Horizontal cell centers of a target grid, or `nothing` for a grid with no one-dimensional
# longitude/latitude axes.
horizontal_centers(grid) = nothing

horizontal_centers(grid::ImmersedBoundaryGrid) = horizontal_centers(grid.underlying_grid)

horizontal_centers(grid::LatitudeLongitudeGrid) =
    (λnodes(grid, Center(), Center(), Center()), φnodes(grid, Center(), Center(), Center()))

"""
    regrid_in_tiles!(target, metadata, native, tile_bytes)

Fill `target` from `metadata` in pieces holding at most `tile_bytes` of source each, reading only
the window each piece needs.
"""
function regrid_in_tiles!(target, metadata, native, tile_bytes = default_tile_bytes)
    grid = target.grid
    LX, LY, LZ = location(metadata)

    λ, φ = read_file_coords(metadata)
    mangling = mangling_for(metadata, length(φ))

    native_longitude = λnodes(native, Center(), Center(), Center())
    native_latitude = φnodes(native, Center(), Center(), Center())
    target_longitude, target_latitude = horizontal_centers(grid)

    offset_i = native_file_offset(λ, native_longitude)
    offset_j = native_file_offset(φ, native_latitude)
    Nx, Ny, _ = size(grid)

    tiles = tile_count(native, eltype(metadata), tile_bytes)
    ni = min(tiles, Nx)
    nj = min(tiles, Ny)

    for tile_j in 1:nj, tile_i in 1:ni
        target_i = tile_indices(Nx, ni, tile_i)
        target_j = tile_indices(Ny, nj, tile_j)

        west  = target_longitude[first(target_i)]
        east  = target_longitude[last(target_i)]
        south = target_latitude[first(target_j)]
        north = target_latitude[last(target_j)]

        source_i = bracketing_indices(native_longitude, west, east)
        source_j = bracketing_indices(native_latitude, south, north)

        window_i = (first(source_i) + offset_i):(last(source_i) + offset_i)
        window_j = (first(source_j) + offset_j):(last(source_j) + offset_j)

        data, window_longitude, window_latitude = retrieve_window(metadata, window_i, window_j)

        source = Field{LX, LY, LZ}(native; indices = (source_i, source_j, :))
        set_region_data!(source, data, window_longitude, window_latitude, metadata; mangling,
                         region = BoundingBoxOffset(1 - first(source_i), 1 - first(source_j)),
                         parameters = KernelParameters(interior_indices(source)))

        interpolate_physical!(view(target, target_i, target_j, :), source, metadata)
    end

    fill_halo_regions!(target)

    return target
end

# The native grid to regrid `metadata` onto `target` in tiles, or `nothing` where tiling would not
# reproduce the single-pass regrid.
function tiled_native_grid(target, metadata, inpainting, halo)
    windowed_retrieval(metadata.dataset) || return nothing
    isnothing(inpainting) || return nothing
    metadata.region isa BoundingBox || return nothing
    isnothing(horizontal_centers(target.grid)) && return nothing

    native = native_grid(metadata, child_architecture(target.grid); halo)
    topology(native, 1) === Bounded || return nothing

    return native
end

# Up to 1% mismatch is allowed for pressure-level datasets with time-varying geopotential heights,
# whose per-timestep vertical extent can be slightly smaller than the temporal-mean extent used
# for the target grid.
function validate_vertical_extent(target_grid, native, metadata)
    is_three_dimensional(metadata) && target_grid.Lz > native.Lz * (1 + 1e-2) &&
        throw("The vertical range of the $(metadata.dataset) dataset ($(native.Lz) m) is smaller " *
              "than the target grid ($(target_grid.Lz) m)")
    return nothing
end

"""
    regrid_from_metadata!(target, metadata; tile_bytes = default_tile_bytes, kw...)

Fill `target` from `metadata`, regridding in tiles of at most `tile_bytes` of source each where
the dataset reads windows, and in a single pass otherwise. Remaining keyword arguments go to
`Field(metadata, arch; …)`.
"""
function regrid_from_metadata!(target, metadata;
                               tile_bytes = default_tile_bytes,
                               inpainting = default_inpainting(metadata),
                               halo = (3, 3, 3),
                               kw...)

    Downloads.download(metadata)

    native = tiled_native_grid(target, metadata, inpainting, halo)

    if isnothing(native)
        source = Field(metadata, architecture(target.grid); inpainting, halo, kw...)
        validate_vertical_extent(target.grid, source.grid, metadata)
        return interpolate_physical!(target, source, metadata)
    end

    validate_vertical_extent(target.grid, native, metadata)

    return regrid_in_tiles!(target, metadata, native, tile_bytes)
end
