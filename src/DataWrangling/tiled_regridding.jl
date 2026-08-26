using Oceananigans.Utils: KernelParameters

#####
##### Reading one horizontal window of a dataset file
#####

"""
    windowed_retrieval(dataset)

Whether `dataset` can read one horizontal window of its file without materializing the whole of
it. Default `false`.
"""
windowed_retrieval(dataset) = false

"""
    retrieve_window(metadata, longitude_indices, latitude_indices)

Return `(data, λ, φ)` for one horizontal window of `metadata`'s file: the data, oriented as
`retrieve_data` orients it and always three-dimensional, together with the cell centers the
window spans. The default reads the whole file and views into it.
"""
function retrieve_window(metadata::Metadatum, longitude_indices, latitude_indices)
    data = retrieve_data(metadata)
    data = ndims(data) == 2 ? reshape(data, size(data, 1), size(data, 2), 1) : data
    λ, φ = read_file_coords(metadata)
    return (view(data, longitude_indices, latitude_indices, :),
            view(λ, longitude_indices),
            view(φ, latitude_indices))
end

"""
    file_latitude_rows(latitude_count, latitude_indices, reversed)

Rows of a file holding the ascending `latitude_indices` a window asks for. A `reversed`
(north-first) file stores them mirrored.
"""
function file_latitude_rows(latitude_count, latitude_indices, reversed)
    reversed || return latitude_indices
    return (latitude_count - last(latitude_indices) + 1):(latitude_count - first(latitude_indices) + 1)
end

"""
    netcdf_retrieve_window(metadata, longitude_indices, latitude_indices)

`retrieve_window` for a dataset whose file is the plain lon/lat NetCDF `retrieve_data` reads,
serving the hyperslab without the rest of the variable being touched.
"""
function netcdf_retrieve_window(metadata, longitude_indices, latitude_indices)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    reversed = reversed_latitude_axis(metadata.dataset)

    return Dataset(path) do ds
        λ = ds[longitude_name(metadata)][:]
        φ = ds[latitude_name(metadata)][:]

        file_rows = file_latitude_rows(length(φ), latitude_indices, reversed)

        data = if is_three_dimensional(metadata)
            ds[name][longitude_indices, file_rows, :, 1]
        else
            ds[name][longitude_indices, file_rows, 1]
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

# Regrid into one rectangle of the target instead of all of it.
function interpolate_physical!(to_field, from_field, metadata, params::KernelParameters)
    to_grid = to_field.grid
    from_grid = from_field.grid
    arch = child_architecture(to_grid)
    from_location = Tuple(L() for L in location(from_field))
    to_location = Tuple(L() for L in location(to_field))

    launch!(arch, to_grid, params, _interpolate_physical!,
            to_field, to_grid, to_location, from_field, from_grid, from_location)

    return to_field
end

#####
##### Tiled regridding — bounding what is resident by the tile, not by the window
#####

# Index of the file cell holding native cell 1; zero when file and native grid start together.
function native_file_offset(coordinates, native_centers)
    i₁, _ = compute_bounding_indices((native_centers[1], native_centers[end]), coordinates)
    return i₁ - 1
end

# Source data one tile may hold.
const default_tile_bytes = 64 * 1024^2

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

# Split `N` target cells into `n` contiguous ranges.
function tile_range(N, n, k)
    first_index = 1 + ((k - 1) * N) ÷ n
    last_index = (k * N) ÷ n
    return first_index:last_index
end

# Tiles per side, chosen so one tile's source stays under `tile_bytes`.
function tile_count(source_cells, vertical_cells, tile_bytes)
    total = 4 * prod(source_cells) * vertical_cells
    return max(1, ceil(Int, sqrt(total / tile_bytes)))
end

"""
    regrid_in_tiles!(target, metadata, native, tile_bytes)

Fill `target` from `metadata` in pieces sized to hold at most `tile_bytes` of source each,
reading only the window each piece needs, so peak residency is one tile rather than the whole
window. Each tile's source is a windowed `Field` over `native`, so the result is bitwise
identical to an untiled regrid at any tiling.
"""
function regrid_in_tiles!(target, metadata, native, tile_bytes = default_tile_bytes)
    grid = target.grid
    arch = child_architecture(grid)
    LX, LY, LZ = location(metadata)

    λ, φ = read_file_coords(metadata)
    mangling = mangling_for(metadata, length(φ))

    λn = λnodes(native, Center(), Center(), Center())
    φn = φnodes(native, Center(), Center(), Center())
    λt, φt = horizontal_centers(grid)

    offset_i = native_file_offset(λ, λn)
    offset_j = native_file_offset(φ, φn)
    Nx, Ny, Nz = size(grid)

    tiles = tile_count(size(native)[1:2], Nz, tile_bytes)
    ni = min(tiles, Nx)
    nj = min(tiles, Ny)

    for jt in 1:nj, it in 1:ni
        target_i = tile_range(Nx, ni, it)
        target_j = tile_range(Ny, nj, jt)

        source_i = bracketing_indices(λn, λt[first(target_i)], λt[last(target_i)])
        source_j = bracketing_indices(φn, φt[first(target_j)], φt[last(target_j)])

        window_i = (first(source_i) + offset_i):(last(source_i) + offset_i)
        window_j = (first(source_j) + offset_j):(last(source_j) + offset_j)

        data, λw, φw = retrieve_window(metadata, window_i, window_j)
        source = Field{LX, LY, LZ}(native; indices = (source_i, source_j, :))
        set_region_data!(source, data, λw, φw, metadata; mangling,
                         region = BoundingBoxOffset(1 - first(source_i), 1 - first(source_j)),
                         parameters = KernelParameters(source_i, source_j, 1:Nz))

        interpolate_physical!(target, source, metadata,
                              KernelParameters(target_i, target_j, 1:size(target, 3)))
    end

    fill_halo_regions!(target)

    return target
end

# Horizontal cell centers of a target grid, or `nothing` for a grid with no one-dimensional
# longitude/latitude axes — a rectilinear or curvilinear target.
horizontal_centers(grid::AbstractGrid) =
    hasproperty(grid, :underlying_grid) ? horizontal_centers(grid.underlying_grid) : nothing

horizontal_centers(grid::LatitudeLongitudeGrid) =
    (λnodes(grid, Center(), Center(), Center()), φnodes(grid, Center(), Center(), Center()))

"""
    tiled_native_grid(target, metadata, inpainting; halo)

The native grid to regrid `metadata` onto `target` in tiles, or `nothing` unless all of: the
dataset reads a window ([`windowed_retrieval`](@ref)), `inpainting` is `nothing`, the region is a
`BoundingBox`, `target` carries one-dimensional longitude/latitude axes, and the native window is
non-periodic.
"""
function tiled_native_grid(target, metadata, inpainting; halo = (3, 3, 3))
    windowed_retrieval(metadata.dataset) || return nothing
    isnothing(inpainting) || return nothing
    metadata.region isa BoundingBox || return nothing
    isnothing(horizontal_centers(target.grid)) && return nothing

    native = native_grid(metadata, child_architecture(target.grid); halo)
    topology(native, 1) === Bounded || return nothing

    return native
end

"""
    regrid_from_metadata!(target, metadata; tile_bytes = default_tile_bytes, kw...)

Fill `target` from `metadata`, regridding in tiles where possible and in a single pass
otherwise. Keyword arguments other than `tile_bytes` go to `Field(metadata, arch; …)`.
"""
function regrid_from_metadata!(target, metadata; tile_bytes = default_tile_bytes, kw...)
    options = values(kw)
    inpainting = get(options, :inpainting, default_inpainting(metadata))
    halo = get(options, :halo, (3, 3, 3))

    Downloads.download(metadata)
    native = tiled_native_grid(target, metadata, inpainting; halo)

    isnothing(native) ||
        return regrid_in_tiles!(target, metadata, native, tile_bytes)

    return interpolate_physical!(target, Field(metadata, architecture(target.grid); kw...), metadata)
end
