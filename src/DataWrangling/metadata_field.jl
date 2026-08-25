using NCDatasets
using JLD2
using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: λnodes, φnodes, Face, Periodic, Bounded, AbstractMutableGrid, interior_indices, ξnode, ηnode, znode
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interpolate, interpolate!
using Oceananigans.Utils: launch!, KernelParameters

#####
##### Location with automatic restriction based on region
#####

Oceananigans.location(metadata::Metadata) = restrict_location(dataset_location(metadata.dataset, metadata.name), metadata.region)

restrict_location(loc, ::Nothing) = loc
restrict_location(loc, ::BoundingBox) = loc
restrict_location((LX, LY, LZ), ::Column) = (Nothing, Nothing, LZ)

#####
##### Native grid construction — dispatches on region type
#####

restrict(::Nothing, interfaces, N) = interfaces, N
restrict(::Nothing, interfaces::NTuple{2,Any}, N) = interfaces, N
restrict(::Nothing, interfaces::AbstractVector, N) = interfaces, N

# Snap so the native cell *centers* bracket the bbox: include the cell whose
# center is at or below the lower edge and the one whose center is at or above the
# upper edge. This keeps the bbox inside the center hull so it stays interpolatable
# at its edges (downscaling clamps outside the hull). Pads by 0 or 1 cell depending
# on where the edge falls within a native cell (an edge in a cell's first half — or
# exactly on a face — needs the extra cell; an edge past the center does not).
function restrict(bbox_interfaces, interfaces::NTuple{2,Any}, N)
    left, right = interfaces
    Δ = (right - left) / N
    i⁻ = clamp(floor(Int, (bbox_interfaces[1] - left) / Δ - 1/2), 0, N)
    i⁺ = clamp(ceil( Int, (bbox_interfaces[2] - left) / Δ + 1/2), 0, N)
    if i⁺ ≤ i⁻
        i⁺ = min(i⁻ + 1, N)
        i⁻ = max(i⁺ - 1, 0)
    end
    return (left + i⁻ * Δ, left + i⁺ * Δ), i⁺ - i⁻
end

# Stretched native grid: same center-bracketing on irregular interfaces.
function restrict(bbox_interfaces, interfaces::AbstractVector, N)
    lo, hi = bbox_interfaces
    n = length(interfaces)
    k  = clamp(searchsortedlast(interfaces,  lo), 1, n - 1)
    i⁻ = (interfaces[k]   + interfaces[k+1]) / 2 ≤ lo ? k : max(k - 1, 1)
    m  = clamp(searchsortedfirst(interfaces, hi), 2, n)
    i⁺ = (interfaces[m-1] + interfaces[m])   / 2 ≥ hi ? m : min(m + 1, n)
    rN = max(i⁺ - i⁻, 1)
    return interfaces[i⁻:i⁺], rN
end

native_convention_longitude(::Nothing, native) = nothing

# Map a bbox longitude into the native longitude convention
function native_convention_longitude(bbox_longitude, native)
    λ⁻ = convert_to_λ₀_λ₀_plus360(bbox_longitude[1], native[1])
    return (λ⁻, λ⁻ + (bbox_longitude[2] - bbox_longitude[1]))
end

restrict_longitude(bbox_interfaces, interfaces, N) =
    restrict(bbox_interfaces, interfaces, N)

restrict_longitude(::Nothing, interfaces::NTuple{2,Any}, N) = interfaces, N

function restrict_longitude(bbox_interfaces, interfaces::NTuple{2,Any}, N)
    left, right = interfaces
    Δ = (right - left) / N

    if bbox_interfaces[2] - bbox_interfaces[1] == 360
        return interfaces, N
    elseif bbox_interfaces[1] ≥ left && bbox_interfaces[2] > right
        i⁻ = max(floor(Int, (bbox_interfaces[1] - left) / Δ - 1/2), 0)
        i⁺ = ceil(Int, (bbox_interfaces[2] - left) / Δ + 1/2)
        return (left + i⁻ * Δ, left + i⁺ * Δ), i⁺ - i⁻
    else
        return restrict(bbox_interfaces, interfaces, N)
    end
end

"""
    native_region_grid(region::BoundingBox, Δλ, Δφ; pad = 2)

Regular lat/lon raster of cell steps `Δλ`/`Δφ` (degrees) covering `region`, snapped to the global
lattice anchored at `(-180, -90)` and padded by `pad` cells on each side. Returns
`(; west, south, Δλ, Δφ, Nx, Ny)`.

Datasets distributed as vector or tiled files lay out the raster they burn onto with this, so the
result is a sub-window of the global lattice `Field(::Metadatum)` assumes.
"""
function native_region_grid(region::BoundingBox, Δλ, Δφ; pad = 2)
    west, east   = region.longitude
    south, north = region.latitude
    i₀ = floor(Int, (west  + 180) / Δλ) - pad
    j₀ = floor(Int, (south +  90) / Δφ) - pad
    i₁ = ceil(Int,  (east  + 180) / Δλ) + pad
    j₁ = ceil(Int,  (north +  90) / Δφ) + pad
    return (; west = -180 + i₀ * Δλ, south = -90 + j₀ * Δφ, Δλ, Δφ, Nx = i₁ - i₀, Ny = j₁ - j₀)
end

"""
    native_grid(metadata::Metadata, arch=CPU(); halo = (3, 3, 3))

Return the native grid corresponding to `metadata` with `halo` size.
Returns a `LatitudeLongitudeGrid` for global or `BoundingBox` regions,
and a column `RectilinearGrid` for `Column` regions.
"""
native_grid(metadata::Metadata, arch=CPU(); halo=(3, 3, 3)) =
    construct_native_grid(metadata, metadata.region, arch; halo)

# 2D-only datasets (surface forcing like JRA55) skip the z dimension.
function construct_native_grid(metadata, ::Nothing, arch; halo)
    FT = eltype(metadata)
    longitude = longitude_interfaces(metadata)
    latitude = latitude_interfaces(metadata)
    Nx, Ny, Nz = size(metadata)

    if is_three_dimensional(metadata)
        z = z_interfaces(metadata)
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                     halo, longitude, latitude, z = z)
    else
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny),
                                     halo = halo[1:2], longitude, latitude,
                                     topology = (Periodic, Bounded, Flat))
    end
end

function construct_native_grid(metadata, bbox::BoundingBox, arch; halo)
    FT = eltype(metadata)
    native_longitude = longitude_interfaces(metadata)
    native_latitude  = latitude_interfaces(metadata)

    # Map the bbox into the native longitude convention.
    bbox_lon = native_convention_longitude(bbox.longitude, native_longitude)

    Nx, Ny, Nz = size(metadata)
    longitude, Nx = restrict_longitude(bbox_lon, native_longitude, Nx)
    latitude,  Ny = restrict(bbox.latitude,  native_latitude,  Ny)

    TX = infer_longitudinal_topology(native_longitude, longitude)

    # Relabel the grid longitudes back to the bbox's convention (data is array-indexed, so the
    # ordering is unchanged). The shift is 0 when the bbox already matches the dataset's native
    # convention and ±360 when they differ (e.g. a [-180, 180] bbox over ERA5's [0, 360] native
    # grid), so a `NestedSimulation` child sees a parent grid labeled in its own convention.
    if !isnothing(bbox.longitude)
        shift = bbox_lon[1] - bbox.longitude[1]
        longitude = longitude .- shift
    end

    if is_three_dimensional(metadata)
        z = z_interfaces(metadata)
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                     halo, longitude, latitude, z = z,
                                     topology = (TX, Bounded, Bounded))
    else
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny),
                                     halo = halo[1:2], longitude, latitude,
                                     topology = (TX, Bounded, Flat))
    end
end

# 2D-only datasets collapse to (Flat, Flat, Flat); 3D keep z Bounded.
function construct_native_grid(metadata, col::Column, arch; halo)
    FT = eltype(metadata)
    x  = FT(col.longitude)
    y  = FT(col.latitude)

    if is_three_dimensional(metadata)
        _, _, Nz, _ = size(metadata)
        z = z_interfaces(metadata)
        return RectilinearGrid(arch, FT; size = Nz, halo = halo[3],
                               x = x, y = y, z = z, topology = (Flat, Flat, Bounded))
    else
        return RectilinearGrid(arch, FT; size = (), halo = (),
                               x, y, topology = (Flat, Flat, Flat))
    end
end

"""
    retrieve_data(metadata)

Retrieve data from netcdf file according to `metadata`.
"""
function retrieve_data(metadata::Metadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)

    # NetCDF shenanigans
    ds = Dataset(path)

    if is_three_dimensional(metadata)
        data = ds[name][:, :, :, 1]

        # Many ocean datasets use a "depth convention" for their vertical axis
        if reversed_vertical_axis(metadata.dataset)
            data = reverse(data, dims=3)
        end
    else
        data = ds[name][:, :, 1]
    end

    close(ds)

    # ERA5 (and some other datasets) store latitude north-to-south;
    # flip to south-to-north to match the grid.
    if reversed_latitude_axis(metadata.dataset)
        data = reverse(data, dims=2)
    end

    return data
end

"""
    windowed_retrieval(dataset)

Whether [`retrieve_window`](@ref) can read one horizontal window of `dataset`'s file without
materializing the whole of it. Datasets large enough that the distinction matters extend this
together with `retrieve_window`; the default is `false`, and `Field(metadatum, grid)` then
regrids the window in a single pass.
"""
windowed_retrieval(dataset) = false

"""
    default_regrid(metadata)

Whether `metadata` reaches its target through the default bilinear `interpolate_physical!`.
A metadatum that extends the regrid hook with another scheme — a conservative area weighting, an
area-majority vote — sets this `false`: tiling drives the bilinear kernel over each tile directly
and would otherwise bypass the override silently.
"""
default_regrid(metadata) = true

"""
    retrieve_window(metadata, longitude_indices, latitude_indices)

Return `(data, λ, φ)` for one horizontal window of `metadata`'s file: the data, oriented as
[`retrieve_data`](@ref) orients it and always three-dimensional, together with the cell centers
the window spans.

The default reads the whole file and views into it — correct for every dataset, but it saves
nothing. Datasets that flag [`windowed_retrieval`](@ref) extend this to read only the window,
which is what lets `Field(metadatum, grid)` regrid in tiles.
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

Rows of a file holding the ascending `latitude_indices` a window asks for. `retrieve_data` flips
a north-first file (`reversed`) after reading it whole, so a window has to map its rows back into
the file's own ordering before slicing and flip just those.
"""
function file_latitude_rows(latitude_count, latitude_indices, reversed)
    reversed || return latitude_indices
    return (latitude_count - last(latitude_indices) + 1):(latitude_count - first(latitude_indices) + 1)
end

"""
    netcdf_retrieve_window(metadata, longitude_indices, latitude_indices)

[`retrieve_window`](@ref) for a dataset whose file is the plain lon/lat NetCDF the default
[`retrieve_data`](@ref) reads, serving the hyperslab without the rest of the variable being
touched. A dataset large enough to want windowed reads opts in with

    DataWrangling.windowed_retrieval(::MyDataset) = true
    DataWrangling.retrieve_window(metadata::MyMetadatum, longitude_indices, latitude_indices) =
        DataWrangling.netcdf_retrieve_window(metadata, longitude_indices, latitude_indices)
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

"""
    Field(metadata::Metadatum, arch=CPU();
          inpainting = default_inpainting(metadata),
          mask = nothing,
          halo = (3, 3, 3),
          cache_inpainted_data = true)

Return a `Field` on `arch`itecture described by `metadata` with `halo` size.
If not `nothing`, the `inpainting` method is used to fill the cells
within the specified `mask`. `mask` is set to `compute_mask` for non-nothing
`inpainting`. Keyword argument `cache_inpainted_data` dictates whether the inpainted
data is cached to avoid recomputing it; default: `true`.
"""
function Oceananigans.Fields.Field(metadata::Metadatum, arch=CPU();
                                   inpainting = default_inpainting(metadata),
                                   mask = nothing,
                                   halo = (3, 3, 3),
                                   cache_inpainted_data = true)

    Downloads.download(metadata)

    # Inpainting on a (Flat, Flat, *) column field is meaningless and the
    # iterative algorithm doesn't terminate gracefully without horizontal
    # neighbors; the NaN-aware bracket-blend in `set_region_data!` handles
    # land cells directly.
    if metadata.region isa Column
        inpainting = nothing
    end

    grid = native_grid(metadata, arch; halo)
    LX, LY, LZ = location(metadata)
    field = Field{LX, LY, LZ}(grid)

    if !isnothing(inpainting)
        inpainted_path = inpainted_metadata_path(metadata)
        if isfile(inpainted_path)
            # apply a load guard for corrupted files
            loaded = false
            try
                jldopen(inpainted_path, "r") do file
                    if haskey(file, "inpainting_maxiter") &&
                       file["inpainting_maxiter"] == inpainting.maxiter
                        copyto!(parent(field), file["data"])
                        loaded = true
                    end
                end
            catch err
                @warn "Could not load existing inpainted data at $inpainted_path; " *
                      "re-inpainting and saving data..." exception=err
                rm(inpainted_path, force=true)
                loaded = false
            end
            loaded && return field
        end
    end

    # Retrieve data from file according to metadata type
    data = retrieve_data(metadata)

    set_metadata_field!(field, data, metadata)
    fill_halo_regions!(field)

    if !isnothing(inpainting)
        # Respect user-supplied mask, but otherwise build default mask for this dataset.
        if isnothing(mask)
            mask = compute_mask(metadata, field)
        end

        # Make sure all values are extended properly
        name = string(metadata.name)
        date = string(metadata.dates)
        dataset = summary(metadata.dataset)
        info_str = string("Inpainting ", dataset, " ", name, " data")
        if date !== "nothing"
            info_str *= string(" from ", date)
        end
        info_str *= "..."
        @info info_str

        start_time = time_ns()

        inpaint_mask!(field, mask; inpainting)
        fill_halo_regions!(field)

        elapsed = 1e-9 * (time_ns() - start_time)
        @info string(" ... (", prettytime(elapsed), ")")

        # We cache the inpainted data to avoid recomputing it
        @root if cache_inpainted_data
            file = jldopen(inpainted_path, "w+")
            file["data"] = on_architecture(CPU(), parent(field))
            file["inpainting_maxiter"] = inpainting.maxiter
            close(file)
        end
    end

    return field
end

@kernel function _interpolate_physical!(to_field, to_grid, to_location, from_field, from_grid, from_location)
    i, j, k = @index(Global, NTuple)
    ℓx, ℓy, ℓz = to_location
    # Sample at the target's deformed `znode`, not the reference coordinate that
    # `_node` (and hence Oceananigans' `interpolate!`) uses on a mutable grid.
    to_node = (ξnode(i, j, k, to_grid, ℓx, ℓy, ℓz),
               ηnode(i, j, k, to_grid, ℓx, ℓy, ℓz),
               znode(i, j, k, to_grid, ℓx, ℓy, ℓz))
    @inbounds to_field[i, j, k] = interpolate(to_node, from_field, from_location, from_grid)
end

"""
    interpolate_physical!(to_field, from_field)

Interpolate `from_field` onto `to_field`. Identical to Oceananigans'
`interpolate!` for ordinary grids, but on a `MutableVerticalDiscretization`
(terrain-following) target it samples the source at the target's *deformed*
`znode` rather than the reference `rnode`. `interpolate!` builds its target node
from `_node`, whose vertical component is `rnode`, so it would place the source
at the LAM's reference heights and ignore the terrain — putting the lowest cells
below the (clipped) surface of a `PressureLevelGrid` source.

!!! note "TODO"
    Drop this once Oceananigans' `interpolate!` resolves the target vertical node
    from the physical `znode` for mutable grids.
"""
function interpolate_physical!(to_field, from_field)
    to_field.grid isa AbstractMutableGrid || return interpolate!(to_field, from_field)

    to_grid       = to_field.grid
    from_grid     = from_field.grid
    arch          = child_architecture(to_grid)
    from_location = Tuple(L() for L in location(from_field))
    to_location   = Tuple(L() for L in location(to_field))
    params        = KernelParameters(interior_indices(to_field))

    launch!(arch, to_grid, params, _interpolate_physical!,
            to_field, to_grid, to_location, from_field, from_grid, from_location)

    fill_halo_regions!(to_field)
    return to_field
end

# Regrid the native-grid field onto the target during `Field(metadata, grid)` and
# `set!`. The default is bilinear `interpolate_physical!`; datasets whose variables
# need a different scheme (e.g. conservative area-weighting for fractions, or an
# area-majority vote for categorical codes) extend this metadatum-dispatched method.
interpolate_physical!(to_field, from_field, metadata) = interpolate_physical!(to_field, from_field)

# Regrid into one rectangle of the target instead of all of it. `_interpolate_physical!` samples
# the source at each target node, so restricting the launch restricts which target cells are
# written; halos are filled once, after the last tile.
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

"""
    regrid_stencil_width(metadata)

Source cells a tile must carry beyond the target points it covers, so the regrid never reaches
outside its window. One suffices for the bilinear `interpolate_physical!` every dataset uses by
default; a metadatum whose regrid reads a wider stencil has to widen this to match, or its tiles
will be served edge-clamped values where a whole-window regrid would have real ones.
"""
regrid_stencil_width(metadata) = 1

"""
    default_tile_bytes()

Bytes of source data one tile of a tiled regrid is allowed to hold. Reading a dataset window in
tiles of this size keeps the resident source bounded by the tile rather than by the window, so a
model domain can exceed what its forcing dataset would occupy at native resolution.
"""
default_tile_bytes() = 64 * 1024^2

# Cells of an ascending coordinate vector whose centers bracket `[lo, hi]`, widened by `margin`
# so that every target point a tile covers is strictly inside the window's center hull and the
# interpolation never has to clamp — which is what makes a tiled regrid agree with a whole one.
function bracketing_indices(coordinates, lo, hi; margin = 1)
    n = length(coordinates)
    n < 2 && return 1:n
    i₁ = clamp(searchsortedlast(coordinates, lo) - margin, 1, n)
    i₂ = clamp(searchsortedfirst(coordinates, hi) + margin, 1, n)
    i₂ = max(i₂, min(i₁ + 1, n))
    return i₁:i₂
end

# The native grid restricted to `longitude_cells` × `latitude_cells`. Cut from the native grid's
# own faces rather than rebuilt from the file's coordinates: the two describe the same lattice in
# exact arithmetic, but a grid carries `Float32` nodes, and rounding the same lattice from two
# different inputs moves the cells by a fraction of a percent of a cell — enough to shift the
# interpolation weights and make a tiled regrid disagree with an untiled one.
function native_subgrid(native, metadata, longitude_cells, latitude_cells, arch; halo)
    FT = eltype(metadata)
    λf = λnodes(native, Face(), Center(), Center())
    φf = φnodes(native, Center(), Face(), Center())

    longitude = (λf[first(longitude_cells)], λf[last(longitude_cells) + 1])
    latitude  = (φf[first(latitude_cells)],  φf[last(latitude_cells) + 1])
    Nx, Ny = length(longitude_cells), length(latitude_cells)

    if is_three_dimensional(metadata)
        z = z_interfaces(metadata)
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, length(z) - 1),
                                     halo, longitude, latitude, z,
                                     topology = (Bounded, Bounded, Bounded))
    else
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny), halo = halo[1:2],
                                     longitude, latitude, topology = (Bounded, Bounded, Flat))
    end
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
    regrid_in_tiles!(target, metadata, native, tiles; halo)

Fill `target` from `metadata` in `tiles` × `tiles` pieces, reading only the window each piece
needs. Peak residency is one tile's source plus `target`, so neither the dataset's native window
nor a field on it is ever materialized whole — which is what lets a model domain exceed the
memory its forcing dataset would occupy at native resolution.

Each tile widens its window by a cell beyond the target points it covers, so the interpolation
stencil stays inside the window and no target cell is served by an edge-clamped source.

The result matches an untiled regrid to within the grid's node precision rather than exactly: a
tile carries a sub-range of the native grid, and rebuilding those cells re-rounds their centers
by up to one `Float32` ulp, perturbing the interpolation weights by a fraction of a percent of a
cell. `regrid_from_metadata!` therefore tiles only when more than one tile is called for.
"""
function regrid_in_tiles!(target, metadata, native, tiles; halo = (3, 3, 3))
    grid = target.grid
    # The source window is never partitioned across ranks, so it is built on the local device.
    arch = child_architecture(grid)
    LX, LY, LZ = location(metadata)

    λ, φ = read_file_coords(metadata)
    # The staggering trait compares the file's latitude count against the dataset's, so it has to
    # be resolved on the whole file and carried into each tile.
    mangling = mangling_for(metadata, length(φ))

    stencil = regrid_stencil_width(metadata)
    λn = λnodes(native, Center(), Center(), Center())
    φn = φnodes(native, Center(), Center(), Center())
    λt, φt = horizontal_centers(grid)
    Nx, Ny, Nz = size(grid)

    ni = min(tiles, Nx)
    nj = min(tiles, Ny)

    for jt in 1:nj, it in 1:ni
        target_i = tile_range(Nx, ni, it)
        target_j = tile_range(Ny, nj, jt)

        # Native cells the tile's target points need, then the file cells those sit on.
        source_i = bracketing_indices(λn, λt[first(target_i)], λt[last(target_i)]; margin = stencil)
        source_j = bracketing_indices(φn, φt[first(target_j)], φt[last(target_j)]; margin = stencil)

        window_i = bracketing_indices(λ, λn[first(source_i)], λn[last(source_i)]; margin = 0)
        window_j = bracketing_indices(φ, φn[first(source_j)], φn[last(source_j)]; margin = 0)

        data, λw, φw = retrieve_window(metadata, window_i, window_j)
        source = Field{LX, LY, LZ}(native_subgrid(native, metadata, source_i, source_j, arch; halo))
        set_region_data!(source, data, λw, φw, metadata; mangling)
        fill_halo_regions!(source)

        interpolate_physical!(target, source, metadata,
                              KernelParameters(target_i, target_j, 1:size(target, 3)))
    end

    fill_halo_regions!(target)

    return target
end

# Horizontal cell centers of a target grid, or `nothing` for a grid that has no one-dimensional
# lon/lat axes to tile over — a rectilinear or curvilinear target, which falls back to one pass.
horizontal_centers(grid::AbstractGrid) =
    hasproperty(grid, :underlying_grid) ? horizontal_centers(grid.underlying_grid) : nothing

horizontal_centers(grid::LatitudeLongitudeGrid) =
    (λnodes(grid, Center(), Center(), Center()), φnodes(grid, Center(), Center(), Center()))

"""
    tiled_native_grid(target, metadata, inpainting; halo)

The native grid to regrid `metadata` onto `target` in tiles, or `nothing` where tiling would not
be equivalent to doing it in one pass and the whole-window path should run instead.

Tiling is an optimization, never a change in result, so it asks for all of: a dataset that can
read a window ([`windowed_retrieval`](@ref)), a target carrying one-dimensional lon/lat axes to
tile over, a bounded region, and a non-periodic native window — a tile of a periodic one would
lose its wraparound. Inpainting is excluded because it is an iterative fill over the whole field,
which no tiling of it reproduces.
"""
function tiled_native_grid(target, metadata, inpainting; halo = (3, 3, 3))
    windowed_retrieval(metadata.dataset) || return nothing
    default_regrid(metadata) || return nothing
    isnothing(inpainting) || return nothing
    metadata.region isa BoundingBox || return nothing
    isnothing(horizontal_centers(target.grid)) && return nothing

    native = native_grid(metadata, child_architecture(target.grid); halo)
    topology(native, 1) === Bounded || return nothing

    return native
end

"""
    regrid_from_metadata!(target, metadata; tile_bytes = default_tile_bytes(), kw...)

Fill `target` from `metadata`, regridding in tiles where that is both possible and exactly
equivalent (see [`regrid_in_tiles!`](@ref)) and in a single pass otherwise. Keyword arguments
other than `tile_bytes` go to the native-grid `Field(metadata, arch; …)`.
"""
function regrid_from_metadata!(target, metadata; tile_bytes = default_tile_bytes(), kw...)
    options = values(kw)
    inpainting = get(options, :inpainting, default_inpainting(metadata))
    halo = get(options, :halo, (3, 3, 3))

    Downloads.download(metadata)
    native = tiled_native_grid(target, metadata, inpainting; halo)

    # Tile only when it buys something. One tile would rebuild the native grid from a sub-range
    # of its own faces, which at `Float32` node precision moves the cells by a fraction of a
    # percent of one — a perturbation worth taking to bound memory, but not for nothing.
    if !isnothing(native)
        tiles = tile_count(size(native)[1:2], size(metadata)[3], tile_bytes)
        tiles > 1 && return regrid_in_tiles!(target, metadata, native, tiles; halo)
    end

    return interpolate_physical!(target, Field(metadata, architecture(target.grid); kw...), metadata)
end

"""
    target_matched_metadata(metadatum, grid)

Return `metadatum` rebuilt on the dataset variant [`matching_resolution_dataset`](@ref) selects
for `grid`. The rebuild re-derives the filename, so a read matched to a coarse target and a
full-resolution read of the same window never share a cached file.
"""
function target_matched_metadata(metadatum::Metadatum, grid)
    dataset = matching_resolution_dataset(metadatum.dataset, grid)
    dataset === metadatum.dataset && return metadatum

    categorical(dataset) &&
        throw(ArgumentError("$(summary(dataset)) stores class codes, which cannot be averaged " *
                            "onto a coarser read lattice; aggregate its native pixels instead."))

    return Metadatum(metadatum.name; dataset, region = metadatum.region,
                     date = metadatum.dates, dir = metadatum.dir)
end

"""
    Field(metadata::Metadatum, grid::AbstractGrid; cache = false, overwrite_cache = false, kw...)

Load `metadata` on its native grid and interpolate onto `grid` — the
`Field` analog of `FieldTimeSeries(metadata, grid)`. Keyword arguments are
forwarded to the native-grid `Field(metadata, arch; …)` (e.g. `inpainting`,
`mask`, `halo`, `cache_inpainted_data`).

Knowing the target lets a dataset that supports it read at a resolution matched to `grid`
instead of at full resolution; see [`matching_resolution_dataset`](@ref).

With `cache = true` the regridded result is cached to disk and reused by later
reads with the same dataset, variable, date, region, target-grid geometry, and
read keywords — skipping the native materialization and regrid entirely; with
`cache = false` (default) the cache is disabled entirely and nothing is read or
written. The key carries a size/mtime stamp of the local dataset file where one
exists, so a re-download invalidates the cache. For streaming datasets with no
local file, pass `overwrite_cache = true` after replacing data upstream: it
skips the lookup and overwrites the entry with a freshly regridded result.
"""
function Oceananigans.Fields.Field(metadata::Metadatum, grid::AbstractGrid;
                                   cache = false, overwrite_cache = false,
                                   tile_bytes = default_tile_bytes(), kw...)
    metadata = target_matched_metadata(metadata, grid)
    LX, LY, LZ = location(metadata)

    if cache && !overwrite_cache
        config = FieldRegridding(grid, metadata, values(kw))
        data = load_field_cache(config)
        if !isnothing(data)
            target = Field{LX, LY, LZ}(grid)
            interior(target) .= on_architecture(architecture(grid), data)
            fill_halo_regions!(target)
            return target
        end
    end

    target = Field{LX, LY, LZ}(grid)
    regrid_from_metadata!(target, metadata; tile_bytes, kw...)
    if cache
        # rebuild the key: the native read may have just downloaded the dataset file it stamps
        config = FieldRegridding(grid, metadata, values(kw))
        save_field_cache(config, Array(interior(target)))
    end
    return target
end

function Oceananigans.Fields.set!(target_field::Field, metadata::Metadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    metadata = target_matched_metadata(metadata, grid)

    # The vertical extent comes from the native grid, which costs nothing to build — the data it
    # would hold is read per tile below.
    Lzt = grid.Lz
    Lzm = native_grid(metadata, arch).Lz

    # Allow up to 1% vertical mismatch for pressure-level datasets with time-varying
    # geopotential heights — the per-timestep vertical extent can be slightly smaller
    # than the temporal-mean extent used for the target grid (e.g. when the atmosphere
    # is compressed). Oceananigans' interpolate! does not extrapolate, so target points
    # just outside the source domain will use the nearest interior values.
    if is_three_dimensional(metadata) && Lzt > Lzm * (1 + 1e-2)
        throw("The vertical range of the $(metadata.dataset) dataset ($(Lzm) m) is smaller than " *
              "the target grid ($(Lzt) m). Some vertical levels cannot be filled with data.")
    end

    regrid_from_metadata!(target_field, metadata; kw...)

    return target_field
end

function set_metadata_field!(field, data, metadatum)
    full_data = ndims(data) == 2 ? reshape(data, size(data, 1), size(data, 2), 1) : data
    λc, φc = read_file_coords(metadatum)
    set_region_data!(field, full_data, λc, φc, metadatum)
    return nothing
end

# Read the lon/lat cell centers from the NetCDF file using the names supplied
# by the dataset's `longitude_name` / `latitude_name` traits.
function read_file_coords(metadatum)
    ds = Dataset(metadata_path(metadatum))
    λc = ds[longitude_name(metadatum)][:]
    φc = ds[latitude_name(metadatum)][:]
    close(ds)
    reversed_latitude_axis(metadatum.dataset) && reverse!(φc)
    return λc, φc
end

#####
##### Helper functions
#####

"""
    centers_to_interfaces(z_centers)

Compute ``z``-interfaces (cell faces) from cell center positions.
`z_centers` should be sorted most negative first (deepest first).
The top face is placed at 0.0 (sea surface). Interior faces are
midpoints between adjacent centers. The bottom face is extrapolated.

Note: the grid's cell centers (midpoints of faces) will approximately
but not exactly match the input centers when spacing is irregular.
"""
function centers_to_interfaces(z_centers)
    Nz = length(z_centers)
    z_faces = zeros(Nz + 1)

    for k in 1:Nz-1
        z_faces[k+1] = (z_centers[k] + z_centers[k+1]) / 2
    end
    # Extrapolate bottom face
    z_faces[1] = z_centers[1] - (z_faces[2] - z_centers[1])
    return z_faces
end

# Convert missing values to NaN
@inline nan_convert_missing(FT, x::Number) = convert(FT, x)
@inline nan_convert_missing(FT, ::Missing) = convert(FT, NaN)
@inline nan_convert_missing(FT, x, ::Missing) = nan_convert_missing(FT, x)
@inline nan_convert_missing(FT, x, missing_val::Number) = ifelse(ismissing(x) || x == missing_val, convert(FT, NaN), nan_convert_missing(FT, x))

# No units conversion
@inline convert_units(T, units) = T

# Just switch sign!
@inline convert_units(T::FT, ::InverseSign) where FT = - T

# Temperature units
@inline convert_units(T::FT, ::Kelvin) where FT = T - convert(FT, 273.15)
@inline convert_units(T::FT, ::Celsius) where FT = T + convert(FT, 273.15)

# Pressure units
@inline convert_units(P::FT, ::Millibar) where FT = P * convert(FT, 100)

# Precipitation rate (assuming ρ_water = 1000 kg/m³, so 1 mm/hr = 1 kg/m²/hr = 1/3600 kg/m²/s)
@inline convert_units(r::FT, ::MillimetersPerHour) where FT = r / convert(FT, 3600)

# ERA5 total precipitation is an hourly accumulated depth (m); m/hr → kg/m²/s.
@inline convert_units(p::FT, ::MetersPerHour) where FT = p * convert(FT, 1000) / convert(FT, 3600)

# ERA5 ssrd/strd are energy accumulated over the previous hour (J/m²); ÷3600 s → mean W/m².
@inline convert_units(ℐ::FT, ::JoulesPerSquareMeterPerHour) where FT = ℐ / convert(FT, 3600)

# Molar units
@inline convert_units(C::FT, ::Union{MolePerLiter, MolePerKilogram})           where FT = C * convert(FT, 1e3)
@inline convert_units(C::FT, ::Union{MillimolePerLiter, MillimolePerKilogram}) where FT = C * convert(FT, 1)
@inline convert_units(C::FT, ::Union{MicromolePerLiter, MicromolePerKilogram}) where FT = C * convert(FT, 1e-3)
@inline convert_units(C::FT, ::Union{NanomolePerLiter, NanomolePerKilogram})   where FT = C * convert(FT, 1e-6)
@inline convert_units(C::FT, ::MilliliterPerLiter)                             where FT = C / convert(FT, 22.3916)
@inline convert_units(C::FT, ::GramPerKilogramMinus35)                         where FT = C + convert(FT, 35)
@inline convert_units(Φ::FT, ::InverseGravity)                                 where FT = Φ / convert(FT, 9.80665)
@inline convert_units(V::FT, ::CentimetersPerSecond)                           where FT = V / convert(FT, 100)

# Mass fractions (convert to kg/kg)
@inline convert_units(χ::FT, ::DecigramPerKilogram) where FT = χ / convert(FT, 1e4)
@inline convert_units(χ::FT, ::GramPerKilogram) where FT = χ / convert(FT, 1e3)
@inline convert_units(χ::FT, ::WeightPercent) where FT = χ / convert(FT, 100)

# Densities (convert to kg/m^3)
@inline convert_units(ρ::FT, ::HectogramPerCubicMeter) where FT = ρ / convert(FT, 10)
@inline convert_units(ρ::FT, ::CentigramPerCubicCentimeter) where FT = ρ * convert(FT, 10)
@inline convert_units(ρ::FT, ::GramPerCubicCentimeter) where FT = ρ * convert(FT, 1000)

#####
##### Masking data for inpainting
#####

# Fallback for lower and higher bounds: 1e5
lower_bound(metadata, name) = -1f5
higher_bound(metadata, name) = 1f5

"""
    compute_mask(metadata::Metadatum, dataset_field,
                 mask_value = default_mask_value(metadata),
                 minimum_value = -1f5,
                 maximum_value = 1f5)

A boolean field where `true` represents a missing value in the dataset_field.
"""
function compute_mask(metadata::Metadatum, dataset_field,
                      mask_value = default_mask_value(metadata.dataset),
                      minimum_value = lower_bound(metadata, Val(metadata.name)),
                      maximum_value = higher_bound(metadata, Val(metadata.name)))

    grid = dataset_field.grid
    arch = Oceananigans.Architectures.architecture(grid)
    LX, LY, LZ = location(dataset_field)
    mask = Field{LX, LY, LZ}(grid, Bool)

    # Set the mask with zeros where field is defined
    launch!(arch, grid, :xyz, _compute_mask!,
            mask, dataset_field, minimum_value, maximum_value, mask_value)

    return mask
end

@kernel function _compute_mask!(mask, field, min_value, max_value, mask_value)
    i, j, k = @index(Global, NTuple)
    @inbounds mask[i, j, k] = is_masked(field[i, j, k], min_value, max_value, mask_value)
end

@inline is_masked(a, min_value, max_value, mask_value) = isnan(a) | (a ≤ min_value) | (a ≥ max_value) | (a == mask_value)

#####
##### Field / FieldTimeSeries for MetadataSet
#####

"""
    Field(mset::MetadataSet, arch=CPU(); kw...)

Build a `NamedTuple` of `Field`s — one per variable in `mset`, keyed by the
verbose dataset variable name. Each value is `Field(mset[name], arch; kw...)`.

Requires `mset` to hold scalar `dates` so each `mset[name]` is a `Metadatum`; for
multi-date sets, build a `NamedTuple` of `FieldTimeSeries` per variable, e.g.
`NamedTuple(name => FieldTimeSeries(mset[name], grid) for name in mset.names)`.
"""
function Oceananigans.Fields.Field(mset::MetadataSet, arch=CPU(); kw...)
    dates = getfield(mset, :dates)
    if !(dates isa AnyDateTime)
        throw(ArgumentError(
            "Field(::MetadataSet) requires a scalar `date`, but this `MetadataSet` carries a multi-date axis. " *
            "For multi-date sets build a NamedTuple of FieldTimeSeries per variable, e.g. " *
            "`NamedTuple(name => FieldTimeSeries(mset[name], grid) for name in mset.names)`."))
    end
    names = getfield(mset, :names)
    return NamedTuple{names}(map(n -> Field(mset[n], arch; kw...), names))
end
