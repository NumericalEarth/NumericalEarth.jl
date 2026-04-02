using NCDatasets
using JLD2
using NumericalEarth.InitialConditions: interpolate!
using Statistics: median
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: fractional_x_index, fractional_y_index

import Oceananigans.Fields: set!, Field, location

#####
##### Location with automatic restriction based on region
#####

location(metadata::Metadata) = restrict_location(dataset_location(metadata.dataset, metadata.name), metadata.region)

restrict_location(loc, ::Nothing) = loc
restrict_location(loc, ::BoundingBox) = loc
restrict_location((LX, LY, LZ), ::Column) = (Nothing, Nothing, LZ)

#####
##### Native grid construction — dispatches on region type
#####

restrict(::Nothing, interfaces, N) = interfaces, N

# TODO support stretched native grids
function restrict(bbox_interfaces, interfaces, N)
    Δ = interfaces[2] - interfaces[1]
    rΔ = bbox_interfaces[2] - bbox_interfaces[1]
    ϵ = rΔ / Δ
    rN = ceil(Int, ϵ * N)  # Round up to ensure bounding box is covered
    return bbox_interfaces, rN
end

"""
    native_grid(metadata::Metadata, arch=CPU(); halo = (3, 3, 3))

Return the native grid corresponding to `metadata` with `halo` size.
Returns a `LatitudeLongitudeGrid` for global or `BoundingBox` regions,
and a column `RectilinearGrid` for `Column` regions.
"""
native_grid(metadata::Metadata, arch=CPU(); halo=(3, 3, 3)) =
    construct_native_grid(metadata, metadata.region, arch; halo)

# Full global grid (no region restriction)
function construct_native_grid(metadata, ::Nothing, arch; halo)
    Nx, Ny, Nz, _ = size(metadata)
    z = z_interfaces(metadata)
    FT = eltype(metadata)
    longitude = longitude_interfaces(metadata)
    latitude = latitude_interfaces(metadata)

    grid = LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                 halo, longitude, latitude, z)
    return grid
end

# BoundingBox-restricted LatitudeLongitudeGrid
function construct_native_grid(metadata, bbox::BoundingBox, arch; halo)
    Nx, Ny, Nz, _ = size(metadata)
    z = z_interfaces(metadata)
    FT = eltype(metadata)
    longitude = longitude_interfaces(metadata)
    latitude = latitude_interfaces(metadata)

    # TODO: can we restrict in `z` as well?
    longitude, Nx = restrict(bbox.longitude, longitude, Nx)
    latitude, Ny = restrict(bbox.latitude, latitude, Ny)

    grid = LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                 halo, longitude, latitude, z)
    return grid
end

# Column RectilinearGrid
function construct_native_grid(metadata, col::Column, arch; halo)
    _, _, Nz, _ = size(metadata)
    z = z_interfaces(metadata)
    FT = eltype(metadata)

    grid = RectilinearGrid(arch, FT;
                           size = Nz,
                           x = FT(col.longitude),
                           y = FT(col.latitude),
                           z,
                           halo = halo[3],
                           topology = (Flat, Flat, Bounded))
    return grid
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
    return data
end

"""
    Field(metadata::Metadatum;
          architecture = CPU(),
          inpainting = default_inpainting(metadata),
          mask = nothing,
          halo = (7, 7, 7),
          cache_inpainted_data = true)

Return a `Field` on `architecture` described by `metadata` with `halo` size.
If not `nothing`, the `inpainting` method is used to fill the cells
within the specified `mask`. `mask` is set to `compute_mask` for non-nothing
`inpainting`. Keyword argument `cache_inpainted_data` dictates whether the inpainted
data is cached to avoid recomputing it; default: `true`.
"""
function Field(metadata::Metadatum, arch=CPU();
               inpainting = default_inpainting(metadata),
               mask = nothing,
               halo = (3, 3, 3),
               cache_inpainted_data = true)

    download_dataset(metadata)

    if is_column(metadata.region)
        return column_field(metadata, arch; inpainting, mask, halo, cache_inpainted_data)
    end

    grid = native_grid(metadata, arch; halo)
    LX, LY, LZ = location(metadata)
    field = Field{LX, LY, LZ}(grid)

    if !isnothing(inpainting)
        inpainted_path = inpainted_metadata_path(metadata)
        if isfile(inpainted_path)
            file = jldopen(inpainted_path, "r")
            maxiter = file["inpainting_maxiter"]

            # read data if generated with the same inpainting
            if maxiter == inpainting.maxiter
                data = file["data"]
                close(file)
                try
                    copyto!(parent(field), data)
                    return field
                catch
                    @warn "Could not load existing inpainted data at $inpainted_path.\n" *
                          "Re-inpainting and saving data..."
                    rm(inpainted_path, force=true)
                end
            end

            close(file)
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

function set!(target_field::Field, metadata::Metadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    meta_field = Field(metadata, arch; kw...)

    Lzt = grid.Lz
    Lzm = meta_field.grid.Lz

    if Lzt > Lzm
        throw("The vertical range of the $(metadata.dataset) dataset ($(Lzm) m) is smaller than " *
              "the target grid ($(Lzt) m). Some vertical levels cannot be filled with data.")
    end

    interpolate!(target_field, meta_field)
    return target_field
end

#####
##### Column field construction
#####

"""Build a column Field from metadata with a Column region.

Internally loads data onto an intermediate LatitudeLongitudeGrid
and interpolates to the column RectilinearGrid."""
function column_field(metadata, arch;
                      inpainting = default_inpainting(metadata),
                      mask = nothing,
                      halo = (3, 3, 3),
                      cache_inpainted_data = true)

    # 1. Build the column grid (the "native grid" for column metadata)
    column_grid = native_grid(metadata, arch; halo)

    # 2. Build an intermediate LatLonGrid from the downloaded data file
    intermediate_grid = intermediate_grid_from_file(metadata, arch; halo)

    # 3. Load data onto intermediate grid
    LX, LY, LZ = dataset_location(metadata.dataset, metadata.name)
    intermediate_field = Field{LX, LY, LZ}(intermediate_grid)

    data = retrieve_data(metadata)
    set_metadata_field!(intermediate_field, data, metadata)
    fill_halo_regions!(intermediate_field)

    # 4. Inpaint on intermediate grid if needed
    if !isnothing(inpainting)
        if isnothing(mask)
            mask = compute_mask(metadata, intermediate_field)
        end
        inpaint_mask!(intermediate_field, mask; inpainting)
        fill_halo_regions!(intermediate_field)
    end

    # 5. Create column field and extract data
    _, _, LZ_col = location(metadata) # (Nothing, Nothing, LZ)
    column_field = Field{Nothing, Nothing, LZ_col}(column_grid)

    extract_column!(column_field, intermediate_field, metadata.region)

    return column_field
end

# Dispatch extraction on interpolation method
function extract_column!(column_field, intermediate_field, col::Column)
    extract_column!(column_field, intermediate_field, col, col.interpolation)
end

function extract_column!(column_field, intermediate_field, col, ::Linear)
    interpolate!(column_field, intermediate_field)
    return nothing
end

function extract_column!(column_field, intermediate_field, col, ::Nearest)
    grid = intermediate_field.grid
    arch = architecture(grid)
    LX, LY, LZ = Oceananigans.Fields.location(intermediate_field)

    # Use Oceananigans' fractional index machinery (handles cyclic longitude etc.)
    i★ = round(Int, fractional_x_index(col.longitude, (LX, LY, LZ), grid))
    j★ = round(Int, fractional_y_index(col.latitude,  (LX, LY, LZ), grid))

    launch!(arch, column_field.grid, :z, copy_column!, column_field, intermediate_field, i★, j★)

    return nothing
end

@kernel function copy_column!(column_field, source_field, i★, j★)
    k = @index(Global, Linear)
    @inbounds column_field[1, 1, k] = source_field[i★, j★, k]
end

"""Build an intermediate LatLonGrid by reading coordinate arrays from the downloaded file."""
function intermediate_grid_from_file(metadata, arch; halo)
    path = metadata_path(metadata)
    ds = Dataset(path)

    # Try common coordinate variable names
    λ = read_longitude(ds)
    φ = read_latitude(ds)
    close(ds)

    Nx = length(λ)
    Ny = length(φ)
    _, _, Nz, _ = size(metadata)
    z = z_interfaces(metadata)
    FT = eltype(metadata)

    # Build interfaces from cell centers
    if Nx > 1
        Δλ = λ[2] - λ[1]
        λf = vcat(λ .- Δλ/2, [λ[end] + Δλ/2])
    else
        Δλ = FT(1) # arbitrary for single cell
        λf = (λ[1] - Δλ/2, λ[1] + Δλ/2)
    end

    if Ny > 1
        Δφ = φ[2] - φ[1]
        φf = vcat(φ .- Δφ/2, [φ[end] + Δφ/2])
    else
        Δφ = FT(1)
        φf = (φ[1] - Δφ/2, φ[1] + Δφ/2)
    end

    grid = LatitudeLongitudeGrid(arch, FT;
                                 size = (Nx, Ny, Nz),
                                 halo,
                                 longitude = λf,
                                 latitude = φf,
                                 z)
    return grid
end

# Helper to read longitude from NetCDF with common variable names
function read_longitude(ds)
    for name in ("longitude", "lon", "LONGITUDE", "LON", "nav_lon")
        if haskey(ds, name)
            return ds[name][:]
        end
    end
    error("Could not find longitude coordinate variable in $(keys(ds))")
end

# Helper to read latitude from NetCDF with common variable names
function read_latitude(ds)
    for name in ("latitude", "lat", "LATITUDE", "LAT", "nav_lat")
        if haskey(ds, name)
            return ds[name][:]
        end
    end
    error("Could not find latitude coordinate variable in $(keys(ds))")
end

# manglings
struct ShiftSouth end
struct AverageNorthSouth end

@inline mangle(i, j, data, ::Nothing) = @inbounds data[i, j]
@inline mangle(i, j, data, ::ShiftSouth) = @inbounds data[i, j-1]
@inline mangle(i, j, data, ::AverageNorthSouth) = @inbounds (data[i, j+1] + data[i, j]) / 2

@inline mangle(i, j, k, data, ::Nothing) = @inbounds data[i, j, k]
@inline mangle(i, j, k, data, ::ShiftSouth) = @inbounds data[i, j-1, k]
@inline mangle(i, j, k, data, ::AverageNorthSouth) = @inbounds (data[i, j+1, k] + data[i, j, k]) / 2

function set_metadata_field!(field, data, metadatum)
    grid = field.grid
    arch = architecture(grid)

    Nx, Ny, Nz = size(metadatum)
    mangling = if size(data, 2) == Ny-1
        ShiftSouth()
    elseif size(data, 2) == Ny+1
        AverageNorthSouth()
    else
        nothing
    end

    conversion = conversion_units(metadatum)

    if ndims(data) == 2
        _kernel = _set_2d_metadata_field!
        spec = :xy
    else
        _kernel = _set_3d_metadata_field!
        spec = :xyz
    end

    data = on_architecture(arch, data)
    Oceananigans.Utils.launch!(arch, grid, spec, _kernel, field, data, mangling, conversion)

    return nothing
end

@kernel function _set_2d_metadata_field!(field, data, mangling, conversion)
    i, j = @index(Global, NTuple)
    FT = eltype(field)
    d = mangle(i, j, data, mangling)
    d = nan_convert_missing(FT, d)
    d = convert_units(d, conversion)
    @inbounds field[i, j, 1] = d
end

@kernel function _set_3d_metadata_field!(field, data, mangling, conversion)
    i, j, k = @index(Global, NTuple)
    FT = eltype(field)
    d = mangle(i, j, k, data, mangling)
    d = nan_convert_missing(FT, d)
    d = convert_units(d, conversion)

    @inbounds field[i, j, k] = d
end

#####
##### Helper functions
#####

@inline nan_convert_missing(FT, ::Missing) = convert(FT, NaN)
@inline nan_convert_missing(FT, d::Number) = convert(FT, d)

# No units conversion
@inline convert_units(T, units) = T

# Just switch sign!
@inline convert_units(T::FT, ::InverseSign) where FT = - T

# Temperature units
@inline convert_units(T::FT, ::Kelvin) where FT = T - convert(FT, 273.15)

# Molar units
@inline convert_units(C::FT, ::Union{MolePerLiter, MolePerKilogram})           where FT = C * convert(FT, 1e3)
@inline convert_units(C::FT, ::Union{MillimolePerLiter, MillimolePerKilogram}) where FT = C * convert(FT, 1)
@inline convert_units(C::FT, ::Union{MicromolePerLiter, MicromolePerKilogram}) where FT = C * convert(FT, 1e-3)
@inline convert_units(C::FT, ::Union{NanomolePerLiter, NanomolePerKilogram})   where FT = C * convert(FT, 1e-6)
@inline convert_units(C::FT, ::MilliliterPerLiter)                             where FT = C / convert(FT, 22.3916)
@inline convert_units(C::FT, ::GramPerKilogramMinus35)                         where FT = C + convert(FT, 35)


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

@inline is_masked(a, min_value, max_value, mask_value) = isnan(a) | (a <= min_value) | (a >= max_value) | (a == mask_value)
