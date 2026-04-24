using NCDatasets
using JLD2
using NumericalEarth.InitialConditions: interpolate!
using Statistics: median

import Oceananigans.Fields: set!, Field

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

Return a grid on `arch` corresponding to the native layout of `metadata`. Dispatch
happens on the `spatial_layout` trait: the default `GriddedLatLon` returns a
`LatitudeLongitudeGrid`; `StationColumn` returns a single-column
`RectilinearGrid(Flat, Flat, Bounded)` anchored at the dataset's (longitude,
latitude) point.
"""
native_grid(metadata::Metadata, arch=CPU(); halo = (3, 3, 3)) = native_grid(spatial_layout(metadata.dataset), metadata, arch; halo)

function native_grid(::GriddedLatLon, metadata::Metadata, arch=CPU(); halo = (3, 3, 3))
    Nx, Ny, Nz, _ = size(metadata)
    z = z_interfaces(metadata)

    FT = eltype(metadata)

    longitude = longitude_interfaces(metadata)
    latitude = latitude_interfaces(metadata)

    # Restrict with BoundingBox
    # TODO: can we restrict in `z` as well?
    bbox = metadata.bounding_box
    if !isnothing(bbox)
        longitude, Nx = restrict(bbox.longitude, longitude, Nx)
        latitude, Ny = restrict(bbox.latitude, latitude, Ny)
    end

    grid = LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                 halo, longitude, latitude, z)

    return grid
end

# StationColumn: single-point (longitude, latitude) with a Bounded z-axis when
# the variable is three-dimensional, otherwise a fully Flat grid for scalars.
function native_grid(::StationColumn, metadata::Metadata, arch=CPU(); halo = (3, 3, 3))
    lon_interfaces = longitude_interfaces(metadata)
    lat_interfaces = latitude_interfaces(metadata)
    longitude = first(lon_interfaces)
    latitude  = first(lat_interfaces)

    if is_three_dimensional(metadata)
        Nz = size(metadata.dataset, metadata.name)[3]
        z  = z_interfaces(metadata)
        hz = length(halo) >= 3 ? halo[3] : 3
        return RectilinearGrid(arch; size = Nz,
                               x = longitude, y = latitude, z = z,
                               topology = (Flat, Flat, Bounded),
                               halo = (hz,))
    else
        return RectilinearGrid(arch; size = (),
                               topology = (Flat, Flat, Flat))
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

    # Retrieve data from file according to metadata type, then apply dataset-specific
    # preprocessing (QC filtering, threshold masking, etc.). Default is identity.
    data = preprocess_data(retrieve_data(metadata), metadata)

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

"""
    set!(target_field::Field, metadata::Metadatum; kw...)

Populate `target_field` from `metadata`. Dispatches on the `spatial_layout` trait:
`GriddedLatLon` uses full 3-D interpolation via Oceananigans' `interpolate!`;
`StationColumn` uses a vertical-only, NaN-skipping interpolation suited to
single-column moored-station data.
"""
set!(target_field::Field, metadata::Metadatum; kw...) =
    set!(spatial_layout(metadata.dataset), target_field, metadata; kw...)

function set!(::GriddedLatLon, target_field::Field, metadata::Metadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    meta_field = Field(metadata, arch; kw...)

    Lzt = grid.Lz
    Lzm = meta_field.grid.Lz

    if Lzt > Lzm && is_three_dimensional(metadata)
        throw("The vertical range of the $(metadata.dataset) dataset ($(Lzm) m) is smaller than " *
              "the target grid ($(Lzt) m). Some vertical levels cannot be filled with data.")
    end

    interpolate!(target_field, meta_field)
    return target_field
end

# StationColumn: vertical-only interpolation that skips NaN sentinels, with
# nearest-value extrapolation outside the source depth range. This exists
# because Oceananigans' `interpolate!` does not skip NaNs and currently has a
# bug for single-column fields (Oceananigans.jl issue #5511).
function set!(::StationColumn, target_field::Field, metadata::Metadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    meta_field = Field(metadata, arch; kw...)

    # Scalar (non-profile) variables: no vertical interpolation, just copy.
    if !is_three_dimensional(metadata)
        parent(target_field) .= parent(meta_field)
        return target_field
    end

    Lzt = grid.Lz
    Lzm = meta_field.grid.Lz
    if Lzt > Lzm
        throw("The vertical range of the $(metadata.dataset) dataset ($(Lzm) m) is smaller than " *
              "the target grid ($(Lzt) m). Some vertical levels cannot be filled with data.")
    end

    z_src = collect(znodes(meta_field.grid, Center()))
    z_dst = collect(znodes(grid, Center()))
    data_profile = Array(interior(meta_field, 1, 1, :))

    interpolated = vertical_interpolate_skip_nan(z_src, data_profile, z_dst)

    interior(target_field, 1, 1, :) .= on_architecture(arch, interpolated)
    return target_field
end

"""
    vertical_interpolate_skip_nan(z_src, data_src, z_dst)

Linearly interpolate a 1-D profile from `z_src` cell centers onto `z_dst`
levels, skipping any `NaN` entries in `data_src`. Targets outside the valid
source range are extrapolated from the nearest valid value (i.e. held
constant). Used by the `StationColumn` implementation of `set!`.
"""
function vertical_interpolate_skip_nan(z_src, data_src, z_dst)
    result = similar(z_dst, Float64)

    valid = .!isnan.(data_src)
    zv = z_src[valid]
    dv = data_src[valid]

    if isempty(zv)
        result .= NaN
        return result
    end

    perm = sortperm(zv)
    zv = zv[perm]
    dv = dv[perm]

    for (i, zt) in enumerate(z_dst)
        if zt <= zv[1]
            result[i] = dv[1]
        elseif zt >= zv[end]
            result[i] = dv[end]
        else
            j = searchsortedlast(zv, zt)
            α = (zt - zv[j]) / (zv[j+1] - zv[j])
            result[i] = dv[j] + α * (dv[j+1] - dv[j])
        end
    end

    return result
end

# Shipped mangle methods. The tag struct definitions and the `mangle` generic
# function live in Datasets (Datasets.jl). Tags here cover off-by-one grid
# staggering and averaging between adjacent rows.
@inline mangle(i, j, data, ::ShiftSouth) = @inbounds data[i, j-1]
@inline mangle(i, j, data, ::AverageNorthSouth) = @inbounds (data[i, j+1] + data[i, j]) / 2

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

"""
    centers_to_interfaces(z_centers)

Compute z-interfaces (cell faces) from cell center positions.
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

@inline nan_convert_missing(FT, ::Missing) = convert(FT, NaN)
@inline nan_convert_missing(FT, d::Number) = convert(FT, d)

# Shipped unit-conversion methods. The tag struct definitions, the generic
# `convert_units` function, and its identity default live in Datasets.

# Just switch sign!
@inline convert_units(T::FT, ::InverseSign) where FT = - T

# Temperature units
@inline convert_units(T::FT, ::Kelvin) where FT = T - convert(FT, 273.15)
@inline convert_units(T::FT, ::Celsius) where FT = T + convert(FT, 273.15)

# Pressure units
@inline convert_units(P::FT, ::Millibar) where FT = P * convert(FT, 100)

# Precipitation rate (assuming ρ_water = 1000 kg/m³, so 1 mm/hr = 1 kg/m²/hr = 1/3600 kg/m²/s)
@inline convert_units(r::FT, ::MillimetersPerHour) where FT = r / convert(FT, 3600)

# Molar units
@inline convert_units(C::FT, ::Union{MolePerLiter, MolePerKilogram})           where FT = C * convert(FT, 1e3)
@inline convert_units(C::FT, ::Union{MillimolePerLiter, MillimolePerKilogram}) where FT = C * convert(FT, 1)
@inline convert_units(C::FT, ::Union{MicromolePerLiter, MicromolePerKilogram}) where FT = C * convert(FT, 1e-3)
@inline convert_units(C::FT, ::Union{NanomolePerLiter, NanomolePerKilogram})   where FT = C * convert(FT, 1e-6)
@inline convert_units(C::FT, ::MilliliterPerLiter)                             where FT = C / convert(FT, 22.3916)
@inline convert_units(C::FT, ::GramPerKilogramMinus35)                         where FT = C + convert(FT, 35)
@inline convert_units(V::FT, ::CentimetersPerSecond)                           where FT = V / convert(FT, 100)


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
