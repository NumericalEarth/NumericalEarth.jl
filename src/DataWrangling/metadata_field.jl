using NCDatasets
using JLD2
using NumericalEarth.InitialConditions: interpolate!
using Statistics: median
using Oceananigans.Grids: λnodes, φnodes, Periodic, Bounded
using Oceananigans.Architectures: on_architecture

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
restrict(::Nothing, interfaces::NTuple{2,Any}, N) = interfaces, N
restrict(::Nothing, interfaces::AbstractVector, N) = interfaces, N

# Snap bbox outward to native cell faces so restricted centers land on native centers
function restrict(bbox_interfaces, interfaces::NTuple{2,Any}, N)
    left, right = interfaces
    Δ = (right - left) / N
    i⁻ = max(floor(Int, (bbox_interfaces[1] - left) / Δ), 0)
    i⁺ = min(ceil( Int, (bbox_interfaces[2] - left) / Δ), N)
    if i⁺ <= i⁻
        i⁺ = min(i⁻ + 1, N)
        i⁻ = max(i⁺ - 1, 0)
    end
    return (left + i⁻ * Δ, left + i⁺ * Δ), i⁺ - i⁻
end

# Stretched native grid: snap outward to the nearest native cell interfaces.
function restrict(bbox_interfaces, interfaces::AbstractVector, N)
    i⁻ = max(searchsortedlast(interfaces,  bbox_interfaces[1]), 1)
    i⁺ = min(searchsortedfirst(interfaces, bbox_interfaces[2]), length(interfaces))
    rN = max(i⁺ - i⁻, 1)
    return interfaces[i⁻:i⁺], rN
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
                                     halo, longitude, latitude, z)
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

    # Map the bbox into the native longitude convention
    bbox_λ⁻ = convert_to_λ₀_λ₀_plus360(bbox.longitude[1], native_longitude[1])
    bbox_λ⁺ = bbox_λ⁻ + (bbox.longitude[2] - bbox.longitude[1])

    Nx, Ny, Nz = size(metadata)
    longitude, Nx = restrict((bbox_λ⁻, bbox_λ⁺), native_longitude, Nx)
    latitude,  Ny = restrict(bbox.latitude,      native_latitude,  Ny)

    TX = infer_longitudinal_topology(native_longitude, longitude)

    if is_three_dimensional(metadata)
        z = z_interfaces(metadata)
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                     halo, longitude, latitude, z,
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
                               x, y, z, topology = (Flat, Flat, Bounded))
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
    Field(metadata::Metadatum;
          architecture = CPU(),
          inpainting = default_inpainting(metadata),
          mask = nothing,
          halo = (3, 3, 3),
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

    # Inpainting on a (Flat, Flat, *) column field is meaningless and the
    # iterative algorithm doesn't terminate gracefully without horizontal
    # neighbours; the NaN-aware bracket-blend in `set_region_data!` handles
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

function set!(target_field::Field, metadata::Metadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    meta_field = Field(metadata, arch; kw...)

    Lzt = grid.Lz
    Lzm = meta_field.grid.Lz

    # Allow up to 1% vertical mismatch for pressure-level datasets with time-varying
    # geopotential heights — the per-timestep vertical extent can be slightly smaller
    # than the temporal-mean extent used for the target grid (e.g. when the atmosphere
    # is compressed). Oceananigans' interpolate! does not extrapolate, so target points
    # just outside the source domain will use the nearest interior values.
    if is_three_dimensional(metadata) && Lzt > Lzm * (1 + 1e-2)
        throw("The vertical range of the $(metadata.dataset) dataset ($(Lzm) m) is smaller than " *
              "the target grid ($(Lzt) m). Some vertical levels cannot be filled with data.")
    end

    interpolate!(target_field, meta_field)

    return target_field
end

function set_metadata_field!(field, data, metadatum)
    full_data = ndims(data) == 2 ? reshape(data, size(data, 1), size(data, 2), 1) : data
    λc, φc = read_file_coords(metadatum)
    set_region_data!(field, full_data, λc, φc, metadatum)
    return nothing
end

# Read the lon/lat cell centres from the NetCDF file using the names supplied
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

# Molar units
@inline convert_units(C::FT, ::Union{MolePerLiter, MolePerKilogram})           where FT = C * convert(FT, 1e3)
@inline convert_units(C::FT, ::Union{MillimolePerLiter, MillimolePerKilogram}) where FT = C * convert(FT, 1)
@inline convert_units(C::FT, ::Union{MicromolePerLiter, MicromolePerKilogram}) where FT = C * convert(FT, 1e-3)
@inline convert_units(C::FT, ::Union{NanomolePerLiter, NanomolePerKilogram})   where FT = C * convert(FT, 1e-6)
@inline convert_units(C::FT, ::MilliliterPerLiter)                             where FT = C / convert(FT, 22.3916)
@inline convert_units(C::FT, ::GramPerKilogramMinus35)                         where FT = C + convert(FT, 35)
@inline convert_units(Φ::FT, ::InverseGravity)                                where FT = Φ / convert(FT, 9.80665)
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
