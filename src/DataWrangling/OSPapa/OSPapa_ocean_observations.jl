import Oceananigans: location
import Oceananigans.Fields: set!

using Oceananigans.Grids: znodes
using Oceananigans.Architectures: architecture

#####
##### OSPapaHourly dataset type
#####

struct OSPapaHourly end

const OSPapaMetadata{D} = Metadata{<:OSPapaHourly, D}
const OSPapaMetadatum = Metadatum{<:OSPapaHourly}

metaprefix(::OSPapaMetadata) = "OSPapaMetadata"

default_download_directory(::OSPapaHourly) = mkpath(download_OSPapa_cache)

available_variables(::OSPapaHourly) = OSPapa_dataset_variable_names

const OSPapa_dataset_variable_names = Dict(
    :temperature        => "TEMP",
    :salinity           => "PSAL",
    :eastward_wind      => "UWND",
    :northward_wind     => "VWND",
    :air_temperature    => "AIRT",
    :relative_humidity  => "RELH",
    :sea_level_pressure => "ATMS",
    :shortwave_radiation => "SW",
    :longwave_radiation  => "LW",
    :rain               => "RAIN",
    :eastward_velocity  => "UCUR",
    :northward_velocity => "VCUR",
)

const OSPapa_depth_variable_names = Dict(
    :temperature        => "DEPTH",
    :salinity           => "DEPPSAL",
    :eastward_velocity  => "DEPCUR",
    :northward_velocity => "DEPCUR",
)

dataset_variable_name(data::OSPapaMetadata) = OSPapa_dataset_variable_names[data.name]

location(::OSPapaMetadata) = (Center, Center, Center)
is_three_dimensional(md::OSPapaMetadata) = md.name in (:temperature, :salinity, :eastward_velocity, :northward_velocity)
reversed_vertical_axis(::OSPapaHourly) = true
function conversion_units(metadatum::OSPapaMetadatum)
    if metadatum.name in (:eastward_velocity, :northward_velocity)
        return CentimetersPerSecond()
    else
        return nothing
    end
end
default_inpainting(::OSPapaMetadata) = nothing

#####
##### Single-file metadata: filename and download
#####

metadata_filename(::OSPapaMetadatum) = OSPAPA_FILENAME
metadata_filename(::OSPapaHourly, name, date, bounding_box) = OSPAPA_FILENAME

function download_dataset(metadata::OSPapaMetadata)
    download_ospapa_file(metadata.dir)
    return nothing
end

function inpainted_metadata_path(metadata::OSPapaMetadata)
    filename = metadata_filename(first(metadata))
    without_ext = filename[1:end-3]
    varname = string(metadata.name)
    return joinpath(metadata.dir, without_ext * "_" * varname * "_inpainted.jld2")
end

#####
##### Epoch, time step, dates, and sizes
#####

metadata_epoch(::OSPapaHourly) = DateTime(2007, 6, 7, 23, 0, 0)
metadata_time_step(::OSPapaHourly) = 3600 # seconds (hourly data)

# Cache the time vector to avoid re-reading the file for every call
const _ospapa_times_cache = Ref{Vector{DateTime}}()
const _ospapa_times_cached = Ref(false)

function ospapa_all_times(dir=download_OSPapa_cache)
    if !_ospapa_times_cached[]
        filepath = download_ospapa_file(dir)
        ds = NCDataset(filepath)
        _ospapa_times_cache[] = DateTime.(ds["TIME"][:])
        close(ds)
        _ospapa_times_cached[] = true
    end
    return _ospapa_times_cache[]
end

all_dates(::OSPapaHourly, variable) = ospapa_all_times()

# Cache depth arrays per variable
const _ospapa_depths_cache = Dict{Symbol, Vector{Float64}}()

function _ospapa_depths(variable, dir=download_OSPapa_cache)
    if !haskey(_ospapa_depths_cache, variable)
        filepath = download_ospapa_file(dir)
        ds = NCDataset(filepath)
        depthvar = OSPapa_depth_variable_names[variable]
        depths = Float64.(ds[depthvar][:])
        close(ds)
        _ospapa_depths_cache[variable] = depths
    end
    return _ospapa_depths_cache[variable]
end

function Base.size(::OSPapaHourly, variable)
    if variable in (:temperature, :salinity, :eastward_velocity, :northward_velocity)
        depths = _ospapa_depths(variable)
        return (1, 1, length(depths))
    else
        return (1, 1, 1)
    end
end

#####
##### Grid construction
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
    z_faces[end] = 0.0  # sea surface
    for k in 1:Nz-1
        z_faces[k+1] = (z_centers[k] + z_centers[k+1]) / 2
    end
    # Extrapolate bottom face
    z_faces[1] = z_centers[1] - (z_faces[2] - z_centers[1])
    return z_faces
end

function z_interfaces(dataset::OSPapaHourly; variable=:temperature)
    depths = _ospapa_depths(variable)
    z_centers = sort(-depths)  # convert to negative, deepest first
    return centers_to_interfaces(z_centers)
end

z_interfaces(md::OSPapaMetadata) = z_interfaces(md.dataset; variable=md.name)

longitude_interfaces(::OSPapaHourly) = (OSPAPA_LONGITUDE, OSPAPA_LONGITUDE)
latitude_interfaces(::OSPapaHourly)  = (OSPAPA_LATITUDE, OSPAPA_LATITUDE)

function native_grid(metadata::OSPapaMetadata, arch=CPU(); halo=(3, 3, 3))
    if is_three_dimensional(metadata)
        Nz = size(metadata.dataset, metadata.name)[3]
        z = z_interfaces(metadata)
        return RectilinearGrid(arch; size=Nz,
                               x=OSPAPA_LONGITUDE, y=OSPAPA_LATITUDE,
                               z=z, topology=(Flat, Flat, Bounded), halo=(halo[3],))
    else
        return RectilinearGrid(arch; size=(),
                               topology=(Flat, Flat, Flat))
    end
end

#####
##### Data retrieval
#####

function retrieve_data(metadata::OSPapaMetadatum)
    filepath = metadata_path(metadata)
    ds = NCDataset(filepath)
    varname = dataset_variable_name(metadata)

    # Find the time index matching metadata.dates
    all_times = ds["TIME"][:]
    t_idx = findfirst(t -> t == metadata.dates, all_times)

    if isnothing(t_idx)
        close(ds)
        error("Date $(metadata.dates) not found in OS Papa dataset")
    end

    if is_three_dimensional(metadata)
        # Read single profile (spatial dims are (1, 1, Nz))
        raw = ds[varname][1, 1, :, t_idx]

        # Apply QC flag filtering: keep only good (1) and probably good (2) data
        qc_varname = varname * "_QC"
        if haskey(ds, qc_varname)
            qc = ds[qc_varname][1, 1, :, t_idx]
            for i in eachindex(raw)
                q = ismissing(qc[i]) ? Int8(9) : Int8(qc[i])
                if q > 2
                    raw[i] = missing
                end
            end
        end

        close(ds)

        data = Float64.(replace(raw, missing => NaN))

        # NetCDF stores shallow-to-deep, but the grid z-axis is bottom-to-top
        # (most negative first), so reverse to match grid ordering
        reverse!(data)

        return reshape(data, 1, 1, :)  # (1, 1, Nz)
    else
        # Surface variable: read scalar at the given time index
        raw = ds[varname][1, 1, 1, t_idx]

        # Apply QC flag filtering
        qc_varname = varname * "_QC"
        if haskey(ds, qc_varname)
            qc = ds[qc_varname][1, 1, 1, t_idx]
            q = ismissing(qc) ? Int8(9) : Int8(qc)
            if q > 2
                raw = missing
            end
        end

        close(ds)

        data = Float64(ismissing(raw) ? NaN : raw)

        return reshape([data], 1, 1, 1)  # (1, 1, 1)
    end
end

#####
##### Custom set! for single-column vertical interpolation
#####

"""
    _vertical_interpolate(metadata::OSPapaMetadatum, z_src, data_src, z_dst)

Linearly interpolate a 1D profile from `z_src` centers onto `z_dst` levels.
NaN values in `data_src` are skipped. Values outside the source range
are extrapolated from the nearest valid value.
"""
function _vertical_interpolate(::OSPapaMetadatum, z_src, data_src, z_dst)
    result = similar(z_dst, Float64)

    # Filter out NaN values
    valid = .!isnan.(data_src)
    zv = z_src[valid]
    dv = data_src[valid]

    if isempty(zv)
        result .= NaN
        return result
    end

    # Sort by depth (most negative first)
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

function set!(target_field::Field, metadata::OSPapaMetadatum; kw...)
    download_dataset(metadata)

    # Read the raw profile data
    data_3d = retrieve_data(metadata)
    data_profile = data_3d[1, 1, :]

    # Source z-centers (buoy depths)
    depths = _ospapa_depths(metadata.name)
    z_src = sort(-depths)  # negative, deepest first

    # Target z-centers
    z_dst = collect(znodes(target_field.grid, Center()))

    # Interpolate vertically
    interpolated = _vertical_interpolate(metadata, z_src, data_profile, z_dst)

    if metadata.name in (:eastward_velocity, :northward_velocity)
        interpolated ./= 100 # cm/s → m/s
    end

    arch = Oceananigans.Architectures.architecture(target_field)
    interior(target_field, 1, 1, :) .= Oceananigans.on_architecture(arch, interpolated)

    return target_field
end
