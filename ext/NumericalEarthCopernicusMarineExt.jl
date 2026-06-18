module NumericalEarthCopernicusMarineExt

using CopernicusMarine: CopernicusMarine
using Dates: DateTime
using Downloads: Downloads
using Oceananigans.DistributedComputations: @root
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Column, Linear, Nearest, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYS, GLORYSMetadata, GLORYSMetadatum
using NumericalEarth.DataWrangling.AVISO: AVISO, AVISOMetadata, AVISOMetadatum

# Download each date individually, instead of downloading the entire dataset at once.
# This is useful for a possible extension of the temporal horizon of the dataset.
const CopernicusMetadata = Union{GLORYSMetadata, AVISOMetadata}
const CopernicusMetadatum = Union{GLORYSMetadatum, AVISOMetadatum}

function copernicusmarine_dataset_id(dataset)
    if dataset isa GLORYS.GLORYSDataset
        return GLORYS.copernicusmarine_dataset_id(dataset)
    elseif dataset isa AVISO.AVISODataset
        return AVISO.copernicusmarine_dataset_id(dataset)
    end
    error("No CopernicusMarine dataset id is defined for $(typeof(dataset)).")
end

function native_horizontal_resolution(dataset)
    if dataset isa GLORYS.GLORYSDataset
        return 1 / 12
    elseif dataset isa AVISO.AVISODataset
        return AVISO.native_horizontal_resolution(dataset)
    end
    error("No native horizontal resolution is defined for $(typeof(dataset)).")
end

function Downloads.download(metadata::CopernicusMetadata; kwargs...)
    paths = Array{String}(undef, length(metadata))
    for (m, metadatum) in enumerate(metadata)
        paths[m] = Downloads.download(metadatum; kwargs...)
    end
    return paths
end

function Downloads.download(meta::CopernicusMetadatum;
                                                       skip_existing=true,
                                                       username=get(ENV, "COPERNICUS_USERNAME", nothing),
                                                       password=get(ENV, "COPERNICUS_PASSWORD", nothing),
                                                       additional_kw...)

    output_directory = meta.dir
    output_filename = meta.filename
    output_path = joinpath(output_directory, output_filename)
    isfile(output_path) && return output_path

    toolbox = CopernicusMarine.copernicusmarine

    variable_name = dataset_variable_name(meta)
    variables = CopernicusMarine.pylist([variable_name])

    dataset_id = copernicusmarine_dataset_id(meta.dataset)
    datetime_kw = if meta.dataset isa GLORYS.GLORYSStatic
        NamedTuple()
    else
        start_datetime = GLORYS.start_date_str(meta.dates)
        end_datetime = GLORYS.end_date_str(meta.dates)
        (; start_datetime, end_datetime)
    end

    lon_kw = longitude_bounds_kw(meta.region, meta.dataset)
    lat_kw = latitude_bounds_kw(meta.region, meta.dataset)
    z_kw = depth_bounds_kw(meta.region)
    selection_method = coordinates_selection_method(meta.region)

    # netcdf3_compatible routes the write through xarray's netcdf4 engine instead of
    # h5netcdf, whose h5py binds to the HDF5_jll libhdf5 (no ROS3 VFD) and fails in-process.
    kw = (; coordinates_selection_method = selection_method,
          netcdf3_compatible = true,
          skip_existing,
          dataset_id,
          variables,
          output_filename,
          output_directory)

    if !isnothing(username) && !isnothing(password)
        kw = merge(kw, (; username, password))
    else
        @warn "No Copernicus credentials found.
        Set the COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables to download data from the Copernicus Marine Service.
        You can sign up for free at: https://data.marine.copernicus.eu/register."
    end

    additional_kw = NamedTuple(name => value for (name, value) in additional_kw)
    kw = merge(kw, datetime_kw, lon_kw, lat_kw, z_kw, additional_kw)

    @root toolbox.subset(; kw...)

    return output_path
end

longitude_bounds_kw(::Nothing, dataset) = NamedTuple()
latitude_bounds_kw(::Nothing, dataset) = NamedTuple()
depth_bounds_kw(::Nothing) = NamedTuple()
coordinates_selection_method(::Nothing) = "outside"

const BBOX = BoundingBox
const COL  = Column
const LIN  = Linear
const NR   = Nearest

# The native grid is built by center-bracketing `restrict`, which can reach one
# native cell past a boundary-aligned bbox edge. CopernicusMarine subsets to the
# requested bounds, so fetch two native cells of margin to guarantee the
# downloaded file always covers the center-bracketed native grid; otherwise
# `set_region_data!` indexes one cell past the file at the domain edge. Over-
# fetching is harmless: `restrict` + `BoundingBoxOffset` select the exact cells
# from the larger file. Mirrors `era5_request_area` in the CDS extension.
longitude_bounds_kw(bbox::BBOX, dataset) = longitude_bounds_kw(pad_bounds(bbox.longitude, dataset))
latitude_bounds_kw(bbox::BBOX, dataset) = latitude_bounds_kw(clamp_latitude(pad_bounds(bbox.latitude, dataset)))
depth_bounds_kw(bbox::BBOX) = depth_bounds_kw(bbox.z)
coordinates_selection_method(::BBOX) = "outside"

pad_bounds(::Nothing, dataset) = nothing
pad_bounds(bounds, dataset) = begin
    Δ = native_horizontal_resolution(dataset)
    return (bounds[1] - 2Δ, bounds[2] + 2Δ)
end

clamp_latitude(::Nothing) = nothing
clamp_latitude(bounds) = (max(bounds[1], -90), min(bounds[2], 90))

# Column with Nearest interpolation: download the single nearest point
longitude_bounds_kw(col::COL{<:Any, <:Any, <:Any, NR}, dataset) = (; minimum_longitude = col.longitude, maximum_longitude = col.longitude)
latitude_bounds_kw(col::COL{<:Any, <:Any, <:Any, NR}, dataset)  = (; minimum_latitude = col.latitude, maximum_latitude = col.latitude)
depth_bounds_kw(col::COL) = depth_bounds_kw(col.z)
coordinates_selection_method(::COL{<:Any, <:Any, <:Any, NR}) = "nearest"

# Column with Linear interpolation: expand by a small margin for interpolation
longitude_bounds_kw(col::COL{<:Any, <:Any, <:Any, LIN}, dataset) = expand_longitude(col.longitude, dataset)
latitude_bounds_kw(col::COL{<:Any, <:Any, <:Any, LIN}, dataset)  = expand_latitude(col.latitude, dataset)
coordinates_selection_method(::COL{<:Any, <:Any, <:Any, LIN}) = "outside"

function expand_longitude(lon, dataset)
    ε = 2 * native_horizontal_resolution(dataset)
    return (; minimum_longitude = lon - ε, maximum_longitude = lon + ε)
end

function expand_latitude(lat, dataset)
    ε = 2 * native_horizontal_resolution(dataset)
    return (; minimum_latitude = lat - ε, maximum_latitude = lat + ε)
end

function longitude_bounds_kw(longitude)
    minimum_longitude = longitude[1]
    maximum_longitude = longitude[2]
    return (; minimum_longitude, maximum_longitude)
end

function latitude_bounds_kw(latitude)
    minimum_latitude = latitude[1]
    maximum_latitude = latitude[2]
    return (; minimum_latitude, maximum_latitude)
end

function depth_bounds_kw(z)
    minimum_depth = - z[2]
    maximum_depth = - z[1]
    return (; minimum_depth, maximum_depth)
end

end # module NumericalEarthCopernicusMarineExt
