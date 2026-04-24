module HadISST

export HadISSTSST, HadISSTICE

using Downloads
using NCDatasets
using Dates
using Oceananigans
using Oceananigans.DistributedComputations: @root
using Scratch
using CodecZlib

using ..DataWrangling:
    Metadata,
    Metadatum,
    metadata_path,
    download_progress

import Oceananigans.Fields: location

import NumericalEarth.DataWrangling:
    all_dates,
    first_date,
    last_date,
    metadata_filename,
    download_dataset,
    default_download_directory,
    dataset_variable_name,
    metaprefix,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    reversed_vertical_axis,
    available_variables,
    retrieve_data,
    is_three_dimensional

download_HadISST_cache::String = ""
function __init__()
    global download_HadISST_cache = @get_scratch!("HadISST")
end

"""
    HadISSTDataset

Abstract supertype for Met Office Hadley Centre Global Sea Ice and SST v1.1
datasets (Rayner et al. 2003). 1° × 1° monthly, January 1870 to present.
"""
abstract type HadISSTDataset end

"""
    HadISSTSST()

Monthly SST (°C) from HadISST v1.1.
"""
struct HadISSTSST <: HadISSTDataset end

"""
    HadISSTICE()

Monthly sea-ice concentration (fraction) from HadISST v1.1.
"""
struct HadISSTICE <: HadISSTDataset end

default_download_directory(::HadISSTDataset) = download_HadISST_cache
longitude_interfaces(::HadISSTDataset) = (-180, 180)
latitude_interfaces(::HadISSTDataset)  = (-90, 90)
reversed_vertical_axis(::HadISSTDataset) = false

# HadISST is monthly 1870-01 to "present month"
function all_dates(::HadISSTDataset, args...)
    start = DateTime(1870, 1, 1)
    stop  = DateTime(Dates.year(today()), Dates.month(today()), 1)
    return start:Month(1):stop
end
first_date(d::HadISSTDataset, args...) = first(all_dates(d))
last_date(d::HadISSTDataset, args...)  = last(all_dates(d))

Base.size(::HadISSTDataset, args...) = (360, 180, 1)

const HadISSTMetadata{D}  = Metadata{<:HadISSTDataset, D}
const HadISSTMetadatum    = Metadatum{<:HadISSTDataset}

metaprefix(::HadISSTMetadata)  = "HadISSTMetadata"
metaprefix(::HadISSTMetadatum) = "HadISSTMetadatum"

const HADISST_VARS = Dict(
    :sea_surface_temperature => "sst",
    :sea_ice_concentration   => "sic",
)

dataset_variable_name(m::HadISSTMetadata) = HADISST_VARS[m.name]
available_variables(::HadISSTDataset)     = HADISST_VARS

# Both SST and ICE ship as single files containing the full time series.
metadata_filename(::HadISSTSST, args...) = "HadISST_sst.nc"
metadata_filename(::HadISSTICE, args...) = "HadISST_ice.nc"

metadata_url(::HadISSTSST) =
    "https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz"
metadata_url(::HadISSTICE) =
    "https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_ice.nc.gz"

z_interfaces(::HadISSTMetadatum) = (0, 1)

function download_dataset(m::HadISSTMetadatum)
    nc_path = metadata_path(m)
    @root if !isfile(nc_path)
        gz_path = nc_path * ".gz"
        url = metadata_url(m.dataset)
        @info "Downloading HadISST $(metadata_filename(m.dataset)) from $url"
        Downloads.download(url, gz_path; progress = download_progress)

        open(GzipDecompressorStream, gz_path) do stream
            open(nc_path, "w") do io
                write(io, read(stream))
            end
        end
        rm(gz_path)
    end
    return nc_path
end

# `Metadata` (collection) downloads by delegating to single-date `Metadatum`.
download_dataset(m::HadISSTMetadata) = [download_dataset(md) for md in m]

location(::HadISSTMetadata) = (Center, Center, Center)
is_three_dimensional(::HadISSTMetadata) = false

function retrieve_data(m::HadISSTMetadatum)
    ds = Dataset(metadata_path(m))
    varname = dataset_variable_name(Metadata(m.name; dataset = m.dataset))
    times = ds["time"][:]

    target = DateTime(Dates.year(m.dates), Dates.month(m.dates), 1)
    k = findfirst(t -> DateTime(Dates.year(t), Dates.month(t), 1) == target, times)
    isnothing(k) && (close(ds); error("HadISST has no record for $target"))

    # Partial read of a single time index, not the full 485 MB variable
    slice = ds[varname][:, :, k]
    close(ds)

    # HadISST ice/no-data sentinel: -1000 is used in addition to the declared
    # _FillValue of -1e30 (which NCDatasets already surfaces as `missing`).
    # Both need to become NaN so reductions ignore them.
    arr = Array{Float32}(undef, size(slice))
    @inbounds for i in eachindex(slice)
        v = slice[i]
        arr[i] = (ismissing(v) || Float32(v) <= -999.0f0) ? NaN32 : Float32(v)
    end

    # HadISST latitude is stored N→S; flip so our grid (which expects
    # S→N for increasing j) sees increasing latitude.
    arr = reverse(arr; dims = 2)

    return reshape(arr, size(arr)..., 1)  # add singleton z
end

end # module
