module ERSST

export ERSSTv5

using Downloads
using NCDatasets
using Dates
using Oceananigans
using Oceananigans.DistributedComputations: @root
using Scratch

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

download_ERSST_cache::String = ""
function __init__()
    global download_ERSST_cache = @get_scratch!("ERSST")
end

"""
    ERSSTv5()

NOAA Extended Reconstructed SST v5 (Huang et al. 2017). Global 2° × 2°
monthly SST, 1854–present, distributed from NOAA NCEI.
"""
struct ERSSTv5 end

default_download_directory(::ERSSTv5) = download_ERSST_cache
longitude_interfaces(::ERSSTv5) = (-180, 180)
latitude_interfaces(::ERSSTv5)  = (-90, 90)
reversed_vertical_axis(::ERSSTv5) = false

function all_dates(::ERSSTv5, args...)
    start = DateTime(1854, 1, 1)
    stop  = DateTime(Dates.year(today()), Dates.month(today()), 1)
    return start:Month(1):stop
end
first_date(d::ERSSTv5, args...) = first(all_dates(d))
last_date(d::ERSSTv5, args...)  = last(all_dates(d))

Base.size(::ERSSTv5, args...) = (180, 89, 1)

const ERSSTMetadata{D}  = Metadata{<:ERSSTv5, D}
const ERSSTMetadatum    = Metadatum{<:ERSSTv5}

metaprefix(::ERSSTMetadata)  = "ERSSTMetadata"
metaprefix(::ERSSTMetadatum) = "ERSSTMetadatum"

const ERSST_VARS = Dict(:sea_surface_temperature => "sst")
dataset_variable_name(m::ERSSTMetadata) = ERSST_VARS[m.name]
available_variables(::ERSSTv5)          = ERSST_VARS

function metadata_filename(::ERSSTv5, name, date, bounding_box)
    y = lpad(string(Dates.year(date)),  4, '0')
    m = lpad(string(Dates.month(date)), 2, '0')
    return "ersst.v5.$(y)$(m).nc"
end

function metadata_url(m::ERSSTMetadatum)
    y = lpad(string(Dates.year(m.dates)),  4, '0')
    mo = lpad(string(Dates.month(m.dates)), 2, '0')
    return "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.$(y)$(mo).nc"
end

z_interfaces(::ERSSTMetadatum) = (0, 1)

function download_dataset(m::ERSSTMetadatum)
    path = metadata_path(m)
    @root if !isfile(path)
        url = metadata_url(m)
        @info "Downloading ERSSTv5 from $url"
        Downloads.download(url, path; progress = download_progress)
    end
    return path
end
download_dataset(m::ERSSTMetadata) = [download_dataset(md) for md in m]

location(::ERSSTMetadata) = (Center, Center, Center)
is_three_dimensional(::ERSSTMetadata) = false

function retrieve_data(m::ERSSTMetadatum)
    ds = Dataset(metadata_path(m))
    raw = ds["sst"][:, :, 1, 1]       # (lon, lat, lev=1, time=1) → (lon, lat)
    lon = Float64.(ds["lon"][:])
    close(ds)

    arr = Array{Float32}(undef, size(raw))
    @inbounds for i in eachindex(raw)
        arr[i] = ismissing(raw[i]) ? NaN32 : Float32(raw[i])
    end

    # ERSST lon is 0..360 → shift to −180..180 to match our longitude_interfaces
    if minimum(lon) ≥ 0
        shift = searchsortedfirst(lon, 180)
        arr   = vcat(arr[shift:end, :], arr[1:shift-1, :])
    end

    return reshape(arr, size(arr)..., 1)
end

end # module
