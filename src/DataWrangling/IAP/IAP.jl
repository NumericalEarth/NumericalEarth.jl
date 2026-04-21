module IAP

export IAPOceanHeatContent

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

download_IAP_cache::String = ""
function __init__()
    global download_IAP_cache = @get_scratch!("IAP")
end

"""
    IAPOceanHeatContent{depth}

IAP ocean heat content estimate (Cheng et al. 2017) integrated from the
surface down to `depth`. Currently `depth = :top300m` is supported.
Data source: <http://www.ocean.iap.ac.cn/>. Units: 10²² J (column
integrated).
"""
struct IAPOceanHeatContent{depth} end
IAPOceanHeatContent(; depth::Symbol = :top300m) = IAPOceanHeatContent{depth}()

default_download_directory(::IAPOceanHeatContent) = download_IAP_cache
longitude_interfaces(::IAPOceanHeatContent) = (-180, 180)
latitude_interfaces(::IAPOceanHeatContent)  = (-90, 90)
reversed_vertical_axis(::IAPOceanHeatContent) = false

function all_dates(::IAPOceanHeatContent, args...)
    start = DateTime(1940, 1, 1)
    stop  = DateTime(Dates.year(today()) - 1, 12, 1)
    return start:Month(1):stop
end
first_date(d::IAPOceanHeatContent, args...) = first(all_dates(d))
last_date(d::IAPOceanHeatContent, args...)  = last(all_dates(d))

Base.size(::IAPOceanHeatContent, args...) = (360, 180, 1)

const IAPMetadata{D}   = Metadata{<:IAPOceanHeatContent, D}
const IAPMetadatum     = Metadatum{<:IAPOceanHeatContent}

metaprefix(::IAPMetadata)  = "IAPMetadata"
metaprefix(::IAPMetadatum) = "IAPMetadatum"

const IAP_VARS = Dict(:ocean_heat_content => "OHC")
dataset_variable_name(m::IAPMetadata) = IAP_VARS[m.name]
available_variables(::IAPOceanHeatContent) = IAP_VARS

metadata_filename(::IAPOceanHeatContent{:top300m}, args...) =
    "IAP_OHC_Monthly_0-300m.nc"

metadata_url(::IAPOceanHeatContent{:top300m}) =
    "http://www.ocean.iap.ac.cn/ftp/cheng/IAP_OHC_Monthly_0-300m.nc"

z_interfaces(::IAPMetadatum) = (0, 1)

function download_dataset(m::IAPMetadatum)
    path = metadata_path(m)
    @root if !isfile(path)
        url = metadata_url(m.dataset)
        @info "Downloading IAP OHC from $url"
        try
            Downloads.download(url, path; progress = download_progress)
        catch err
            error("IAP download failed. IAP occasionally reorganizes its " *
                  "files — manually place the NetCDF at $(path). " *
                  "Original error: $err")
        end
    end
    return path
end
download_dataset(m::IAPMetadata) = [download_dataset(md) for md in m]

location(::IAPMetadata) = (Center, Center, Center)
is_three_dimensional(::IAPMetadata) = false

function retrieve_data(m::IAPMetadatum)
    ds = Dataset(metadata_path(m))
    raw   = ds["OHC"][:, :, :]
    times = ds["time"][:]
    close(ds)

    target = DateTime(Dates.year(m.dates), Dates.month(m.dates), 1)
    k = findfirst(t -> DateTime(Dates.year(t), Dates.month(t), 1) == target, times)
    isnothing(k) && error("IAP has no record for $target")

    slice = raw[:, :, k]
    arr = Array{Float32}(undef, size(slice))
    @inbounds for i in eachindex(slice)
        arr[i] = ismissing(slice[i]) ? NaN32 : Float32(slice[i])
    end
    return reshape(arr, size(arr)..., 1)
end

end # module
