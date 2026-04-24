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

IAP ocean heat content estimate (Cheng et al. 2017, IAPv4.2) integrated
from the surface down to `depth`. Supported values of `depth`: `:top100m`,
`:top300m`, `:top700m`, `:top1500m`, `:top2000m`, `:top3000m`, `:top4000m`,
`:top5000m`, `:top6000m`. Data source:
<http://www.ocean.iap.ac.cn/ftp/cheng/IAPv4.2_Ocean_heat_content_0_6000m/>.
Each monthly file contains OHC integrated to each of the depths above,
as 2-D fields in J/m².
"""
struct IAPOceanHeatContent{depth} end
IAPOceanHeatContent(; depth::Symbol = :top300m) = IAPOceanHeatContent{depth}()

_iap_depth_suffix(::IAPOceanHeatContent{:top100m})  = "100"
_iap_depth_suffix(::IAPOceanHeatContent{:top300m})  = "300"
_iap_depth_suffix(::IAPOceanHeatContent{:top700m})  = "700"
_iap_depth_suffix(::IAPOceanHeatContent{:top1500m}) = "1500"
_iap_depth_suffix(::IAPOceanHeatContent{:top2000m}) = "2000"
_iap_depth_suffix(::IAPOceanHeatContent{:top3000m}) = "3000"
_iap_depth_suffix(::IAPOceanHeatContent{:top4000m}) = "4000"
_iap_depth_suffix(::IAPOceanHeatContent{:top5000m}) = "5000"
_iap_depth_suffix(::IAPOceanHeatContent{:top6000m}) = "6000"

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

const IAP_VARS = Dict(:ocean_heat_content => "OHC300")
dataset_variable_name(m::IAPMetadata) =
    "OHC" * _iap_depth_suffix(m.dataset)
available_variables(::IAPOceanHeatContent) = IAP_VARS

function metadata_filename(::IAPOceanHeatContent, name, date, args...)
    y  = Dates.year(date)
    mm = lpad(Dates.month(date), 2, '0')
    return "OHC_IAP_0_6000m_year_$(y)_month_$(mm).nc"
end

function metadata_url(m::IAPMetadatum)
    y  = Dates.year(m.dates)
    mm = lpad(Dates.month(m.dates), 2, '0')
    return "http://www.ocean.iap.ac.cn/ftp/cheng/" *
           "IAPv4.2_Ocean_heat_content_0_6000m/" *
           "OHC_IAP_0_6000m_year_$(y)_month_$(mm).nc"
end

z_interfaces(::IAPMetadatum) = (0, 1)

function download_dataset(m::IAPMetadatum)
    path = metadata_path(m)
    @root if !isfile(path)
        url = metadata_url(m)
        @info "Downloading IAP OHC from $url"
        try
            Downloads.download(url, path; progress = download_progress)
        catch err
            error("IAP download failed. IAP occasionally reorganizes its " *
                  "files. Manually place the NetCDF at $(path), or update " *
                  "`metadata_url` in NumericalEarth.DataWrangling.IAP. " *
                  "Original error: $err")
        end
    end
    return path
end
download_dataset(m::IAPMetadata) = [download_dataset(md) for md in m]

location(::IAPMetadata) = (Center, Center, Center)
is_three_dimensional(::IAPMetadata) = false

# IAPv4.2 sentinel fill value: unmasked cells carry values around 1e30
# because the _FillValue attribute is not correctly typed in the NC file.
# Anything above 1e20 is treated as a fill.
const _IAP_FILL_THRESHOLD = 1.0f20

function retrieve_data(m::IAPMetadatum)
    ds = Dataset(metadata_path(m))
    varname = dataset_variable_name(m)
    raw = ds[varname][:, :]   # (lon, lat) in Julia column-major
    close(ds)

    arr = Array{Float32}(undef, size(raw))
    @inbounds for i in eachindex(raw)
        v = raw[i]
        arr[i] = (ismissing(v) || abs(Float32(v)) > _IAP_FILL_THRESHOLD) ?
                 NaN32 : Float32(v)
    end

    # Shift 1..360 longitude to -180..180 (same pattern as ERSST)
    shifted = vcat(arr[181:360, :], arr[1:180, :])
    return reshape(shifted, size(shifted)..., 1)
end

end # module
