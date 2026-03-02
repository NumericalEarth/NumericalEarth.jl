module WOA

export WOAClimatology, WOAAnnual, WOAMonthly

using NumericalEarth
using Oceananigans
using NCDatasets
using JLD2
using Scratch
using Adapt
using Dates

using ..DataWrangling:
    Metadata,
    Metadatum,
    BoundingBox,
    inpaint_mask!,
    NearestNeighborInpainting,
    download_progress

using Oceananigans.DistributedComputations: @root

import NumericalEarth.DataWrangling:
    all_dates,
    first_date,
    last_date,
    metadata_filename,
    download_dataset,
    default_download_directory,
    metadata_path,
    dataset_variable_name,
    metaprefix,
    z_interfaces,
    longitude_interfaces,
    latitude_interfaces,
    is_three_dimensional,
    reversed_vertical_axis,
    inpainted_metadata_path,
    available_variables,
    retrieve_data

import Oceananigans.Fields: location

download_WOA_cache::String = ""
function __init__()
    global download_WOA_cache = @get_scratch!("WOA")
end

WOA_variable_names = Dict(
    :temperature      => "t",
    :salinity         => "s",
    :phosphate        => "p",
    :nitrate          => "n",
    :silicate         => "i",
    :dissolved_oxygen => "o",
)

# Dataset types
abstract type WOAClimatology end
struct WOAAnnual <: WOAClimatology end
struct WOAMonthly <: WOAClimatology end

function default_download_directory(::WOAAnnual)
    return mkpath(joinpath(download_WOA_cache, "annual"))
end

function default_download_directory(::WOAMonthly)
    return mkpath(joinpath(download_WOA_cache, "monthly"))
end

# WOA 1° resolution: 360×180 grid, 102 depth levels
Base.size(::WOAClimatology, variable) = (360, 180, 102)

# Annual: single snapshot, no date
all_dates(::WOAAnnual, args...) = nothing
first_date(::WOAAnnual, args...) = nothing
last_date(::WOAAnnual, args...) = nothing

# Monthly: 12 climatological months (year is arbitrary, month matters)
all_dates(::WOAMonthly, args...) = [DateTime(2018, m, 1) for m in 1:12]

# WOA stores depth as positive values, surface first (0 to 5500m)
reversed_vertical_axis(::WOAClimatology) = true

longitude_interfaces(::WOAClimatology) = (-179.5, 180.5)
latitude_interfaces(::WOAClimatology) = (-89.5, 89.5)
available_variables(::WOAClimatology) = WOA_variable_names

# WOA z-interfaces (103 faces for 102 cells).
# Computed as midpoints between consecutive WOA standard depth centers
# (0, 5, 10, ..., 5500 m), converted to negative z, bottom first.
z_interfaces(::WOAClimatology) = [
    -5550.0, -5450.0, -5350.0, -5250.0, -5150.0, -5050.0,
    -4950.0, -4850.0, -4750.0, -4650.0, -4550.0, -4450.0,
    -4350.0, -4250.0, -4150.0, -4050.0, -3950.0, -3850.0,
    -3750.0, -3650.0, -3550.0, -3450.0, -3350.0, -3250.0,
    -3150.0, -3050.0, -2950.0, -2850.0, -2750.0, -2650.0,
    -2550.0, -2450.0, -2350.0, -2250.0, -2150.0, -2050.0,
    -1975.0, -1925.0, -1875.0, -1825.0, -1775.0, -1725.0,
    -1675.0, -1625.0, -1575.0, -1525.0, -1475.0, -1425.0,
    -1375.0, -1325.0, -1275.0, -1225.0, -1175.0, -1125.0,
    -1075.0, -1025.0,  -975.0,  -925.0,  -875.0,  -825.0,
     -775.0,  -725.0,  -675.0,  -625.0,  -575.0,  -525.0,
     -487.5,  -462.5,  -437.5,  -412.5,  -387.5,  -362.5,
     -337.5,  -312.5,  -287.5,  -262.5,  -237.5,  -212.5,
     -187.5,  -162.5,  -137.5,  -112.5,   -97.5,   -92.5,
      -87.5,   -82.5,   -77.5,   -72.5,   -67.5,   -62.5,
      -57.5,   -52.5,   -47.5,   -42.5,   -37.5,   -32.5,
      -27.5,   -22.5,   -17.5,   -12.5,    -7.5,    -2.5,
        0.0,
]

# Type aliases
const WOAMetadata{D} = Metadata{<:WOAClimatology, D}
const WOAMetadatum   = Metadatum{<:WOAClimatology}

metaprefix(::WOAMetadata) = "WOAMetadata"
metaprefix(::WOAMetadatum) = "WOAMetadatum"

# Map from date to WOA period number (used by extension for download)
woa_period(::WOAAnnual, date) = 0
woa_period(::WOAMonthly, date) = Dates.month(date)

function metadata_filename(metadata::Metadatum{<:WOAAnnual})
    varname = WOA_variable_names[metadata.name]
    return "woa_$(varname)_annual.nc"
end

function metadata_filename(metadata::Metadatum{<:WOAMonthly})
    varname = WOA_variable_names[metadata.name]
    m = lpad(Dates.month(metadata.dates), 2, '0')
    return "woa_$(varname)_monthly_$(m).nc"
end

dataset_variable_name(data::WOAMetadata) = string(data.name)
location(::WOAMetadata) = (Center, Center, Center)
is_three_dimensional(::WOAMetadata) = true

function inpainted_metadata_filename(metadata::WOAMetadata)
    original_filename = metadata_filename(metadata)
    without_extension = original_filename[1:end-3]
    var = string(metadata.name)
    return without_extension * "_" * var * "_inpainted.jld2"
end

inpainted_metadata_path(metadata::WOAMetadata) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

# Custom retrieve_data: WOA NetCDF files have no time dimension (3D only)
function retrieve_data(metadata::Metadatum{<:WOAClimatology})
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)

    ds = Dataset(path)
    data = ds[name][:, :, :]
    close(ds)

    # WOA stores depth surface-first; reverse to match grid (bottom-first)
    if reversed_vertical_axis(metadata.dataset)
        data = reverse(data, dims=3)
    end

    return data
end

end # module
