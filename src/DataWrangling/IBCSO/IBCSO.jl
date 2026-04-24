module IBCSO

export IBCSOv2

using Downloads
using Oceananigans
using Oceananigans.DistributedComputations: @root
using Scratch

using ..DataWrangling: download_progress, Metadatum, metadata_path

import NumericalEarth.DataWrangling:
    metadata_filename,
    default_download_directory,
    all_dates,
    first_date,
    last_date,
    dataset_variable_name,
    download_dataset,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    reversed_vertical_axis,
    validate_dataset_coverage


download_IBCSO_cache::String = ""
function __init__()
    global download_IBCSO_cache = @get_scratch!("IBCSO")
end

IBCSO_bathymetry_variable_names = Dict(
    :bottom_height => "z",  # Variable name in NetCDF
)

"""
    IBCSOv2

International Bathymetric Chart of the Southern Ocean Version 2 (2024 Annual Release).
High-resolution (500m) bathymetry for the Southern Ocean (south of 50°S).

Reference: Dorschel et al. (2022), https://doi.org/10.1594/PANGAEA.937574
Data source: https://ibcso.org/ibcso-2024-annual-release/
"""
struct IBCSOv2 end

default_download_directory(::IBCSOv2) = download_IBCSO_cache
reversed_vertical_axis(::IBCSOv2) = false

# WGS84 version covers -180 to 180 longitude, -90 to -50 latitude
longitude_interfaces(::IBCSOv2) = (-180, 180)
latitude_interfaces(::IBCSOv2) = (-90, -50)

# Grid size for WGS84 version (500m resolution)
# lon: 33812, lat: 3757 (from -180 to 180, -90 to -50)
Base.size(::IBCSOv2) = (33812, 3757, 1)
Base.size(dataset::IBCSOv2, variable) = size(dataset)

all_dates(::IBCSOv2, args...) = nothing
first_date(::IBCSOv2, args...) = nothing
last_date(::IBCSOv2, args...) = nothing

# Bathymetry is 2D, so z_interfaces is just a placeholder
z_interfaces(::IBCSOv2) = (0, 1)

const IBCSOMetadatum = Metadatum{<:IBCSOv2}

dataset_variable_name(data::IBCSOMetadatum) = IBCSO_bathymetry_variable_names[data.name]

const IBCSO_pangaea_url = "https://download.pangaea.de/dataset/937574/files/IBCSO_v2_bed_WGS84.nc"

metadata_url(::IBCSOMetadatum) = IBCSO_pangaea_url

# The expected NetCDF filename inside the ZIP or from PANGAEA
const IBCSO_nc_filename = "IBCSO_v2_bed_WGS84.nc"
metadata_filename(::IBCSOv2, name, date, bounding_box) = IBCSO_nc_filename

function validate_dataset_coverage(grid, ::IBCSOMetadatum)
    _, φ_north = Oceananigans.Grids.y_domain(grid)
    if φ_north > -50
        error("IBCSOv2 only covers the Southern Ocean (south of 50°S). " *
              "The grid extends to $(round(φ_north, digits=1))°. " *
              "Use ETOPO2022() or GEBCO2024() for domains that include latitudes north of 50°S.")
    end
end

function download_dataset(metadatum::IBCSOMetadatum)
    filepath = metadata_path(metadatum)
    download_dir = metadatum.dir

    @root if !isfile(filepath)
        @info "Downloading IBCSO data: $(metadatum.name) to $download_dir..."
        Downloads.download(IBCSO_pangaea_url, filepath; progress=download_progress)
    end

    return filepath
end

end # module
