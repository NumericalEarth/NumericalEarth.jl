module IBCAO

export IBCAOv5

using Downloads
using Oceananigans
using Oceananigans.DistributedComputations: @root
using Scratch
using NCDatasets

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


download_IBCAO_cache::String = ""
function __init__()
    global download_IBCAO_cache = @get_scratch!("IBCAO")
end

IBCAO_bathymetry_variable_names = Dict(
    :bottom_height => "z",
)

"""
    IBCAOv5

International Bathymetric Chart of the Arctic Ocean Version 5.1 (2025).
100m resolution bathymetry for the Arctic Ocean (north of 64°N), including
Greenland ice sheet surface elevation. Data is provided in Polar Stereographic
projection (EPSG:3996) and reprojected to WGS84 geographic coordinates at 0.01°
resolution on first use.

Reference: Jakobsson et al. (2024), https://doi.org/10.1038/s41597-024-04278-w
Data source: https://www.gebco.net/data-products/gridded-bathymetry-data/arctic-ocean/
"""
struct IBCAOv5 end

default_download_directory(::IBCAOv5) = download_IBCAO_cache
reversed_vertical_axis(::IBCAOv5) = false

# Geographic bounds after reprojection to WGS84
longitude_interfaces(::IBCAOv5) = (-180, 180)
latitude_interfaces(::IBCAOv5) = (64, 90)

# 0.01° resolution: 36000 × 2600
Base.size(::IBCAOv5) = (36000, 2600, 1)
Base.size(dataset::IBCAOv5, variable) = size(dataset)

all_dates(::IBCAOv5, args...) = nothing
first_date(::IBCAOv5, args...) = nothing
last_date(::IBCAOv5, args...) = nothing

# Bathymetry is 2D; z_interfaces is a placeholder
z_interfaces(::IBCAOv5) = (0, 1)

const IBCAOMetadatum = Metadatum{<:IBCAOv5}

dataset_variable_name(data::IBCAOMetadatum) = IBCAO_bathymetry_variable_names[data.name]

# CEDA BODC direct download — 100m, with Greenland ice sheet elevation (~25 GB)
const IBCAO_tiff_url = "https://dap.ceda.ac.uk/bodc/gebco/ibcao/ibcao_v5.1/" *
    "greenland_ice_sheet_elevation_data/100mx100m_grid_cell_spacing/" *
    "single_complete_bathymetric_grid/ibcao_5_1_2025_ice_100m.tiff?download=1"

const IBCAO_tiff_filename = "ibcao_5_1_2025_ice_100m.tiff"
const IBCAO_nc_filename   = "ibcao_v5_wgs84_0p01deg.nc"

metadata_filename(::IBCAOv5, name, date, bounding_box) = IBCAO_nc_filename

function validate_dataset_coverage(grid, ::IBCAOMetadatum)
    φ_south, _ = Oceananigans.Grids.y_domain(grid)
    if φ_south < 64
        error("IBCAOv5 only covers the Arctic Ocean (north of 64°N). " *
              "The grid extends to $(round(φ_south, digits=1))°N. " *
              "Use ETOPO2022() or GEBCO2024() for domains that extend south of 64°N.")
    end
end

function download_dataset(metadatum::IBCAOMetadatum)
    nc_path   = metadata_path(metadatum)
    tiff_path = joinpath(metadatum.dir, IBCAO_tiff_filename)

    @root if !isfile(nc_path)
        if !isfile(tiff_path)
            @info "Downloading IBCAO V5.1 GeoTIFF (100m, with Greenland ice, ~25 GB)..."
            Downloads.download(IBCAO_tiff_url, tiff_path; progress=download_progress)
        end

        @info "Reprojecting IBCAO from Polar Stereographic (EPSG:3996) to WGS84 at 0.01°..."
        reproject_ibcao_to_netcdf(tiff_path, nc_path)
        @info "Reprojection complete. Removing raw GeoTIFF to save disk space..."
        rm(tiff_path; force=true)
    end

    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl when ArchGDAL is loaded.
function _reproject_ibcao_to_netcdf end

end # module
