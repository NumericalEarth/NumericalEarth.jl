module GEBCO

export GEBCO

using Downloads: Downloads
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, DownloadProgress, Metadatum, metadata_path, AbstractStaticBathymetry

import ..DataWrangling:
    metadata_filename,
    default_download_directory,
    dataset_variable_name,
    longitude_interfaces,
    latitude_interfaces,
    reversed_vertical_axis

download_GEBCO_cache::String = ""
function __init__()
    global download_GEBCO_cache = DataWrangling.download_cache("GEBCO")
end

const GEBCO_bathymetry_variable_names = Dict(:bottom_height => "elevation")

"""
    GEBCO

General Bathymetric Chart of the Oceans (latest release).
Global bathymetry and topography at 15 arc-second resolution.

Data source: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
"""
struct GEBCO <: AbstractStaticBathymetry end

const GEBCOMetadatum = Metadatum{<:GEBCO}

default_download_directory(::GEBCO)   = download_GEBCO_cache
reversed_vertical_axis(::GEBCO)       = false
longitude_interfaces(::GEBCO)         = (-180, 180)
latitude_interfaces(::GEBCO)          = (-90, 90)
Base.size(::GEBCO)                    = (86400, 43200, 1)

dataset_variable_name(::GEBCOMetadatum) = GEBCO_bathymetry_variable_names[:bottom_height]

const GEBCO_2026_nc_url      = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.nc"
const GEBCO_2026_nc_filename = "GEBCO_2026.nc"

DataWrangling.metadata_url(::GEBCOMetadatum)                        = GEBCO_2026_nc_url
metadata_filename(::GEBCO, name, date, bounding_box)                = GEBCO_2026_nc_filename

function Downloads.download(metadatum::GEBCOMetadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading GEBCO to $(metadatum.dir)..."
        @info "Note: GEBCO is a large dataset (~7.5 GB). This may take a while."
        Downloads.download(GEBCO_2026_nc_url, filepath; progress=DownloadProgress())
        @info "GEBCO download complete."
    end

    return filepath
end

end # module
