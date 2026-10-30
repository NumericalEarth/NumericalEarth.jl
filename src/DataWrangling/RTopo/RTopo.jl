module RTopo

export RTopo2

using Downloads: Downloads
using NCDatasets: Dataset
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, DownloadProgress, Metadatum, metadata_path, AbstractStaticBathymetry,
                       download_with_retries

import ..DataWrangling:
    metadata_filename,
    default_download_directory,
    dataset_variable_name,
    available_variables,
    longitude_interfaces,
    latitude_interfaces,
    reversed_vertical_axis,
    retrieve_bathymetry

download_RTopo_cache::String = ""
function __init__()
    global download_RTopo_cache = DataWrangling.download_cache("RTopo")
    return nothing
end

RTopo_variable_names = Dict(
    :bottom_height => "bedrock_topography",
    :ice_draft     => "ice_base_topography",
)

"""
    RTopo2

RTopo-2.0.4 global topography at 30 arc-second resolution, including the ice-shelf cavity geometry
of Antarctica and Greenland. Supports `:bottom_height` (bedrock, also under grounded and floating ice)
and `:ice_draft` (ice base height; zero where there is no ice).

Reference: Schaffer et al. (2016), https://doi.org/10.5194/essd-8-543-2016
Data: https://doi.org/10.1594/PANGAEA.905295 (CC-BY-3.0), served from the E3SM data mirror.
"""
struct RTopo2 <: AbstractStaticBathymetry end

const RTopoMetadatum = Metadatum{<:RTopo2}

# The files store nodes at -180:Δ:180 and -90:Δ:90 (Δ = 1/120°). Dropping the repeated 180° column
# and the two pole rows leaves those nodes at the centers of a periodic grid.
const RTopo_Δ = 1 / 120

default_download_directory(::RTopo2) = download_RTopo_cache
reversed_vertical_axis(::RTopo2)     = false
available_variables(::RTopo2)        = RTopo_variable_names
longitude_interfaces(::RTopo2)       = (-180 - RTopo_Δ / 2, 180 - RTopo_Δ / 2)
latitude_interfaces(::RTopo2)        = (-90 + RTopo_Δ / 2, 90 - RTopo_Δ / 2)
Base.size(::RTopo2)                  = (43200, 21599, 1)

dataset_variable_name(metadata::RTopoMetadatum) = RTopo_variable_names[metadata.name]

metadata_filename(::RTopo2, name, date, bounding_box) =
    "RTopo-2.0.4_30sec_" * RTopo_variable_names[name] * ".nc"

const RTopo_url = "https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database/"

DataWrangling.metadata_url(metadata::RTopoMetadatum) = RTopo_url * metadata.filename

function retrieve_bathymetry(metadata::RTopoMetadatum)
    return Dataset(metadata_path(metadata), "r") do dataset
        dataset[dataset_variable_name(metadata)][1:end-1, 2:end-1]
    end
end

function Downloads.download(metadatum::RTopoMetadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading RTopo-2 $(metadatum.name) (~3.7 GB) to $(metadatum.dir)..."
        download_with_retries(DataWrangling.metadata_url(metadatum), filepath; progress=DownloadProgress())
    end

    return filepath
end

end # module
