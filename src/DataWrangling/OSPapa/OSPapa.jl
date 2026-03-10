module OSPapa

export OSPapaPrescribedAtmosphere
export OSPapaHourly

using Oceananigans
using NCDatasets
using Dates
using Scratch
using Downloads

using NumericalEarth.DataWrangling: download_progress
using NumericalEarth.Atmospheres: PrescribedAtmosphere, TwoBandDownwellingRadiation

using NumericalEarth.DataWrangling:
    Metadata,
    Metadatum,
    metadata_path,
    NearestNeighborInpainting,
    DatasetRestoring,
    Temperature,
    Salinity

import NumericalEarth.DataWrangling:
    default_download_directory,
    all_dates,
    metadata_filename,
    download_dataset,
    dataset_variable_name,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    is_three_dimensional,
    inpainted_metadata_path,
    available_variables,
    retrieve_data,
    native_grid,
    default_inpainting,
    conversion_units,
    metaprefix

const OSPAPA_S3_URL  = "https://noaa-oar-keo-papa-pds.s3.amazonaws.com/PAPA/"
const OSPAPA_FILENAME = "OS_PAPA_200706_M_TSVMBP_50N145W_hr.nc"
const OSPAPA_LONGITUDE = -144.9
const OSPAPA_LATITUDE  = 50.1

download_OSPapa_cache::String = ""

function __init__()
    global download_OSPapa_cache = @get_scratch!("OSPapa")
end

function download_ospapa_file(dir=download_OSPapa_cache)
    filepath = joinpath(dir, OSPAPA_FILENAME)
    if !isfile(filepath)
        url = OSPAPA_S3_URL * OSPAPA_FILENAME
        @info "Downloading Ocean Station Papa data from AWS S3..."
        Downloads.download(url, filepath; progress=download_progress)
    end
    return filepath
end

include("OSPapa_prescribed_atmosphere.jl")
include("OSPapa_ocean_observations.jl")

end # module
