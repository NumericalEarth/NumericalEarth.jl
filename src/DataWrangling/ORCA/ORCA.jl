module ORCA

export ORCA1

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
    reversed_vertical_axis

download_ORCA_cache::String = ""
function __init__()
    global download_ORCA_cache = @get_scratch!("ORCA")
end

struct ORCA1 end

default_download_directory(::ORCA1) = download_ORCA_cache
reversed_vertical_axis(::ORCA1) = false
longitude_interfaces(::ORCA1) = (-180, 180)
latitude_interfaces(::ORCA1) = (-80, 90)

all_dates(::ORCA1, args...) = nothing
first_date(::ORCA1, args...) = nothing
last_date(::ORCA1, args...) = nothing

const ORCA1Metadatum = Metadatum{<:ORCA1}

ORCA1_variable_names = Dict(
    :bottom_height => "Bathymetry",
    :mesh_mask     => "glamt",
)

dataset_variable_name(data::ORCA1Metadatum) = ORCA1_variable_names[data.name]

# Zenodo record 4436658: eORCA1 mesh_mask and bathymetry
const ORCA1_mesh_mask_url   = "https://zenodo.org/records/4436658/files/eORCA1.2_mesh_mask.nc"
const ORCA1_bathymetry_url  = "https://zenodo.org/records/4436658/files/eORCA_R1_bathy_meter_v2.2.nc"

function metadata_url(metadatum::ORCA1Metadatum)
    if metadatum.name == :mesh_mask
        return ORCA1_mesh_mask_url
    elseif metadatum.name == :bottom_height
        return ORCA1_bathymetry_url
    else
        error("Unknown ORCA1 variable: $(metadatum.name)")
    end
end

function metadata_filename(metadatum::ORCA1Metadatum)
    if metadatum.name == :mesh_mask
        return "eORCA1.2_mesh_mask.nc"
    elseif metadatum.name == :bottom_height
        return "eORCA_R1_bathy_meter_v2.2.nc"
    else
        error("Unknown ORCA1 variable: $(metadatum.name)")
    end
end

z_interfaces(::ORCA1Metadatum) = nothing

function download_dataset(metadatum::ORCA1Metadatum)
    fileurl  = metadata_url(metadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading ORCA1 data: $(metadatum.name) to $(metadatum.dir)..."
        Downloads.download(fileurl, filepath; progress=download_progress)
    end

    return filepath
end

end # module
