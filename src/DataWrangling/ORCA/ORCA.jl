module ORCA

export ORCA1, ORCA12

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
abstract type OrcaDataset end

struct ORCA1 <: OrcaDataset end
struct ORCA12 <: OrcaDataset end

default_download_directory(::ORCA1) = download_ORCA_cache
default_download_directory(::ORCA12) = download_ORCA_cache

reversed_vertical_axis(::ORCA1) = false
reversed_vertical_axis(::ORCA12) = false

longitude_interfaces(::ORCA1) = (-180, 180)
longitude_interfaces(::ORCA12) = (-180, 180)

latitude_interfaces(::ORCA1) = (-80, 90)
latitude_interfaces(::ORCA12) = (-80, 90)

all_dates(::ORCA1, args...) = nothing
all_dates(::ORCA12, args...) = nothing

first_date(::ORCA1, args...) = nothing
first_date(::ORCA12, args...) = nothing

last_date(::ORCA1, args...) = nothing
last_date(::ORCA12, args...) = nothing

const ORCA1Metadatum = Metadatum{<:ORCA1}
const ORCA12Metadatum = Metadatum{<:ORCA12}

ORCA1_variable_names = Dict(
    :bottom_height => "Bathymetry",
    :mesh_mask     => "glamt",
)

ORCA12_variable_names = Dict(
    :bottom_height => "Bathymetry",
    :mesh_mask     => "e1t",
)

dataset_variable_name(data::ORCA1Metadatum) = ORCA1_variable_names[data.name]
dataset_variable_name(data::ORCA12Metadatum) = ORCA12_variable_names[data.name]

# Zenodo record 4436658: eORCA1 mesh_mask and bathymetry
const ORCA1_mesh_mask_url  = "https://zenodo.org/records/4436658/files/eORCA1.2_mesh_mask.nc"
const ORCA1_bathymetry_url = "https://zenodo.org/records/4436658/files/eORCA_R1_bathy_meter_v2.2.nc"

# Google Drive records for GLORYS12 (ORCA12) mesh mask and bathymetry
const ORCA12_mesh_mask_url  = "https://zenodo.org/records/15495870/files/grid_mask_eORCA12-GO6.nc"
const ORCA12_bathymetry_url = "https://zenodo.org/records/15495870/files/bathy_eORCA12_noclosea_from_GEBCO2021_FillZero_S21TT_CloseaCopy.nc"

function metadata_url(metadatum::ORCA1Metadatum)
    if metadatum.name == :mesh_mask
        return ORCA1_mesh_mask_url
    elseif metadatum.name == :bottom_height
        return ORCA1_bathymetry_url
    else
        error("Unknown ORCA1 variable: $(metadatum.name)")
    end
end

function metadata_url(metadatum::ORCA12Metadatum)
    if metadatum.name == :mesh_mask
        return ORCA12_mesh_mask_url
    elseif metadatum.name == :bottom_height
        return ORCA12_bathymetry_url
    else
        error("Unknown ORCA12 variable: $(metadatum.name)")
    end
end

function metadata_filename(::ORCA1, name, date, bounding_box)
    if name == :mesh_mask
        return "eORCA1.2_mesh_mask.nc"
    elseif name == :bottom_height
        return "eORCA_R1_bathy_meter_v2.2.nc"
    else
        error("Unknown ORCA1 variable: $name")
    end
end

function metadata_filename(::ORCA12, name, date, bounding_box)
    if name == :mesh_mask
        return "grid_mask_eORCA12-GO6.nc"
    elseif name == :bottom_height
        return "bathy_eORCA12_noclosea_from_GEBCO2021_FillZero_S21TT_CloseaCopy.nc"
    else
        error("Unknown ORCA12 variable: $name")
    end
end

z_interfaces(::ORCA1Metadatum) = nothing
z_interfaces(::ORCA12Metadatum) = nothing

function download_dataset(metadatum::Union{ORCA1Metadatum, ORCA12Metadatum})
    fileurl  = metadata_url(metadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        dataset_name = nameof(typeof(metadatum.dataset))
        @info "Downloading $(dataset_name) data: $(metadatum.name) to $(metadatum.dir)..."
        Downloads.download(fileurl, filepath; progress=download_progress)
    end

    return filepath
end

default_south_rows_to_remove(::ORCA1) = 35
# Conservative default: keep all rows unless user requests trimming.
default_south_rows_to_remove(::ORCA12) = 0

end # module
