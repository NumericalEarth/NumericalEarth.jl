module ORCA

export ORCAOne, ORCAQuarter, ORCATwelfth

using Downloads: Downloads
using Oceananigans: Oceananigans
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, DownloadProgress, atomic_download, Metadatum, metadata_path, metadata_url

import ..DataWrangling:
    metadata_filename,
    default_download_directory,
    all_dates,
    first_date,
    last_date,
    dataset_variable_name,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    reversed_vertical_axis

download_ORCA_cache::String = ""

function __init__()
    global download_ORCA_cache = DataWrangling.download_cache("ORCA")
end

abstract type ORCADataset end

# Names spell out the nominal horizontal resolution: 1°, 1/4° (eORCA025), and 1/12° (eORCA12).
struct ORCAOne <: ORCADataset end
struct ORCAQuarter <: ORCADataset end
struct ORCATwelfth <: ORCADataset end

default_download_directory(::ORCADataset) = download_ORCA_cache
reversed_vertical_axis(::ORCADataset) = false
longitude_interfaces(::ORCADataset) = (-180, 180)
latitude_interfaces(::ORCADataset) = (-80, 90)
all_dates(::ORCADataset, args...) = nothing
first_date(::ORCADataset, args...) = nothing
last_date(::ORCADataset, args...) = nothing

const ORCAOneMetadatum = Metadatum{<:ORCAOne}
const ORCAQuarterMetadatum = Metadatum{<:ORCAQuarter}
const ORCATwelfthMetadatum = Metadatum{<:ORCATwelfth}
const ORCAMetadatum = Metadatum{<:ORCADataset}

const ORCA_variable_names = Dict(
    :bottom_height => "Bathymetry",
    :mesh_mask     => "glamt",
)

const ORCATwelfth_variable_names = Dict(
    :bottom_height => "Bathymetry",
    :mesh_mask     => "e1t",
)

dataset_variable_name(data::ORCAMetadatum)         = ORCA_variable_names[data.name]
dataset_variable_name(data::ORCATwelfthMetadatum)  = ORCATwelfth_variable_names[data.name]

# Zenodo record 4436658: eORCA1 mesh_mask and bathymetry
mesh_mask_url(::ORCAOne)  = "https://zenodo.org/records/4436658/files/eORCA1.2_mesh_mask.nc"
bathymetry_url(::ORCAOne) = "https://zenodo.org/records/4436658/files/eORCA_R1_bathy_meter_v2.2.nc"

# Zenodo record 15494369: eORCA025 mesh_mask and bathymetry
mesh_mask_url(::ORCAQuarter)  = "https://zenodo.org/records/15494369/files/grid_mask_eORCA025-GO6.nc"
bathymetry_url(::ORCAQuarter) = "https://zenodo.org/records/15494369/files/bathy_eORCA025_noclosea_from_GEBCO2021_S21TT_CloseaCopy_edit.nc"

# Zenodo record 15495870: eORCA12 mesh_mask and bathymetry
mesh_mask_url(::ORCATwelfth)  = "https://zenodo.org/records/15495870/files/grid_mask_eORCA12-GO6.nc"
bathymetry_url(::ORCATwelfth) = "https://zenodo.org/records/15495870/files/bathy_eORCA12_noclosea_from_GEBCO2021_FillZero_S21TT_CloseaCopy.nc"

function dataset_url(dataset::ORCADataset, name)
    if name == :mesh_mask
        return mesh_mask_url(dataset)
    elseif name == :bottom_height
        return bathymetry_url(dataset)
    else
        error("Unknown $(dataset) variable: $(name)")
    end
end

DataWrangling.metadata_url(metadatum::ORCAMetadatum) = dataset_url(metadatum.dataset, metadatum.name)

metadata_filename(dataset::ORCADataset, name, date, region) = basename(dataset_url(dataset, name))

z_interfaces(::ORCAMetadatum) = nothing

function Downloads.download(metadatum::ORCAMetadatum)
    fileurl  = metadata_url(metadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        dataset_name = nameof(typeof(metadatum.dataset))
        @info "Downloading $(dataset_name) data: $(metadatum.name) to $(metadatum.dir)..."
        atomic_download(fileurl, filepath; progress=DownloadProgress())
    end

    return filepath
end

default_south_rows_to_remove(::ORCAOne)     = 35
default_south_rows_to_remove(::ORCAQuarter) = 155
default_south_rows_to_remove(::ORCATwelfth) = 460

end # module
