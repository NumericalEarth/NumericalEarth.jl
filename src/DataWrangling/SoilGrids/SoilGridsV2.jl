module SoilGrids

export SoilGridsV2

using Downloads: Downloads
using Oceananigans: CPU, Field, Center, location, fill_halo_regions!
using Oceananigans.DistributedComputations: @root
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: DataWrangling, DownloadProgress, AbstractStaticDataset, Metadatum,
    metadata_path, metadata_url, native_grid, retrieve_data, set_metadata_field!

import Oceananigans

@enum SoilGridsStat mean Q5 Q50 Q95

download_SoilGridsV2_cache::String = ""
function __init__()
    return global download_SoilGridsV2_cache = @get_scratch!("SoilGridsV2")
end

struct SoilGridsV2 <: AbstractStaticDataset end

# Variable name mappings from NumericalEarth names to SoilGridsV2 variable names
SoilGridsV2_dataset_variable_names = Dict(
    :sand_content => "sand",
    :silt_content => "silt",
    :clay_content => "clay",
    :bulk_density => "bdod",
    :organic_carbon_density => "ocd",
    :soil_organic_carbon => "soc"
)

Base.size(::SoilGridsV2, variable) = (3956, 1979, 6, 4)

const SoilGridsV2Metadatum = Metadatum{<:SoilGridsV2}

const SoilGridsV2_url = "https://syncandshare.lrz.de/dl/fiVMyHskjL3FNbceuUFJev/soilgrids2_clenshaw989_10km.nc"

# Dataset methods
DataWrangling.available_variables(::SoilGridsV2) = SoilGridsV2_dataset_variable_names
DataWrangling.default_download_directory(::SoilGridsV2) = download_SoilGridsV2_cache
DataWrangling.reversed_vertical_axis(::SoilGridsV2) = true
DataWrangling.reversed_latitude_axis(::SoilGridsV2) = true
DataWrangling.longitude_interfaces(::SoilGridsV2) = (-180, 180)
DataWrangling.latitude_interfaces(::SoilGridsV2) = (-90, 90)
DataWrangling.z_interfaces(::SoilGridsV2) = [-200.0, -100.0, -60.0, -30.0, -15.0, -5.0, 0.0]
DataWrangling.metadata_filename(::SoilGridsV2, name, date, region) = "SoilGridsV2_clenshaw_10km_full.nc"

# Metadatum methods
DataWrangling.is_three_dimensional(::SoilGridsV2Metadatum) = true
DataWrangling.dataset_variable_name(data::SoilGridsV2Metadatum) = SoilGridsV2_dataset_variable_names[data.name]
DataWrangling.metadata_url(::SoilGridsV2Metadatum) = SoilGridsV2_url
DataWrangling.longitude_name(::SoilGridsV2Metadatum) = "lon"
DataWrangling.latitude_name(::SoilGridsV2Metadatum) = "lat"

function Downloads.download(metadatum::SoilGridsV2Metadatum)
    fileurl = metadata_url(metadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading SoilGridsV2 (~10 km) data: $(metadatum.name) in $(metadatum.dir)..."
        Downloads.download(fileurl, filepath; progress = DownloadProgress())
    end
    return filepath
end

Oceananigans.Fields.location(::SoilGridsV2Metadatum) = (Center, Center, Center)

function Oceananigans.Fields.Field(
        metadatum::SoilGridsV2Metadatum, arch = CPU();
        halo = (3, 3, 3),
        stat::SoilGridsStat = mean
    )
    Downloads.download(metadatum)

    grid = native_grid(metadatum, arch; halo)
    LX, LY, LZ = location(metadatum)
    field = Field{LX, LY, LZ}(grid)

    # Retrieve data from file according to metadatum type
    data = retrieve_data(metadatum, Int(stat) + 1)
    # Replace fill value
    data = replace(data, -32768 => 0)

    set_metadata_field!(field, data, metadatum)
    fill_halo_regions!(field)

    return field
end

end # module
