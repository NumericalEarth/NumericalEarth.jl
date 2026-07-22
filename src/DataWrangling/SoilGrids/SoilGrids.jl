module SoilGrids

export SoilGrids2

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling,
    Dataset, DownloadProgress, atomic_download, AbstractStaticDataset, Metadatum,
    GramPerKilogram, CentigramPerCubicCentimeter, HectogramPerCubicMeter, DecigramPerKilogram,
    metadata_path, metadata_url, dataset_variable_name

import Oceananigans

"""
    Statistic

Enum corresponding to the various prediction layers provided by the SoilGrids 2.0 dataset.
Since SoilGrids was produced using quantile regression, multiple output statistics are available
that capture uncertainty in estimates for each variable. The default is `Mean` which represents the
most likely value. The 5th (`Q5`), 50th (`Q50`), and 95% (`Q90`) percentiles/quantiles represent the
lower, middle, and upper bound of the predictions for each variable. These could be used in sensitivity
analyses to check how sensitive model simulations are to uncertainty in soil properties. Note that
the spread of the distribution should be expected to be larger in data-sparse regions such as
remote islands, deserts, and the Arctic.
"""
@enum Statistic Mean Q5 Q50 Q95

download_SoilGrids2_cache::String = ""
function __init__()
    return global download_SoilGrids2_cache = DataWrangling.download_cache("SoilGrids2")
end

@kwdef struct SoilGrids2 <: AbstractStaticDataset
    "Specifies which dataset layer to load variables from; see [SoilGrids.Statistic](@ref). Defaults to `Mean`"
    statistic::Statistic = Mean
end

# Variable name mappings from NumericalEarth names to SoilGrids2 variable names
SoilGrids2_dataset_variable_names = Dict(
    :sand_fraction           => "sand",
    :silt_fraction           => "silt",
    :clay_fraction           => "clay",
    :coarse_fraction         => "cfvo",
    :bulk_density            => "bdod",
    :organic_carbon_density  => "ocd",
    :soil_organic_carbon     => "soc"
)

Base.size(::SoilGrids2, variable) = (3956, 1979, 6, 4)

const SoilGrids2Metadatum = Metadatum{<:SoilGrids2}

const SoilGrids2_url = "https://syncandshare.lrz.de/dl/fiVMyHskjL3FNbceuUFJev/soilgrids2_clenshaw989_10km.nc"

# Dataset methods
DataWrangling.available_variables(::SoilGrids2) = SoilGrids2_dataset_variable_names
DataWrangling.default_download_directory(::SoilGrids2) = download_SoilGrids2_cache
DataWrangling.reversed_vertical_axis(::SoilGrids2) = true
DataWrangling.reversed_latitude_axis(::SoilGrids2) = true
DataWrangling.longitude_interfaces(::SoilGrids2) = (0, 360)
DataWrangling.latitude_interfaces(::SoilGrids2) = (-90, 90)
DataWrangling.z_interfaces(::SoilGrids2) = [-200, -100, -60, -30, -15, -5, 0]
DataWrangling.metadata_filename(::SoilGrids2, name, date, region) = "SoilGrids2_clenshaw_10km_full.nc"

# Metadatum methods
DataWrangling.is_three_dimensional(::SoilGrids2Metadatum) = true
DataWrangling.dataset_variable_name(data::SoilGrids2Metadatum) = SoilGrids2_dataset_variable_names[data.name]
DataWrangling.metadata_url(::SoilGrids2Metadatum) = SoilGrids2_url
DataWrangling.longitude_name(::SoilGrids2Metadatum) = "lon"
DataWrangling.latitude_name(::SoilGrids2Metadatum) = "lat"
DataWrangling.default_inpainting(md::SoilGrids2Metadatum) = nothing
DataWrangling.missing_value(md::SoilGrids2Metadatum) = -32768

# Unit conversions
function DataWrangling.conversion_units(metadatum::SoilGrids2Metadatum)
    if metadatum.name ∈ (:sand_fraction, :silt_fraction, :clay_fraction, :coarse_fraction)
        # Note that coarse_fraction is technically in cm³/dm³, but the conversion factor is the same, so we ignore that here
        return GramPerKilogram()
    elseif metadatum.name == :bulk_density
        return CentigramPerCubicCentimeter()
    elseif metadatum.name == :organic_carbon_density
        return HectogramPerCubicMeter()
    elseif metadatum.name == :soil_organic_carbon
        return DecigramPerKilogram()
    else
        return nothing
    end
end

function Downloads.download(metadatum::SoilGrids2Metadatum)
    fileurl = metadata_url(metadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading SoilGrids2 (~10 km) data: $(metadatum.name) in $(metadatum.dir)..."
        atomic_download(fileurl, filepath; progress = DownloadProgress())
    end
    return filepath
end

function DataWrangling.retrieve_data(metadata::SoilGrids2Metadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)

    # Open NetCDF file
    data = Dataset(path) do ds
        l = Int(metadata.dataset.statistic) + 1
        data = ds[name][:, :, :, l]
        # Reverse vertical axis to be increasing upwards
        reverse(data, dims = 3)
    end

    # Reverse latitude axis
    data = reverse(data, dims = 2)
    return data
end

Oceananigans.Fields.location(::SoilGrids2Metadatum) = (Center, Center, Center)

end # module
