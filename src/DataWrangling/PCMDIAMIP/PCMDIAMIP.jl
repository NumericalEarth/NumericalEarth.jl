module PCMDIAMIP

export PCMDIAMIPBoundaryConditions, PCMDIAMIPObservations

using Dates: Dates, DateTime, Month
using DocStringExtensions: TYPEDEF
using Downloads: Downloads
using Oceananigans.DistributedComputations: @root
using Oceananigans.Fields: Center
using NCDatasets: Dataset

using ..DataWrangling: DataWrangling, Metadata, Metadatum, DownloadProgress, Celsius, Percent,
                       dataset_variable_name, metadata_path, metadata_url, download_with_retries

download_PCMDIAMIP_cache::String = ""
function __init__()
    global download_PCMDIAMIP_cache = DataWrangling.download_cache("PCMDIAMIP")
    return nothing
end

abstract type PCMDIAMIPDataset end

"""
$(TYPEDEF)

A release of the PCMDI-AMIP product on input4MIPs, located by its version in the product name
(`"1-1-10"`), the `mip_era` it is published under (`"CMIP7"`), its dataset version (`"v20250807"`),
and the last month it holds.
"""
struct PCMDIAMIPRelease
    source_version :: String
    mip_era :: String
    dataset_version :: String
    last_date :: DateTime
end

const PCMDIAMIP_default_release = PCMDIAMIPRelease("1-1-10", "CMIP7", "v20250807", DateTime(2022, 12, 1))

"""
$(TYPEDEF)

The sea surface temperature and sea ice concentration boundary conditions of AMIP simulations, from
the input4MIPs PCMDI-AMIP product (`tosbcs`, `siconcbcs`): one value per month from January 1870 on a
global 1° grid. The keyword argument `release` selects a [`PCMDIAMIPRelease`](@ref); the default is
PCMDI-AMIP 1.1.10, which ends in December 2022.

`:sea_area_fraction` (`sftof`) is the fraction of each cell of that grid that is sea, the same at every
date: the land-sea mask the two series were built with.

The values are the mid-month nodes of [Taylor et al. (2000)](@cite taylor2000sea), constructed so that
the monthly mean of the series interpolated linearly in time between them is the observed monthly
mean of [`PCMDIAMIPObservations`](@ref). They are therefore not themselves observations and overshoot
them, so that the sea ice concentration leaves `[0, 1]` and the sea surface temperature falls below
freezing: the bounds are to be imposed after interpolating in time.
"""
struct PCMDIAMIPBoundaryConditions <: PCMDIAMIPDataset
    release :: PCMDIAMIPRelease
end

"""
$(TYPEDEF)

The observed monthly mean sea surface temperature and sea ice concentration of the input4MIPs
PCMDI-AMIP product (`tos`, `siconc`), merged from HadISST and NCEP OI2, from January 1870 on a global
1° grid. The keyword argument `release` selects a [`PCMDIAMIPRelease`](@ref); the default is
PCMDI-AMIP 1.1.10, which ends in December 2022.
"""
struct PCMDIAMIPObservations <: PCMDIAMIPDataset
    release :: PCMDIAMIPRelease
end

PCMDIAMIPBoundaryConditions(; release=PCMDIAMIP_default_release) = PCMDIAMIPBoundaryConditions(release)
PCMDIAMIPObservations(; release=PCMDIAMIP_default_release) = PCMDIAMIPObservations(release)

const PCMDIAMIPMetadata{D} = Metadata{<:PCMDIAMIPDataset, D}
const PCMDIAMIPMetadatum   = Metadatum{<:PCMDIAMIPDataset}

const PCMDIAMIP_boundary_condition_variable_names = Dict(
    :sea_surface_temperature => "tosbcs",
    :sea_ice_concentration   => "siconcbcs",
    :sea_area_fraction       => "sftof",
)

const PCMDIAMIP_observation_variable_names = Dict(
    :sea_surface_temperature => "tos",
    :sea_ice_concentration   => "siconc",
    :sea_area_fraction       => "sftof",
)

DataWrangling.available_variables(::PCMDIAMIPBoundaryConditions) = PCMDIAMIP_boundary_condition_variable_names
DataWrangling.available_variables(::PCMDIAMIPObservations) = PCMDIAMIP_observation_variable_names
DataWrangling.dataset_variable_name(metadata::PCMDIAMIPMetadata) =
    DataWrangling.available_variables(metadata.dataset)[metadata.name]

DataWrangling.default_download_directory(::PCMDIAMIPDataset) = download_PCMDIAMIP_cache

Base.size(::PCMDIAMIPDataset, variable) = (360, 180, 1)
DataWrangling.longitude_interfaces(::PCMDIAMIPDataset) = (0, 360)
DataWrangling.latitude_interfaces(::PCMDIAMIPDataset) = (-90, 90)
DataWrangling.longitude_name(::PCMDIAMIPMetadata) = "lon"
DataWrangling.latitude_name(::PCMDIAMIPMetadata) = "lat"
DataWrangling.dataset_location(::PCMDIAMIPDataset, name) = (Center, Center, Nothing)
DataWrangling.is_three_dimensional(::PCMDIAMIPMetadata) = false
DataWrangling.default_inpainting(::PCMDIAMIPMetadata) = nothing

DataWrangling.all_dates(dataset::PCMDIAMIPDataset, variable) = DateTime(1870, 1, 1) : Month(1) : dataset.release.last_date

DataWrangling.averaging_window(metadatum::PCMDIAMIPMetadatum) = DataWrangling.calendar_month_window(metadatum)

function DataWrangling.conversion_units(metadatum::PCMDIAMIPMetadatum)
    if metadatum.name == :sea_surface_temperature
        return Celsius()
    else
        return Percent()
    end
end

# The sea area fraction of the grid (`sftof`) is a fixed field of the release: one record, whatever the date
is_fixed_field(name) = name == :sea_area_fraction

# One file per variable holds every month.
function DataWrangling.metadata_filename(dataset::PCMDIAMIPDataset, name, date, region)
    release = dataset.release
    is_fixed_field(name) &&
        return string("sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-", release.source_version, "_gn.nc")

    return string(DataWrangling.available_variables(dataset)[name], "_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-",
                  release.source_version, "_gn_187001-", Dates.format(release.last_date, "yyyymm"), ".nc")
end

DataWrangling.build_filename(dataset::PCMDIAMIPDataset, name, dates::AbstractArray, region) =
    DataWrangling.metadata_filename(dataset, name, first(dates), region)

const PCMDIAMIP_url = "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/input4MIPs"

function DataWrangling.metadata_url(metadata::PCMDIAMIPMetadata)
    release = metadata.dataset.release
    realm = metadata.name == :sea_ice_concentration ? "seaIce" : "ocean"
    frequency = is_fixed_field(metadata.name) ? "fx" : "mon"
    return join((PCMDIAMIP_url, release.mip_era, "CMIP/PCMDI", "PCMDI-AMIP-" * release.source_version, realm, frequency,
                 dataset_variable_name(metadata), "gn", release.dataset_version, metadata.filename), "/")
end

function Downloads.download(metadata::PCMDIAMIPMetadata)
    filepath = metadata_path(metadata)

    @root if !isfile(filepath)
        download_with_retries(metadata_url(metadata), filepath; progress=DownloadProgress())
    end

    return filepath
end

function DataWrangling.retrieve_data(metadatum::PCMDIAMIPMetadatum)
    date = metadatum.dates
    record = 12 * (Dates.year(date) - 1870) + Dates.month(date)

    return Dataset(metadata_path(metadatum)) do ds
        variable = ds[dataset_variable_name(metadatum)]
        is_fixed_field(metadatum.name) ? variable[:, :] : variable[:, :, record]
    end
end

end # module
