module ECCO

export ECCOMetadatum, adjusted_ECCO_tracers, initialize!
export ECCO2Monthly, ECCO4Monthly, ECCO2Daily
export ECCOPrescribedAtmosphere, ECCOPrescribedRadiation

export ECCO2DarwinMonthly, ECCO4DarwinMonthly
export retrieve_data

import Oceananigans
import NumericalEarth
using NCDatasets: NCDatasets
using Dates: Dates, DateTime, Day, Month
using Adapt: Adapt
using Scratch: Scratch, @get_scratch!
import Downloads

using Oceananigans.DistributedComputations: @root
using Oceananigans.Architectures: CPU

using NumericalEarth.DataWrangling:
    netrc_downloader,
    NearestNeighborInpainting,
    Column,
    metadata_path,
    GramPerKilogramMinus35,
    MicromolePerLiter,
    Metadata,
    Metadatum,
    download_progress,
    native_grid,
    location,
    extract_column!,
    first_date,
    last_date,
    default_download_directory

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field
using Oceananigans.OutputReaders: FieldTimeSeries, OutputReaders, Cyclical
using JLD2: JLD2
using Glob: Glob
using MeshArrays: MeshArrays, GridLoad, GridLoadVar, GridSpec, interpolation_setup, land_mask

download_ECCO_cache::String = ""
function __init__()
    global download_ECCO_cache = @get_scratch!("ECCO")
end

# Datasets
abstract type ECCODataset end
struct ECCO2Monthly <:ECCODataset end
struct ECCO2Daily   <:ECCODataset end
struct ECCO4Monthly <:ECCODataset end

include("ECCO_darwin.jl")

function NumericalEarth.DataWrangling.default_download_directory(::ECCO2Monthly)
    path = joinpath(download_ECCO_cache, "v2", "monthly")
    return mkpath(path)
end

function NumericalEarth.DataWrangling.default_download_directory(::ECCO2Daily)
    path = joinpath(download_ECCO_cache, "v2", "daily")
    return mkpath(path)
end

function NumericalEarth.DataWrangling.default_download_directory(::ECCO4Monthly)
    path = joinpath(download_ECCO_cache, "v4")
    return mkpath(path)
end

Base.size(::ECCO2Daily, variable)   = (1440, 720, 50)
Base.size(::ECCO2Monthly, variable) = (1440, 720, 50)
Base.size(::ECCO4Monthly, variable) = (720,  360, 50)

NumericalEarth.DataWrangling.default_mask_value(::ECCO4Monthly) = 0
NumericalEarth.DataWrangling.reversed_vertical_axis(::ECCODataset) = true

const ECCO2_url = "https://ecco.jpl.nasa.gov/drive/files/ECCO2/cube92_latlon_quart_90S90N/"
const ECCO4_url = "https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/interp_monthly/"

# The whole range of dates in the different dataset datasets
metadata_epoch(::ECCODataset) = DateTime(1992, 1, 1)

NumericalEarth.DataWrangling.all_dates(dataset::ECCODataset) =
    NumericalEarth.DataWrangling.all_dates(dataset, nothing)
NumericalEarth.DataWrangling.all_dates(dataset::ECCO4Monthly, variable) = metadata_epoch(dataset) : Month(1) : DateTime(2017, 12, 1)
NumericalEarth.DataWrangling.all_dates(dataset::ECCO2Monthly, variable) = metadata_epoch(dataset) : Month(1) : DateTime(2024, 12, 1)
NumericalEarth.DataWrangling.all_dates(dataset::ECCO2Daily,   variable) = metadata_epoch(dataset) : Day(1)   : DateTime(2024, 12, 31)

NumericalEarth.DataWrangling.longitude_interfaces(::ECCODataset) = (0, 360)
NumericalEarth.DataWrangling.longitude_interfaces(::ECCO4Monthly) = (-180, 180)
NumericalEarth.DataWrangling.latitude_interfaces(::ECCODataset) = (-90, 90)

NumericalEarth.DataWrangling.z_interfaces(::ECCODataset) = [
    -6128.75,
    -5683.75,
    -5250.25,
    -4839.75,
    -4452.25,
    -4087.75,
    -3746.25,
    -3427.75,
    -3132.25,
    -2859.75,
    -2610.25,
    -2383.74,
    -2180.13,
    -1999.09,
    -1839.64,
    -1699.66,
    -1575.64,
    -1463.12,
    -1357.68,
    -1255.87,
    -1155.72,
    -1056.53,
    -958.45,
    -862.10,
    -768.43,
    -678.57,
    -593.72,
    -515.09,
    -443.70,
    -380.30,
    -325.30,
    -278.70,
    -240.09,
    -208.72,
    -183.57,
    -163.43,
    -147.11,
    -133.45,
    -121.51,
    -110.59,
    -100.20,
    -90.06,
    -80.01,
    -70.0,
    -60.0,
    -50.0,
    -40.0,
    -30.0,
    -20.0,
    -10.0,
      0.0,
]

NumericalEarth.DataWrangling.available_variables(::ECCO2Monthly) = ECCO2_dataset_variable_names
NumericalEarth.DataWrangling.available_variables(::ECCO2Daily)   = ECCO2_dataset_variable_names
NumericalEarth.DataWrangling.available_variables(::ECCO4Monthly) = ECCO4_dataset_variable_names

ECCO4_dataset_variable_names = Dict(
    :temperature            => "THETA",
    :salinity               => "SALT",
    :u_velocity             => "EVEL",
    :v_velocity             => "NVEL",
    :free_surface           => "SSH",
    :sea_ice_thickness      => "SIheff",
    :sea_ice_concentration  => "SIarea",
    :net_heat_flux          => "oceQnet",
    :sensible_heat_flux     => "EXFhs",
    :latent_heat_flux       => "EXFhl",
    :net_longwave           => "EXFlwnet",
    :downwelling_shortwave  => "oceQsw",
    :downwelling_longwave   => "EXFlwdn",
    :air_temperature        => "EXFatemp",
    :air_specific_humidity  => "EXFaqh",
    :sea_level_pressure     => "EXFpress",
    :eastward_wind          => "EXFewind",
    :northward_wind         => "EXFnwind",
    :rain_freshwater_flux   => "EXFpreci",
    :zonal_wind_stress      => "EXFtaue",
    :meridional_wind_stress => "EXFtaun",
)

ECCO2_dataset_variable_names = Dict(
    :temperature           => "THETA",
    :salinity              => "SALT",
    :u_velocity            => "UVEL",
    :v_velocity            => "VVEL",
    :free_surface          => "SSH",
    :sea_ice_thickness     => "SIheff",
    :sea_ice_concentration => "SIarea",
    :net_heat_flux         => "oceQnet",
)

ECCO_location = Dict(
    :temperature            => (Center, Center, Center),
    :salinity               => (Center, Center, Center),
    :u_velocity             => (Face,   Center, Center),
    :v_velocity             => (Center, Face,   Center),
    :free_surface           => (Center, Center, Nothing),
    :sea_ice_thickness      => (Center, Center, Nothing),
    :sea_ice_concentration  => (Center, Center, Nothing),
    :net_heat_flux          => (Center, Center, Nothing),
    :sensible_heat_flux     => (Center, Center, Nothing),
    :latent_heat_flux       => (Center, Center, Nothing),
    :net_longwave           => (Center, Center, Nothing),
    :downwelling_longwave   => (Center, Center, Nothing),
    :downwelling_shortwave  => (Center, Center, Nothing),
    :air_temperature        => (Center, Center, Nothing),
    :air_specific_humidity  => (Center, Center, Nothing),
    :sea_level_pressure     => (Center, Center, Nothing),
    :eastward_wind          => (Center, Center, Nothing),
    :northward_wind         => (Center, Center, Nothing),
    :rain_freshwater_flux   => (Center, Center, Nothing),
    :zonal_wind_stress      => (Center, Center, Nothing),
    :meridional_wind_stress => (Center, Center, Nothing),
)

const ECCOMetadata{D} = Metadata{<:ECCODataset, D}
const ECCOMetadatum   = Metadatum{<:ECCODataset}

# sea surface pressure can exceed 1e5 (the default higher bound for datasets data)
NumericalEarth.DataWrangling.higher_bound(::ECCOMetadata, ::Val{:sea_level_pressure}) = 1f10

# Note: ECCO downwelling radiation variables (oceQsw, EXFlwdn) are already
# in positive-downwelling convention, so no sign conversion is needed.
NumericalEarth.DataWrangling.conversion_units(metadatum::ECCOMetadatum) = nothing

"""
    ECCOMetadatum(name;
                  date = NumericalEarth.DataWrangling.first_date(ECCO4Monthly(), name),
                  dir = download_ECCO_cache)

An alias to construct a [`Metadatum`](@ref) of `ECCO4Monthly()`.
"""
function ECCOMetadatum(name;
                       date = NumericalEarth.DataWrangling.first_date(ECCO4Monthly(), name),
                       dir = download_ECCO_cache)

    return Metadatum(name; date, dir, dataset=ECCO4Monthly())
end

NumericalEarth.DataWrangling.metaprefix(::ECCOMetadata) = "ECCOMetadata"

# File name generation specific to each dataset
function NumericalEarth.DataWrangling.metadata_filename(::ECCO4Monthly, name, date, region)
    shortname = ECCO4_dataset_variable_names[name]
    yearstr   = string(Dates.year(date))
    monthstr  = string(Dates.month(date), pad=2)
    return shortname * "_" * yearstr * "_" * monthstr * ".nc"
end

ecco2_is_three_dimensional(name) =
    name == :temperature ||
    name == :salinity ||
    name == :u_velocity ||
    name == :v_velocity

function NumericalEarth.DataWrangling.metadata_filename(dataset::Union{ECCO2Daily, ECCO2Monthly}, name, date, region)
    shortname = ECCO2_dataset_variable_names[name]
    yearstr   = string(Dates.year(date))
    monthstr  = string(Dates.month(date), pad=2)
    postfix   = ecco2_is_three_dimensional(name) ? ".1440x720x50." : ".1440x720."

    if dataset isa ECCO2Monthly
        return shortname * postfix * yearstr * monthstr * ".nc"
    elseif dataset isa ECCO2Daily
        daystr = ecco2_is_three_dimensional(name) ? string(Dates.day(date), pad=2) : ""
        return shortname * postfix * yearstr * monthstr * daystr * ".nc"
    end
end

# Convenience functions
NumericalEarth.DataWrangling.dataset_variable_name(data::Metadata{<:ECCO2Daily})   = ECCO2_dataset_variable_names[data.name]
NumericalEarth.DataWrangling.dataset_variable_name(data::Metadata{<:ECCO2Monthly}) = ECCO2_dataset_variable_names[data.name]
NumericalEarth.DataWrangling.dataset_variable_name(data::Metadata{<:ECCO4Monthly}) = ECCO4_dataset_variable_names[data.name]
NumericalEarth.DataWrangling.dataset_location(::ECCODataset, name) = ECCO_location[name]

NumericalEarth.DataWrangling.is_three_dimensional(data::ECCOMetadata) =
    data.name == :temperature ||
    data.name == :salinity ||
    data.name == :u_velocity ||
    data.name == :v_velocity

# URLs for the ECCO datasets specific to each dataset
metadata_url(m::Metadata{<:ECCO2Monthly}) =
    ECCO2_url * "monthly/" * NumericalEarth.DataWrangling.dataset_variable_name(m) * "/" * m.filename
metadata_url(m::Metadata{<:ECCO2Daily}) =
    ECCO2_url * "daily/"   * NumericalEarth.DataWrangling.dataset_variable_name(m) * "/" * m.filename

function metadata_url(m::Metadata{<:ECCO4Monthly})
    year = string(Dates.year(m.dates))
    return ECCO4_url * NumericalEarth.DataWrangling.dataset_variable_name(m) * "/" * year * "/" * m.filename
end

function NumericalEarth.DataWrangling.download_dataset(metadata::ECCOMetadata)
    username = get(ENV, "ECCO_USERNAME", nothing)
    password = get(ENV, "ECCO_WEBDAV_PASSWORD", nothing)
    dir = metadata.dir

    # Create a temporary directory to store the .netrc file
    # The directory will be deleted after the download is complete
    @root mktempdir(dir) do tmp

        # Write down the username and password in a .netrc file
        downloader = netrc_downloader(username, password, "ecco.jpl.nasa.gov", tmp)
        ntasks = Threads.nthreads()

        asyncmap(metadata; ntasks) do metadatum # Distribute the download among tasks

            fileurl  = metadata_url(metadatum)
            filepath = metadata_path(metadatum)

            if !isfile(filepath)
                instructions_msg = "\n See NumericalEarth.jl/src/DataWrangling/ECCO/README.md for instructions."
                if isnothing(username)
                    msg = "Could not find the ECCO_USERNAME environment variable. \
                            See NumericalEarth.jl/src/DataWrangling/ECCO/README.md for instructions on obtaining \
                            and setting your ECCO_USERNAME and ECCO_WEBDAV_PASSWORD." * instructions_msg
                    throw(ArgumentError(msg))
                elseif isnothing(password)
                    msg = "Could not find the ECCO_WEBDAV_PASSWORD environment variable. \
                            See NumericalEarth.jl/src/DataWrangling/ECCO/README.md for instructions on obtaining \
                            and setting your ECCO_USERNAME and ECCO_WEBDAV_PASSWORD." * instructions_msg
                    throw(ArgumentError(msg))
                end
                @info "Downloading ECCO data: $(metadatum.name) in $(metadatum.dir)..."
                Downloads.download(fileurl, filepath; downloader, progress=download_progress)
            end
        end
    end

    return metadata_path(metadata)
end

function inpainted_metadata_filename(metadata::ECCOMetadatum)
    without_extension = metadata.filename[1:end-3]
    return without_extension * "_inpainted.jld2"
end

ECCO_atmosphere_variables = (
    :downwelling_shortwave,
    :downwelling_longwave,
    :air_temperature,
    :air_specific_humidity,
    :sea_level_pressure,
    :eastward_wind,
    :northward_wind,
    :rain_freshwater_flux,
)

function NumericalEarth.DataWrangling.default_inpainting(metadata::ECCOMetadata)
    if metadata.name in (:temperature, :salinity) || metadata.name in ECCO_atmosphere_variables
        return NearestNeighborInpainting(Inf)
    elseif metadata.name in (:sea_ice_thickness, :sea_ice_concentration)
        return nothing
    else
        return NearestNeighborInpainting(5)
    end
end

NumericalEarth.DataWrangling.inpainted_metadata_path(metadata::ECCOMetadatum) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

include("ECCO_atmosphere.jl")
include("ECCO_radiation.jl")

#####
##### Column Field for ECCO datasets (which always download globally)
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!

const ECCOColumnMetadatum = Metadatum{<:ECCODataset, <:Any, <:Column}

function Oceananigans.Fields.Field(metadata::ECCOColumnMetadatum, arch=CPU();
                                   inpainting = NumericalEarth.DataWrangling.default_inpainting(metadata),
                                   mask = nothing,
                                   halo = (3, 3, 3),
                                   cache_inpainted_data = true)

    NumericalEarth.DataWrangling.download_dataset(metadata)
    column_grid = native_grid(metadata, arch; halo)

    # Build a full-grid Field without a region to load the global data
    global_metadatum = Metadatum(metadata.name;
                                 dataset = metadata.dataset,
                                 date = metadata.dates)

    intermediate_field = Field(global_metadatum, arch; inpainting, mask, halo, cache_inpainted_data)
    fill_halo_regions!(intermediate_field)

    # Extract the column
    _, _, LZ = location(metadata)
    column_field = Field{Nothing, Nothing, LZ}(column_grid)
    extract_column!(column_field, intermediate_field, metadata.region)

    return column_field
end

end # Module
