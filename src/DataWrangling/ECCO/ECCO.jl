module ECCO

export ECCOMetadatum, adjusted_ECCO_tracers, initialize!
export ECCO2Monthly, ECCO4Monthly, ECCO2Daily
export ECCOPrescribedAtmosphere, ECCOPrescribedRadiation

export ECCO2DarwinMonthly, ECCO4DarwinMonthly
export retrieve_data

using Adapt: Adapt
using Dates: Dates, DateTime, Day, Month
using Downloads: Downloads
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.DistributedComputations: @root
using Oceananigans.Grids: Face, Center
using Oceananigans.OutputReaders: OutputReaders, Cyclical, FieldTimeSeries
using NCDatasets: NCDatasets
using Scratch: Scratch, @get_scratch!

using ...NumericalEarth: NumericalEarth
using ..DataWrangling: DataWrangling, binary_data_grid, binary_data_size, default_mask_value,
                       dataset_variable_name, default_download_directory, longitude_interfaces,
                       latitude_interfaces, netrc_downloader, NearestNeighborInpainting, metadata_path,
                       GramPerKilogramMinus35, Metadata, Metadatum, DownloadProgress,
                       metadata_url, first_date, last_date, all_dates

download_ECCO_cache::String = ""
function __init__()
    global download_ECCO_cache = @get_scratch!("ECCO")
    DataWrangling.DataModes.register_dataset!(ECCO2Monthly, "ECCO2Monthly")
    DataWrangling.DataModes.register_dataset!(ECCO2Daily, "ECCO2Daily")
    DataWrangling.DataModes.register_dataset!(ECCO4Monthly, "ECCO4Monthly")
    DataWrangling.DataModes.register_dataset!(ECCO2DarwinMonthly, "ECCO2DarwinMonthly")
    DataWrangling.DataModes.register_dataset!(ECCO4DarwinMonthly, "ECCO4DarwinMonthly")
end

# Datasets
abstract type ECCODataset end
struct ECCO2Monthly <:ECCODataset end
struct ECCO2Daily   <:ECCODataset end
struct ECCO4Monthly <:ECCODataset end

include("ECCO_darwin.jl")

function DataWrangling.default_download_directory(::ECCO2Monthly)
    path = joinpath(download_ECCO_cache, "v2", "monthly")
    return mkpath(path)
end

function DataWrangling.default_download_directory(::ECCO2Daily)
    path = joinpath(download_ECCO_cache, "v2", "daily")
    return mkpath(path)
end

function DataWrangling.default_download_directory(::ECCO4Monthly)
    path = joinpath(download_ECCO_cache, "v4")
    return mkpath(path)
end

Base.size(::ECCO2Daily, variable)   = (1440, 720, 50)
Base.size(::ECCO2Monthly, variable) = (1440, 720, 50)
Base.size(::ECCO4Monthly, variable) = (720,  360, 50)

DataWrangling.default_mask_value(::ECCO4Monthly) = 0
DataWrangling.reversed_vertical_axis(::ECCODataset) = true

const ECCO2_url = "https://ecco.jpl.nasa.gov/drive/files/ECCO2/cube92_latlon_quart_90S90N/"
const ECCO4_url = "https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/interp_monthly/"

# The whole range of dates in the different dataset datasets
metadata_epoch(::ECCODataset) = DateTime(1992, 1, 1)

DataWrangling.all_dates(dataset::ECCODataset) = all_dates(dataset, nothing)
DataWrangling.all_dates(dataset::ECCO4Monthly, variable) = metadata_epoch(dataset) : Month(1) : DateTime(2017, 12, 1)
DataWrangling.all_dates(dataset::ECCO2Monthly, variable) = metadata_epoch(dataset) : Month(1) : DateTime(2024, 12, 1)
DataWrangling.all_dates(dataset::ECCO2Daily,   variable) = metadata_epoch(dataset) : Day(1)   : DateTime(2024, 12, 31)

DataWrangling.longitude_interfaces(::ECCODataset) = (0, 360)
DataWrangling.longitude_interfaces(::ECCO4Monthly) = (-180, 180)
DataWrangling.latitude_interfaces(::ECCODataset) = (-90, 90)

DataWrangling.longitude_name(::Metadata{<:ECCODataset})        = "LONGITUDE_T"
DataWrangling.latitude_name(::Metadata{<:ECCODataset})         = "LATITUDE_T"
DataWrangling.longitude_name(::Metadata{<:ECCO4Monthly})       = "longitude"
DataWrangling.latitude_name(::Metadata{<:ECCO4Monthly})        = "latitude"
DataWrangling.longitude_name(::Metadata{<:ECCO4DarwinMonthly}) = "longitude"
DataWrangling.latitude_name(::Metadata{<:ECCO4DarwinMonthly})  = "latitude"

DataWrangling.z_interfaces(::ECCODataset) = [
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

DataWrangling.available_variables(::ECCO2Monthly) = ECCO2_dataset_variable_names
DataWrangling.available_variables(::ECCO2Daily)   = ECCO2_dataset_variable_names
DataWrangling.available_variables(::ECCO4Monthly) = ECCO4_dataset_variable_names

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
DataWrangling.higher_bound(::ECCOMetadata, ::Val{:sea_level_pressure}) = 1f10

"""
    ECCOMetadatum(name;
                  date = first_date(ECCO4Monthly(), name),
                  dir = download_ECCO_cache)

An alias to construct a [`Metadatum`](@ref) of `ECCO4Monthly()`.
"""
function ECCOMetadatum(name;
                       date = first_date(ECCO4Monthly(), name),
                       dir = download_ECCO_cache)

    return Metadatum(name; date, dir, dataset=ECCO4Monthly())
end

DataWrangling.metaprefix(::ECCOMetadata) = "ECCOMetadata"

# File name generation specific to each dataset
function DataWrangling.metadata_filename(::ECCO4Monthly, name, date, region)
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

function DataWrangling.metadata_filename(dataset::Union{ECCO2Daily, ECCO2Monthly}, name, date, region)
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

DataWrangling.dataset_variable_name(data::Metadata{<:ECCO2Daily})   = ECCO2_dataset_variable_names[data.name]
DataWrangling.dataset_variable_name(data::Metadata{<:ECCO2Monthly}) = ECCO2_dataset_variable_names[data.name]
DataWrangling.dataset_variable_name(data::Metadata{<:ECCO4Monthly}) = ECCO4_dataset_variable_names[data.name]
DataWrangling.dataset_location(::ECCODataset, name) = name in keys(ECCO_location) ? ECCO_location[name] : (Center, Center, Center)
  
DataWrangling.is_three_dimensional(data::ECCOMetadata) =
    data.name == :temperature ||
    data.name == :salinity ||
    data.name == :u_velocity ||
    data.name == :v_velocity

# URLs for the ECCO datasets specific to each dataset
DataWrangling.metadata_url(m::Metadata{<:ECCO2Monthly}) = ECCO2_url * "monthly/" * dataset_variable_name(m) * "/" * m.filename
DataWrangling.metadata_url(m::Metadata{<:ECCO2Daily})   = ECCO2_url * "daily/"   * dataset_variable_name(m) * "/" * m.filename

function DataWrangling.metadata_url(m::Metadata{<:ECCO4Monthly})
    year = string(Dates.year(m.dates))
    return ECCO4_url * dataset_variable_name(m) * "/" * year * "/" * m.filename
end

function Downloads.download(metadata::ECCOMetadata)
    # Skip download if all files already exist
    all(isfile(metadata_path(m)) for m in metadata) && return metadata_path(metadata)

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
                Downloads.download(fileurl, filepath; downloader, progress=DownloadProgress())
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

function DataWrangling.default_inpainting(metadata::ECCOMetadata)
    if metadata.name in (:temperature, :salinity) || metadata.name in ECCO_atmosphere_variables
        return NearestNeighborInpainting(Inf)
    elseif metadata.name in (:sea_ice_thickness, :sea_ice_concentration)
        return nothing
    else
        return NearestNeighborInpainting(5)
    end
end

DataWrangling.inpainted_metadata_path(metadata::ECCOMetadatum) = 
    joinpath(metadata.dir, inpainted_metadata_filename(metadata))

function DataWrangling.read_file_coords(metadata::ECCOMetadatum)
    Nx, Ny, _, _ = size(metadata)
    resolution_X = 360/Nx
    resolution_Y = 180/Ny
    longitudes = longitude_interfaces(metadata.dataset)
    latitudes  = latitude_interfaces(metadata.dataset)
    lon = [i for i = longitudes[1]+resolution_X/2:resolution_X:longitudes[2]-resolution_X/2]
    lat = [j for j = latitudes[1]+resolution_Y/2:resolution_Y:latitudes[2]-resolution_Y/2]

    return lon, lat
end

include("ECCO_atmosphere.jl")
include("ECCO_radiation.jl")

end # Module
