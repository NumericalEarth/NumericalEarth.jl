using MeshArrays: MeshArrays, GridSpec, GridLoad, GridLoadVar, interpolation_setup, land_mask

struct ECCO2DarwinMonthly <:ECCODataset end
struct ECCO4DarwinMonthly <:ECCODataset end

const ECCODarwin = Union{ECCO2DarwinMonthly, ECCO4DarwinMonthly}
const ECCODarwinMetadata = Metadata{<:Union{ECCO2DarwinMonthly, ECCO4DarwinMonthly}}
const ECCODarwinMetadatum = Metadatum{<:Union{ECCO2DarwinMonthly, ECCO4DarwinMonthly}}

# URLs for the ECCO datasets specific to each version
const ECCO4Darwin_url = "https://ecco.jpl.nasa.gov/drive/files/ECCO2/LLC90/ECCO-Darwin/"
const ECCO2Darwin_url = "https://ecco.jpl.nasa.gov/drive/files/ECCO2/LLC270/ECCO-Darwin_extension/"

Base.size(data::Metadata{<:ECCO4DarwinMonthly}) = (720,  360, 50, length(data.dates))
Base.size(::Metadatum{<:ECCO4DarwinMonthly})    = (720,  360, 50, 1)
Base.size(data::Metadata{<:ECCO2DarwinMonthly}) = (1440, 720, 50, length(data.dates))
Base.size(::Metadatum{<:ECCO2DarwinMonthly})    = (1440, 720, 50, 1)

metadata_time_step(::ECCO4DarwinMonthly) = 3600
metadata_epoch(::ECCO4DarwinMonthly) = DateTime(1992, 1, 1, 12, 0, 0)

metadata_time_step(::ECCO2DarwinMonthly) = 1200
metadata_epoch(::ECCO2DarwinMonthly) = DateTime(1992, 1, 1, 0, 0, 0)

# The whole range of dates in the different dataset datasets
DataWrangling.all_dates(dataset::ECCO4DarwinMonthly, name) = metadata_epoch(dataset) : Month(1) : DateTime(2023, 3, 1)
DataWrangling.all_dates(dataset::ECCO2DarwinMonthly, name) = metadata_epoch(dataset) : Month(1) : DateTime(2025, 5, 1)

# ECCO4Darwin is stamped at noon on the first of the month, so the window is the calendar month
# containing the stamp rather than a month starting at it.
DataWrangling.averaging_window(metadatum::ECCODarwinMetadatum) = DataWrangling.calendar_month_window(metadatum)

# File name generation specific to each Dataset dataset
"""
    metadata_filename(dataset, name, date, region)

Generate the filename for a given ECCO Darwin dataset and date.

The filename is constructed using the dataset variable name, and the iteration number is calculated
from the date and epoch.
"""
function DataWrangling.metadata_filename(dataset::ECCODarwin, name, date, region)
    shortname = ECCO_darwin_dataset_variable_names[name]

    reference_date = metadata_epoch(dataset)
    timestep_size  = metadata_time_step(dataset)

    # Explicitly convert to Int to avoid return of a float
    iternum = Int(Dates.value((date - reference_date) / (timestep_size * 1e3)))
    iterstr = string(iternum, pad=10)

    return shortname * "." * iterstr * ".data"
end

# Convenience functions
DataWrangling.dataset_variable_name(data::ECCODarwinMetadata) = ECCO_darwin_dataset_variable_names[data.name]
DataWrangling.longitude_name(           ::ECCODarwinMetadata) = "longitude"
DataWrangling.latitude_name(            ::ECCODarwinMetadata) = "latitude"
DataWrangling.default_mask_value(       ::ECCODarwinMetadata) = NaN
DataWrangling.missing_value(            ::ECCODarwinMetadata) = NaN
DataWrangling.is_three_dimensional(     ::ECCODarwinMetadata) = true
variable_is_three_dimensional(          ::ECCODarwinMetadata) = true

ECCO_darwin_dataset_variable_names = Dict(
    :temperature                    => "THETA",
    :salinity                       => "SALTanom",
    :dissolved_inorganic_carbon     => "DIC",
    :alkalinity                     => "ALK",
    :phosphate                      => "PO4",
    :nitrate                        => "NO3",
    :dissolved_organic_phosphorus   => "DOP",
    :particulate_organic_phosphorus => "POP",
    :dissolved_iron                 => "FeT",
    :dissolved_silicate             => "SiO2",
    :dissolved_oxygen               => "O2",
)

"""
    conversion_units(metadatum::Metadatum{<:ECCODarwin})

Set up conversion from the ECCODarwin output data to standard units
  -  salinity = SALTanom + 35
  -  biogeochemical tracer concentrations are in umol => umol/L in the output files from Darwin
"""
function DataWrangling.conversion_units(metadatum::Union{ECCODarwinMetadata, ECCODarwinMetadatum})
    if dataset_variable_name(metadatum) == "SALTanom"
        return GramPerKilogramMinus35()
    elseif dataset_variable_name(metadatum) != "THETA"
        return MicromolePerLiter() # or mmol/m3, but we choose the more conventional oceanographic units for biogeochemical tracers converted with a factor of 1000 to mol/m3
    else
        return nothing
    end
end

function DataWrangling.default_download_directory(::ECCO4DarwinMonthly)
    path = joinpath(download_ECCO_cache, "v4_darwin", "monthly")
    return mkpath(path)
end

function DataWrangling.default_download_directory(::ECCO2DarwinMonthly)
    path = joinpath(download_ECCO_cache, "v2_darwin", "monthly")
    return mkpath(path)
end

DataWrangling.metadata_url(m::Metadata{<:ECCO4DarwinMonthly}) = ECCO4Darwin_url * "monthly/" * dataset_variable_name(m) * "/" * m.filename
DataWrangling.metadata_url(m::Metadata{<:ECCO2DarwinMonthly}) = ECCO2Darwin_url * "monthly/" * dataset_variable_name(m) * "/" * m.filename

# Functions for reading the ECCO binary files using MeshArrays
DataWrangling.binary_data_grid(::ECCO4DarwinMonthly) = GridSpec(ID=:LLC90)
DataWrangling.binary_data_size(::ECCO4DarwinMonthly) = (90, 1170, 50)
DataWrangling.binary_data_grid(::ECCO2DarwinMonthly) = GridSpec(ID=:LLC270)
DataWrangling.binary_data_size(::ECCO2DarwinMonthly) = (270, 3510, 50)

DataWrangling.longitude_interfaces(::ECCO4DarwinMonthly) = (-180, 180)

"""
    retrieve_data(metadata::Metadatum{<:Union{ECCO4DarwinMonthly, ECCO2DarwinMonthly}})

Read an ECCO Darwin data file and regrid using MeshArrays onto a regular lat-lon grid.
"""
function DataWrangling.retrieve_data(metadata::Metadatum{<:Union{ECCO4DarwinMonthly, ECCO2DarwinMonthly}})
    native_size = binary_data_size(metadata.dataset)
    native_grid = binary_data_grid(metadata.dataset)
    native_data = zeros(Float32, prod(native_size)) # Native LLC grid at precision of the input binary file

    read!(metadata_path(metadata), native_data)
    native_data = bswap.(native_data)

    meshed_data   = read(reshape(native_data, native_size...), native_grid)
    Nx, Ny, Nz, _ = size(metadata)
    data          = zeros(Float32, Nx, Ny, Nz) # Native LLC grid at precision of the input binary file
    mask          = zeros(Float32, Nx, Ny, Nz)

    # Download the native grid data from MeshArrays repo (only if not in already in datadeps)
    native_grid_coords = GridLoad(native_grid; option="full")

    # We can download interpolation weights for LLC90 and LLC270 grids to 1 or 
    #  0.5 degree lat-lon grids from the MeshArrays Artifacts list (since MeshArrays v0.5.7)
    coeffs = interpolation_setup(native_grid)

    # Read continental mask on the native model grid (1 on ocean, NaN on land)
    native_grid_fac_center = GridLoadVar("hFacC", native_grid)

    # Mask land as NaN on the native grid *before* interpolating: MeshArrays.Interpolate
    # excludes NaN neighbors from its weighted average, so this prevents land values
    # from contaminating coastal ocean cells during interpolation.
    for k in 1:Nz
        masked_layer = meshed_data[:, k] .* land_mask(native_grid_fac_center[:, k])
        i, j, c = MeshArrays.Interpolate(masked_layer, coeffs)
        data[:, :, k] = c
    end

    # Reverse the z-axis
    data = reverse(data, dims=3)

    if metadata.name ∉ (:THETA, :SALTanom)
       # Negative values are unphysical for concentration data; treat them as missing so
       # they get inpainted
       data[data .<= 0] .= NaN
    end

    # Cells with no valid ocean neighbor stay NaN; mark them so compute_mask detects them
    data[isnan.(data)] .= default_mask_value(metadata.dataset)

    return data
end
