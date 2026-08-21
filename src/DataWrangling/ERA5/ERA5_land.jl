using DocStringExtensions: TYPEDSIGNATURES

"""
$(TYPEDSIGNATURES)

Hourly ERA5-Land reanalysis (`reanalysis-era5-land`): a global 0.1° land-surface
product covering skin/soil temperature, soil moisture, 2 m temperature/dewpoint, and
snow depth/water-equivalent. Ocean cells are masked and never inpainted.
"""
struct ERA5HourlyLand <: ERA5Dataset end

"""
$(TYPEDSIGNATURES)

Monthly-mean counterpart of [`ERA5HourlyLand`](@ref).
"""
struct ERA5MonthlyLand <: ERA5Dataset end

dataset_name(::ERA5HourlyLand)  = "ERA5HourlyLand"
dataset_name(::ERA5MonthlyLand) = "ERA5MonthlyLand"

const ERA5LandDataset     = Union{ERA5HourlyLand, ERA5MonthlyLand}
const ERA5LandMetadata{D} = Metadata{<:ERA5LandDataset, D}
const ERA5LandMetadatum   = Metadatum{<:ERA5LandDataset}

#####
##### ERA5-Land data availability
#####

DataWrangling.all_dates(::ERA5HourlyLand,  var) = range(DateTime("1950-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
DataWrangling.all_dates(::ERA5MonthlyLand, var) = range(DateTime("1950-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

# ERA5-Land is a spatially 2-D dataset
DataWrangling.is_three_dimensional(::ERA5LandMetadata) = false

# 0.1° global grid; the file's 1801 latitude rows fold to 1800 cells via AverageNorthSouth mangling (see set_region_data.jl).
Base.size(::ERA5LandDataset, variable) = (3600, 1800, 1)

#####
##### Grid interfaces (0.1° resolution)
#####

# Half a 0.1° cell offset, mirroring the single-level -0.125/359.875 convention at 0.25°.
DataWrangling.longitude_interfaces(::ERA5LandMetadata) = (-0.05, 359.95)
DataWrangling.latitude_interfaces(::ERA5LandMetadata)  = (-90, 90)
DataWrangling.z_interfaces(::ERA5LandMetadata)         = (0, 1)

# Two native 0.1° cells; the base ERA5Dataset default (1/2) is sized for the 0.25° single-level grid.
DataWrangling.default_horizontal_padding(::ERA5LandDataset) = 1/5

#####
##### ERA5-Land variable name mappings
#####

# Variable name mappings from NumericalEarth names to ERA5-Land/CDS API variable names
ERA5Land_dataset_variable_names = Dict(
    :skin_temperature              => "skin_temperature",
    :soil_temperature_level_1      => "soil_temperature_level_1",
    :soil_temperature_level_2      => "soil_temperature_level_2",
    :soil_temperature_level_3      => "soil_temperature_level_3",
    :soil_temperature_level_4      => "soil_temperature_level_4",
    :volumetric_soil_water_layer_1 => "volumetric_soil_water_layer_1",
    :volumetric_soil_water_layer_2 => "volumetric_soil_water_layer_2",
    :volumetric_soil_water_layer_3 => "volumetric_soil_water_layer_3",
    :volumetric_soil_water_layer_4 => "volumetric_soil_water_layer_4",
    :temperature                   => "2m_temperature",
    :dewpoint_temperature          => "2m_dewpoint_temperature",
    :snow_water_equivalent         => "snow_depth_water_equivalent",
    :snow_depth                    => "snow_depth",
)

# NetCDF short variable names (what's actually in the downloaded files), verified against a real ERA5-Land download.
ERA5Land_netcdf_variable_names = Dict(
    :skin_temperature              => "skt",
    :soil_temperature_level_1      => "stl1",
    :soil_temperature_level_2      => "stl2",
    :soil_temperature_level_3      => "stl3",
    :soil_temperature_level_4      => "stl4",
    :volumetric_soil_water_layer_1 => "swvl1",
    :volumetric_soil_water_layer_2 => "swvl2",
    :volumetric_soil_water_layer_3 => "swvl3",
    :volumetric_soil_water_layer_4 => "swvl4",
    :temperature                   => "t2m",
    :dewpoint_temperature          => "d2m",
    :snow_water_equivalent         => "sd",
    :snow_depth                    => "sde",
)

DataWrangling.available_variables(::ERA5LandDataset)      = ERA5Land_dataset_variable_names
DataWrangling.dataset_variable_name(md::ERA5LandMetadata) = ERA5Land_netcdf_variable_names[md.name]

# Instantaneous analysis fields — no unit conversion (accumulated fields like precipitation aren't supported yet).
DataWrangling.conversion_units(md::ERA5LandMetadata) = nothing

# Never inpaint land-only targets: ocean cells are masked and must stay masked.
DataWrangling.default_inpainting(md::ERA5LandMetadata) = nothing

# Monthly means span the calendar month; ERA5HourlyLand variables are all instantaneous.
DataWrangling.sample_window(md::Metadatum{<:ERA5MonthlyLand}) = DataWrangling.calendar_month_window(md)

# reversed_latitude_axis and ocean masking (nan_convert_missing) are inherited unchanged from ERA5Dataset.

#####
##### One file per variable per year, like ERA5YearlySingleLevel.
#####

function DataWrangling.metadata_filename(dataset::ERA5LandDataset, name, date, region)
    var = ERA5Land_dataset_variable_names[name]
    year = Dates.year(date)
    suffix = region_suffix(region)
    return string(var, "_", dataset_name(dataset), "_", year, suffix, ".nc")
end

"""
    build_filename(dataset::ERA5LandDataset, name, dates::AbstractArray, region)

One yearly file covers every date within that year, so a request spanning
multiple years resolves to multiple (repeated) filenames — one per date,
naming that date's year file. `set!` for the resulting `FieldTimeSeries`
groups consecutive same-year dates together and opens each yearly file once
(see `read_era5_yearly_series` in `ERA5_field_time_series.jl`).
"""
function DataWrangling.build_filename(dataset::ERA5LandDataset, name, dates::AbstractArray, region)
    return DatewiseFilename([DataWrangling.metadata_filename(dataset, name, d, region) for d in dates])
end

#####
##### Download interface
#####

"""
$(TYPEDSIGNATURES)

Return the path of the yearly ERA5-Land file containing `metadatum`, downloading it
via [`download_era5_land`](@ref) unless `skip_existing` (default `true`) finds it
already on disk.
"""
function Downloads.download(metadatum::ERA5LandMetadatum; skip_existing = true, kw...)
    path = metadata_path(metadatum)
    skip_existing && isfile(path) && return path
    return download_era5_land(metadatum; skip_existing, kw...)
end

"""
$(TYPEDSIGNATURES)

Download ERA5-Land data for each date in `metadata`, returning the paths of the
yearly files (repeated for dates that share a year).
"""
function Downloads.download(metadata::ERA5LandMetadata; kw...)
    return [Downloads.download(metadatum; kw...) for metadatum in metadata]
end

"""
$(TYPEDSIGNATURES)

Download the yearly ERA5-Land file containing `metadatum` from the Copernicus
Climate Data Store and return its path.

Implemented in `ext/NumericalEarthCopernicusClimateDataStoreExt.jl` when
CopernicusClimateDataStore is loaded; the fallback below fires only when the
extension is not active.
"""
download_era5_land(metadatum; kw...) =
    error("Downloading ERA5-Land requires the CopernicusClimateDataStore package; ",
          "load it with `using CopernicusClimateDataStore`.")
