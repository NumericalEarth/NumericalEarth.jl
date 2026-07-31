#####
##### ERA5-Land: the land component of ERA5 rerun at 0.1° (~9 km), 1950–present.
##### Hourly (`reanalysis-era5-land`) and monthly-mean (`reanalysis-era5-land-monthly-means`)
##### CDS catalogue entries. Ocean cells carry no data and load as `NaN`.
#####

"""
    ERA5HourlyLand()

Hourly [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
reanalysis on its native 0.1° (~9 km) grid: soil temperature and moisture on the four
ECMWF soil levels (0–7, 7–28, 28–100, and 100–289 cm), skin temperature, snow, and 2 m
air state. ERA5-Land is land-only — ocean cells load as `NaN`.

Downloading requires CDS credentials in `~/.cdsapirc` and acceptance of the ERA5-Land
licence on the CDS portal (separate from the ERA5 licence).

```jldoctest
using NumericalEarth

size(ERA5HourlyLand(), :soil_temperature_level_4)

# output
(3600, 1800, 1)
```
"""
struct ERA5HourlyLand <: ERA5Dataset end

"""
    ERA5MonthlyLand()

Monthly-mean [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means)
reanalysis on the native 0.1° grid — the `monthly_averaged_reanalysis` companion of
[`ERA5HourlyLand`](@ref), with one field per calendar month. Suited to climatological
soil state, e.g. the deep soil temperature that closes a slab land model's energy
budget from below.
"""
struct ERA5MonthlyLand <: ERA5Dataset end

dataset_name(::ERA5HourlyLand)  = "ERA5HourlyLand"
dataset_name(::ERA5MonthlyLand) = "ERA5MonthlyLand"

const ERA5LandDataset = Union{ERA5HourlyLand, ERA5MonthlyLand}
const ERA5LandMetadata{D} = Metadata{<:ERA5LandDataset, D}

DataWrangling.all_dates(::ERA5HourlyLand, variable)  = range(DateTime("1950-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
DataWrangling.all_dates(::ERA5MonthlyLand, variable) = range(DateTime("1950-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

# 0.1° global grid. As on the 0.25° single-level grid, the file's poleward rows sit on
# the ±90 faces, so the native grid carries one fewer latitude row than the file
# (`AverageNorthSouth` mangling folds the extra row in on global reads).
Base.size(::ERA5LandDataset, variable) = (3600, 1800, 1)

DataWrangling.longitude_interfaces(::ERA5LandMetadata) = (-0.05, 359.95)
DataWrangling.latitude_interfaces(::ERA5LandMetadata) = (-90, 90)
DataWrangling.default_horizontal_padding(::ERA5LandDataset) = 1/5  # two native (0.1°) cells

#####
##### ERA5-Land variable name mappings
#####

# Variable name mappings from NumericalEarth names to CDS API variable names.
# Soil levels 1–4 span 0–7, 7–28, 28–100, and 100–289 cm.
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

# NetCDF short variable names (what's actually in the downloaded files)
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

DataWrangling.available_variables(::ERA5LandDataset) = ERA5Land_dataset_variable_names
DataWrangling.dataset_variable_name(metadata::ERA5LandMetadata) = ERA5Land_netcdf_variable_names[metadata.name]

# All supported ERA5-Land variables are instantaneous states in SI units. Unlike the
# single-level product, ERA5-Land accumulations (radiation, precipitation) reset daily
# rather than hourly, so the single-level accumulation conversions must not be inherited
# if accumulated variables are ever added here.
DataWrangling.conversion_units(::ERA5LandMetadata) = nothing
