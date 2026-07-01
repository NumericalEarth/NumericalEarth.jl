struct ERA5HourlyLand  <: ERA5Dataset end
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

# ERA5-Land is a global 0.1° product: the file carries 3600 longitudes
# (0:0.1:359.9) and 1801 latitudes (90:-0.1:-90). As with the single-level grid,
# the latitude cell count is one less than the file's row count, so the extra row
# folds in through `AverageNorthSouth` mangling — placing the 1800 cell centers at
# exactly -89.95:0.1:89.95 over (-90, 90). The inherited single-level method
# returns the 0.25° (1440, 720) size, so this override is mandatory.
Base.size(::ERA5LandDataset, variable) = (3600, 1800, 1)

#####
##### Grid interfaces (0.1° resolution)
#####

# Half a 0.1° cell offset, mirroring the single-level -0.125/359.875 convention at 0.25°.
DataWrangling.longitude_interfaces(::ERA5LandMetadata) = (-0.05, 359.95)
DataWrangling.latitude_interfaces(::ERA5LandMetadata)  = (-90, 90)
DataWrangling.z_interfaces(::ERA5LandMetadata)         = (0, 1)

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

# NetCDF short variable names (what's actually expected in the downloaded files).
# As with ERA5 single levels, the CDS "shortName" does not always match the netCDF
# variable name — these expected names must be confirmed against a real download
# before relying on them.
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

# ERA5-Land state variables are instantaneous analysis fields — no unit conversion.
DataWrangling.conversion_units(md::ERA5LandMetadata)  = nothing

# Never inpaint land-only targets: ocean cells are masked and must stay masked.
DataWrangling.default_inpainting(md::ERA5LandMetadata) = nothing

# `retrieve_data(::ERA5Metadatum)` (the (lon, lat, time) layout + latitude reversal)
# and `reversed_latitude_axis(::ERA5Dataset) = true` are inherited unchanged.
# Ocean masking is handled for free by `nan_convert_missing` in `set_region_data.jl`
# with the default `missing_value = missing`.
