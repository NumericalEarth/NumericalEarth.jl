# ERA5 variable name mappings
# Maps symbolic names to CDS parameter names

"""
CDS variable name mapping for ERA5 datasets.
Maps common atmospheric variable symbols to CDS parameter names.

Common ERA5 variables:
- Wind: u10, v10 (10m wind components)
- Temperature: t2m (2m temperature), d2m (2m dewpoint)
- Pressure: msl (mean sea level pressure), sp (surface pressure)
- Radiation: ssrd (solar down), strd (thermal down), ssr (solar net), str (thermal net)
- Precipitation: tp (total precipitation), cp (convective precipitation)
- Clouds: tcc (total cloud cover), lcc/mcc/hcc (low/mid/high cloud cover)
- Humidity: q (specific humidity at levels)
"""
const ERA5_VARIABLE_MAP = Dict{Symbol, String}(
    # Wind components
    :u10 => "10m_u_component_of_wind",
    :v10 => "10m_v_component_of_wind",

    # Temperature
    :t2m => "2m_temperature",
    :d2m => "2m_dewpoint_temperature",
    :skt => "skin_temperature",

    # Pressure
    :msl => "mean_sea_level_pressure",
    :sp => "surface_pressure",

    # Radiation (downward)
    :ssrd => "surface_solar_radiation_downwards",
    :strd => "surface_thermal_radiation_downwards",

    # Radiation (net)
    :ssr => "surface_net_solar_radiation",
    :str => "surface_net_thermal_radiation",

    # Precipitation
    :tp => "total_precipitation",
    :cp => "convective_precipitation",

    # Cloud cover
    :tcc => "total_cloud_cover",
    :lcc => "low_cloud_cover",
    :mcc => "medium_cloud_cover",
    :hcc => "high_cloud_cover",

    # Boundary layer
    :blh => "boundary_layer_height",

    # Evaporation
    :e => "evaporation",

    # Snow
    :sd => "snow_depth",
    :snowc => "snow_cover",

    # Waves (for ocean applications)
    :swh => "significant_height_of_combined_wind_waves_and_swell",
    :mwd => "mean_wave_direction",
    :mwp => "mean_wave_period"
)

"""
    cds_variable_name(var::Symbol)

Convert symbolic variable name to CDS parameter name.

# Example
```julia
cds_variable_name(:u10)  # Returns "10m_u_component_of_wind"
```
"""
function cds_variable_name(var::Symbol)
    haskey(ERA5_VARIABLE_MAP, var) || error("Unknown ERA5 variable: $var. Available: $(keys(ERA5_VARIABLE_MAP))")
    return ERA5_VARIABLE_MAP[var]
end

"""
    cds_variable_name(vars::Vector{Symbol})

Convert vector of symbolic variable names to CDS parameter names.
"""
cds_variable_name(vars::Vector{Symbol}) = [cds_variable_name(v) for v in vars]

# Reverse mapping for convenience
const CDS_TO_SYMBOL = Dict{String, Symbol}(v => k for (k, v) in ERA5_VARIABLE_MAP)

"""
    symbol_from_cds(cds_name::String)

Convert CDS parameter name back to symbolic variable name.
"""
function symbol_from_cds(cds_name::String)
    haskey(CDS_TO_SYMBOL, cds_name) || error("Unknown CDS variable: $cds_name")
    return CDS_TO_SYMBOL[cds_name]
end
