abstract type ERA5PressureLevelsDataset <: ERA5Dataset end

# ERA5PressureMetadata is a subtype of ERA5Metadata
const ERA5PressureMetadata{D} = Metadata{<:ERA5PressureLevelsDataset, D}
const ERA5PressureMetadatum   = Metadatum{<:ERA5PressureLevelsDataset}

struct ERA5HourlyPressureLevels <: ERA5PressureLevelsDataset
    pressure_levels        :: Vector{Float64}
    z                      :: Union{Nothing, Vector{Float64}}
    mean_geopotential_height :: Bool
    ERA5HourlyPressureLevels(pressure_levels, z=nothing; mean_geopotential_height=true) =
        new(sort(pressure_levels, rev=true), z, mean_geopotential_height)
end
ERA5HourlyPressureLevels(; pressure_levels=ERA5_all_pressure_levels, z=nothing, mean_geopotential_height=true) =
    ERA5HourlyPressureLevels(pressure_levels, z; mean_geopotential_height)

struct ERA5MonthlyPressureLevels <: ERA5PressureLevelsDataset
    pressure_levels        :: Vector{Float64}
    z                      :: Union{Nothing, Vector{Float64}}
    mean_geopotential_height :: Bool
    ERA5MonthlyPressureLevels(pressure_levels, z=nothing; mean_geopotential_height=true) =
        new(sort(pressure_levels, rev=true), z, mean_geopotential_height)
end
ERA5MonthlyPressureLevels(; pressure_levels=ERA5_all_pressure_levels, z=nothing, mean_geopotential_height=true) =
    ERA5MonthlyPressureLevels(pressure_levels, z; mean_geopotential_height)

dataset_name(::ERA5HourlyPressureLevels)  = "ERA5HourlyPressureLevels"
dataset_name(::ERA5MonthlyPressureLevels) = "ERA5MonthlyPressureLevels"

#####
##### ERA5 pressure-level data availability
#####

# ERA5 reanalysis data available from 1940 to present (we use a practical range here)
all_dates(::ERA5HourlyPressureLevels,  var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
all_dates(::ERA5MonthlyPressureLevels, var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

# ERA5 pressure-level data is a spatially 3-D dataset
is_three_dimensional(::ERA5PressureMetadata) = true

# TODO: drop once Oceananigans.Units exports `hPa` in a tagged release
const hPa = 100

const ERA5_all_pressure_levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,
    175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000]hPa

# ERA5 stores pressure levels bottom-to-top
reversed_vertical_axis(::ERA5PressureLevelsDataset) = false

Base.size(ds::ERA5PressureLevelsDataset, variable) = (1440, 720, length(ds.pressure_levels))

#####
##### ERA5 pressure-level variable name mappings
#####

ERA5PL_dataset_variable_names = Dict(
    :temperature                         => "temperature",
    :eastward_velocity                   => "u_component_of_wind",
    :northward_velocity                  => "v_component_of_wind",
    :vertical_velocity                   => "vertical_velocity",
    :geopotential                        => "geopotential",
    :geopotential_height                 => "geopotential",
    :specific_humidity                   => "specific_humidity",
    :relative_humidity                   => "relative_humidity",
    :vorticity                           => "vorticity",
    :divergence                          => "divergence",
    :potential_vorticity                 => "potential_vorticity",
    :ozone_mass_mixing_ratio             => "ozone_mass_mixing_ratio",
    :fraction_of_cloud_cover             => "fraction_of_cloud_cover",
    :specific_cloud_liquid_water_content => "specific_cloud_liquid_water_content",
    :specific_cloud_ice_water_content    => "specific_cloud_ice_water_content",
    :specific_rain_water_content         => "specific_rain_water_content",
    :specific_snow_water_content         => "specific_snow_water_content",
)

# NetCDF short variable names (what's actually in the downloaded files)
# These differ from the CDS API variable names above
ERA5PL_netcdf_variable_names = Dict(
    :temperature                         => "t",
    :eastward_velocity                   => "u",
    :northward_velocity                  => "v",
    :vertical_velocity                   => "w",
    :geopotential                        => "z",
    :geopotential_height                 => "z",
    :specific_humidity                   => "q",
    :relative_humidity                   => "r",
    :vorticity                           => "vo",
    :divergence                          => "d",
    :potential_vorticity                 => "pv",
    :ozone_mass_mixing_ratio             => "o3",
    :fraction_of_cloud_cover             => "cc",
    :specific_cloud_liquid_water_content => "clwc",
    :specific_cloud_ice_water_content    => "ciwc",
    :specific_rain_water_content         => "crwc",
    :specific_snow_water_content         => "cswc",
)

# Variables available for download
available_variables(::ERA5PressureLevelsDataset) = ERA5PL_dataset_variable_names

# `dataset_variable_name` returns the short name as stored in the NetCDF file
# (e.g. "u"). The CDS API catalog name (e.g. "u_component_of_wind") used in
# download requests is accessed via the `ERA5PL_dataset_variable_names` dict
# directly in `NumericalEarthCDSAPIExt`.
dataset_variable_name(md::ERA5PressureMetadata) = ERA5PL_netcdf_variable_names[md.name]

conversion_units(md::ERA5PressureMetadata) =
    md.name == :geopotential_height ? InverseGravity() : nothing

default_inpainting(md::ERA5PressureMetadata) = nothing

"""
    retrieve_data(metadata::ERA5PressureMetadatum)

Retrieve ERA5 pressure-level data from a NetCDF file.
Returns a 3D array (lon, lat, level) with levels ordered bottom-to-top
(highest pressure at k=1, lowest pressure at k=Nz).
"""
function retrieve_data(metadata::ERA5PressureMetadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    ds   = NCDatasets.Dataset(path)
    data = ds[name][:, :, :, 1]   # (lon, lat, pressure_level, time=1)
    close(ds)
    return reverse(data, dims=2)  # Latitude is stored from 90°N → 90°S
end

#####
##### Metadata filename construction
#####

function metadata_prefix(md::ERA5PressureMetadata)
    var = ERA5PL_dataset_variable_names[md.name]
    dataset = dataset_name(md.dataset)
    start_date = start_date_str(md.dates)
    end_date = end_date_str(md.dates)
    bbox = md.region

    if !isnothing(bbox)
        w, e = bbox_strs(bbox.longitude)
        s, n = bbox_strs(bbox.latitude)
        suffix = string(w, e, s, n)
    else
        suffix = ""
    end

    if start_date == end_date
        prefix = string(var, "_", dataset, "_", start_date, suffix)
    else
        prefix = string(var, "_", dataset, "_", start_date, "_", end_date, suffix)
    end
    prefix = colon2dash(prefix)
    prefix = underscore_spaces(prefix)
    return prefix
end

#####
##### Pressure-level vertical coordinate
#####

const ERA5_gravitational_acceleration = 9.80665

# International Standard Atmosphere height (m) for a given pressure
function standard_atmosphere_geopotential_height(p)
    g = ERA5_gravitational_acceleration
    T⁰ = 288.15 # K
    p⁰ = 101325
    Rᵈ = 287.0528 # J/(kg-K)

    return (Rᵈ * T⁰ / g) * log(p⁰ / p)
end

# Build z-interfaces (Nz+1 values) from pressure levels.
# Levels may be in any order; output is sorted so k=1 is highest pressure (lowest altitude).
function standard_atmosphere_z_interfaces(levels)
    sorted_levels = sort(levels, rev=true)   # highest pressure first → k=1 is bottom
    heights = standard_atmosphere_geopotential_height.(Float64.(sorted_levels))
    Nz = length(heights)

    interfaces = Vector{Float64}(undef, Nz + 1)

    if Nz == 1
        interfaces[1] = heights[1] - 0.5
        interfaces[2] = heights[1] + 0.5
    else
        interfaces[1] = heights[1] - (heights[2] - heights[1]) / 2
        for k in 2:Nz
            interfaces[k] = (heights[k-1] + heights[k]) / 2
        end
        interfaces[Nz+1] = heights[Nz] + (heights[Nz] - heights[Nz-1]) / 2
    end

    return interfaces
end

# ERA5 pressure-levels (3-D) data product
function z_interfaces(metadata::ERA5PressureMetadata)
    # Return cached z if already set
    !isnothing(metadata.dataset.z) && return metadata.dataset.z

    # If mean_geopotential_height is enabled, try to download and compute
    if metadata.dataset.mean_geopotential_height
        ϕ_metadata = Metadata(:geopotential; dataset=metadata.dataset,
                                dates=metadata.dates, region=metadata.region,
                                dir=metadata.dir)
        try
            download_dataset(ϕ_metadata)
            return mean_geopotential_z_interfaces(metadata)
        catch e
            @warn "Failed to derive geopotential heights; falling back to standard atmosphere" exception=(e, catch_backtrace())
        end
    end

    # Fallback
    return standard_atmosphere_z_interfaces(metadata.dataset.pressure_levels)
end

#####
##### mean_geopotential_heights — data-derived static z-coordinate
#####

"""
    mean_geopotential_heights(metadata::ERA5PressureMetadata)

Compute spatially and temporally averaged geopotential heights (m) for each
pressure level in `metadata`.

Downloads the `:geopotential` field for every date in `metadata`, divides by g,
averages over the horizontal domain and all dates, and returns one representative
height per pressure level in bottom-to-top order (k=1 is highest pressure).
"""
function mean_geopotential_heights(metadata::ERA5PressureMetadata)
    ϕ_metadata = Metadata(:geopotential; dataset=metadata.dataset,
                          dates=metadata.dates, region=metadata.region,
                          dir=metadata.dir)
    Nz = length(metadata.dataset.pressure_levels)
    heights = zeros(Nz)
    # average over time
    for ϕ_datum in ϕ_metadata
        data = retrieve_data(ϕ_datum) ./ Float32(ERA5_gravitational_acceleration)   # Φ → Z (m)
        if size(data, 3) != Nz
            error("Cached geopotential file at $(metadata_path(ϕ_datum)) has " *
                  "$(size(data, 3)) pressure levels, but the dataset configuration " *
                  "expects $Nz. This is most likely a stale cache from a previous " *
                  "run with different `pressure_levels`. Delete the file and re-run.")
        end
        # average over horizontal dims
        data_mean = mean(data; dims=(1, 2))
        heights .+= dropdims(data_mean; dims=(1, 2))
    end
    heights ./= length(ϕ_metadata)

    return sort(heights)
end

function mean_geopotential_z_interfaces(metadata::ERA5PressureMetadata)
    return stagger(mean_geopotential_heights(metadata))
end

function stagger(zc::AbstractVector)
    # heights are ascending (k=1 = highest pressure = lowest altitude,
    # consistent with retrieve_data's reverse() and Oceananigans bottom-to-top
    # convention); bottom and top interfaces are extrapolated
    zf = (zc[1:end-1] .+ zc[2:end]) / 2     # Nz-1 interior interfaces
    pushfirst!(zf, zc[1] - (zf[1] - zc[1])) # bottom interface
    push!(zf, zc[end] + (zc[end] - zf[end])) # top interface
    return zf
end

#####
##### pressure_field — synthetic pressure coordinate field
#####

"""
    pressure_field(metadata::ERA5PressureMetadatum, arch=CPU(); halo=(3,3,3))

Return a `Field{Nothing, Nothing, Center}` on the native grid of `metadata`
holding the pressure value (Pa) at each vertical level. Levels are ordered
bottom-to-top (k=1 is the highest pressure level). The `Nothing` horizontal
locations make this field broadcast against full 3-D fields without copying.
"""
function pressure_field(metadata::ERA5PressureMetadatum, arch=CPU(); halo=(3,3,3))
    grid = native_grid(metadata, arch; halo)
    field = Field{Nothing, Nothing, Center}(grid)
    reversed_levels = sort(metadata.dataset.pressure_levels, rev=true)   # highest pressure → k=1
    set!(field, reversed_levels)
    fill_halo_regions!(field)
    return field
end

