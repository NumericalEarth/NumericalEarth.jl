abstract type ERA5PressureLevelsDataset <: ERA5Dataset end

# ERA5PressureMetadata is a subtype of ERA5Metadata
const ERA5PressureMetadata{D} = Metadata{<:ERA5PressureLevelsDataset, D}
const ERA5PressureMetadatum = Metadatum{<:ERA5PressureLevelsDataset}

struct ERA5HourlyPressureLevels{Z} <: ERA5PressureLevelsDataset
    pressure_levels :: Vector{Float64}
    z :: Z   # `Nothing` → build per-column from Φ; or a precomputed vertical coord
end

ERA5HourlyPressureLevels(pressure_levels, z = nothing) =
    ERA5HourlyPressureLevels{typeof(z)}(sort(pressure_levels, rev=true), z)

struct ERA5MonthlyPressureLevels{Z} <: ERA5PressureLevelsDataset
    pressure_levels :: Vector{Float64}
    z :: Z
end

ERA5MonthlyPressureLevels(pressure_levels, z = nothing) =
    ERA5MonthlyPressureLevels{typeof(z)}(sort(pressure_levels, rev=true), z)

"""
    ERA5HourlyPressureLevels(; pressure_levels = ERA5_all_pressure_levels, z = nothing)

Hourly ERA5 pressure-levels dataset metadata. By default (`z = nothing`) the
native grid's vertical coordinate is a 3-D
[`PressureLevelVerticalDiscretization`](@ref) built from the **time-evolving**
geopotential Φ(λ,φ,p,t)/g with sub-surface levels clipped at the surface — the
right thing over terrain and synoptic Φ-displacement (issue #236). Each pressure
level's height follows the reanalysis in time as the atmosphere's clock advances.

To pin the z-coordinate to something static — e.g. for a quick test without an
extra Φ download — pass a precomputed vector:
`z = standard_atmosphere_z_interfaces(pressure_levels)`.
"""
ERA5HourlyPressureLevels(; pressure_levels = ERA5_all_pressure_levels, z = nothing) =
    ERA5HourlyPressureLevels(pressure_levels, z)

ERA5MonthlyPressureLevels(; pressure_levels = ERA5_all_pressure_levels, z = nothing) =
    ERA5MonthlyPressureLevels(pressure_levels, z)

dataset_name(::ERA5HourlyPressureLevels) = "ERA5HourlyPressureLevels"
dataset_name(::ERA5MonthlyPressureLevels) = "ERA5MonthlyPressureLevels"

#####
##### ERA5 pressure-level data availability
#####

# ERA5 reanalysis data available from 1940 to present (we use a practical range here)
DataWrangling.all_dates(::ERA5HourlyPressureLevels, var) = range(DateTime("1940-01-01"), stop=DateTime("2025-12-31"), step=Hour(1))
DataWrangling.all_dates(::ERA5MonthlyPressureLevels, var) = range(DateTime("1940-01-01"), stop=DateTime("2025-12-01"), step=Month(1))

# ERA5 pressure-level data is a spatially 3-D dataset
DataWrangling.is_three_dimensional(::ERA5PressureMetadata) = true

# TODO: drop once Oceananigans.Units exports `hPa` in a tagged release
const hPa = 100

const ERA5_all_pressure_levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,
    175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000]hPa

# ERA5 stores pressure levels bottom-to-top
DataWrangling.reversed_vertical_axis(::ERA5PressureLevelsDataset) = false

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
DataWrangling.available_variables(::ERA5PressureLevelsDataset) = ERA5PL_dataset_variable_names

# `dataset_variable_name` returns the short name as stored in the NetCDF file
# (e.g. "u"). The CDS API catalog name (e.g. "u_component_of_wind") used in
# download requests is accessed via the `ERA5PL_dataset_variable_names` dict
# directly in `NumericalEarthCDSAPIExt`.
DataWrangling.dataset_variable_name(md::ERA5PressureMetadata) = ERA5PL_netcdf_variable_names[md.name]

DataWrangling.conversion_units(md::ERA5PressureMetadata) =
    md.name == :geopotential_height ? InverseGravity() : nothing

DataWrangling.default_inpainting(md::ERA5PressureMetadata) = nothing

"""
    retrieve_data(metadata::ERA5PressureMetadatum)

Retrieve ERA5 pressure-level data from a NetCDF file.
Returns a 3D array (lon, lat, level) with levels ordered bottom-to-top
(highest pressure at k=1, lowest pressure at k=Nz).
"""
function DataWrangling.retrieve_data(metadata::ERA5PressureMetadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    ds   = NCDatasets.Dataset(path)
    data = ds[name][:, :, :, 1]   # (lon, lat, pressure_level, time=1)
    close(ds)
    return reverse(data, dims=2)  # Latitude is stored from 90°N → 90°S
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

# ERA5 pressure-levels (3-D) data product. If the caller pre-supplied a `z`,
# use it; otherwise build a per-column geopotential discretization.
function DataWrangling.z_interfaces(metadata::ERA5PressureMetadata)
    !isnothing(metadata.dataset.z) && return metadata.dataset.z
    return per_column_geopotential_discretization(metadata)
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
##### per_column_geopotential_discretization — 3-D, time-evolving z-coordinate from Φ(t)
#####

"""
    per_column_geopotential_discretization(metadata::ERA5PressureMetadata;
                                           clock = Clock{eltype(metadata)}(time = 0))

Build a [`PressureLevelVerticalDiscretization`](@ref) whose per-column heights
follow the ERA5 geopotential in time. The geopotential Φ(λ,φ,p,t) is loaded over
the whole date window of `metadata` (all snapshots in memory) and wrapped in a
`TimeSeriesInterpolation` bound to `clock`; `znode` reads Φ interpolated to
`clock.time` and divides by `g`. Sub-surface levels are clipped to the local
surface geopotential so that columns stay monotonic. The single-level surface
geopotential (orography × g, static in time) is downloaded from the matching
`ERA5*SingleLevel` dataset and used as the clip source.

A static-z copy of the pressure-level dataset backs the Φ `FieldTimeSeries`, to
break the recursive `Field(ϕ) → native_grid → z_interfaces` chain. The static z
values are arbitrary placeholders; only their shape matters.

Pass the atmosphere's `clock` so that stepping the atmosphere advances the grid's
heights; the default (a detached clock at `t = 0`) yields a first-date-static grid.
"""
function per_column_geopotential_discretization(metadata::ERA5PressureMetadata;
                                                 clock = Clock{eltype(metadata)}(time = zero(eltype(metadata))))
    static_z = standard_atmosphere_z_interfaces(metadata.dataset.pressure_levels)
    static_ds = _with_z(metadata.dataset, static_z)
    sl_ds = DataWrangling.matching_single_level_dataset(metadata.dataset)

    # Φ(λ, φ, p, t) over the whole window, held in memory on the static-z grid. A
    # `TimeSeriesInterpolation` bound to `clock` turns it into a time-varying z-coordinate.
    ϕ_meta = Metadata(:geopotential; dataset=static_ds,
                      dates=metadata.dates, region=metadata.region, dir=metadata.dir)
    Nt = length(ϕ_meta)
    Φ_fts = FieldTimeSeries(ϕ_meta, CPU(); time_indices_in_memory = Nt, time_indexing = Cyclical())
    Φ = TimeSeriesInterpolation(Φ_fts, Φ_fts.grid; clock)

    # Surface geopotential is orography × g — static in time — so a single snapshot is the clip source.
    date = first(metadata).dates
    ϕ_sl_datum = Metadatum(:geopotential; dataset=sl_ds, date, region=metadata.region, dir=metadata.dir)
    Downloads.download(ϕ_sl_datum)
    Φ_sfc = Field(ϕ_sl_datum)  # 2-D surface geopotential, m²/s²

    return PressureLevelVerticalDiscretization(Φ;
                                               gravitational_acceleration = ERA5_gravitational_acceleration,
                                               surface_geopotential = Φ_sfc)
end

_with_z(ds::ERA5HourlyPressureLevels, z) = ERA5HourlyPressureLevels(ds.pressure_levels, z)
_with_z(ds::ERA5MonthlyPressureLevels, z) = ERA5MonthlyPressureLevels(ds.pressure_levels, z)

# Match each pressure-level dataset to the single-level dataset at the same
# temporal cadence so surface fields (the geopotential clip-source, the surface
# pressure anchor) match the data they accompany.
DataWrangling.matching_single_level_dataset(::ERA5HourlyPressureLevels) = ERA5HourlySingleLevel()
DataWrangling.matching_single_level_dataset(::ERA5MonthlyPressureLevels) = ERA5MonthlySingleLevel()

#####
##### Raw per-level native load
#####

# A `FieldTimeSeries` of native-grid ERA5 pressure-level data. It is loaded raw —
# each pressure level's values go straight into the matching k-slot with no vertical
# interpolation — because the grid's geopotential heights move in time (see
# `per_column_geopotential_discretization`): the data belongs to pressure levels, and
# the level→height mapping is applied later, at query time, from the current clock.
# Interpolating at load onto one reference-height frame (as the generic `set!` does)
# would freeze the data to that frame and desync it from the moving grid.
const NativeERA5PressureFTS =
    FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{true, <:Any, <:Any, <:ERA5PressureMetadata}}

function Oceananigans.Fields.set!(fts::NativeERA5PressureFTS, backend=fts.backend)
    for t in time_indices(fts)
        metadatum = @inbounds backend.metadata[t]
        set_metadata_field!(fts[t], retrieve_data(metadatum), metadatum)
    end
    fill_halo_regions!(fts)
    return nothing
end

# Load a variable's native pressure-level `FieldTimeSeries` on the pre-built,
# time-evolving `grid`. `on_native_grid = true` selects the raw `set!` above and
# skips the generic `FieldTimeSeries(metadata, grid)` constructor's redundant
# `native_grid` rebuild + column-mean equality check (once per variable).
function era5_native_pressure_fts(metadata, grid;
                                  time_indexing = Cyclical(),
                                  time_indices_in_memory = nothing)
    Downloads.download(metadata)
    times = convert.(eltype(grid), native_times(metadata))
    Nt = length(times)
    time_indices_in_memory = min(something(time_indices_in_memory, Nt), Nt)
    loc = LX, LY, LZ = location(metadata)
    boundary_conditions = FieldBoundaryConditions(grid, instantiate.(loc))
    backend = DatasetBackend(time_indices_in_memory, metadata; on_native_grid = true,
                             inpainting = nothing, cache_inpainted_data = false)
    fts = FieldTimeSeries{LX, LY, LZ}(grid, times; backend, time_indexing, boundary_conditions)
    set!(fts)
    return fts
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

