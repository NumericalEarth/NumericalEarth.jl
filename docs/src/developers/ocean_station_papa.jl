# # Implementing a Station Dataset: Ocean Station Papa
#
# This tutorial walks through the implementation of a custom station dataset that plugs into NumericalEarth's
# Metadata machinery. Ocean Station Papa (OSP) is a moored buoy at (−144.9°E, 50.1°N) in the North-East Pacific
# that has been producing time-series of ocean and surface atmospheric observations since June 2007. The OSP
# file is a single water column rather than a global grid, and for this reason it is the natural case study for
# the `Column` region: the user wraps the metadata in `region = Column(longitude, latitude)` and the generic
# field pipeline does the rest. No `native_grid` or `set!` override is needed.
#
# The tutorial is organised in three parts. The first part implements the `OceanStationPapa` dataset and
# verifies that it satisfies the contract via `test_dataset_contract`. The second part loads a multi-week time
# series and produces a Hovmöller diagram of the temperature evolution. The third part composes a
# `PrescribedAtmosphere` from the buoy surface variables: this is the pattern a companion package such as
# `NumericalEarthMoorings.jl` would expose as a public helper.
#
# The dataset interface methods listed below are the ones any `AbstractDataset` subtype must (or may)
# implement. For OS Papa the file structure is fixed and known, so the date range, the per-variable depth
# arrays, the variable name table, and the URL are all hardcoded constants — not lazily read from the file.
# This makes the tutorial easy to follow line by line.

using NumericalEarth
using NumericalEarth.DataWrangling:
    AbstractDataset,
    Metadata, Metadatum, Column,
    dataset_variable_name, all_dates,
    longitude_interfaces, latitude_interfaces, z_interfaces,
    is_three_dimensional, reversed_vertical_axis,
    dataset_location,
    conversion_units, Celsius, Millibar, MillimetersPerHour,
    retrieve_data,
    metadata_filename, default_download_directory, metadata_path,
    available_variables, dataset_url,
    centers_to_interfaces, default_inpainting,
    test_dataset_contract

import NumericalEarth.DataWrangling:
    dataset_variable_name, all_dates,
    longitude_interfaces, latitude_interfaces, z_interfaces,
    is_three_dimensional, reversed_vertical_axis,
    dataset_location,
    conversion_units, retrieve_data,
    metadata_filename, default_download_directory,
    available_variables, dataset_url,
    default_inpainting

using NumericalEarth.Atmospheres:
    PrescribedAtmosphere, TwoBandDownwellingRadiation,
    AtmosphereThermodynamicsParameters

using Oceananigans
using Oceananigans.Fields: Center
using Dates: DateTime, Hour, Day
using NCDatasets
using Thermodynamics: q_vap_from_RH, Liquid

# ## The dataset type
#
# `OceanStationPapa` is a singleton: the mooring has only one product, so no parameters are needed.

struct OceanStationPapa <: AbstractDataset end

const OSPMetadata{D}  = Metadata{<:OceanStationPapa, D}
const OSPMetadatum    = Metadatum{<:OceanStationPapa}

# ## File store and URL
#
# The OSP archive is a single netCDF file hosted on AWS. We cache it on the local directory and expose the URL
# through `dataset_url`; the default `download_dataset` orchestrator then handles the rest.

default_download_directory(::OceanStationPapa) = "./"
dataset_url(::Metadatum{<:OceanStationPapa}) = "https://noaa-oar-keo-papa-pds.s3.amazonaws.com/PAPA/OS_PAPA_200706_M_TSVMBP_50N145W_hr.nc"

# A single file holds every variable for every date, so `metadata_filename` returns the same string regardless
# of `name`, `date`, or `region`.

metadata_filename(::OceanStationPapa, name, date, region) = "OS_PAPA_200706_M_TSVMBP_50N145W_hr.nc"

# ## Variables
#
# The variable table pairs canonical names (`:temperature`, `:salinity`, `:eastward_wind`, ...) with the
# native netCDF variable names. Profile variables are listed separately because they share a vertical axis;
# scalar surface variables occupy the same horizontal location but reduce to a single level.

const OSP_PROFILE_VARIABLES = Dict(
    :temperature => "TEMP",
    :salinity    => "PSAL",
)

const OSP_SURFACE_VARIABLES = Dict(
    :eastward_wind       => "UWND",
    :northward_wind      => "VWND",
    :air_temperature     => "AIRT",
    :relative_humidity   => "RELH",
    :sea_level_pressure  => "ATMS",
    :shortwave_radiation => "SW",
    :longwave_radiation  => "LW",
    :rain                => "RAIN",
)

const OSP_VARIABLE_NAMES = merge(OSP_PROFILE_VARIABLES, OSP_SURFACE_VARIABLES)

available_variables(::OceanStationPapa) = OSP_VARIABLE_NAMES
dataset_variable_name(md::OSPMetadata)  = OSP_VARIABLE_NAMES[md.name]
is_three_dimensional(md::OSPMetadata)   = md.name in keys(OSP_PROFILE_VARIABLES)

# ## Time axis (hardcoded)
#
# The OSP archive in this file runs from 2007-06-07 23:00 UTC to 2023-06-07 00:00 UTC at one-hour resolution.
# A handful of hours are flagged as bad for at least one variable (most often the wind sensors went offline);
# rather than keep those entries and patch them downstream, we list them in `OSP_OUTAGES` and remove them
# from the date axis returned by `all_dates`. The list below covers the Oct 2012 demo window only — extend it
# with the bad hours for any other window you simulate.

const OSP_FULL_DATES = DateTime(2007, 6, 7, 23) : Hour(1) : DateTime(2023, 6, 7, 0)
const OSP_OUTAGES = (DateTime(2012, 10, 10, 23), DateTime(2012, 10, 11, 0))

all_dates(::OceanStationPapa, name) = [t for t in OSP_FULL_DATES if !(t in OSP_OUTAGES)]

# ## Depth axes (hardcoded)
#
# Temperature and salinity are reported on slightly different depth grids; the values below are read directly
# from the file headers and pasted in as `const`s so the rest of the dataset interface does not need to open
# the file just to know how many levels to expect.

osp_depths(::Val{:temperature}) = Float64[1, 5, 8, 10, 13, 14, 17, 20, 25, 30, 31, 32, 35, 36, 37, 45, 60, 80, 100, 120, 150, 175, 200, 300]
osp_depths(::Val{:salinity})    = Float64[1, 5, 8, 10, 14, 20, 25, 30, 35, 36, 37, 45, 60, 80, 100, 120, 150, 175, 200, 300]

# ## Native grid parameters
#
# The mooring sits at one fixed `(longitude, latitude)` point. Profile variables get cell faces derived from
# the depth array (top at z = 0, interior faces at midpoints, bottom extrapolated) via the
# `centers_to_interfaces` helper; surface scalars get a degenerate z ∈ (-1, 0) range that the
# `column_field_from_file` path expects. The file stores depths shallow-first; `reversed_vertical_axis = true`
# tells the generic pipeline to flip the vertical dimension so the user does not need to think about it in
# `retrieve_data`.

longitude_interfaces(::OceanStationPapa) = (-144.9, -144.9)
latitude_interfaces(::OceanStationPapa)  = (50.1, 50.1)

reversed_vertical_axis(::OceanStationPapa) = true

z_interfaces(md::OSPMetadata) = is_three_dimensional(md) ? centers_to_interfaces(sort(-osp_depths(Val(md.name)))) : (-1.0, 0.0)
Base.size(::OceanStationPapa, name) = haskey(OSP_PROFILE_VARIABLES, name) ? (1, 1, length(osp_depths(Val(name)))) : (1, 1, 1)

# ## Locations
#
# Every variable lives at cell centers in z. Surface scalars are conventionally 0-D, but the OSP file stores
# them as `(1, 1, 1, t)`; we treat that single cell as a 1-cell column at z ∈ (-1, 0). `Center` for the
# horizontal locations is reduced to `Nothing` automatically once the user wraps the metadata in
# `region = Column(...)`.

dataset_location(::OceanStationPapa, name) = (Center, Center, Center)

# ## Unit conversions
#
# OSP reports air temperature in °C, sea-level pressure in mbar, and rain rate in mm/hr; NumericalEarth
# expects SI internally. Returning the matching unit tag from `conversion_units` registers the conversion; the
# generic field-population kernel then calls `convert_units(value, tag)` for every grid point. No conversion
# is needed for ocean temperature (already °C, which is what the ocean component uses).

function conversion_units(md::OSPMetadatum)
    md.name === :air_temperature    && return Celsius()
    md.name === :sea_level_pressure && return Millibar()
    md.name === :rain               && return MillimetersPerHour()
    return nothing
end

# Inpainting fills land-masked cells in a global gridded field with neighbouring ocean values. A station
# dataset has no horizontal neighbours, so inpainting is switched off entirely.

default_inpainting(::OSPMetadata) = nothing

# ## Reading one snapshot
#
# `retrieve_data` is the dataset's I/O hook: it returns the data for one `Metadatum` (one variable, one date)
# as a CPU array of shape `(1, 1, Nz)`. For OS Papa the file layout is the same for profiles and scalars —
# both are stored as `(1, 1, Nz, Nt)` with `Nz = 1` for the scalars — so the function does not need to branch
# on `is_three_dimensional`.
#
# The implementation is built around two small helpers that act element-wise on the vertical column.
# `mask_with_qc` turns suspect or missing entries into `NaN`, following the Argo convention (`1` = good,
# `2` = probably good, anything else is suspect). `fill_nans_vertically!` then walks the column and replaces
# each `NaN` with the value at the nearest non-`NaN` index, so the data leaves `retrieve_data` clean.

mask_with_qc(value, qc) = (ismissing(value) || ismissing(qc) || qc > 2) ? NaN : Float64(value)

function fill_nans_vertically!(profile)
    valid = findall(!isnan, profile)
    isempty(valid) && return profile
    for i in eachindex(profile)
        if isnan(profile[i])
            nearest = valid[argmin(abs.(valid .- i))]
            profile[i] = profile[nearest]
        end
    end
    return profile
end

# `retrieve_data` itself is then a short linear pipeline: open the file, locate the snapshot along `TIME`,
# read the vertical column for the requested variable, read the matching `_QC` column (or fall back to a
# vector of `Int8(1)` for variables without QC), close the file, apply the two helpers, and reshape the
# result to `(1, 1, Nz)`.

function retrieve_data(md::OSPMetadatum)
    file    = NCDataset(metadata_path(md))
    varname = dataset_variable_name(md)

    time_index = findfirst(==(md.dates), file["TIME"][:])
    isnothing(time_index) && (close(file); error("Date $(md.dates) not found in OSP file"))

    values   = vec(file[varname][1, 1, :, time_index])
    qc_flags = haskey(file, varname * "_QC") ?
                   vec(file[varname * "_QC"][1, 1, :, time_index]) :
                   fill(Int8(1), length(values))

    close(file)

    cleaned = mask_with_qc.(values, qc_flags)
    fill_nans_vertically!(cleaned)

    return reshape(cleaned, 1, 1, :)
end

# ## Conformance check
#
# At this point the dataset should satisfy the extension contract. The `test_dataset_contract` helper walks
# every required and optional method and produces a human-readable conformance report.

report = test_dataset_contract(OceanStationPapa(); verbose=true)

# ## A time series of temperature profiles
#
# With the dataset implemented, a multi-week temperature time series is a few lines: a `Metadata` with a date
# range and `region = Column(...)`, combined with a target column grid, builds a `FieldTimeSeries` whose
# backend lazily loads and regrids each hourly snapshot onto the user's column. The `Column` region is what
# tells the generic `Field` pipeline to extract a single water column from the file.

start_date = DateTime(2012, 10, 1)
end_date   = DateTime(2012, 10, 15)
osp_column = Column(-144.9, 50.1) 
md_series  = Metadata(:temperature; dataset = OceanStationPapa(), region = osp_column, start_date, end_date)

target_grid = RectilinearGrid(size = 40,
                              x = -144.9, 
                              y = 50.1,
                              z = (-300, 0),
                              topology = (Flat, Flat, Bounded))

Tts = FieldTimeSeries(md_series, target_grid; time_indices_in_memory = length(md_series))
nothing #hide

# A Hovmöller diagram with time on the x-axis and depth on the y-axis makes the evolution of the mixed layer
# and the thermocline immediately visible.

using CairoMakie

z = znodes(target_grid, Center())
t = Tts.times ./ 86400.0  # seconds → days since start_date
T = [Tts[n][1, 1, k] for k in 1:length(z), n in 1:length(t)]

fig = Figure(size = (900, 400))
ax  = Axis(fig[1, 1];
           xlabel = "Days from $start_date",
           ylabel = "Depth (m)",
           title  = "Ocean Station Papa — temperature evolution")
hm = heatmap!(ax, t, z, T'; colormap = :thermal)
Colorbar(fig[1, 2], hm; label = "Temperature (°C)")
save(joinpath(@__DIR__, "ocean_station_papa_temperature.png"), fig)
fig

# In the Hovmöller diagram a mixed layer of roughly 30 to 40 meters at the start of October 2012, 
# a gradual cooling at the surface over the two-week window, and the thermocline located
# between 50 and 100 meters, all consistent with the observed evolution at the mooring.
#
# ## Bonus: a `PrescribedAtmosphere` from buoy observations
#
# The tutorial ends with a helper that assembles a full `PrescribedAtmosphere` from the buoy surface
# variables. The implementation loads each surface variable as a `FieldTimeSeries` on a 1-cell column grid at
# the mooring point. The unit conversions registered through `conversion_units` above turn `Ta` into Kelvin,
# `Pa` into Pascal, and `rain` into kg/m²/s automatically as the fields are populated.
#
# Specific humidity is the one tracer not directly reported by the buoy: it is derived from relative
# humidity, pressure, and temperature via `Thermodynamics.q_vap_from_RH`. Everything is then handed to
# `PrescribedAtmosphere`, which the rest of NumericalEarth consumes identically to a global
# reanalysis-derived atmosphere.

function OceanStationPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                              start_date = DateTime(2012, 10, 1),
                                              end_date   = DateTime(2012, 10, 15),
                                              surface_layer_height = 2.5)

    dataset = OceanStationPapa()
    region  = Column(-144.9, 50.1)

    surface_grid = RectilinearGrid(architecture, FT;
                                   size = 1,
                                   x = -144.9, 
                                   y = 50.1,
                                   z = (-1, 0),
                                   topology = (Flat, Flat, Bounded))

    function surface_fts(name)
        md = Metadata(name; dataset, region, start_date, end_date)
        return FieldTimeSeries(md, surface_grid; time_indices_in_memory = length(md))
    end

    ua   = surface_fts(:eastward_wind)
    va   = surface_fts(:northward_wind)
    Ta   = surface_fts(:air_temperature)
    Pa   = surface_fts(:sea_level_pressure)
    swa  = surface_fts(:shortwave_radiation)
    lwa  = surface_fts(:longwave_radiation)
    rain = surface_fts(:rain)
    RHa  = surface_fts(:relative_humidity)

    thermo_params = AtmosphereThermodynamicsParameters(FT)
    LX, LY, LZ    = location(Ta)
    qa            = FieldTimeSeries{LX, LY, LZ}(Ta.grid, Ta.times)
    parent(qa)   .= q_vap_from_RH.(Ref(thermo_params), parent(Pa), parent(Ta), parent(RHa) ./ 100, Ref(Liquid()))

    return PrescribedAtmosphere(ua.grid, ua.times;
                                velocities = (u = ua, v = va),
                                tracers = (T = Ta, q = qa),
                                pressure = Pa,
                                freshwater_flux = (; rain),
                                downwelling_radiation = TwoBandDownwellingRadiation(shortwave = swa, longwave  = lwa),
                                thermodynamics_parameters = thermo_params,
                                surface_layer_height = convert(FT, surface_layer_height))
end

atmos = OceanStationPapaPrescribedAtmosphere()
nothing #hide

# `atmos` is a fully-formed `PrescribedAtmosphere` that can be passed to `OceanOnlyModel` or
# `AtmosphereOceanModel` exactly like any global reanalysis-derived atmosphere, with the single difference
# that the underlying data is a single-column observational record rather than a global reanalysis field.
