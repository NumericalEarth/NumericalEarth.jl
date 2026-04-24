# # Implementing a Station Dataset: Ocean Station Papa
#
# This tutorial walks through the implementation of a custom station dataset that
# plugs into NumericalEarth's Metadata machinery through the public extension API.
# Ocean Station Papa (OSP) is a moored buoy at (−144.9°E, 50.1°N) in the North-East
# Pacific that has been producing long time-series of ocean and surface atmospheric
# observations since 2007. It is a *station* dataset, that is, a single water column
# and not a global grid, and for this reason it is the natural case study for the
# `StationColumn` spatial-layout trait.
#
# The goal of the tutorial consists in showing that a station dataset can be
# implemented from outside the package, by subtyping `AbstractDataset`, declaring
# a `StationColumn` spatial layout, and implementing a small, documented set of
# methods. The generic field-construction pipeline, including vertical-only
# interpolation and NaN-skipping, is then inherited automatically; no custom
# `set!` or `native_grid` method is needed.
#
# The tutorial is organised as follows. In the first part we implement the
# `OceanStationPapa` dataset and verify that it satisfies the contract via
# `test_dataset_contract`. In the second part we load a multi-week time series
# and produce a Hovmöller diagram showing the evolution of the temperature
# profile. In the third part we compose a `PrescribedAtmosphere` from the buoy
# observations, which is the pattern a companion package such as
# `NumericalEarthMoorings.jl` would expose as a public helper.
#
# ## The dataset contract
#
# An `AbstractDataset` subtype is recognised by the Metadata machinery when it
# implements a small number of required methods and, optionally, a few hooks.
# The required set covers variable naming (`dataset_variable_name`), the date
# axis (`all_dates`), the disk I/O (`retrieve_data`), the native grid interfaces
# (`longitude_interfaces`, `latitude_interfaces`, `z_interfaces`), and the
# Metadata-construction prerequisites (`default_download_directory`,
# `metadata_filename`, `Base.size(::Dataset, ::Symbol)`). Overriding
# `spatial_layout` to `StationColumn()` is what switches the pipeline to the
# single-column path; `dataset_url` enables the default download orchestrator;
# `conversion_units` and `preprocess_data` handle per-variable cleanup.

using NumericalEarth
using NumericalEarth.DataWrangling:
    AbstractDataset, StationColumn, spatial_layout,
    Metadata, Metadatum,
    dataset_variable_name, all_dates,
    longitude_interfaces, latitude_interfaces, z_interfaces,
    is_three_dimensional, reversed_vertical_axis, location,
    conversion_units, Celsius, Millibar, MillimetersPerHour,
    preprocess_data, retrieve_data,
    metadata_filename, default_download_directory, metadata_path,
    available_variables, dataset_url,
    centers_to_interfaces, default_inpainting, fill_gaps!

import NumericalEarth.DataWrangling:
    dataset_variable_name, all_dates,
    longitude_interfaces, latitude_interfaces, z_interfaces,
    is_three_dimensional, reversed_vertical_axis, location,
    conversion_units,
    preprocess_data, retrieve_data,
    metadata_filename, default_download_directory,
    available_variables, dataset_url, spatial_layout,
    default_inpainting

using NumericalEarth.Atmospheres:
    PrescribedAtmosphere, TwoBandDownwellingRadiation,
    AtmosphereThermodynamicsParameters

using Oceananigans
using Oceananigans.Fields: Center
using Dates: DateTime, Hour, Day
using NCDatasets
using Scratch
using Downloads
using Thermodynamics: q_vap_from_RH, Liquid

# ## The dataset type
#
# `OceanStationPapa` is a zero-field singleton. The mooring only has one
# "product", so no parameters are needed. Contrarily to global products such
# as `ECCO4Monthly`, a station type does not encode a temporal frequency in
# its name: the archive is hourly and that information lives in `all_dates`.

struct OceanStationPapa <: AbstractDataset end

# The mooring is at a single point, and for this reason we declare a
# `StationColumn` spatial layout. This single line instructs the generic
# pipeline to build a `RectilinearGrid(Flat, Flat, Bounded)` native grid and
# to regrid with vertical-only, NaN-skipping interpolation.

spatial_layout(::OceanStationPapa) = StationColumn()

# ## The file store
#
# The archive is a single netCDF file hosted on AWS. We cache it with
# `Scratch.jl`, use a constant filename, and expose the URL via
# `dataset_url`. The default `download_dataset` orchestrator defined in
# `NumericalEarth.DataWrangling` composes `dataset_url`, `authenticate`
# (no-op by default), and `download_file!` (plain `Downloads.download` by
# default) into a single per-file download, which is the right behaviour for
# a static one-file archive.

const OSP_FILENAME  = "OS_PAPA_200706_M_TSVMBP_50N145W_hr.nc"
const OSP_URL       = "https://noaa-oar-keo-papa-pds.s3.amazonaws.com/PAPA/" * OSP_FILENAME
const OSP_LONGITUDE = -144.9
const OSP_LATITUDE  = 50.1

_download_cache::String = ""
function __init__()
    global _download_cache = @get_scratch!("OceanStationPapa")
end

default_download_directory(::OceanStationPapa) = _download_cache

metadata_filename(::OceanStationPapa, name, date, bounding_box) = OSP_FILENAME

dataset_url(::Metadatum{<:OceanStationPapa}) = OSP_URL

# ## Variables and naming
#
# The variable table pairs canonical names (`:temperature`, `:salinity`,
# `:eastward_wind`, ...) with the native netCDF variable names (`"TEMP"`,
# `"PSAL"`, `"UWND"`, ...). Profile variables are listed separately because
# they share a vertical axis; scalar surface variables occupy the same
# horizontal position but reduce to a single level.

const OSP_PROFILE_VARIABLES = Dict(
    :temperature        => "TEMP",
    :salinity           => "PSAL",
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

const OSP_DEPTH_VARIABLE = Dict(
    :temperature => "DEPTH",
    :salinity    => "DEPPSAL",
)

available_variables(::OceanStationPapa) = OSP_VARIABLE_NAMES

const OSPMetadata{D} = Metadata{<:OceanStationPapa, D}
const OSPMetadatum   = Metadatum{<:OceanStationPapa}

dataset_variable_name(md::OSPMetadata) = OSP_VARIABLE_NAMES[md.name]

location(::OSPMetadata) = (Center, Center, Center)

is_three_dimensional(md::OSPMetadata) = md.name in keys(OSP_PROFILE_VARIABLES)

# The netCDF archive stores depths shallow-first, whereas the grid z-axis is
# bottom-to-top; `reversed_vertical_axis = true` tells the generic pipeline to
# flip the vertical dimension of the retrieved array, so the user does not
# need to think about it in `retrieve_data`.

reversed_vertical_axis(::OceanStationPapa) = true

# ## Unit conversions
#
# OSP reports air temperature in °C, sea-level pressure in mbar, and rain rate
# in mm/hr, while NumericalEarth expects SI internally. Scalar conversions are
# registered by returning the relevant unit tag from `conversion_units`; the
# generic field-population kernel then calls `convert_units(value, tag)` for
# every grid point. No conversion is needed for ocean temperature (°C) since
# NumericalEarth's ocean component works in °C.

function conversion_units(md::OSPMetadatum)
    md.name === :air_temperature     && return Celsius()
    md.name === :sea_level_pressure  && return Millibar()
    md.name === :rain                && return MillimetersPerHour()
    return nothing
end

# Inpainting is a post-processing step that fills land-masked cells in a
# global gridded field with neighbouring ocean values. The default for ocean
# tracers turns it on, because that is the right behaviour for ECCO, EN4 and
# similar products. On the other hand, a station dataset sees only a single
# water column with no land boundary, and for this reason inpainting is
# switched off entirely.

default_inpainting(::OSPMetadata) = nothing

# ## Time axis and depth arrays
#
# The first read of the netCDF file populates two caches: the full hourly
# time vector, shared by all variables, and the per-variable depth array for
# profile variables. The file is downloaded by the default orchestrator when
# `Field(metadatum)` is first called, but the helpers below trigger an
# explicit download if a caller needs the time or depth axes before
# constructing a `Metadatum`.

function _download_file()
    path = joinpath(_download_cache, OSP_FILENAME)
    isfile(path) || Downloads.download(OSP_URL, path)
    return path
end

const _OSP_TIMES_CACHE = Ref{Vector{DateTime}}()
const _OSP_TIMES_READY = Ref(false)

function _osp_times()
    if !_OSP_TIMES_READY[]
        ds = NCDataset(_download_file())
        _OSP_TIMES_CACHE[] = DateTime.(ds["TIME"][:])
        close(ds)
        _OSP_TIMES_READY[] = true
    end
    return _OSP_TIMES_CACHE[]
end

all_dates(::OceanStationPapa, variable) = _osp_times()

const _OSP_DEPTHS_CACHE = Dict{Symbol, Vector{Float64}}()

function _osp_depths(variable)
    if !haskey(_OSP_DEPTHS_CACHE, variable)
        ds = NCDataset(_download_file())
        depth_var = OSP_DEPTH_VARIABLE[variable]
        _OSP_DEPTHS_CACHE[variable] = Float64.(ds[depth_var][:])
        close(ds)
    end
    return _OSP_DEPTHS_CACHE[variable]
end

# ## Native grid parameters
#
# Longitude and latitude degenerate to a single point: the mooring location.
# `z_interfaces` converts cell centers to interfaces using the
# `centers_to_interfaces` helper exposed by `DataWrangling`, and returns a
# degenerate vertical axis for scalar variables.

longitude_interfaces(::OceanStationPapa) = (OSP_LONGITUDE, OSP_LONGITUDE)
latitude_interfaces(::OceanStationPapa)  = (OSP_LATITUDE, OSP_LATITUDE)

function z_interfaces(md::OSPMetadata)
    if is_three_dimensional(md)
        depths = _osp_depths(md.name)
        z_centers = sort(-depths)
        return centers_to_interfaces(z_centers)
    else
        return (-1.0, 0.0)
    end
end

z_interfaces(::OceanStationPapa) = (-1.0, 0.0)

function Base.size(::OceanStationPapa, variable)
    if variable in keys(OSP_PROFILE_VARIABLES)
        return (1, 1, length(_osp_depths(variable)))
    else
        return (1, 1, 1)
    end
end

# ## Reading one snapshot
#
# `retrieve_data` extracts a single time index and returns a CPU array of
# shape `(1, 1, Nz)` for profile variables or `(1, 1, 1)` for scalars. The
# OS-Papa QC flags follow the Argo convention (`1` = good, `2` = probably
# good, higher values are suspect). Suspect or missing entries are replaced
# with `NaN`, and the `StationColumn` vertical-interpolation routine skips
# them automatically downstream.

function retrieve_data(md::OSPMetadatum)
    filepath = metadata_path(md)
    ds = NCDataset(filepath)
    varname = dataset_variable_name(md)

    times = ds["TIME"][:]
    t = findfirst(==(md.dates), times)
    isnothing(t) && (close(ds); error("Date $(md.dates) not found in $filepath"))

    if is_three_dimensional(md)
        raw = ds[varname][1, 1, :, t]
        qc  = haskey(ds, varname * "_QC") ? ds[varname * "_QC"][1, 1, :, t] : nothing
        close(ds)
        return reshape(Float64.(_mask_qc(raw, qc)), 1, 1, :)
    else
        raw = ds[varname][1, 1, 1, t]
        qc  = haskey(ds, varname * "_QC") ? ds[varname * "_QC"][1, 1, 1, t] : nothing
        close(ds)
        value = _mask_qc([raw], qc === nothing ? nothing : [qc])
        return reshape(Float64.(value), 1, 1, 1)
    end
end

function _mask_qc(values, qc)
    out = [ismissing(v) ? NaN : Float64(v) for v in values]
    if !isnothing(qc)
        for i in eachindex(out)
            q = ismissing(qc[i]) ? Int8(9) : Int8(qc[i])
            q > 2 && (out[i] = NaN)
        end
    end
    return out
end

# `preprocess_data` is the generic extension point for data-cleanup that
# happens *after* `retrieve_data` returns and *before* the kernel that writes
# the native field. For OS Papa it is a no-op; the QC masking above is done
# in `retrieve_data` to keep reading and cleaning together. Kept as
# documentation of the hook's existence.

preprocess_data(data, ::OSPMetadatum) = data

# ## Conformance check
#
# At this point the dataset should satisfy the extension contract. The
# `test_dataset_contract` helper walks every required and optional method and
# produces a human-readable conformance report.

using NumericalEarth.DataWrangling: test_dataset_contract
report = test_dataset_contract(OceanStationPapa(); verbose=true)

# ## A time series of temperature profiles
#
# With the dataset type implemented, a multi-week temperature time series is
# a one-liner: a `Metadata` with a date range, combined with a target grid,
# builds a `FieldTimeSeries` whose backend lazily loads and regrids each
# hourly snapshot onto the user's column grid. It is possible to notice that
# nothing in the code below is station-specific: the exact same call pattern
# works for any `AbstractDataset`.

start_date = DateTime(2012, 10, 1)
end_date   = DateTime(2012, 10, 15)

md_series = Metadata(:temperature;
                     dataset = OceanStationPapa(),
                     start_date, end_date)

target_grid = RectilinearGrid(size = 40,
                              x = OSP_LONGITUDE, y = OSP_LATITUDE,
                              z = (-200, 0),
                              topology = (Flat, Flat, Bounded),
                              halo = (3,))

Tts = FieldTimeSeries(md_series, target_grid; time_indices_in_memory = length(md_series))
nothing #hide

# We now have `Tts`, a `FieldTimeSeries` covering two weeks of hourly
# temperature profiles, each regridded onto the 40-level target column.
# A Hovmöller diagram, that is, a time on the x-axis and depth on the
# y-axis contour, makes the evolution of the mixed layer and the thermocline
# immediately visible.

using CairoMakie

z = znodes(target_grid, Center())
t = Tts.times ./ 86400.0  # seconds → days since start_date
T = [Tts[n][1, 1, k] for k in 1:length(z), n in 1:length(t)]

fig = Figure(size = (900, 400))
ax = Axis(fig[1, 1];
            xlabel = "Days from $start_date",
            ylabel = "Depth (m)",
            title  = "Ocean Station Papa — temperature evolution")
hm = heatmap!(ax, t, z, T'; colormap = :thermal)
Colorbar(fig[1, 2], hm; label = "Temperature (°C)")
save(joinpath(@__DIR__, "ocean_station_papa_temperature.png"), fig)
fig

# It is possible to notice in the Hovmöller diagram a mixed layer of roughly
# 30 to 40 meters at the start of October 2012, a gradual cooling at the
# surface over the two-week window, and the thermocline located between 50
# and 100 meters, all consistent with the observed evolution at the mooring.
#
# ## Bonus: a `PrescribedAtmosphere` from buoy observations
#
# The tutorial ends with a helper that assembles a full `PrescribedAtmosphere`
# from the buoy surface variables. This is the pattern a companion package
# such as `NumericalEarthMoorings.jl` would expose as a public function. The
# implementation consists in loading each surface variable as a
# `FieldTimeSeries` on the surface-scalar grid, filling short observational
# gaps by linear interpolation via `fill_gaps!`, deriving specific humidity
# from relative humidity, pressure and temperature via
# `Thermodynamics.q_vap_from_RH`, and wrapping the result in a
# `PrescribedAtmosphere`.

function OceanStationPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                              start_date = DateTime(2012, 10, 1),
                                              end_date   = DateTime(2012, 10, 15),
                                              surface_layer_height = 2.5,
                                              max_gap_hours = 72)

    dataset = OceanStationPapa()
    surface_grid = RectilinearGrid(architecture, FT;
                                   size = (), topology = (Flat, Flat, Flat))

    function surface_fts(name)
        md = Metadata(name; dataset, start_date, end_date)
        fts = FieldTimeSeries(md, surface_grid;
                              time_indices_in_memory = length(md))
        fill_gaps!(fts; max_gap = max_gap_hours)
        return fts
    end

    ua   = surface_fts(:eastward_wind)
    va   = surface_fts(:northward_wind)
    Ta   = surface_fts(:air_temperature)     # K  (Celsius conversion applied)
    Pa   = surface_fts(:sea_level_pressure)  # Pa (Millibar conversion applied)
    swa  = surface_fts(:shortwave_radiation)
    lwa  = surface_fts(:longwave_radiation)
    rain = surface_fts(:rain)                # kg/m²/s (MillimetersPerHour conversion)
    RHa  = surface_fts(:relative_humidity)

    # Derive specific humidity from RH, P, T.
    thermo_params = AtmosphereThermodynamicsParameters(FT)
    LX, LY, LZ = location(Ta)
    qa = FieldTimeSeries{LX, LY, LZ}(Ta.grid, Ta.times)
    pqa, pPa, pTa, pRHa = parent(qa), parent(Pa), parent(Ta), parent(RHa)
    pqa .= q_vap_from_RH.(Ref(thermo_params), pPa, pTa, pRHa ./ 100, Ref(Liquid()))

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

# `atmos` is now a fully-formed `PrescribedAtmosphere` that can be passed to
# `OceanOnlyModel` or `AtmosphereOceanModel` exactly like any global
# reanalysis-derived atmosphere, with the single difference that the
# underlying data is a single-column observational record rather than a
# global reanalysis field.
