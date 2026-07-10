using CFTime: AbstractCFDateTime, CFTime
using Dates: Dates, Date, DateTime
using Base: @propagate_inbounds

struct BoundingBox{X, Y, Z}
    longitude :: X
    latitude :: Y
    z :: Z
end

latitude_summary(::Nothing) = "latitude=nothing"
longitude_summary(::Nothing) = "longitude=nothing"
latitude_summary(lat) = string("latitude=(", prettysummary(lat[1]), ", ", prettysummary(lat[2]), ")")
longitude_summary(lon) = string("longitude=(", prettysummary(lon[1]), ", ", prettysummary(lon[2]), ")")
Base.summary(bbox::BoundingBox) = string("BoundingBox(",
                                         longitude_summary(bbox.longitude), ", ",
                                         latitude_summary(bbox.latitude), ")")

"""
    BoundingBox(; longitude=nothing, latitude=nothing, z=nothing)

Create a bounding box with `latitude`, `longitude`, and `z` bounds on the sphere.
A non-`nothing` `z = (z_bottom, z_top)` restricts both the download (for backends
that support it, e.g. CopernicusMarine/GLORYS) and the vertical extent of the
native grid built by [`native_grid`](@ref).
"""
BoundingBox(; longitude=nothing, latitude=nothing, z=nothing) =
    BoundingBox(longitude, latitude, z)

#####
##### Column region and interpolation types
#####

struct Linear end
struct Nearest end

"""
    Column(longitude, latitude; z=nothing, interpolation=Linear())

Create a column region at a single horizontal point `(longitude, latitude)`.
When used as a `Metadata` region, `native_grid` returns a single-column
`RectilinearGrid` and `location` reduces horizontal dimensions to `Nothing`.

Keyword Arguments
=================

- `z`: depth range tuple `(z_bottom, z_top)` that restricts both the download
  (used by CopernicusMarine/GLORYS) and the vertical extent of the native grid.
  Default: `nothing` (full depth).
- `interpolation`: method for extracting data from the surrounding grid
  cells. `Linear()` (default) bilinearly interpolates to the exact point;
  `Nearest()` selects the closest grid cell.
"""
struct Column{X, Y, Z, I}
    longitude :: X
    latitude :: Y
    z :: Z
    interpolation :: I
end

Column(longitude, latitude; z=nothing, interpolation=Linear()) =
    Column(longitude, latitude, z, interpolation)

Base.summary(col::Column) = string("Column(longitude=", prettysummary(col.longitude),
                                   ", latitude=", prettysummary(col.latitude), ")")

struct DatewiseFilename{A}
    filenames :: A
end

Base.getindex(f::DatewiseFilename, i::Int) = f.filenames[i]
Base.length(f::DatewiseFilename) = length(f.filenames)
Base.unique(f::DatewiseFilename) = unique(f.filenames)

getfilename(f::DatewiseFilename, i) = f.filenames[i]
getfilename(f::String, i) = f
getfilename(::Nothing, i) = nothing

struct Metadata{V, D, R, S, F}
    name :: S
    dataset :: V
    dates :: D
    region :: R
    dir :: String
    filename :: F
end

Metadata(name, dataset, dates, region, dir) = Metadata(name, dataset, dates, region, dir, nothing)

is_three_dimensional(::Metadata) = true
z_interfaces(md::Metadata) = z_interfaces(md.dataset)

# NetCDF coordinate-variable names. Default follows CF standard; datasets
# whose files use different names (e.g. JRA55 uses `lon`/`lat`) override.
longitude_name(::Metadata) = "longitude"
latitude_name(::Metadata)  = "latitude"
longitude_interfaces(md::Metadata) = longitude_interfaces(md.dataset)
latitude_interfaces(md::Metadata) = latitude_interfaces(md.dataset)

"""
    Metadata(variable_name;
             dataset,
             dates = all_dates(dataset, variable_name),
             dir = default_download_directory(dataset),
             region = nothing,
             filename = nothing,
             start_date = nothing,
             end_date = nothing)

Metadata holding a specific dataset information.

Argument
========
- `variable_name`: a symbol representing the name of the variable (for example, `:temperature`,
                   `:salinity`, `:u_velocity`, etc)

Keyword Arguments
=================

- `dataset`: Supported datasets are returned by [`supported_datasets`](@ref).

- `dates`: The dates of the dataset (`Dates.AbstractDateTime` or `CFTime.AbstractCFDateTime`).
           Note that `dates` can either be a range or a vector of dates, representing a time-series.
           For a single date, use [`Metadatum`](@ref).

- `start_date`: If `dates = nothing`, we can prescribe the first date of metadata as a date
                (`Dates.AbstractDateTime` or `CFTime.AbstractCFDateTime`). If outside the
                date range of the dataset, the first allowable date is chosen. Default: nothing.

- `end_date`: If `dates = nothing`, we can prescribe the last date of metadata as a date
              (`Dates.AbstractDateTime` or `CFTime.AbstractCFDateTime`). If outside the
                date range of the dataset, the last allowable date is chosen. Default: nothing.

- `region`: Specifies the spatial region of the dataset. Can be a [`BoundingBox`](@ref)
            for a rectangular region, a [`Column`](@ref) for a single horizontal location,
            or `nothing` for the full domain.

- `filename`: The filename(s) for the dataset. If `nothing`, the filename is computed from
              the dataset type. Can be a `String` (single file for all dates) or a
              `DatewiseFilename` (one file per date).

- `dir`: The directory where the dataset is stored.
"""
function Metadata(variable_name;
                  dataset,
                  dates = all_dates(dataset, variable_name),
                  dir = default_download_directory(dataset),
                  region = nothing,
                  filename = nothing,
                  start_date = nothing,
                  end_date = nothing)

    # crop dates if _either_ a start date or an end date is provided
    if !isnothing(start_date) || !isnothing(end_date)

        # If one of the two is nothing, take the native limits
        start_date = isnothing(start_date) ? dates[1]   : start_date
        end_date   = isnothing(end_date)   ? dates[end] : end_date

        # Crop the dates to fit start_date and end_date
        dates = compute_native_date_range(dates, start_date, end_date)
    end

    if isnothing(filename)
        filename = build_filename(dataset, variable_name, dates, region)
    end

    return Metadata(variable_name, dataset, dates, region, dir, filename)
end

const AnyDateTime  = Union{AbstractCFDateTime, Dates.AbstractDateTime}
const Metadatum{V} = Metadata{V, <:Union{AnyDateTime, Nothing}} where V

function Base.size(metadata::Metadata)
    Nx, Ny, Nz = size(metadata.dataset, metadata.name)

    if metadata.dates isa AbstractArray
        Nt = length(metadata.dates)
    else
        Nt = 1
    end
    return (Nx, Ny, Nz, Nt)
end

"""
    Metadatum(variable_name;
              dataset,
              region = nothing,
              date = first_date(dataset, variable_name),
              filename = nothing,
              dir = default_download_directory(dataset))

A specialized constructor for a [`Metadata`](@ref) object with a single date, representative of a snapshot in time.
"""
function Metadatum(variable_name;
                   dataset,
                   region = nothing,
                   date = first_date(dataset, variable_name),
                   filename = nothing,
                   dir = default_download_directory(dataset))

    if date isa Date
        date = DateTime(date)
    end

    if !isnothing(date) && !(date isa AnyDateTime)
        msg = "`date` must be `nothing`, a `Dates.AbstractDateTime`, or `CFTime.AbstractCFDateTime`, received $(typeof(date))"
        throw(ArgumentError(msg))
    end

    if isnothing(filename)
        filename = metadata_filename(dataset, variable_name, date, region)
    end

    return Metadata(variable_name, dataset, date, region, dir, filename)
end

datestr(md::Metadata) = string(first(md.dates), "--", last(md.dates))
datestr(md::Metadatum) = string(md.dates)
datasetstr(md::Metadata) = string(md.dataset)
metaprefix(md::Metadata) = string("Metadata{", md.dataset, "}")

Oceananigans.Utils.prettysummary(dt::DateTime) = Dates.format(dt, "yyyy-mm-dd HH:MM:SS")

function Base.show(io::IO, metadata::Metadata)
    V = typeof(metadata.dataset)
    D = typeof(metadata.dates)

    name = if metadata isa Metadatum
        "Metadatum"
    else
        "Metadata"
    end

    print(io, "$name{$V, $D}:", '\n',
    "├── name: $(metadata.name)", '\n',
    "├── dataset: ", prettysummary(metadata.dataset), '\n',
    "├── dates: ", prettysummary(metadata.dates), '\n')

    rgn = metadata.region
    if !isnothing(rgn)
        print(io, "├── region: ", summary(rgn), '\n')
    end

    print(io, "├── filename: $(metadata.filename)", '\n')
    print(io, "└── dir: $(metadata.dir)")
end

# Treat Metadata as an array to allow iteration over the dates.
Base.length(metadata::Metadata) = isnothing(metadata.dates) ? 1 : length(metadata.dates)
Base.eltype(metadata::Metadata) = Float32

Base.summary(md::Metadata) = string(metaprefix(md),
                                    "{", datasetstr(md), "} of ",
                                    md.name, " for ", datestr(md))

# If only one date, it's a single element array
Base.length(metadata::Metadatum) = 1

@propagate_inbounds Base.getindex(m::Metadata, i::Int) =
    Metadata(m.name, m.dataset, m.dates[i], m.region, m.dir, getfilename(m.filename, i))

@propagate_inbounds Base.first(m::Metadata) =
    Metadata(m.name, m.dataset, m.dates[1], m.region, m.dir, getfilename(m.filename, 1))

@propagate_inbounds Base.last(m::Metadata) =
    Metadata(m.name, m.dataset, m.dates[end], m.region, m.dir, getfilename(m.filename, lastindex(m.dates)))

@inline function Base.iterate(m::Metadata, i=1)
   if (i % UInt) - 1 < length(m)
        return Metadata(m.name, m.dataset, m.dates[i], m.region, m.dir, getfilename(m.filename, i)), i + 1
    else
        return nothing
    end
end

# Implementation for 1 date
Base.axes(metadata::Metadatum)    = 1
Base.first(metadata::Metadatum)   = metadata
Base.last(metadata::Metadatum)    = metadata
Base.iterate(metadata::Metadatum) = (metadata, nothing)
Base.iterate(::Metadatum, ::Any)  = nothing

metadata_path(metadata::Metadatum) = joinpath(metadata.dir, metadata.filename)

function metadata_path(metadata::Metadata)
    fn = metadata.filename
    if fn isa DatewiseFilename
        return [joinpath(metadata.dir, f) for f in fn.filenames]
    else
        # Single filename (String) — one file for all dates
        return joinpath(metadata.dir, fn)
    end
end

#####
##### MetadataSet — a bundle of `Metadata` sharing dataset, dates, region, and dir.
#####
##### An `mset::MetadataSet` is keyed by *verbose* dataset variable names. Iteration is
##### over variables (orthogonal to `Metadata`'s date-axis iteration); every
##### element returned by `mset[name]` / `mset[i]` is itself a `Metadata` or
##### `Metadatum`, so all existing per-`Metadata` machinery (`Field`, `set!`,
##### `download`, ...) keeps working unchanged on the elements.
#####

struct MetadataSet{V, D, R, N, F}
    names :: N      # NTuple{K, Symbol} — verbose dataset variable names
    dataset :: V    # shared
    dates :: D      # shared; scalar or AbstractVector
    region :: R     # shared
    dir :: String   # shared
    filenames :: F  # NamedTuple keyed by `names`, one entry per variable
end

"""
    MetadataSet(variable_names::Symbol...;
                dataset,
                dates = all_dates(dataset, first(variable_names)),
                date = nothing,
                dir = default_download_directory(dataset),
                region = nothing,
                filenames = nothing,
                start_date = nothing,
                end_date = nothing)

A bundle of [`Metadata`](@ref) for many variables that share `dataset`, `dates`,
`region`, and `dir` — differing only in variable name.

Each element of an `mset::MetadataSet`, e.g., `mset[name]` (or equivalently `mset.name` or
`mset[i]`) is itself a `Metadata` — or a `Metadatum` when `dates` is a single date.
Iteration walks the variable axis, yielding one `Metadata` per variable.

Arguments
=========
- `variable_names`: one or more `Symbol`s naming the dataset variables to bundle
  (e.g. `:temperature, :salinity`). Verbose dataset-internal names — no aliases.
  It can also be a tuple of Symbols, e.g., `(:temperature, :salinity)`, but not a vector of Symbols.

Keyword Arguments
=================
- `dataset`: The shared dataset. Supported datasets are returned by [`supported_datasets`](@ref).
- `dates`: Shared date axis. Either a single `AbstractDateTime`/`AbstractCFDateTime`
           or an `AbstractVector` of dates. Defaults to `all_dates(dataset, first(variable_names))`.
- `date`: Convenience scalar form; cannot be used together with `dates`.
- `region`: Shared spatial region — `BoundingBox`, `Column`, or `nothing`.
- `dir`: Shared download directory.
- `filenames`: An optional `NamedTuple` keyed by `variable_names` overriding the
               auto-computed per-variable filenames.
- `start_date`, `end_date`: Optional date cropping, matching [`Metadata`](@ref).

Example
=======

```jldoctest
using NumericalEarth, Dates

mset = MetadataSet(:temperature, :salinity;
                   dataset = ECCO4Monthly(),
                   date = DateTime(1995, 1, 1))

mset[2] # Metadata for :salinity

using NumericalEarth, Dates

mset = MetadataSet(:temperature, :salinity;
                   dataset = ECCO4Monthly(),
                   date = DateTime(1995, 1, 1))

mset[2] # Metadata for :salinity

# output
Metadatum{ECCO4Monthly, DateTime}:
├── name: salinity
├── dataset: ECCO4Monthly
├── dates: 1995-01-01 00:00:00
├── filename: SALT_1995_01.nc
└── dir: /.julia/scratchspaces/904d977b-046a-4731-8b86-9235c0d1ef02/ECCO/v4
```

See also [`Metadata`](@ref), [`Metadatum`](@ref).
"""
function MetadataSet(variable_names::Symbol...;
                     dataset,
                     dates = nothing,
                     date = nothing,
                     dir = default_download_directory(dataset),
                     region = nothing,
                     filenames = nothing,
                     start_date = nothing,
                     end_date = nothing)

    isempty(variable_names) &&
        throw(ArgumentError("MetadataSet requires at least one variable name"))

    if !isnothing(date) && !isnothing(dates)
        throw(ArgumentError("Specify either `date` (scalar) or `dates` (vector), not both"))
    end

    # Resolve the effective date axis.
    effective_dates = if !isnothing(date)
        date isa Date ? DateTime(date) : date
    elseif !isnothing(dates)
        dates
    else
        all_dates(dataset, first(variable_names))
    end

    if !isnothing(date) && !(effective_dates isa AnyDateTime)
        msg = "`date` must be a `Dates.AbstractDateTime` or `CFTime.AbstractCFDateTime`, received $(typeof(date))"
        throw(ArgumentError(msg))
    end

    # Optional date cropping (parallels Metadata).
    if !isnothing(start_date) || !isnothing(end_date)
        effective_dates isa AnyDateTime &&
            throw(ArgumentError("`start_date`/`end_date` are not compatible with a scalar `date`"))
        sd = isnothing(start_date) ? effective_dates[1]   : start_date
        ed = isnothing(end_date)   ? effective_dates[end] : end_date
        effective_dates = compute_native_date_range(effective_dates, sd, ed)
    end

    # Auto-build per-variable filenames if not supplied.
    if isnothing(filenames)
        filename_values = map(n -> build_filename(dataset, n, effective_dates, region), variable_names)
        filenames = NamedTuple{variable_names}(filename_values)
    else
        filenames isa NamedTuple ||
            throw(ArgumentError("`filenames` must be a NamedTuple keyed by variable names"))
        keys(filenames) === variable_names ||
            throw(ArgumentError("`filenames` keys $(keys(filenames)) must match variable names $variable_names"))
    end

    return MetadataSet(variable_names, dataset, effective_dates, region, dir, filenames)
end

MetadataSet(names::NTuple{<:Any, <:Symbol}; kw...) = MetadataSet(names...; kw...)

# Property access: variables first via filenames lookup, struct fields second.
function Base.getproperty(mset::MetadataSet, name::Symbol)
    if name in fieldnames(MetadataSet)
        return getfield(mset, name)
    elseif name in getfield(mset, :names)
        return getindex(mset, name)
    else
        throw(KeyError(name))
    end
end

Base.propertynames(mset::MetadataSet) =
    (getfield(mset, :names)..., fieldnames(MetadataSet)...)

# Indexed access. We use `getfield` here so subsequent edits to `getproperty`
# can't make these recursive.
function Base.getindex(mset::MetadataSet, name::Symbol)
    fname = getfield(mset, :filenames)[name]
    return Metadata(name,
                    getfield(mset, :dataset),
                    getfield(mset, :dates),
                    getfield(mset, :region),
                    getfield(mset, :dir),
                    fname)
end

@propagate_inbounds Base.getindex(mset::MetadataSet, i::Int) =
    getindex(mset, getfield(mset, :names)[i])

Base.length(mset::MetadataSet) = length(getfield(mset, :names))
Base.keys(mset::MetadataSet)   = getfield(mset, :names)
Base.eltype(::Type{<:MetadataSet}) = Metadata
Base.firstindex(::MetadataSet) = 1
Base.lastindex(mset::MetadataSet) = length(mset)

@inline function Base.iterate(mset::MetadataSet, state::Int=1)
    state > length(mset) && return nothing
    return mset[state], state + 1
end

Base.NamedTuple(mset::MetadataSet) =
    NamedTuple{getfield(mset, :names)}(map(n -> mset[n], getfield(mset, :names)))

"""
    metadata_path(mset::MetadataSet)

Return a `NamedTuple` keyed by the set's variable names whose values are the
file paths of each variable's `Metadata` (a `String` for single-date sets,
a `Vector{String}` for multi-date sets — matching `metadata_path(::Metadata)`).
"""
function metadata_path(mset::MetadataSet)
    names = getfield(mset, :names)
    return NamedTuple{names}(map(n -> metadata_path(mset[n]), names))
end

function Base.show(io::IO, mset::MetadataSet)
    V = typeof(getfield(mset, :dataset))
    D = typeof(getfield(mset, :dates))

    print(io, "MetadataSet{$V, $D}:", '\n',
          "├── names: ", getfield(mset, :names), '\n',
          "├── dataset: ", prettysummary(getfield(mset, :dataset)), '\n',
          "├── dates: ", prettysummary(getfield(mset, :dates)), '\n')

    rgn = getfield(mset, :region)
    if !isnothing(rgn)
        print(io, "├── region: ", summary(rgn), '\n')
    end

    print(io, "└── dir: $(getfield(mset, :dir))")
end

Base.summary(mset::MetadataSet) =
    string("MetadataSet{", typeof(getfield(mset, :dataset)), "} of ",
           length(mset), " variables")

"""
    set!(fields::NamedTuple, mset::MetadataSet)

Set each `fields[name]` from the corresponding `mset[name]`. The NamedTuple's
keys must be a subset of the set's variable names; extra fields are ignored.

This is the explicit form that takes verbose dataset names on both sides — no
glossary translation. For the auto-routing form (short model field-names), see
`set!(model, ::MetadataSet)`.
"""
function Fields.set!(fields::NamedTuple, mset::MetadataSet)
    for name in keys(fields)
        in(name, getfield(mset, :names)) ||
            throw(ArgumentError("Field $(name) is not in MetadataSet variables $(getfield(mset, :names))"))
        set!(fields[name], mset[name])
    end
    return fields
end

"""
    set!(model, mset::MetadataSet, names=keys(variable_glossary))

Route variables from `mset` to `set!(model; kwargs...)`, translating verbose
dataset names to short model field-names via [`variable_glossary`](@ref).
Only the intersection of `names` and `mset.names` is forwarded.

The default `names` is every glossary key — fine for permissive models. Models
that throw on unknown kwargs (`HydrostaticFreeSurfaceModel`, `SeaIceModel`)
override the 2-argument form to pass a narrower `names`, letting a single
multi-component MetadataSet drive both an ocean and a sea-ice model.
Each model's override of `set!(model, ::MetadataSet)` filters the same `mset`
down to the variables that model knows how to consume:

```jldoctest
using NumericalEarth
using Oceananigans
using Statistics
using Dates

grid = LatitudeLongitudeGrid(size = (60, 30, 5),
                             longitude = (-180, 180),
                             latitude  = (-60, 60),
                             z = (-5000, 0),
                             halo = (7, 7, 7))

ocean = ocean_simulation(grid)

mset = MetadataSet(:temperature, :salinity;
                   dataset = ECCO4Monthly(),
                   date    = DateTime(1993, 1, 1))

# Ocean override routes :temperature → T, :salinity → S; sea-ice vars are
# filtered out. A `set!(sea_ice.model, mset)` call against the same `mset`
# would route :sea_ice_thickness → h, :sea_ice_concentration → ℵ.
set!(ocean.model, mset)

T = ocean.model.tracers.T
(min = round(minimum(T), digits = 2),
 max = round(maximum(T), digits = 2),
 mean = round(mean(T), digits = 2))

# output
(min = -1.06, max = 21.41, mean = 3.3)
```
"""
function Fields.set!(model, mset::MetadataSet, names=keys(variable_glossary))
    routed = filter(in(names), getfield(mset, :names))
    isempty(routed) && return model
    kwargs = NamedTuple{Tuple(variable_glossary[n] for n in routed)}(Tuple(mset[n] for n in routed))
    return set!(model; kwargs...)
end

# Ocean: route only variables whose short name appears in velocities,
# tracers, or free_surface — the three places HydrostaticFreeSurfaceModel's
# `set!` looks up kwargs.
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
function Fields.set!(model::HydrostaticFreeSurfaceModel, mset::MetadataSet)
    hfsm_short_names = (propertynames(model.velocities)..., propertynames(model.tracers)..., :η)
    valid_long_names = Tuple(long_name for (long_name, short_name) in variable_glossary if short_name in hfsm_short_names)
    return set!(model, mset, valid_long_names)
end

# Sea ice: ClimaSeaIce's `set!(::SeaIceModel; h, ℵ)` only accepts these two.
using ClimaSeaIce: SeaIceModel
function Fields.set!(model::SeaIceModel, mset::MetadataSet)
    return set!(model, mset, (:sea_ice_thickness, :sea_ice_concentration))
end

"""
    download(mset::MetadataSet; kwargs...)

Download every variable in `mset`. The default is a per-variable loop calling
`download(mset[name]; kwargs...)`; backends that support batched multi-variable
requests (e.g. the ERA5 pressure-level CDS path) override this to route through
a single batched call.

Returns a `NamedTuple` keyed by the set's variable names, whose values are the
results of each per-variable `download` call (typically the file path(s)).
"""
function Downloads.download(mset::MetadataSet; kwargs...)
    names = getfield(mset, :names)
    return NamedTuple{names}(map(n -> Downloads.download(mset[n]; kwargs...), names))
end

"""
    native_times(metadata; start_time=first(metadata).dates)

Extract the time values from the given `metadata`, calculate the time difference
from the `start_time`, and return an array of time differences in seconds.

Argument
========
- `metadata`: The metadata containing the date information.

Keyword Argument
================
- `start_time`: The start time for calculating the time difference. Defaults to the first
                date in the metadata.
"""
function native_times(metadata; start_time=first(metadata).dates)
    times = zeros(length(metadata))
    for (t, data) in enumerate(metadata)
        date = data.dates
        delta = date - start_time
        delta = Second(delta).value
        times[t] = delta
    end

    return times
end

####
#### Metadata interface
####

"""
    default_download_directory(dataset)

Return the default directory to which `dataset` is downloaded.
"""
function default_download_directory end

"""
    dataset_variable_name(metadata)

Return the name used for the variable `metadata.name` in its raw dataset file.
"""
function dataset_variable_name end

"""
    validate_dataset_coverage(grid, metadata)

Check that `grid` lies within the spatial coverage of `metadata`'s dataset.
Throws an error if the grid extends outside the dataset's domain.
The default implementation does nothing (all grids are accepted).
Dataset-specific methods can override this to enforce coverage constraints.
"""
validate_dataset_coverage(grid, metadata) = nothing

"""
    dataset_location(dataset, variable_name)

Return the native field location `(LX, LY, LZ)` for `variable_name` in
`dataset`. Defaults to `(Center, Center, Center)`. Only datasets with
staggered variables (e.g., ECCO velocity fields) need to extend this.
"""
dataset_location(dataset, variable_name) = (Center, Center, Center)

# Note: all_dates needs to be extended for any new dataset.
"""
    all_dates(metadata)

Extract all the dates of the given `metadata` formatted using the `DateTime` type.
"""
all_dates(metadata) = all_dates(metadata.dataset, metadata.name)

"""
    first_date(dataset, variable_name)

Extracts the first date of the given `dataset` and variable name formatted using the `DateTime` type.
"""
first_date(dataset, variable_name) = first(all_dates(dataset, variable_name))

"""
    last_date(dataset, variable_name)

Extracts the last date of the given dataset and variable name formatted using the `DateTime` type.
"""
last_date(dataset, variable_name) = last(all_dates(dataset, variable_name))

"""
    metadata_filename(metadata)

Return the stored filename(s) of `metadata`.
"""
metadata_filename(metadata::Metadata) = metadata.filename

"""
    metadata_filename(dataset, name, date, region)

Compute the filename for a single date. Extended by each dataset module.
"""
function metadata_filename end

"""
    metadata_url(metadata)

Return the URL for the dataset described by `metadata`. Extended by each dataset module.
"""
function metadata_url end

# Internal: build filename for construction.
# Single date: delegate to metadata_filename
build_filename(dataset, name, date, region) =
    metadata_filename(dataset, name, date, region)

# Multi-date: one filename per date, wrapped in DatewiseFilename
build_filename(dataset, name, dates::AbstractArray, region) =
    DatewiseFilename([metadata_filename(dataset, name, d, region) for d in dates])

"""
    available_variables(metadata)

Return the available variables in the dataset.
"""
available_variables(metadata) = available_variables(metadata.dataset)

struct Celsius end
struct Kelvin end

struct MolePerKilogram end
struct MolePerLiter end
struct MillimolePerKilogram end
struct MillimolePerLiter end
struct MicromolePerKilogram end
struct MicromolePerLiter end
struct NanomolePerKilogram end
struct NanomolePerLiter end
struct CentigramPerCubicCentimeter end
struct HectogramPerCubicMeter end
struct GramPerKilogram end
struct DecigramPerKilogram end

struct InverseSign end
struct InverseGravity end

struct GramPerKilogramMinus35 end # Salinity anomaly
struct MilliliterPerLiter end # Sometimes for disssolved_oxygen
struct CentimetersPerSecond end
struct Millibar end               # pressure in mbar (hPa) → Pa
struct MillimetersPerHour end     # liquid precipitation rate in mm/hr → kg/m²/s
struct MetersPerHour end          # liquid precipitation depth in m/hr → kg/m²/s (ERA5 total_precipitation)
struct JoulesPerSquareMeterPerHour end # radiative energy accumulated over 1 hr, J/m² → mean flux W/m² (ERA5 ssrd/strd)

"""
    conversion_units(metadatum)

Return the units of the source variable in the given dataset referenced by `metadatum`.
These units will be used to apply automatic conversions to standard units for `NumericalEarth`.
"""
conversion_units(metadatum) = nothing

"""
    missing_value(metadatum)

Return the value used by the underlying dataset to represent missing data. Defaults to `missing`.
"""
missing_value(metadatum) = missing

#####
##### Utilities
#####

"""
    compute_native_date_range(native_dates, start_date, end_date)

Compute the range of `native_dates` that fall within the specified `start_date` and `end_date`.
"""
comparable_datetime(date::Dates.AbstractDateTime) = DateTime(date)
comparable_datetime(date::AbstractCFDateTime) = DateTime(date)

function compute_native_date_range(native_dates, start_date, end_date)
    start_datetime = comparable_datetime(start_date)
    end_datetime = comparable_datetime(end_date)
    first_native_datetime = comparable_datetime(first(native_dates))
    last_native_datetime = comparable_datetime(last(native_dates))

    if last_native_datetime < end_datetime
        @warn "`end_date` ($end_date) is after the last date in the dataset $last_native_datetime"
    end

    if start_datetime < first_native_datetime
       @warn "`start_date` ($start_date) is before the first date in the dataset $first_native_datetime"
    end

    if end_datetime < start_datetime
       @warn "`end_date` ($end_date) is before the `start_date` ($start_date)"
    end

    if start_datetime < first_native_datetime && end_datetime < first_native_datetime
        throw(ArgumentError("both `start_date` ($start_date) and `end_date` ($end_date) are before the first date in the dataset $first_native_datetime"))
    end

    if last_native_datetime < start_datetime && last_native_datetime < end_datetime
        throw(ArgumentError("both `start_date` ($start_date) and `end_date` ($end_date) are after the last date in the dataset $last_native_datetime"))
    end

    start_idx = findfirst(x -> comparable_datetime(x) ≥ start_datetime, native_dates)
    end_idx   = findfirst(x -> comparable_datetime(x) ≥ end_datetime, native_dates)
    start_idx = (start_idx > 1 && comparable_datetime(native_dates[start_idx]) > start_datetime) ? start_idx - 1 : start_idx
    end_idx   = isnothing(end_idx) ? length(native_dates) : end_idx

    return native_dates[start_idx:end_idx]
end
