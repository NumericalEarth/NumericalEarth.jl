module Datasets

export AbstractDataset
export SpatialLayout, GriddedLatLon, StationColumn
export spatial_layout
export dataset_url, authenticate, download_file!, download_dataset
export preprocess_data
export conversion_units, convert_units, mangle
export Celsius, Kelvin, Millibar, InverseSign, MillimetersPerHour, CentimetersPerSecond
export GramPerKilogramMinus35, MilliliterPerLiter
export MolePerKilogram, MolePerLiter, MillimolePerKilogram, MillimolePerLiter
export MicromolePerKilogram, MicromolePerLiter, NanomolePerKilogram, NanomolePerLiter
export ShiftSouth, AverageNorthSouth

"""
    AbstractDataset

Supertype for every dataset type recognised by NumericalEarth's Metadata machinery.

Third-party packages define concrete dataset types by subtyping `AbstractDataset`
and implementing the methods listed in the developer guide. A minimum-viable
dataset implements `dataset_variable_name`, `all_dates`, `retrieve_data`, and
either a native-grid constructor or the three `*_interfaces` functions.
"""
abstract type AbstractDataset end

"""
    SpatialLayout

Supertype for the spatial-layout trait that describes how a dataset lives in space.
Concrete subtypes (`GriddedLatLon`, `StationColumn`) drive pipeline dispatch for
grid construction and field population.
"""
abstract type SpatialLayout end

"""
    GriddedLatLon()

Spatial-layout trait for datasets defined on a latitude-longitude grid
(global or regional with a bounding box). This is the default for
`AbstractDataset`.
"""
struct GriddedLatLon <: SpatialLayout end

"""
    StationColumn()

Spatial-layout trait for single-column station datasets (e.g. moorings, towers).
Datasets with this layout have a `RectilinearGrid{Flat, Flat, Bounded}` native
grid at one fixed `(longitude, latitude)` point.
"""
struct StationColumn <: SpatialLayout end

"""
    spatial_layout(dataset)

Return the [`SpatialLayout`](@ref) of `dataset`. Defaults to `GriddedLatLon()`
for any [`AbstractDataset`](@ref). Override this for station / column datasets.
"""
spatial_layout(::AbstractDataset) = GriddedLatLon()

#####
##### Download contract
#####

"""
    dataset_url(metadatum) -> Union{String, Nothing}

Return the URL (as `String`) from which the file for `metadatum` should be
downloaded, or `nothing` if the dataset does not expose a public URL. Called
by the default [`download_dataset`](@ref) orchestrator. Override this for any
dataset that has a one-file-per-(variable, date) public download.
"""
dataset_url(metadatum) = nothing

"""
    authenticate(dataset)

Hook invoked once before [`download_file!`](@ref). The default is a no-op.
Override for datasets that require credentials (e.g. ECCO netrc, CDS API
tokens) to stage them in the environment before the transport layer runs.
"""
authenticate(dataset) = nothing

"""
    download_file!(path, url, dataset)

Transport layer: download the file at `url` to `path` for `dataset`. The
default (defined in `DataWrangling`) calls `Downloads.download(url, path)`.
Override for custom transports such as WebDAV, S3 SDKs, or the Copernicus
Marine SDK.
"""
function download_file! end

"""
    download_dataset(metadatum) -> path

Orchestrator for the download step. The default composes [`authenticate`](@ref),
[`dataset_url`](@ref), and [`download_file!`](@ref) into a single per-file
download, and iterates over all dates for a multi-date `Metadata`. Override
only when the orchestration is unusual (bulk archives, parallel transports,
SDK-driven flows).
"""
function download_dataset end

"""
    preprocess_data(data, metadatum)

Transform the raw CPU array returned by `retrieve_data` before it enters the
native-field population step. Default is identity. Override for lightweight
cleanup such as QC-flag filtering or threshold masking that is not
representable as a scalar `conversion_units`.
"""
preprocess_data(data, metadatum) = data

#####
##### Unit-conversion contract
#####

# Unit tags advertise that a variable in a particular dataset is stored in a
# non-canonical unit system; `convert_units(x, tag)` performs the conversion
# to the canonical value. Both the tag set and the conversion function are
# open to extension: third-party datasets can add new tag types and register
# a `convert_units` method without touching core.

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

struct InverseSign end

struct GramPerKilogramMinus35 end    # Salinity anomaly
struct MilliliterPerLiter end        # Sometimes for dissolved oxygen
struct CentimetersPerSecond end
struct Millibar end                  # pressure in mbar (hPa) → Pa
struct MillimetersPerHour end        # liquid precipitation rate in mm/hr → kg/m²/s

"""
    conversion_units(metadatum) -> tag_or_nothing

Return the unit tag for `metadatum`, or `nothing` when no conversion is needed.
The returned tag is dispatched on by [`convert_units`](@ref) inside the field
population kernel. Default: `nothing`.
"""
conversion_units(metadatum) = nothing

"""
    convert_units(value, tag) -> converted_value

Apply the unit conversion identified by `tag` to a single scalar `value`.
Default: identity for unrecognised tags. Extension datasets define new tags
and add `convert_units` methods to register conversions.
"""
@inline convert_units(value, tag) = value

#####
##### Data-mangling contract
#####

# `mangle` performs index-space transforms of raw data arrays inside the
# field population kernel. The `Nothing` tag is the default (no mangling).
# Shipped transforms cover off-by-one grid staggering (`ShiftSouth`) and
# averaging between adjacent rows (`AverageNorthSouth`); extension datasets
# can register new tags.

struct ShiftSouth end
struct AverageNorthSouth end

"""
    mangle(i, j, data, tag)
    mangle(i, j, k, data, tag)

Return the raw-array value at index `(i, j[, k])` transformed by `tag`. The
`Nothing` tag returns the value unchanged. Extension datasets add methods on
`mangle` to register new tag types.
"""
@inline mangle(i, j,    data, ::Nothing) = @inbounds data[i, j]
@inline mangle(i, j, k, data, ::Nothing) = @inbounds data[i, j, k]

end # module Datasets
