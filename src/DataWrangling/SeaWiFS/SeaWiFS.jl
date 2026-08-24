module SeaWiFS

export SeaWiFSMonthly, SeaWiFSMetadata, SeaWiFSMetadatum

using Dates: Dates, DateTime, Month
using Downloads: Downloads
using NCDatasets: Dataset
using Oceananigans.DistributedComputations: @root
using Oceananigans.Fields: Center

using ...NumericalEarth: NumericalEarth
using ..DataWrangling: DataWrangling, Metadata, Metadatum, metadata_filename, metadata_path,
                       dataset_variable_name

download_SeaWiFS_cache::String = ""
function __init__()
    global download_SeaWiFS_cache = DataWrangling.download_cache("SeaWiFS")
    return nothing
end

"""
    SeaWiFSMonthly(; resolution = 1)

Monthly composites of near-surface chlorophyll-a from SeaWiFS, reprocessing R2018.0, covering the
mission record September 1997 to December 2010. Provides `:chlorophyll` in mg m⁻³, the input
`TwoColorRadiation` needs to set how deep shortwave radiation penetrates.

The native product is mapped at 1/12°; `resolution` is the degree spacing actually requested, since
a global monthly record at full resolution is several gigabytes. Files come from the NOAA CoastWatch
ERDDAP server, which needs no credentials.

Pair a twelve-month window with the default `Cyclical()` time indexing to drive a run of any length
with a repeating seasonal cycle:

```julia
dates = (DateTime(2000, 1, 1), DateTime(2000, 12, 1))
chlorophyll = FieldTimeSeries(Metadata(:chlorophyll; dataset=SeaWiFSMonthly(), dates), grid)
radiative_forcing = TwoColorRadiation(grid; chlorophyll)
```

```jldoctest
using NumericalEarth.DataWrangling.SeaWiFS

SeaWiFSMonthly()

# output

SeaWiFSMonthly chlorophyll at 1° resolution
```
"""
struct SeaWiFSMonthly
    resolution :: Int
end

SeaWiFSMonthly(; resolution = 1) = SeaWiFSMonthly(resolution)

Base.show(io::IO, dataset::SeaWiFSMonthly) =
    print(io, "SeaWiFSMonthly chlorophyll at ", dataset.resolution, "° resolution")

const SeaWiFSMetadata{D} = Metadata{<:SeaWiFSMonthly, D}
const SeaWiFSMetadatum   = Metadatum{<:SeaWiFSMonthly}

const SeaWiFS_variable_names = Dict(:chlorophyll => "chlorophyll")

# Product identity: where the record is served from, at what spacing, and over what period.
native_horizontal_resolution(::SeaWiFSMonthly) = 1/12
erddap_server(::SeaWiFSMonthly) = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
erddap_dataset_identifier(::SeaWiFSMonthly) = "erdSW2018chlamday"
first_available_date(::SeaWiFSMonthly) = DateTime(1997, 9, 1)
last_available_date(::SeaWiFSMonthly) = DateTime(2010, 12, 1)

DataWrangling.default_download_directory(::SeaWiFSMonthly) = download_SeaWiFS_cache

DataWrangling.all_dates(dataset::SeaWiFSMonthly, args...) =
    first_available_date(dataset) : Month(1) : last_available_date(dataset)

DataWrangling.available_variables(::SeaWiFSMonthly) = SeaWiFS_variable_names
DataWrangling.dataset_variable_name(metadata::SeaWiFSMetadata) = SeaWiFS_variable_names[metadata.name]
DataWrangling.dataset_location(::SeaWiFSMonthly, name) = (Center, Center, Nothing)
DataWrangling.is_three_dimensional(::SeaWiFSMetadata) = false
DataWrangling.longitude_name(::SeaWiFSMetadata) = "longitude"
DataWrangling.latitude_name(::SeaWiFSMetadata) = "latitude"
DataWrangling.longitude_interfaces(::SeaWiFSMetadata) = (-180, 180)
DataWrangling.latitude_interfaces(::SeaWiFSMetadata) = (-90, 90)
DataWrangling.metaprefix(::SeaWiFSMetadata) = "SeaWiFSMetadata"
DataWrangling.metaprefix(::SeaWiFSMetadatum) = "SeaWiFSMetadatum"

Base.size(dataset::SeaWiFSMonthly, variable) =
    (round(Int, 360 / dataset.resolution), round(Int, 180 / dataset.resolution), 1)

DataWrangling.metadata_filename(dataset::SeaWiFSMonthly, name, date, region) =
    string("SeaWiFS_", name, "_", dataset.resolution, "deg_", Dates.format(date, "yyyy-mm"), ".nc")

# Cloud gaps are inpainted once and cached beside the download, as every other dataset does.
DataWrangling.inpainted_metadata_path(metadatum::SeaWiFSMetadatum) =
    joinpath(metadatum.dir, replace(metadata_filename(metadatum), ".nc" => "_inpainted.jld2"))

# ERDDAP subsets server side: a value range picks the single composite inside the month, and the
# index strides thin the native 1/12° grid to the requested resolution.
function erddap_url(metadatum::SeaWiFSMetadatum)
    dataset = metadatum.dataset
    stride = round(Int, dataset.resolution / native_horizontal_resolution(dataset))
    Nx = round(Int, 360 / native_horizontal_resolution(dataset))
    Ny = round(Int, 180 / native_horizontal_resolution(dataset))
    first_day = Dates.format(metadatum.dates, "yyyy-mm-01")
    last_day = Dates.format(Dates.lastdayofmonth(metadatum.dates), "yyyy-mm-dd")

    return string(erddap_server(dataset), "/", erddap_dataset_identifier(dataset), ".nc?",
                  dataset_variable_name(metadatum),
                  "%5B($first_day):1:($last_day)%5D",
                  "%5B0:", stride, ":", Ny - 1, "%5D",
                  "%5B0:", stride, ":", Nx - 1, "%5D")
end

# Datasets define `download` on the whole `Metadata` and loop over its elements; a single
# `Metadatum` is a `Metadata` with one date, so this covers both.
function Downloads.download(metadata::SeaWiFSMetadata; kwargs...)
    for metadatum in metadata
        path = metadata_path(metadatum)
        isfile(path) && continue
        @root begin
            @info "Downloading SeaWiFS chlorophyll for $(Dates.format(metadatum.dates, "yyyy-mm"))"
            Downloads.download(erddap_url(metadatum), path; kwargs...)
        end
    end
    return nothing
end

# ERDDAP serves SeaWiFS with latitude descending and cloud gaps as `missing`.
function DataWrangling.retrieve_data(metadatum::SeaWiFSMetadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    raw = Dataset(path) do ds
        ds[name][:, :, 1]
    end

    data = [ismissing(chlorophyll) ? NaN32 : Float32(chlorophyll) for chlorophyll in raw]

    return reverse(data, dims=2)
end

end # module
