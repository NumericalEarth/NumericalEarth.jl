module GloFAS

# Datasets
export GloFASReanalysis

# Prescribed components
export GloFASPrescribedLand

using Dates: Dates, DateTime, Day
using Oceananigans.OutputReaders: Cyclical, FieldTimeSeries
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: DataWrangling, Metadata, Metadatum,
                       available_variables, first_date, last_date
using ...Lands: PrescribedLand, build_river_routing, coastal_outlet_indices

download_GloFAS_cache::String = ""

function __init__()
    global download_GloFAS_cache = @get_scratch!("GloFAS")
end

#####
##### GloFAS datasets
#####

# The GloFAS (Global Flood Awareness System) river-discharge reanalysis is produced
# by routing ERA5-forced LISFLOOD runoff through a channel-routing model. It provides
# river discharge already accumulated downstream to river mouths — the ERA5-consistent
# analogue of JRA55's pre-routed `river_freshwater_flux`. See Harrigan et al. (2020),
# https://essd.copernicus.org/articles/12/2043/2020/.
abstract type GloFASDataset end

DataWrangling.default_download_directory(::GloFASDataset) = download_GloFAS_cache

# GloFAS files store latitude north-to-south (90 → -60); flip on read.
DataWrangling.reversed_latitude_axis(::GloFASDataset) = true

const GloFASMetadata{D} = Metadata{<:GloFASDataset, D}
const GloFASMetadatum = Metadatum{<:GloFASDataset}

#####
##### Grid interfaces
#####

# GloFAS v4 global coverage: -180 to 180 longitude, -60 to 90 latitude at 0.05° resolution.
DataWrangling.longitude_interfaces(::GloFASMetadata) = (-180, 180)
DataWrangling.latitude_interfaces(::GloFASMetadata)  = (-60, 90)

# GloFAS is a spatially 2-D surface dataset.
DataWrangling.z_interfaces(::GloFASMetadata) = (0, 1)
DataWrangling.is_three_dimensional(::GloFASMetadata) = false

Base.eltype(::GloFASMetadata) = Float32

#####
##### Filename utilities
#####

function date_str(date::DateTime)
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    d = lpad(Dates.day(date),   2, '0')
    return "$(y)-$(m)-$(d)"
end

date_str(dates::AbstractVector) = string(date_str(first(dates)), "_", date_str(last(dates)))

region_suffix(::Nothing) = ""
region_suffix(region) = string("_", region.longitude[1], "_", region.longitude[2],
                               "_", region.latitude[1], "_", region.latitude[2])

function DataWrangling.metadata_filename(dataset::GloFASDataset, name, date, region)
    var = available_variables(dataset)[name]
    ds = dataset_name(dataset)
    return string(var, "_", ds, "_", date_str(date), region_suffix(region), ".nc")
end

function inpainted_metadata_filename(metadata::GloFASMetadatum)
    without_extension = metadata.filename[1:end-3]
    return without_extension * "_inpainted.jld2"
end

DataWrangling.inpainted_metadata_path(metadata::GloFASMetadatum) =
    joinpath(metadata.dir, inpainted_metadata_filename(metadata))

include("glofas_reanalysis.jl")
include("glofas_prescribed_land.jl")

end # module GloFAS
