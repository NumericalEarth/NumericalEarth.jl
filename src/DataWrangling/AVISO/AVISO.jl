module AVISO

export AVISOMetadata, AVISOMetadatum, AVISODaily, AVISOMonthly

using Dates: Dates, DateTime, Day, Month
using Oceananigans.Fields: Center
using NCDatasets: Dataset
using Scratch: @get_scratch!

using ...NumericalEarth: NumericalEarth
using ..DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path, first_date

download_AVISO_cache::String = ""
function __init__()
    global download_AVISO_cache = @get_scratch!("AVISO")
end

abstract type AVISODataset end
struct AVISODaily <: AVISODataset end
struct AVISOMonthly <: AVISODataset end

# AVISODaily and AVISOMonthly are the daily- and monthly-resolution datasets in
# the delayed-time Multi-Year product. The product is extended only a few times
# per year, so its temporal coverage cannot be inferred from today's date.
const AVISO_FIRST_DATE = DateTime(1993, 1, 1)
const AVISO_DAILY_LAST_DATE = DateTime(2026, 1, 16)
const AVISO_MONTHLY_LAST_DATE = DateTime(2025, 12, 1)

function DataWrangling.default_download_directory(::AVISODaily)
    return mkpath(joinpath(download_AVISO_cache, "daily"))
end

function DataWrangling.default_download_directory(::AVISOMonthly)
    return mkpath(joinpath(download_AVISO_cache, "monthly"))
end

Base.size(::AVISODataset, variable) = (2880, 1440, 1)

DataWrangling.all_dates(::AVISODaily, variable) = AVISO_FIRST_DATE : Day(1) : AVISO_DAILY_LAST_DATE
DataWrangling.all_dates(::AVISOMonthly, variable) = AVISO_FIRST_DATE : Month(1) : AVISO_MONTHLY_LAST_DATE

const AVISO_dataset_variable_names = Dict(
    :free_surface => "adt",
    :sea_level_anomaly => "sla",
    :zonal_geostrophic_velocity => "ugos",
    :meridional_geostrophic_velocity => "vgos",
)

DataWrangling.available_variables(::AVISODataset) = AVISO_dataset_variable_names
DataWrangling.dataset_variable_name(metadata::Metadata{<:AVISODataset}) = AVISO_dataset_variable_names[metadata.name]
DataWrangling.dataset_location(::AVISODataset, name) = (Center, Center, Nothing)
DataWrangling.is_three_dimensional(::Metadata{<:AVISODataset}) = false
DataWrangling.default_inpainting(metadata::Metadata{<:AVISODataset}) = nothing

DataWrangling.longitude_name(::Metadata{<:AVISODataset}) = "longitude"
DataWrangling.latitude_name(::Metadata{<:AVISODataset}) = "latitude"

coord_spacing(coords) = length(coords) > 1 ? Float64(coords[2] - coords[1]) : 1.0

function read_coordinate(path, name)
    ds = Dataset(path)
    coords = Float64.(ds[name][:])
    close(ds)
    return coords
end

function DataWrangling.longitude_interfaces(metadata::Metadata{<:AVISODataset})
    path = metadata_path(first(metadata))
    λ = read_coordinate(path, DataWrangling.longitude_name(metadata))
    Δλ = coord_spacing(λ)
    return (λ[1] - Δλ / 2, λ[end] + Δλ / 2)
end

function DataWrangling.latitude_interfaces(metadata::Metadata{<:AVISODataset})
    path = metadata_path(first(metadata))
    φ = read_coordinate(path, DataWrangling.latitude_name(metadata))
    Δφ = coord_spacing(φ)
    return (φ[1] - Δφ / 2, φ[end] + Δφ / 2)
end

function DataWrangling.metadata_filename(::AVISODaily, name, date, region)
    var = AVISO_dataset_variable_names[name]
    return string(var, "_AVISODaily_", Dates.format(date, "yyyy-mm-dd"), ".nc")
end

function DataWrangling.metadata_filename(::AVISOMonthly, name, date, region)
    var = AVISO_dataset_variable_names[name]
    return string(var, "_AVISOMonthly_", Dates.format(date, "yyyy-mm-dd"), ".nc")
end

function inpainted_metadata_filename(metadata::Metadatum{<:AVISODataset})
    return replace(metadata.filename, ".nc" => "_inpainted.jld2")
end

DataWrangling.inpainted_metadata_path(metadata::Metadatum{<:AVISODataset}) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

const AVISOMetadata{D} = Metadata{<:AVISODataset, D}
const AVISOMetadatum = Metadatum{<:AVISODataset}

function AVISOMetadatum(name; date = first_date(AVISOMonthly(), name), dir = DataWrangling.default_download_directory(AVISOMonthly()))
    return Metadatum(name; date, dir, dataset = AVISOMonthly())
end

DataWrangling.metaprefix(::AVISOMetadata) = "AVISOMetadata"
DataWrangling.metaprefix(::AVISOMetadatum) = "AVISOMetadatum"

copernicusmarine_dataset_id(::AVISODaily) = get(ENV, "AVISO_DAILY_DATASET_ID", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D")
copernicusmarine_dataset_id(::AVISOMonthly) = get(ENV, "AVISO_MONTHLY_DATASET_ID", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m")

# Pin the catalogue version advertised by the product data-access page. This is
# important during Copernicus double-distribution transitions, when resolving
# the bare dataset id can select a version with a different variable table.
copernicusmarine_dataset_version(::AVISODataset) = get(ENV, "AVISO_DATASET_VERSION", "202411")

native_horizontal_resolution(::AVISODataset) = 1 / 8

end # module
