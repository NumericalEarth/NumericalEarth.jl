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

function last_complete_day()
    d = Dates.today() - Day(2)
    return DateTime(Dates.year(d), Dates.month(d), Dates.day(d))
end

function last_complete_month()
    d = Dates.firstdayofmonth(Dates.today()) - Month(2)
    return DateTime(Dates.year(d), Dates.month(d), 1)
end

function DataWrangling.default_download_directory(::AVISODaily)
    return mkpath(joinpath(download_AVISO_cache, "daily"))
end

function DataWrangling.default_download_directory(::AVISOMonthly)
    return mkpath(joinpath(download_AVISO_cache, "monthly"))
end

Base.size(::AVISODataset, variable) = (1440, 720, 1)

DataWrangling.all_dates(::AVISODaily, variable) = DateTime(1993, 1, 1) : Day(1) : last_complete_day()
DataWrangling.all_dates(::AVISOMonthly, variable) = DateTime(1993, 1, 1) : Month(1) : last_complete_month()

const AVISO_dataset_variable_names = Dict(
    :free_surface => "adt",
    :ssh => "adt",
    :sea_surface_height => "adt",
    :absolute_dynamic_topography => "adt",
    :adt => "adt",
    :sea_level_anomaly => "sla",
    :sla => "sla",
    :zonal_geostrophic_velocity => "ugos",
    :ugos => "ugos",
    :meridional_geostrophic_velocity => "vgos",
    :vgos => "vgos",
)

DataWrangling.available_variables(::AVISODataset) = AVISO_dataset_variable_names
DataWrangling.dataset_variable_name(metadata::Metadata{<:AVISODataset}) = AVISO_dataset_variable_names[metadata.name]
DataWrangling.dataset_location(::AVISODataset, name) = (Center, Center, Nothing)
DataWrangling.is_three_dimensional(::Metadata{<:AVISODataset}) = false
DataWrangling.z_interfaces(::AVISODataset) = (-1.0, 0.0)
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

copernicusmarine_dataset_id(::AVISODaily) = get(ENV, "AVISO_DAILY_DATASET_ID", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D-m")
copernicusmarine_dataset_id(::AVISOMonthly) = get(ENV, "AVISO_MONTHLY_DATASET_ID", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1M-m")

native_horizontal_resolution(::AVISODataset) = 1 / 4

end # module
