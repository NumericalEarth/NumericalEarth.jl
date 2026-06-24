module GLORYS

export GLORYSStatic, GLORYSDaily, GLORYSMonthly,
       GLORYSAnalysisForecastBGCDaily, GLORYSAnalysisForecastBGCMonthly

using Dates: DateTime, Day, Month
using Oceananigans.Fields: Center
using NCDatasets: NCDatasets, Dataset
using Printf: Printf, @sprintf

using ..DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path

import ..DataWrangling:
    all_dates,
    dataset_variable_name,
    dataset_location,
    default_download_directory,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    metadata_filename,
    inpainted_metadata_path,
    reversed_vertical_axis,
    available_variables,
    is_three_dimensional

download_GLORYS_cache::String = ""
function __init__()
    global download_GLORYS_cache = DataWrangling.download_cache("GLORYS")
end

# Datasets
abstract type AbstractGLORYSDataset end

default_download_directory(::AbstractGLORYSDataset) = download_GLORYS_cache

# This contains "static" variables -- eg the grid
struct GLORYSStatic <: AbstractGLORYSDataset end
struct GLORYSDaily <: AbstractGLORYSDataset end
struct GLORYSMonthly <: AbstractGLORYSDataset end

dataset_name(::GLORYSStatic) = "GLORYSStatic"
dataset_name(::GLORYSDaily) = "GLORYSDaily"
dataset_name(::GLORYSMonthly) = "GLORYSMonthly"

Base.size(::AbstractGLORYSDataset, variable) = (4320, 2040, 50)

all_dates(::GLORYSStatic, var) = [nothing]
all_dates(::GLORYSDaily, var) = range(DateTime("1993-01-01"), stop=DateTime("2026-05-26"), step=Day(1))
all_dates(::GLORYSMonthly, var) = range(DateTime("1993-01-01"), stop=DateTime("2026-04-01"), step=Month(1))

copernicusmarine_dataset_id(dataset, name) = copernicusmarine_dataset_id(dataset) #fallback
copernicusmarine_dataset_id(::GLORYSStatic) = "cmems_mod_glo_phy_my_0.083deg_static"
copernicusmarine_dataset_id(::GLORYSDaily) = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
copernicusmarine_dataset_id(::GLORYSMonthly) = "cmems_mod_glo_phy_my_0.083deg_P1M-m"

struct CMEMSHourlyAnalysis <: AbstractGLORYSDataset end
copernicusmarine_dataset_id(::CMEMSHourlyAnalysis) = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

const GLORYSMetadata{D} = Metadata{<:AbstractGLORYSDataset, D}
const GLORYSMetadatum = Metadatum{<:AbstractGLORYSDataset}

Base.size(::GLORYSMetadatum) = (4320, 2040, 50, 1)

reversed_vertical_axis(::AbstractGLORYSDataset) = true

available_variables(::AbstractGLORYSDataset) = GLORYS_dataset_variable_names

GLORYS_dataset_variable_names = Dict(
    :temperature => "thetao",
    :depth => "deptho",
    :salinity => "so",
    :sea_ice_concentration => "siconc",
    :sea_ice_thickness => "sithick",
    :u_velocity=> "uo",
    :v_velocity=> "vo",
    :sea_ice_u_velocity => "usi",
    :sea_ice_v_velocity => "vsi",
    :free_surface => "zos",
)

start_date_str(date) = string(date)
end_date_str(date) = string(date)
start_date_str(dates::AbstractVector) = first(dates) |> string
end_date_str(dates::AbstractVector) = last(dates) |> string

dataset_variable_name(metadata::Metadata{<:AbstractGLORYSDataset}) = available_variables(metadata.dataset)[metadata.name]

bbox_strs(::Nothing) = "_nothing", "_nothing"
bbox_strs(c::Number) = @sprintf("_%.1f", c), @sprintf("_%.1f", c)

function bbox_strs(c)
    first = @sprintf("_%.1f", c[1])
    second = @sprintf("_%.1f", c[2])
    return first, second
end

colon2dash(s::String) = replace(s, ":" => "-")

function region_suffix(::Nothing)
    return ""
end

function region_suffix(region)
    w, e = bbox_strs(region.longitude)
    s, n = bbox_strs(region.latitude)
    return string(w, e, s, n)
end

function metadata_prefix(dataset::AbstractGLORYSDataset, name, date, region)
    var = available_variables(dataset)[name]
    ds = dataset_name(dataset)
    start_date = start_date_str(date)
    end_date = end_date_str(date)
    suffix = region_suffix(region)
    return string(var, "_",
                  ds, "_",
                  start_date, "_",
                  end_date, suffix) |> colon2dash
end

function metadata_filename(dataset::AbstractGLORYSDataset, name, date, region)
    prefix = metadata_prefix(dataset, name, date, region)
    return string(prefix, ".nc")
end

function inpainted_metadata_filename(metadata::GLORYSMetadatum)
    prefix = metadata_prefix(metadata.dataset, metadata.name, metadata.dates, metadata.region)
    return string(prefix, "_inpainted.jld2")
end

inpainted_metadata_path(metadata::GLORYSMetadatum) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

function dataset_location(::AbstractGLORYSDataset, name)
    name == :free_surface && return (Center, Center, Nothing)
    return (Center, Center, Center)
end

is_three_dimensional(metadata::GLORYSMetadata) = metadata.name != :free_surface

longitude_interfaces(::GLORYSMetadata) = (-180, 180)
latitude_interfaces(::GLORYSMetadata) = (-80, 90)

function z_interfaces(metadata::GLORYSMetadata)
    is_three_dimensional(metadata) || return (-1.0, 0.0)
    paths = metadata_path(metadata)
    path = paths isa AbstractVector ? first(paths) : paths
    ds = Dataset(path)
    zc = - reverse(ds["depth"][:])
    close(ds)
    dz = zc[2] - zc[1]
    zf = (zc[1:end-1] .+ zc[2:end]) / 2
    push!(zf, 0)
    pushfirst!(zf, zf[1] - dz)
    return zf
end

include("GLORYS_bgc.jl")

end # module GLORYS

