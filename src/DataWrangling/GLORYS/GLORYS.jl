module GLORYS

export GLORYSStatic, GLORYSDaily, GLORYSMonthly

using NCDatasets
using Printf

using Oceananigans.Fields: Center
using NumericalEarth.DataWrangling: Metadata, Metadatum, metadata_path
using Dates: DateTime, Day, Month

import Oceananigans.Fields:
    location

import NumericalEarth.DataWrangling:
    all_dates,
    dataset_variable_name,
    default_download_directory,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    metadata_filename,
    inpainted_metadata_path,
    reversed_vertical_axis,
    available_variables

using Scratch

download_GLORYS_cache::String = ""
function __init__()
    global download_GLORYS_cache = @get_scratch!("GLORYS")
end

# Datasets
abstract type GLORYSDataset end

default_download_directory(::GLORYSDataset) = download_GLORYS_cache

# This contains "static" variables -- eg the grid
struct GLORYSStatic <: GLORYSDataset end
struct GLORYSDaily <: GLORYSDataset end
struct GLORYSMonthly <: GLORYSDataset end

dataset_name(::GLORYSStatic) = "GLORYSStatic"
dataset_name(::GLORYSDaily) = "GLORYSDaily"
dataset_name(::GLORYSMonthly) = "GLORYSMonthly"

Base.size(::GLORYSDataset, variable) = (4320, 2040, 50)

all_dates(::GLORYSStatic, var) = [nothing]
all_dates(::GLORYSDaily, var) = range(DateTime("1993-01-01"), stop=DateTime("2021-06-30"), step=Day(1))
all_dates(::GLORYSMonthly, var) = range(DateTime("1993-01-01"), stop=DateTime("2021-06-01"), step=Month(1))

copernicusmarine_dataset_id(::GLORYSStatic) = "cmems_mod_glo_phy_my_0.083deg_static"
copernicusmarine_dataset_id(::GLORYSDaily) = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
copernicusmarine_dataset_id(::GLORYSMonthly) = "cmems_mod_glo_phy_my_0.083deg_P1M-m"

struct CMEMSHourlyAnalysis <: GLORYSDataset end
copernicusmarine_dataset_id(::CMEMSHourlyAnalysis) = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

const GLORYSMetadata{D} = Metadata{<:GLORYSDataset, D}
const GLORYSMetadatum = Metadatum{<:GLORYSDataset}

Base.size(::GLORYSMetadatum) = (4320, 2040, 50, 1)

reversed_vertical_axis(::GLORYSDataset) = true

available_variables(::GLORYSDataset) = GLORYS_dataset_variable_names

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

dataset_variable_name(metadata::GLORYSMetadata) = GLORYS_dataset_variable_names[metadata.name]

bbox_strs(::Nothing) = "_nothing", "_nothing"

function bbox_strs(c)
    first = @sprintf("_%.1f", c[1])
    second = @sprintf("_%.1f", c[2])
    return first, second
end

colon2dash(s::String) = replace(s, ":" => "-")

function metadata_prefix(dataset::GLORYSDataset, name, date, bounding_box)
    var = GLORYS_dataset_variable_names[name]
    ds = dataset_name(dataset)
    start_date = start_date_str(date)
    end_date = end_date_str(date)
    if !isnothing(bounding_box)
        w, e = bbox_strs(bounding_box.longitude)
        s, n = bbox_strs(bounding_box.latitude)
        suffix = string(w, e, s, n)
    else
        suffix = ""
    end
    return string(var, "_",
                  ds, "_",
                  start_date, "_",
                  end_date, suffix) |> colon2dash
end

function metadata_filename(dataset::GLORYSDataset, name, date, bounding_box)
    prefix = metadata_prefix(dataset, name, date, bounding_box)
    return string(prefix, ".nc")
end

function inpainted_metadata_filename(metadata::GLORYSMetadatum)
    prefix = metadata_prefix(metadata.dataset, metadata.name, metadata.dates, metadata.bounding_box)
    return string(prefix, "_inpainted.jld2")
end

inpainted_metadata_path(metadata::GLORYSMetadatum) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

location(::GLORYSMetadata) = (Center, Center, Center)
longitude_interfaces(::GLORYSMetadata) = (-180, 180)
latitude_interfaces(::GLORYSMetadata) = (-80, 90)

function z_interfaces(metadata::GLORYSMetadata)
    path = metadata_path(metadata)
    ds = Dataset(path)
    zc = - reverse(ds["depth"][:])
    close(ds)
    dz = zc[2] - zc[1]
    zf = (zc[1:end-1] .+ zc[2:end]) / 2
    push!(zf, 0)
    pushfirst!(zf, zf[1] - dz)
    return zf
end

end # module GLORYS

