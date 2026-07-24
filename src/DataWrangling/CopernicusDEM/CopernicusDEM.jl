module CopernicusDEM

export GLO30, GLO90

using Downloads: Downloads
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticBathymetry, Metadatum,
                       metadata_path, BoundingBox

download_CopernicusDEM_cache::String = ""
function __init__()
    global download_CopernicusDEM_cache = DataWrangling.download_cache("CopernicusDEM")
end

# Variable name in the regional NetCDF we materialize from the Zarr store; this is
# what `_regrid_bathymetry` reads back. The name inside the Zarr store itself is
# `dataset_zarr_variable_name`, read within the Zarr extension.
CopernicusDEM_bathymetry_variable_names = Dict(:bottom_height => "z")

const dataset_zarr_variable_name = "dsm"

"""
    GLO30

Copernicus DEM GLO-30: global 30 m (1 arc-second) Digital Surface Model (DSM),
representing the Earth's surface including buildings, infrastructure, and
vegetation. Heights are referenced to the EGM2008 geoid; ocean is set to 0.

Because GLO-30 is a global 30 m product (≈ 1.3M × 0.65M cells), it is read in
regional windows only: construct the `Metadatum` with a longitude/latitude
`BoundingBox` and use it with [`regrid_topography`](@ref).

Data is read from the Earth Data Hub Zarr store, which requires a (free) DestinE
personal access token in the `DESTINE_ACCESS_TOKEN` environment variable. Register
at https://platform.destine.eu/ and create a token at
https://earthdatahub.destine.eu/account-settings#my-personal-access-tokens.

Reading the Zarr store requires the `Zarr` package to be loaded (`using Zarr`).

Data source: https://earthdatahub.destine.eu/collections/copernicus-dem/datasets/GLO-30
"""
struct GLO30 <: AbstractStaticBathymetry end

"""
    GLO90

Copernicus DEM GLO-90: global 90 m (3 arc-second) Digital Surface Model, the
coarser sibling of [`GLO30`](@ref). Same source, access, and usage; lower
resolution and a correspondingly smaller native grid.

Data source: https://earthdatahub.destine.eu/collections/copernicus-dem/datasets/GLO-90
"""
struct GLO90 <: AbstractStaticBathymetry end

const CopernicusDEMDataset = Union{GLO30, GLO90}

DataWrangling.default_download_directory(::CopernicusDEMDataset) = download_CopernicusDEM_cache
DataWrangling.reversed_vertical_axis(::CopernicusDEMDataset) = false
DataWrangling.longitude_interfaces(::CopernicusDEMDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::CopernicusDEMDataset) = (-90, 90)

# GLO-30 is 1 arc-second (360 * 3600 × 180 * 3600); GLO-90 is 3 arc-second.
Base.size(::GLO30) = (1296000, 648000, 1)
Base.size(::GLO90) = (432000, 216000, 1)

dataset_prefix(::GLO30) = "GLO30"
dataset_prefix(::GLO90) = "GLO90"

# Earth Data Hub (DestinE) Zarr endpoints, without scheme/credentials. The Zarr
# extension prepends `https://edh:<token>@` using the DESTINE_ACCESS_TOKEN env var.
zarr_host_path(::GLO30) = "api.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr"
zarr_host_path(::GLO90) = "api.earthdatahub.destine.eu/copernicus-dem/GLO-90-v0.zarr"

const CopernicusDEMMetadatum = Metadatum{<:CopernicusDEMDataset}

DataWrangling.dataset_variable_name(data::CopernicusDEMMetadatum) =
    CopernicusDEM_bathymetry_variable_names[data.name]

DataWrangling.metadata_filename(dataset::CopernicusDEMDataset, name, date, region) =
    string(dataset_prefix(dataset), "_", DataWrangling.bounded_region_suffix(region), ".nc")

function DataWrangling.validate_dataset_coverage(grid, metadata::CopernicusDEMMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("$(dataset_prefix(metadata.dataset))() must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:bottom_height; dataset = $(dataset_prefix(metadata.dataset))(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    regrid_topography(grid, metadatum)")
    end
    return nothing
end

function Downloads.download(metadatum::CopernicusDEMMetadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        zarr_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthZarrExt.jl once `Zarr` is loaded. The fallback
# below fires only when the extension is not active.
zarr_to_netcdf(metadatum, nc_path) =
    error("Reading the Copernicus DEM Zarr store requires the Zarr package. " *
          "Load it with `using Zarr` and set the DESTINE_ACCESS_TOKEN environment variable.")

end # module CopernicusDEM
