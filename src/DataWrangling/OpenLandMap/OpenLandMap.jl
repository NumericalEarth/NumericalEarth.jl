module OpenLandMap

export OpenLandMapSoilDB

using Downloads: Downloads
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling,
    AbstractStaticDataset, Metadatum, BoundingBox, Dataset,
    WeightPercent, GramPerCubicCentimeter,
    metadata_path, dataset_variable_name

import Oceananigans

download_OpenLandMap_cache::String = ""
function __init__()
    return global download_OpenLandMap_cache = DataWrangling.download_cache("OpenLandMap")
end

"""
    OpenLandMapSoilDB <: AbstractStaticDataset

OpenLandMap-soilDB global soil properties at 30 m (Hengl et al., 2026), predicted
from spatiotemporal machine learning over Landsat/MODIS/Sentinel covariates.

Delivers static soil texture mass fractions (`:sand_fraction`, `:silt_fraction`,
`:clay_fraction`, in kg/kg) and fine-earth bulk density (`:bulk_density`, in
kg/m³) over the three native depth intervals 0–30, 30–60, and 60–100 cm — stored
as a three-dimensional field whose vertical axis carries the depths (deepest
first), mirroring [`SoilGrids2`](@ref). Data are plain geographic EPSG:4326 so no
reprojection is needed.

Because the global grid is ~1.44M × 528k cells, this dataset is read in regional
windows only: construct the [`Metadatum`](@ref) with a longitude/latitude
[`BoundingBox`](@ref). Reading the cloud-optimized GeoTIFFs requires ArchGDAL to
be loaded (`using ArchGDAL`); access is anonymous, no credentials needed.

Coverage spans latitudes −56° to 76° (Antarctica excluded); permanent ice and
sand deserts are masked to `NaN` (pass `inpainting` to `Field` to fill them).

Data source: https://stac.openlandmap.org/ (CC-BY 4.0).

```jldoctest
using NumericalEarth

metadatum = Metadatum(:clay_fraction;
                      dataset = OpenLandMapSoilDB(),
                      region = BoundingBox(longitude = (-112.3, -111.9),
                                           latitude  = (36.0, 36.4)))

metadatum.filename

# output
"OpenLandMap_clay_fraction_lon_-112.3_-111.9_lat_36.0_36.4.nc"
```
"""
struct OpenLandMapSoilDB <: AbstractStaticDataset end

const OpenLandMapSoilDBMetadatum = Metadatum{<:OpenLandMapSoilDB}

# Variable-name mapping from NumericalEarth names to the short names used inside
# the regional NetCDF we materialize. These match SoilGrids2's scheme so a future
# pedotransfer function is source-agnostic.
OpenLandMap_dataset_variable_names = Dict(
    :sand_fraction => "sand",
    :silt_fraction => "silt",
    :clay_fraction => "clay",
    :bulk_density  => "bd")

# Per-variable cloud-optimized GeoTIFF location. Texture and bulk density live
# under different versioned mosaic directories, so each is pinned individually.
# Resolved from the OpenLandMap STAC at build time; scale/offset/nodata are read
# from each COG band directly (not hardcoded here).
OpenLandMap_cog_sources = Dict(
    :sand_fraction => (slug = "sand.tot_iso.11277.2020.wpct", directory = "global_soil_props_v20250523",        version = "v20250523"),
    :silt_fraction => (slug = "silt.tot_iso.11277.2020.wpct", directory = "global_soil_props_v20250523",        version = "v20250523"),
    :clay_fraction => (slug = "clay.tot_iso.11277.2020.wpct", directory = "global_soil_props_v20250523",        version = "v20250523"),
    :bulk_density  => (slug = "bd.core_iso.11272.2017.g.cm3", directory = "global_soil_props_v20250204_mosaics", version = "v20250204"))

const OpenLandMap_s3_base = "https://s3.opengeohub.org/global-soil"

# Depth intervals, deepest first, so the stacked vertical axis increases upward
# and matches `z_interfaces`.
const OpenLandMap_depths = ("b60cm..100cm", "b30cm..60cm", "b0cm..30cm")

const OpenLandMap_date_range = "20200101_20221231"

cog_url(source, depth) = string(OpenLandMap_s3_base, "/", source.directory, "/",
    source.slug, "_m_30m_", depth, "_", OpenLandMap_date_range,
    "_g_epsg.4326_", source.version, ".tif")

#####
##### Dataset traits
#####

DataWrangling.available_variables(::OpenLandMapSoilDB) = OpenLandMap_dataset_variable_names
DataWrangling.default_download_directory(::OpenLandMapSoilDB) = download_OpenLandMap_cache

# True COG extent (EPSG:4326 cell faces); Δλ = Δφ = 0.00025°. Coverage is
# latitudes −56° to 76°, not the full globe.
Base.size(::OpenLandMapSoilDB, variable) = (1440004, 528004, 3)
DataWrangling.longitude_interfaces(::OpenLandMapSoilDB) = (-180.0005, 180.0005)
DataWrangling.latitude_interfaces(::OpenLandMapSoilDB)  = (-56.0005, 76.0005)

# Faces of the 60–100 / 30–60 / 0–30 cm intervals, increasing upward (m).
DataWrangling.z_interfaces(::OpenLandMapSoilDB) = [-1.0, -0.6, -0.3, 0.0]
DataWrangling.reversed_vertical_axis(::OpenLandMapSoilDB) = false

#####
##### Metadatum traits
#####

DataWrangling.is_three_dimensional(::OpenLandMapSoilDBMetadatum) = true
DataWrangling.dataset_variable_name(data::OpenLandMapSoilDBMetadatum) = OpenLandMap_dataset_variable_names[data.name]
DataWrangling.longitude_name(::OpenLandMapSoilDBMetadatum) = "lon"
DataWrangling.latitude_name(::OpenLandMapSoilDBMetadatum)  = "lat"

# The windowed reader already decodes COG integers to physical units (percent for
# texture, g/cm³ for bulk density) and masks nodata to NaN. `conversion_units`
# applies only the final unit conversion to model units.
function DataWrangling.conversion_units(metadatum::OpenLandMapSoilDBMetadatum)
    if metadatum.name ∈ (:sand_fraction, :silt_fraction, :clay_fraction)
        return WeightPercent()
    elseif metadatum.name == :bulk_density
        return GramPerCubicCentimeter()
    else
        return nothing
    end
end

Oceananigans.Fields.location(::OpenLandMapSoilDBMetadatum) = (Center, Center, Center)

# Masked cells (ice, sand deserts, water, outside −56°–76°) stay NaN by default,
# mirroring SoilGrids2: shallow nearest-neighbor inpainting would 0-fill masks it
# cannot reach in a few passes, and 0 is a spurious soil value. Pass an explicit
# `inpainting = NearestNeighborInpainting(n)` to `Field` to fill them.
DataWrangling.default_inpainting(::OpenLandMapSoilDBMetadatum) = nothing

DataWrangling.inpainted_metadata_path(metadata::OpenLandMapSoilDBMetadatum) =
    joinpath(metadata.dir, replace(metadata.filename, ".nc" => "_inpainted.jld2"))

#####
##### Regional-window filename (variable + region)
#####

DataWrangling.metadata_filename(::OpenLandMapSoilDB, name, date, region) =
    string("OpenLandMap_", name, "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::OpenLandMapSoilDBMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("OpenLandMapSoilDB() must be used with a bounded region. " *
              "The global 30 m grid is ~1.44M × 528k cells and is never read in full. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))")
    end

    # Coverage is latitudes −56° to 76° (longitude is global); reject a box with no overlap.
    φ_south, φ_north = DataWrangling.latitude_interfaces(metadata.dataset)
    if region.latitude[2] < φ_south || region.latitude[1] > φ_north
        error("OpenLandMapSoilDB latitude coverage is $(φ_south)° to $(φ_north)°; " *
              "requested latitude = $(region.latitude).")
    end
    return nothing
end

#####
##### Download: window each depth COG into a stacked regional NetCDF
#####

function Downloads.download(metadatum::OpenLandMapSoilDBMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)

    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        source = OpenLandMap_cog_sources[metadatum.name]
        sources = ["/vsicurl/" * cog_url(source, depth) for depth in OpenLandMap_depths]
        name = dataset_variable_name(metadatum)
        @info "Downloading OpenLandMap-soilDB (30 m) $(metadatum.name) over $(summary(metadatum.region))..."
        cog_window_to_netcdf(sources, nc_path, name, metadatum.region)
    end
    return nc_path
end

# Read the stacked (lon, lat, depth) regional NetCDF; the vertical axis is already
# deepest-first (increasing upward), so no reversal is needed.
function DataWrangling.retrieve_data(metadata::OpenLandMapSoilDBMetadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    data = Dataset(path) do ds
        Array(ds[name][:, :, :])
    end
    return data
end

"""
    read_cog_window(source, bbox)

Read the `bbox` longitude/latitude window from a single-band EPSG:4326
cloud-optimized GeoTIFF `source`, decode raw integers to physical units (mask
nodata → `NaN`, then apply the band `scale`/`offset`), and return
`(longitude, latitude, data)` with ascending, cell-center coordinates (latitude
south-to-north, per CF convention).

Implemented in `ext/NumericalEarthArchGDALExt.jl` when ArchGDAL is loaded; the
fallback below fires only when the extension is not active.
"""
read_cog_window(source, bbox) =
    error("Reading OpenLandMap COGs requires the ArchGDAL package. Load it with `using ArchGDAL`.")

# Window each depth COG in `sources` (deepest-first) over `bbox` and stack them
# into a `(lon, lat, depth)` NetCDF at `nc_path`. All depths share one grid, so
# the coordinate axes come from the first window.
function cog_window_to_netcdf(sources, nc_path, variable_name, bbox)
    windows   = [read_cog_window(source, bbox) for source in sources]
    longitude = windows[1][1]
    latitude  = windows[1][2]
    Nx, Ny, Nz = length(longitude), length(latitude), length(windows)

    data = Array{Float32}(undef, Nx, Ny, Nz)
    for (k, (_, _, layer)) in enumerate(windows)
        data[:, :, k] = layer
    end

    # Interval midpoints (m) from the dataset's depth faces, deepest first.
    z = DataWrangling.z_interfaces(OpenLandMapSoilDB())
    depth_centers = Nz == length(z) - 1 ? (z[1:end-1] .+ z[2:end]) ./ 2 : collect(1.0:Nz)

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "depth", Nz)

        lon_var   = defVar(ds, "lon", Float64, ("lon",);
                           attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var   = defVar(ds, "lat", Float64, ("lat",);
                           attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        depth_var = defVar(ds, "depth", Float64, ("depth",);
                           attrib = ["units" => "m", "long_name" => "depth interval midpoint"])
        data_var  = defVar(ds, variable_name, Float32, ("lon", "lat", "depth"))

        lon_var[:]        = longitude
        lat_var[:]        = latitude
        depth_var[:]      = depth_centers
        data_var[:, :, :] = data
    end

    return nothing
end

end # module OpenLandMap
