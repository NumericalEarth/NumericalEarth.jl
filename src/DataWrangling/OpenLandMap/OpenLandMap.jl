module OpenLandMap

export OpenLandMapSoilDB

using Downloads: Downloads
using DocStringExtensions: TYPEDSIGNATURES
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans: Bounded, Center, Face, LatitudeLongitudeGrid
using Oceananigans.Architectures: architecture
using Oceananigans.DistributedComputations: @root, all_reduce
using Oceananigans.Fields: Field, interior, set!
using Oceananigans.Grids: λnodes, φnodes, λspacings, φspacings

using ..DataWrangling: DataWrangling,
    AbstractStaticDataset, Metadatum, BoundingBox, Dataset,
    WeightPercent, GramPerCubicCentimeter,
    metadata_path, dataset_variable_name, bounding_box_suffix,
    default_download_directory, inpaint_mask!

using ...Lands: Lands

import Oceananigans

download_OpenLandMap_cache::String = ""
function __init__()
    global download_OpenLandMap_cache = DataWrangling.download_cache("OpenLandMap")
    return nothing
end

"""
    OpenLandMapSoilDB(; aggregation_factor = 1)

OpenLandMap-soilDB global soil properties at 30 m (Hengl et al., 2026), predicted
from spatiotemporal machine learning over Landsat/MODIS/Sentinel covariates.

Delivers static soil texture mass fractions (`:sand_fraction`, `:silt_fraction`,
`:clay_fraction`, in kg/kg) and fine-earth bulk density (`:bulk_density`, in
kg/m³) over the three native depth intervals 0–30, 30–60, and 60–100 cm — stored
as a three-dimensional field whose vertical axis carries the depths (deepest
first). Data are plain geographic EPSG:4326 so no reprojection is needed.

`aggregation_factor` is the integer number of 30 m pixels reduced per side into one read cell.
The default `1` reads at full resolution. Pass `nothing` to size the read to the target instead:
`Field(metadatum, grid)` then picks the coarsest lattice that still oversamples `grid` twofold,
while `Field(metadatum, arch)`, which has no target, still reads at full resolution. Coarse reads
come from the GeoTIFFs' average-resampled overview pyramid, so they average rather than sample.

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
struct OpenLandMapSoilDB{F} <: AbstractStaticDataset
    aggregation_factor :: F
end

function OpenLandMapSoilDB(; aggregation_factor = 1)
    if !isnothing(aggregation_factor) && aggregation_factor < 1
        throw(ArgumentError("OpenLandMapSoilDB aggregation_factor must be a positive number of " *
                            "30 m pixels per read cell side, got $aggregation_factor"))
    end
    return OpenLandMapSoilDB(aggregation_factor)
end

Base.summary(dataset::OpenLandMapSoilDB) =
    string("OpenLandMapSoilDB(aggregation_factor = ", dataset.aggregation_factor, ")")

Base.show(io::IO, dataset::OpenLandMapSoilDB) = print(io, summary(dataset))

const OpenLandMapSoilDBMetadatum = Metadatum{<:OpenLandMapSoilDB}

"""
$(TYPEDSIGNATURES)

Native 30 m pixels reduced per side into one read cell; `1` when sized to a target.
"""
aggregation_factor(dataset::OpenLandMapSoilDB) = something(dataset.aggregation_factor, 1)

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

# Depth intervals, deepest first.
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
const OpenLandMap_native_step = 0.00025
const OpenLandMap_native_size = (1440004, 528004)
const OpenLandMap_longitude_interfaces = (-180.0005, 180.0005)
const OpenLandMap_latitude_interfaces = (-56.0005, 76.0005)

"""
$(TYPEDSIGNATURES)

Return the cell size (degrees) `dataset` is read at: the 30 m native step times the aggregation factor.
"""
read_step(dataset::OpenLandMapSoilDB) = OpenLandMap_native_step * aggregation_factor(dataset)

# Read cells tile from the file origin, so a remainder too small to fill one falls at the east
# and the south.
read_size(dataset::OpenLandMapSoilDB) = map(N -> fld(N, aggregation_factor(dataset)), OpenLandMap_native_size)

Base.size(dataset::OpenLandMapSoilDB, variable) = (read_size(dataset)..., 3)

function DataWrangling.longitude_interfaces(dataset::OpenLandMapSoilDB)
    west, east = OpenLandMap_longitude_interfaces
    aggregation_factor(dataset) == 1 && return (west, east)
    return (west, west + read_size(dataset)[1] * read_step(dataset))
end

function DataWrangling.latitude_interfaces(dataset::OpenLandMapSoilDB)
    south, north = OpenLandMap_latitude_interfaces
    aggregation_factor(dataset) == 1 && return (south, north)
    return (north - read_size(dataset)[2] * read_step(dataset), north)
end

"""
$(TYPEDSIGNATURES)

Return the smallest horizontal cell spacing of `grid` in degrees.
"""
function minimum_horizontal_spacing(grid)
    Δλ = minimum(λspacings(grid, Center(), Center(), Center()))
    Δφ = minimum(φspacings(grid, Center(), Center(), Center()))
    return all_reduce(min, min(Δλ, Δφ), architecture(grid))
end

# Twofold oversampling of the target, rounded down to a power of two so the read lands on a
# pyramid level.
function DataWrangling.coarsest_resolving_dataset(dataset::OpenLandMapSoilDB{Nothing}, grid)
    pixels = minimum_horizontal_spacing(grid) / (2 * OpenLandMap_native_step)
    factor = pixels < 2 ? 1 : prevpow(2, floor(Int, pixels))
    return OpenLandMapSoilDB(factor)
end

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

# Masked cells (ice, sand deserts, water, outside −56°–76°) stay NaN by default.
# Pass an explicit `inpainting = NearestNeighborInpainting(n)` to `Field` to fill them.
DataWrangling.default_inpainting(::OpenLandMapSoilDBMetadatum) = nothing

#####
##### Regional-window filename (variable + aggregation factor + region)
#####

# A full-resolution read carries no factor token.
function DataWrangling.metadata_filename(dataset::OpenLandMapSoilDB, name, date, region)
    factor = aggregation_factor(dataset)
    suffix = factor == 1 ? "" : string("_f", factor)
    return string("OpenLandMap_", name, suffix, "_", bounding_box_suffix(region), ".nc")
end

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
        dataset = metadatum.dataset
        source = OpenLandMap_cog_sources[metadatum.name]
        sources = ["/vsicurl/" * cog_url(source, depth) for depth in OpenLandMap_depths]
        name = dataset_variable_name(metadatum)
        resolution = string(30 * aggregation_factor(dataset), " m")
        @info "Downloading OpenLandMap-soilDB ($resolution) $(metadatum.name) over $(summary(metadatum.region))..."
        cog_window_to_netcdf(sources, nc_path, name, metadatum.region, aggregation_factor(dataset))
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

# The 30 m regional window is large and regridded by window.
DataWrangling.windowed_retrieval(::OpenLandMapSoilDB) = true

"""
    read_cog_window(source, bbox, factor = 1)

Read the `bbox` longitude/latitude window from a single-band EPSG:4326
cloud-optimized GeoTIFF `source`, decode raw integers to physical units (mask
nodata → `NaN`, then apply the band `scale`/`offset`), and return
`(longitude, latitude, data)` with ascending, cell-center coordinates (latitude
south-to-north, per CF convention).

`factor` is the number of native pixels reduced per side into one returned cell. Above `1` the
window is snapped to whole `factor`-pixel blocks and served from the COG's overview pyramid, so
the values are averages of the underlying pixels and the read costs a fraction of the bytes.
Averages only make sense for continuous fields — never read a class-code raster this way.

Implemented in `ext/NumericalEarthArchGDALExt/openlandmap.jl` when ArchGDAL is loaded; the
fallback below fires only when the extension is not active.
"""
read_cog_window(source, bbox, factor = 1) =
    error("Reading OpenLandMap COGs requires the ArchGDAL package. Load it with `using ArchGDAL`.")

# Window each depth COG in `sources` (deepest-first) over `bbox` and stack them into a
# `(lon, lat, depth)` NetCDF at `nc_path`. All depths share one grid, so the coordinate axes come
# from the first window. Layers are read and written one at a time, so the peak memory is one
# window rather than every depth at once.
function cog_window_to_netcdf(sources, nc_path, variable_name, bbox, factor = 1)
    Nz = length(sources)

    # Interval midpoints (m) from the dataset's depth faces, deepest first.
    z = DataWrangling.z_interfaces(OpenLandMapSoilDB())
    depth_centers = Nz == length(z) - 1 ? (z[1:end-1] .+ z[2:end]) ./ 2 : collect(1.0:Nz)

    NCDataset(nc_path, "c") do ds
        for (k, source) in enumerate(sources)
            longitude, latitude, layer = read_cog_window(source, bbox, factor)

            if k == 1
                defDim(ds, "lon", length(longitude))
                defDim(ds, "lat", length(latitude))
                defDim(ds, "depth", Nz)

                defVar(ds, "lon", Float64, ("lon",);
                       attrib = ["units" => "degrees_east", "long_name" => "longitude"])[:] = longitude
                defVar(ds, "lat", Float64, ("lat",);
                       attrib = ["units" => "degrees_north", "long_name" => "latitude"])[:] = latitude
                defVar(ds, "depth", Float64, ("depth",);
                       attrib = ["units" => "m", "long_name" => "depth interval midpoint"])[:] = depth_centers

                chunk = [min(512, length(longitude)), min(512, length(latitude)), Nz]
                defVar(ds, variable_name, Float32, ("lon", "lat", "depth"); chunksizes = chunk)
            end

            ds[variable_name][:, :, k] = layer
        end
    end

    return nothing
end

#####
##### Windowing and decoding, independent of the GDAL reader
#####

# The windowing math and the north→south row reversal below assume a north-up,
# axis-aligned geographic (EPSG:4326, degrees) grid.
function validate_geographic_northup(geotransform)
    _, dx, rx, _, ry, dy = geotransform
    (rx == 0 && ry == 0) ||
        error("Windowed COG reader requires an axis-aligned grid (no rotation/shear); " *
              "got geotransform $geotransform.")
    (dx > 0 && dy < 0) ||
        error("Windowed COG reader assumes west→east (Δλ > 0) and north→south (Δφ < 0) " *
              "pixel order; got Δλ = $dx, Δφ = $dy.")
    return nothing
end

"""
$(TYPEDSIGNATURES)

Check that `coordinate_system_code` identifies WGS84 longitude and latitude in
degrees (EPSG:4326). WGS84 means World Geodetic System 1984, a standard reference
system defining Earth's shape and how coordinates locate points on it.
EPSG:4326 is the catalog code for its longitude/latitude representation.
Accept `nothing` when no code is available, leaving the coordinate system unverified.
"""
function validate_wgs84_longitude_latitude(coordinate_system_code)
    isnothing(coordinate_system_code) || coordinate_system_code == 4326 ||
        error("Expected WGS84 longitude/latitude in degrees (EPSG:4326), " *
              "but the source declares EPSG:$coordinate_system_code.")
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return `(column_offset, row_offset, Nx, Ny)` for a rectangular raster patch covering
the longitude/latitude bounds `bbox`, snapped outward to whole `factor`-pixel blocks of the
image's own lattice, padded by one block, and clipped to the image.
Offsets count columns and rows from zero; `Nx` and `Ny` are the patch's pixel counts.
The full image has `width` columns and `height` rows, ordered west to east and north
to south, with `geotransform = [western_edge, longitude_spacing, 0,
northern_edge, 0, latitude_spacing]` in degrees and negative `latitude_spacing`.
"""
function raster_window_indices(geotransform, width, height, bbox, factor = 1)
    x0, dx, _, y0, _, dy = geotransform
    W, E = bbox.longitude
    S, N = bbox.latitude

    column_offset, Nx = block_aligned_range(W, E, x0, dx, width, factor)
    row_offset, Ny    = block_aligned_range(N, S, y0, dy, height, factor)  # Δφ < 0: north comes first

    return column_offset, row_offset, Nx, Ny
end

"""
$(TYPEDSIGNATURES)

Return `(offset, N)` for the pixels `offset` through `offset + N - 1` along one image axis of
`pixel_count` pixels that span `first_coordinate` through `last_coordinate`. The axis starts at
`origin` (the outer edge of pixel zero) and advances by `spacing` per pixel. The range starts and
ends on edges of `factor`-pixel blocks counted from `origin`, is padded by one block on each side,
and is clipped to the whole blocks of the axis, so the `pixel_count % factor` pixels at the far
end are never read.
"""
function block_aligned_range(first_coordinate, last_coordinate, origin, spacing, pixel_count, factor)
    # The pad makes the window a strict superset of the framework's center-bracketed native grid;
    # otherwise the grid can hold one more cell than the file, forcing a clamped read that shifts
    # the whole window and duplicates the outermost cell.
    first_pixel = factor * (fld(floor(Int, (first_coordinate - origin) / spacing), factor) - 1)
    last_pixel  = factor * (cld(ceil( Int, (last_coordinate  - origin) / spacing), factor) + 1)
    last_face   = factor * fld(pixel_count, factor)
    first_pixel = clamp(first_pixel, 0, last_face - factor)
    last_pixel  = clamp(last_pixel, first_pixel + factor, last_face)
    return first_pixel, last_pixel - first_pixel
end

"""
$(TYPEDSIGNATURES)

Return a `Float32` array of physical values with the same shape as `raw`.
Replace values equal to `missing_value` with `NaN`, then decode all other values
as `value * value_scale + value_offset`. Use `missing_value = nothing` when
the raster has no missing-value marker.
"""
function decode_raster_values(raw, value_scale, value_offset, missing_value)
    decoded = Array{Float32}(undef, size(raw))
    @inbounds for idx in eachindex(raw)
        value = Float64(raw[idx])
        is_nodata = !isnothing(missing_value) && isequal(value, missing_value)
        decoded[idx] = is_nodata ? NaN32 : Float32(value * value_scale + value_offset)
    end
    return decoded
end

"""
$(TYPEDSIGNATURES)

Return `(longitude, latitude, data)` for a rectangular raster patch `raw`, whose
first dimension runs west to east and second dimension runs north to south.
The `geotransform` contains the full image's western and northern edges and pixel
spacing in degrees as `[western_edge, longitude_spacing, 0, northern_edge, 0,
latitude_spacing]`. `column_offset` and `row_offset` locate the patch's first
pixel within that image, counting from zero. Each cell of `raw` covers `factor × factor`
image pixels.

Compute pixel-center coordinates and reorder latitude and data from south to north.
Replace `missing_value` with `NaN`. Convert each remaining value `v` to
`Float32(v * value_scale + value_offset)`.
"""
function assemble_raster_window(raw, geotransform, column_offset, row_offset, value_scale, value_offset, missing_value,
                                factor = 1)
    x0, dx, _, y0, _, dy = geotransform
    Nx, Ny = size(raw)

    # Cell centers lie half a cell inward from their western and northern edges.
    longitude = [x0 + (column_offset + (i - 0.5) * factor) * dx for i in 1:Nx]
    # Reverse north-first rows so latitude and data run south to north.
    latitude  = reverse([y0 + (row_offset + (j - 0.5) * factor) * dy for j in 1:Ny])
    data = reverse(decode_raster_values(raw, value_scale, value_offset, missing_value), dims = 2)

    return longitude, latitude, data
end

#####
##### Hydraulic parameters straight from the dataset
#####

"""
$(TYPEDSIGNATURES)

Effective van Genuchten parameter fields for `grid` computed from `dataset`: the four
texture variables (`:sand_fraction`, `:silt_fraction`, `:clay_fraction`, `:bulk_density`)
are read over `region` onto a lattice with `grid`'s horizontal cells and the dataset's
depth layers, their gaps are inpainted, and the per-layer pedotransfer reduction runs
over `slab_depth`. Returns a NamedTuple of `porosity`, `residual_liquid_fraction`,
`inverse_air_entry_head`, `pore_size_uniformity`, `matching_point_conductivity`, and
`pore_connectivity_exponent`, each a `Field{Center, Center, Nothing}` on `grid`.
`dir` is the download directory; remaining
keyword arguments (`ptf`, `matching_heads`) pass to the reduction.
"""
function Lands.soil_hydraulic_properties(grid, dataset::OpenLandMapSoilDB;
                                         slab_depth,
                                         region = BoundingBox(grid),
                                         dir = default_download_directory(dataset),
                                         kw...)
    z = DataWrangling.z_interfaces(dataset)
    Nx, Ny, _ = size(grid)
    lattice = LatitudeLongitudeGrid(architecture(grid), eltype(grid);
                                    size = (Nx, Ny, length(z) - 1),
                                    longitude = λnodes(grid, Face(), Center(), Center()),
                                    latitude = φnodes(grid, Center(), Face(), Center()),
                                    z,
                                    topology = (Bounded, Bounded, Bounded))

    texture = map((:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)) do name
        field = Field(Metadatum(name; dataset, region, dir), lattice)
        gaps  = Field{Center, Center, Center}(lattice, Bool)
        interior(gaps) .= .!isfinite.(interior(field))
        inpaint_mask!(field, gaps)
        return field
    end

    layered = Lands.soil_hydraulic_properties(texture...; slab_depth, kw...)

    surface = (porosity = Field{Center, Center, Nothing}(grid),
               residual_liquid_fraction = Field{Center, Center, Nothing}(grid),
               inverse_air_entry_head = Field{Center, Center, Nothing}(grid),
               pore_size_uniformity = Field{Center, Center, Nothing}(grid),
               matching_point_conductivity = Field{Center, Center, Nothing}(grid),
               pore_connectivity_exponent = Field{Center, Center, Nothing}(grid))
    set!(surface, layered)
    return surface
end

end # module OpenLandMap
