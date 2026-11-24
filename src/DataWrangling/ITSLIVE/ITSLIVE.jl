module ITSLIVE

export ITSLIVEAntarctica

using Downloads: Downloads
using NumericalEarth: TYPEDSIGNATURES
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum, Dataset,
                       metadata_path, dataset_variable_name, bounding_box_suffix, projected_window,
                       projected_window_centers, validate_projected_region, write_projected_window,
                       x_interfaces, y_interfaces

using ..OpenLandMap: validate_geographic_northup


download_ITSLIVE_cache::String = ""
function __init__()
    global download_ITSLIVE_cache = DataWrangling.download_cache("ITSLIVE")
    return nothing
end

"""
    ITSLIVEAntarctica <: AbstractStaticDataset

ITS_LIVE version 2 static mosaic of Antarctic ice-flow velocity at 120 m, on the
Antarctic polar stereographic projection (EPSG:3031). Delivers the velocity components along the projection axes
(`:velocity_x`, `:velocity_y`), the speed (`:speed`) and their errors
(`:velocity_x_error`, `:velocity_y_error`, `:speed_error`), all in m/a.

The data stay on their native grid: construct the [`Metadatum`](@ref) with a
[`ProjectedBoundingBox`](@ref) in EPSG:3031 metres, and the native grid is a
`RectilinearGrid` on those coordinates. Cells without observations are `NaN`.
Reading the cloud-optimized GeoTIFFs requires ArchGDAL to be loaded
(`using ArchGDAL`); access is anonymous, no credentials needed.

Data source: https://its-live.jpl.nasa.gov/

```jldoctest
using NumericalEarth

metadatum = Metadatum(:velocity_x;
                      dataset = ITSLIVEAntarctica(),
                      region = ProjectedBoundingBox(x = (-1.7e6, -1.5e6), y = (-4e5, -2e5)))

metadatum.filename

# output
"ITSLIVE_Antarctica_velocity_x_x_-1.7e6_-1.5e6_y_-400000.0_-200000.0.nc"
```
"""
struct ITSLIVEAntarctica <: AbstractStaticDataset end

const ITSLIVEAntarcticaMetadatum = Metadatum{<:ITSLIVEAntarctica}

ITSLIVE_dataset_variable_names = Dict(
    :velocity_x       => "vx",
    :velocity_y       => "vy",
    :speed            => "v",
    :velocity_x_error => "vx_error",
    :velocity_y_error => "vy_error",
    :speed_error      => "v_error")

const ITSLIVE_cog_base = "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/cog"

cog_url(name) = string(ITSLIVE_cog_base, "/ITS_LIVE_velocity_120m_RGI19A_0000_v02_", name, ".tif")

#####
##### Dataset traits: the 120 m lattice of the RGI19A (Antarctica) mosaic, cell faces in metres
#####

const ITSLIVE_coordinate_system_code = 3031

DataWrangling.available_variables(::ITSLIVEAntarctica) = ITSLIVE_dataset_variable_names
DataWrangling.default_download_directory(::ITSLIVEAntarctica) = download_ITSLIVE_cache

Base.size(::ITSLIVEAntarctica, variable) = (45834, 38334, 1)
DataWrangling.x_interfaces(::ITSLIVEAntarctica) = (-2700007.5, 2800072.5)
DataWrangling.y_interfaces(::ITSLIVEAntarctica) = (-2200072.5, 2400007.5)

#####
##### Metadatum traits
#####

DataWrangling.is_three_dimensional(::ITSLIVEAntarcticaMetadatum) = false
DataWrangling.dataset_variable_name(data::ITSLIVEAntarcticaMetadatum) = ITSLIVE_dataset_variable_names[data.name]
DataWrangling.longitude_name(::ITSLIVEAntarcticaMetadatum) = "x"
DataWrangling.latitude_name(::ITSLIVEAntarcticaMetadatum)  = "y"

# Cells without observations (ocean, rock, gaps) stay NaN.
DataWrangling.default_inpainting(::ITSLIVEAntarcticaMetadatum) = nothing

DataWrangling.metadata_filename(::ITSLIVEAntarctica, name, date, region) =
    string("ITSLIVE_Antarctica_", name, "_", bounding_box_suffix(region), ".nc")

DataWrangling.validate_dataset_coverage(grid, metadatum::ITSLIVEAntarcticaMetadatum) =
    validate_projected_region(metadatum.region, metadatum.dataset)

#####
##### Download: window one cloud-optimized GeoTIFF into a regional NetCDF
#####

function Downloads.download(metadatum::ITSLIVEAntarcticaMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)

    path = metadata_path(metadatum)
    @root if !isfile(path)
        @info "Downloading ITS_LIVE Antarctica (120 m) $(metadatum.name) over $(summary(metadatum.region))..."
        source = "/vsicurl/" * cog_url(dataset_variable_name(metadatum))
        cog_window_to_netcdf(source, path, dataset_variable_name(metadatum), metadatum.dataset, metadatum.region)
    end
    return path
end

function DataWrangling.retrieve_data(metadatum::ITSLIVEAntarcticaMetadatum)
    data = Dataset(metadata_path(metadatum)) do ds
        Array(ds[dataset_variable_name(metadatum)][:, :])
    end
    return data
end

"""
$(TYPEDSIGNATURES)

Check that a source raster with `geotransform` and EPSG `coordinate_system_code` lies on
the lattice of `dataset`, so that the window computed from the dataset traits reads the
intended cells.
"""
function validate_source_lattice(geotransform, coordinate_system_code, dataset)
    validate_geographic_northup(geotransform)
    coordinate_system_code == ITSLIVE_coordinate_system_code ||
        error("Expected the Antarctic polar stereographic projection (EPSG:3031), " *
              "but the source declares EPSG:$coordinate_system_code.")

    Nx, Ny, _ = size(dataset, nothing)
    x₁, x₂ = x_interfaces(dataset)
    y₁, y₂ = y_interfaces(dataset)
    x₀, Δx, _, y₀, _, Δy = geotransform
    expected = (x₁, (x₂ - x₁) / Nx, y₂, -(y₂ - y₁) / Ny)
    all(isapprox.((x₀, Δx, y₀, Δy), expected)) ||
        error("The source lattice (x₀, Δx, y₀, Δy) = $((x₀, Δx, y₀, Δy)) differs from the " *
              "ITSLIVEAntarctica lattice $expected.")
    return nothing
end

"""
    read_cog_window(source, window)

Read the raster `window` (see [`projected_window`](@ref)) from the single-band cloud-optimized GeoTIFF
`source`, mask its nodata to `NaN`, apply the band scale and offset, and return
`(; geotransform, coordinate_system_code, data)` with `data` ordered west to east and south
to north.

Implemented in `ext/NumericalEarthArchGDALExt/itslive.jl` when ArchGDAL is loaded; the
fallback below fires only when the extension is not active.
"""
read_cog_window(source, window) =
    error("Reading ITS_LIVE COGs requires the ArchGDAL package. Load it with `using ArchGDAL`.")

function cog_window_to_netcdf(source, path, variable_name, dataset, region)
    window = projected_window(dataset, region)
    raster = read_cog_window(source, window)
    validate_source_lattice(raster.geotransform, raster.coordinate_system_code, dataset)
    x, y = projected_window_centers(dataset, window)

    write_projected_window(path, variable_name, x, y, raster.data; units = "m/a")

    return nothing
end

end # module ITSLIVE
