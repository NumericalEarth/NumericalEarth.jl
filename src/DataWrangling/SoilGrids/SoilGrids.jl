module SoilGrids

export SoilGrids2

using Downloads: Downloads
using Oceananigans: Center, CPU
using Oceananigans.DistributedComputations: @root
using Oceananigans.Grids: x_domain, y_domain, λnodes, φnodes

using ..DataWrangling: DataWrangling,
    Dataset, DownloadProgress, AbstractStaticDataset, Metadatum, BoundingBox,
    GramPerKilogram, CentigramPerCubicCentimeter, HectogramPerCubicMeter, DecigramPerKilogram,
    metadata_path, metadata_url, dataset_variable_name, download_with_retries,
    bounding_box_suffix, native_grid

import Oceananigans

"""
    Statistic

Enum corresponding to the various prediction layers provided by the SoilGrids 2.0 dataset.
Since SoilGrids was produced using quantile regression, multiple output statistics are available
that capture uncertainty in estimates for each variable. The default is `Mean` which represents the
most likely value. The 5th (`Q5`), 50th (`Q50`), and 95% (`Q90`) percentiles/quantiles represent the
lower, middle, and upper bound of the predictions for each variable. These could be used in sensitivity
analyses to check how sensitive model simulations are to uncertainty in soil properties. Note that
the spread of the distribution should be expected to be larger in data-sparse regions such as
remote islands, deserts, and the Arctic.
"""
@enum Statistic Mean Q5 Q50 Q95

"""
    Resolution

The three ways [`SoilGrids2`](@ref) can source its data:

- `Clenshaw10km`: a small preprocessed global NetCDF projected onto a standard Clenshaw-Curtis grid.
  No ArchGDAL dependency required.
- `Grid1000m`: downloads directly from ISRIC (`files.isric.org`) at 1000 m, reprojecting
  each depth's Homolosine (IGH) VRT to EPSG:4326 with GDAL. Global by default (no region needed);
  at 1000 m the full global 6-depth stack for one variable+statistic is ~19 GB.
- `Grid250m`: the same ISRIC-direct pipeline at native resolution. A global read at 250 m
  would be ~300 GB per variable, so `Grid250m` requires a bounded `region::BoundingBox`.

Note that `Grid1000m`/`Grid250m` require `using ArchGDAL` (for the Homolosine → EPSG:4326
reprojection).
"""
@enum Resolution Grid250m Grid1000m Clenshaw10km

download_SoilGrids2_cache::String = ""
function __init__()
    global download_SoilGrids2_cache = DataWrangling.download_cache("SoilGrids2")
    return nothing
end

"""
    SoilGrids2(; statistic = Mean, resolution = Grid1000m)

SoilGrids 2.0 global soil properties (ISRIC, Poggio et al., 2021), predicted by quantile
regression forest over a large set of environmental covariates.

Provides `:sand_fraction`, `:silt_fraction`, `:clay_fraction`, `:coarse_fraction`, `:bulk_density`,
`:organic_carbon_density`, and `:soil_organic_carbon` over the six standard SoilGrids depth
intervals 0–5, 5–15, 15–30, 30–60, 60–100, and 100–200 cm — stored as a three-dimensional field
whose vertical axis carries the depths (deepest first).

`statistic` selects which prediction layer to load (see [`Statistic`](@ref)); `resolution`
selects how the data is sourced (see [`Resolution`](@ref)) — the default
`Grid1000m` downloads directly from ISRIC and requires `using ArchGDAL`.

Data source: https://www.isric.org/explore/soilgrids (CC-BY 4.0).

```jldoctest
using NumericalEarth

SoilGrids2()

# output
SoilGrids2(statistic = Mean, resolution = Grid1000m)
```
"""
@kwdef struct SoilGrids2 <: AbstractStaticDataset
    "Specifies which dataset layer to load variables from; see [SoilGrids.Statistic](@ref). Defaults to `Mean`"
    statistic::Statistic = Mean
    "Specifies how the data is sourced; see [`Resolution`](@ref). Defaults to `Grid1000m`"
    resolution::Resolution = Grid1000m
end

Base.summary(dataset::SoilGrids2) = string("SoilGrids2(statistic = ", dataset.statistic, ", resolution = ", dataset.resolution, ")")
Base.show(io::IO, dataset::SoilGrids2) = print(io, summary(dataset))

# Variable name mappings from NumericalEarth names to SoilGrids2 variable names
SoilGrids2_dataset_variable_names = Dict(
    :sand_fraction           => "sand",
    :silt_fraction           => "silt",
    :clay_fraction           => "clay",
    :coarse_fraction         => "cfvo",
    :bulk_density            => "bdod",
    :organic_carbon_density  => "ocd",
    :soil_organic_carbon     => "soc"
)

# ISRIC depth-range URL fragments, deepest first (matches SoilGrids2_z_interfaces above).
const SoilGrids2_depth_ranges = ("100-200cm", "60-100cm", "30-60cm", "15-30cm", "5-15cm", "0-5cm")

const SoilGrids2Metadatum = Metadatum{<:SoilGrids2}

const SoilGrids2_10km_url = "https://syncandshare.lrz.de/dl/fiVMyHskjL3FNbceuUFJev/soilgrids2_clenshaw989_10km.nc"

"""
    soilgrids_spacing_meters(resolution::Resolution)

The native pixel size in meters of a [`Resolution`](@ref). Note that, for the postprocessed `Clenshaw10km`
dataset, the native grid is spherical so the grid spacing in meters is only approximate.
"""
function soilgrids_spacing_meters(resolution::Resolution)
    resolution === Grid250m  && return 250
    resolution === Grid1000m && return 1000
    resolution === Clenshaw10km && return 10_000
    error("unsupported resolution $resolution")
end

"""
    soilgrids_statistic_url_name(stat::Statistic)

The ISRIC VRT filename fragment for `stat` (e.g. `Q0.05` for `Q5`), as used in
`https://files.isric.org/soilgrids/latest/data/{var}/{var}_{depth}_{stat}.vrt`.
"""
soilgrids_statistic_url_name(stat::Statistic) =
    stat === Mean ? "mean"  :
    stat === Q5   ? "Q0.05" :
    stat === Q50  ? "Q0.5"  :
    stat === Q95  ? "Q0.95" :
    error("unhandled SoilGrids2 Statistic $stat")

"""
    soilgrids_vsicurl_source(var, depth, stat_name)

The `/vsicurl` GDAL source string for one ISRIC SoilGrids2 depth VRT, e.g.
`soilgrids_vsicurl_source("clay", "0-5cm", "mean")`.

```jldoctest
using NumericalEarth.DataWrangling.SoilGrids: soilgrids_vsicurl_source

soilgrids_vsicurl_source("clay", "0-5cm", "mean")

# output
"/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt"
```
"""
soilgrids_vsicurl_source(var, depth, stat_name) =
    string("/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=",
           "https://files.isric.org/soilgrids/latest/data/", var, "/",
           var, "_", depth, "_", stat_name, ".vrt")

#####
##### Dataset methods
#####

DataWrangling.available_variables(::SoilGrids2) = SoilGrids2_dataset_variable_names
DataWrangling.default_download_directory(::SoilGrids2) = download_SoilGrids2_cache

function Base.size(dataset::SoilGrids2, variable)
    if dataset.resolution === Clenshaw10km
        return (3956, 1979, 6, 4)
    else
        Δ = soilgrids_spacing_meters(dataset.resolution) / 111320
        Nx = round(Int, 360 / Δ)
        Ny = round(Int, 180 / Δ)
        return (Nx, Ny, 6)
    end
end

function DataWrangling.longitude_interfaces(dataset::SoilGrids2)
    dataset.resolution === Clenshaw10km && return (0, 360)
    return (-180, 180)
end

function DataWrangling.z_interfaces(dataset::SoilGrids2)
    dataset.resolution === Clenshaw10km && return [-200, -100, -60, -30, -15, -5, 0]
    return [-2.0, -1.0, -0.6, -0.3, -0.15, -0.05, 0]
end

DataWrangling.latitude_interfaces(::SoilGrids2) = (-90, 90)
DataWrangling.reversed_latitude_axis(dataset::SoilGrids2)  = dataset.resolution === Clenshaw10km
DataWrangling.reversed_vertical_axis(dataset::SoilGrids2)  = dataset.resolution === Clenshaw10km
DataWrangling.windowed_retrieval(dataset::SoilGrids2)      = dataset.resolution === Grid250m

function DataWrangling.metadata_filename(dataset::SoilGrids2, name, date, region)
    dataset.resolution === Clenshaw10km && return "SoilGrids2_clenshaw_10km_full.nc"
    resolution_tag = dataset.resolution === Grid1000m ? "1000m" : "250m"
    return string("SoilGrids2_", name, "_", dataset.statistic, "_", resolution_tag, "_",
                 bounding_box_suffix(region), ".nc")
end

"""
    validate_dataset_coverage(grid, metadata::SoilGrids2Metadatum)

`SoilGrids2(resolution = Grid250m)` must be used with a bounded region; at the native 250m
resolution, a global read would be ~300 GB per variable. `Grid1000m` and `Clenshaw10km` are
unrestricted and default to global coverage.
"""
function DataWrangling.validate_dataset_coverage(grid, metadata::SoilGrids2Metadatum)
    dataset = metadata.dataset
    dataset.resolution === Grid250m || return nothing

    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("SoilGrids2(resolution = Grid250m) must be used with a bounded region. " *
              "At native 250 m a global read would be ~300 GB per variable; it is never read " *
              "in full. Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:$(metadata.name); dataset = SoilGrids2(resolution = Grid250m),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### Metadatum methods
#####

DataWrangling.is_three_dimensional(::SoilGrids2Metadatum) = true
DataWrangling.dataset_variable_name(data::SoilGrids2Metadatum) = SoilGrids2_dataset_variable_names[data.name]
DataWrangling.longitude_name(::SoilGrids2Metadatum) = "lon"
DataWrangling.latitude_name(::SoilGrids2Metadatum) = "lat"
DataWrangling.default_inpainting(md::SoilGrids2Metadatum) = nothing

# Only the pre-baked LRZ file carries a raw integer sentinel; the other two resolutions are
# already NaN-masked on disk.
DataWrangling.missing_value(md::SoilGrids2Metadatum) =
    md.dataset.resolution === Clenshaw10km ? -32768 : missing

function DataWrangling.conversion_units(metadatum::SoilGrids2Metadatum)
    if metadatum.name ∈ (:sand_fraction, :silt_fraction, :clay_fraction, :coarse_fraction)
        # Note that coarse_fraction is technically in cm³/dm³, but the conversion factor is the same, so we ignore that here
        return GramPerKilogram()
    elseif metadatum.name == :bulk_density
        return CentigramPerCubicCentimeter()
    elseif metadatum.name == :organic_carbon_density
        return HectogramPerCubicMeter()
    elseif metadatum.name == :soil_organic_carbon
        return DecigramPerKilogram()
    else
        return nothing
    end
end

#####
##### Regional/global raster geometry for the ISRIC-direct pipeline
#####

"""
    soilgrids_raster_geometry(metadatum)

Geometry of the raster to materialize for `metadatum`'s ISRIC-direct pipeline (`Grid1000m`
or `Grid250m`), taken from [`native_grid`](@ref) so the file the reprojection writes lands
on the cells the read path indexes — global when `metadatum.region` is `nothing`
(`Grid1000m`'s default), or the region's window when it is a `BoundingBox`
(`Grid250m`'s required case).
"""
function soilgrids_raster_geometry(metadatum::SoilGrids2Metadatum)
    grid = native_grid(metadatum, CPU())
    west, east   = x_domain(grid)
    south, north = y_domain(grid)
    Nx, Ny, _ = size(grid)

    return (; west, south, east, north, Nx, Ny,
              longitude = collect(λnodes(grid, Center())),
              latitude  = collect(φnodes(grid, Center())))
end

#####
##### Download
#####

function DataWrangling.metadata_url(m::SoilGrids2Metadatum)
    m.dataset.resolution === Clenshaw10km && return SoilGrids2_10km_url
    error("metadata_url is only defined for SoilGrids2(resolution = Clenshaw10km); the " *
          "$(m.dataset.resolution) pipeline downloads directly from ISRIC per depth inside " *
          "`Downloads.download`, with no single URL.")
end

function Downloads.download(metadatum::SoilGrids2Metadatum)
    dataset = metadatum.dataset
    filepath = metadata_path(metadatum)

    if dataset.resolution === Clenshaw10km
        @root if !isfile(filepath)
            @info "Downloading SoilGrids2 (~10 km) data: $(metadatum.name) in $(metadatum.dir)..."
            download_with_retries(metadata_url(metadatum), filepath; progress = DownloadProgress())
        end
        return filepath
    end

    DataWrangling.validate_dataset_coverage(nothing, metadatum)
    @root if !isfile(filepath)
        @info "Downloading SoilGrids2 ($(soilgrids_spacing_meters(dataset.resolution)) m) " *
              "data: $(metadatum.name) ($(dataset.statistic)) over $(summary(metadatum.region))..."
        soilgrids_variable_to_netcdf(metadatum, filepath)
    end
    return filepath
end

"""
    soilgrids_variable_to_netcdf(metadatum, nc_path)

Download and reproject the six SoilGrids2 depth VRTs for `metadatum` directly from ISRIC
(`files.isric.org`, via GDAL's `/vsicurl`), stacking them into a `(lon, lat, depth)` NetCDF at
`nc_path`.

Implemented in `ext/NumericalEarthArchGDALExt/soilgrids.jl` when ArchGDAL is loaded; the
fallback below fires only when the extension is not active.
"""
soilgrids_variable_to_netcdf(metadatum, nc_path) =
    error("Reading SoilGrids2 ISRIC data ($(metadatum.dataset.resolution)) requires the ArchGDAL " *
          "package (for the Homolosine → EPSG:4326 reprojection). Load it with `using ArchGDAL`.")

#####
##### Read
#####

function DataWrangling.retrieve_data(metadata::SoilGrids2Metadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    dataset = metadata.dataset

    if dataset.resolution === Clenshaw10km
        data = Dataset(path) do ds
            l = Int(dataset.statistic) + 1
            data = ds[name][:, :, :, l]
            # Reverse vertical axis to be increasing upwards
            reverse(data, dims = 3)
        end
        # Reverse latitude axis
        data = reverse(data, dims = 2)
        return data
    end

    # ISRIC-direct NetCDFs are already stored deepest-first, ascending latitude — no reversal.
    data = Dataset(path) do ds
        Array(ds[name][:, :, :])
    end
    return data
end

Oceananigans.Fields.location(::SoilGrids2Metadatum) = (Center, Center, Center)

end # module
