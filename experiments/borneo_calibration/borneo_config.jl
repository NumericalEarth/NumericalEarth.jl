# Domain, window and cache layout shared by the ingestion and the model scripts.

using Oceananigans
using JLD2
import Dates: DateTime, Hour

refinement = parse(Int, get(ENV, "REFINEMENT", "1"))
resolution_km = round(Int, 111.32 / (9 * refinement))

latitude  = (0.5, 2.5)
longitude = (113.0, 115.0)
Nx = Ny = 18 * refinement

start_date = DateTime(2020, 4, 1)
end_date   = DateTime(2020, 4, 8)       # forcing window; runs stop before it
landcover_year = 2020

surface_layer_height  = 10
boundary_layer_height = 800

ingest_region = BoundingBox(longitude = (longitude[1] - 0.2, longitude[2] + 0.2),
                            latitude  = (latitude[1] - 0.2, latitude[2] + 0.2))
era5_region = BoundingBox(; latitude, longitude)

land_grid(arch = CPU(), FT = Float64) =
    LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny), latitude, longitude, topology = (Bounded, Bounded, Flat))

cache_directory = "surface_cache"
cache_file(name) = joinpath(cache_directory, "$(name)_r$(refinement).jld2")
load_cache(name) = jldopen(file -> file["data"], cache_file(name))
load_static() = jldopen(file -> NamedTuple{Tuple(Symbol.(keys(file)))}(Tuple(file[k] for k in keys(file))), cache_file("static"))

# ERA5-Land 0–28 cm volumetric water (layers 1 and 2, thickness-weighted) at hourly index `n`.
era5_land_soil_water(era5_land, n) = 0.25 .* era5_land.layer_1[n, :, :] .+ 0.75 .* era5_land.layer_2[n, :, :]
