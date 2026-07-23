module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NumericalEarth: NumericalEarth

using NumericalEarth.DataWrangling: BoundingBox
const GloBFP3D = NumericalEarth.DataWrangling.GloBFP3D

function NumericalEarth.DataWrangling.IBCAO.reproject_ibcao_to_netcdf(tiff_path, nc_path)
    ArchGDAL.read(tiff_path) do src
        # Warp from EPSG:3996 (Polar Stereographic) to EPSG:4326 (WGS84)
        # at 0.01° resolution, clipping to 64–90°N
        ArchGDAL.gdalwarp([src],
            ["-t_srs", "EPSG:4326",
             "-te",    "-180", "64", "180", "90",  # xmin ymin xmax ymax
             "-tr",    "0.01", "0.01",             # target resolution (degrees)
             "-r",     "bilinear",                 # resampling method
             "-ot",    "Float32"]) do warped

            # ArchGDAL returns data as (Nx, Ny) with y from north to south (GDAL convention)
            data = Float32.(ArchGDAL.read(warped, 1))
            data = reverse(data, dims=2)

            Nx, Ny = size(data)  # expected: (36000, 2600)

            NCDataset(nc_path, "c") do ds
                defDim(ds, "lon", Nx)
                defDim(ds, "lat", Ny)

                lon_var = defVar(ds, "lon", Float64, ("lon",);
                                attrib = ["units" => "degrees_east",
                                          "long_name" => "longitude"])
                lat_var = defVar(ds, "lat", Float64, ("lat",);
                                attrib = ["units" => "degrees_north",
                                          "long_name" => "latitude"])
                z_var   = defVar(ds, "z",   Float32, ("lon", "lat");
                                attrib = ["long_name" => "elevation",
                                          "units"     => "m"])

                lon_var[:] = range(-180 + 0.005, 180 - 0.005; length=Nx)
                lat_var[:] = range(64 + 0.005, 90 - 0.005; length=Ny)
                z_var[:, :] = data
            end
        end
    end

    return nothing
end

#####
##### 3D-GloBFP building-footprint ingest (per-tile shapefiles → per-cell morphometry)
#####
#####
##### Discovers the figshare tile `.zip`s intersecting a BoundingBox, downloads them, reads
##### the EPSG:4326 footprint polygons under a spatial filter (never the whole continental
##### layer), reduces them to per-cell morphometry with the pure `GloBFP3D` core, and writes
##### a regional multi-variable lat/lon NetCDF that the shared `Field(::Metadatum, grid)`
##### path reads and regrids.
#####

# Fetch and cache the tile catalog: (name, download_url, west, south, east, north) for every
# tile across the ten figshare parts, parsed from the filenames. Cached as a TSV so the ten
# figshare API calls happen only once per machine.
function globfp3d_tile_catalog(cache_dir)
    catalog_path = joinpath(cache_dir, "tile_catalog.tsv")
    if isfile(catalog_path)
        entries = NamedTuple[]
        for line in eachline(catalog_path)
            name, url, W, S, E, N = split(line, '\t')
            push!(entries, (; name = String(name), url = String(url),
                              west = parse(Float64, W), south = parse(Float64, S),
                              east = parse(Float64, E), north = parse(Float64, N)))
        end
        return entries
    end

    mkpath(cache_dir)
    entries = NamedTuple[]
    file_regex = r"\"name\":\s*\"([^\"]+\.zip)\"[^{}]*?\"download_url\":\s*\"([^\"]+)\""
    for id in GloBFP3D.FIGSHARE_ARTICLE_IDS
        json = sprint() do io
            Downloads.download(GloBFP3D.figshare_article_url(id), io)
        end
        for m in eachmatch(file_regex, json)
            bounds = GloBFP3D.parse_tile_bounds(m[1])
            isnothing(bounds) && continue
            push!(entries, (; name = String(m[1]), url = String(m[2]),
                              west = bounds.west, south = bounds.south,
                              east = bounds.east, north = bounds.north))
        end
    end
    isempty(entries) && error("Could not build the 3D-GloBFP tile catalog from figshare.")

    open(catalog_path, "w") do io
        for e in entries
            println(io, join((e.name, e.url, e.west, e.south, e.east, e.north), '\t'))
        end
    end
    return entries
end

# Download a tile archive (idempotent, staged rename), returning the `/vsizip/` path to the
# shapefile inside it.
function globfp3d_download_tile(entry, cache_dir)
    zip_path = joinpath(cache_dir, entry.name)
    if !isfile(zip_path)
        staging = zip_path * ".part"
        try
            Downloads.download(entry.url, staging)
            mv(staging, zip_path; force = true)
        finally
            rm(staging; force = true)
        end
    end
    inner_shp = replace(entry.name, r"\.zip$" => ".shp")
    return string("/vsizip/", zip_path, "/", inner_shp)
end

# Burn the footprint `Height` attribute of one tile onto the region raster grid, combining
# with what is already there by max (tiles are disjoint, so this is just the union).
function globfp3d_rasterize_tile!(height, vsi_path, grid)
    east  = grid.west  + grid.Nx * grid.Δ
    north = grid.south + grid.Ny * grid.Δ
    ArchGDAL.read(vsi_path) do dataset
        ArchGDAL.gdalrasterize(dataset,
            ["-a", "Height",
             "-init", "0", "-a_nodata", "0",
             "-te", string(grid.west), string(grid.south), string(east), string(north),
             "-ts", string(grid.Nx), string(grid.Ny),
             "-ot", "Float32"]) do raster
            tile_height = Float64.(ArchGDAL.read(raster, 1))
            tile_height = reverse(tile_height, dims = 2)  # GDAL writes y north→south
            @. height = max(height, tile_height)
        end
    end
    return nothing
end

function NumericalEarth.DataWrangling.GloBFP3D.globfp3d_rasterize_to_netcdf(
        metadatum::GloBFP3D.BuildingFootprints3DMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("globfp3d_rasterize_to_netcdf requires a BoundingBox region.")

    cache_dir = joinpath(dirname(nc_path), "tiles")
    catalog = globfp3d_tile_catalog(cache_dir)
    tiles = filter(e -> GloBFP3D.tile_intersects(e, region), catalog)
    isempty(tiles) &&
        error("No 3D-GloBFP tiles intersect the requested region $(summary(region)).")

    grid = GloBFP3D.native_region_grid(region, GloBFP3D.native_cell_size(dataset))
    height = zeros(Float64, grid.Nx, grid.Ny)
    for tile in tiles
        vsi_path = globfp3d_download_tile(tile, cache_dir)
        globfp3d_rasterize_tile!(height, vsi_path, grid)
    end

    longitude = [grid.west  + (i - 1/2) * grid.Δ for i in 1:grid.Nx]
    latitude  = [grid.south + (j - 1/2) * grid.Δ for j in 1:grid.Ny]

    staging = nc_path * ".part"
    NCDataset(staging, "c") do ds
        defDim(ds, "lon", grid.Nx)
        defDim(ds, "lat", grid.Ny)
        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        lon_var[:] = longitude
        lat_var[:] = latitude
        var = defVar(ds, "building_height", Float64, ("lon", "lat"))
        var[:, :] = height
    end
    mv(staging, nc_path; force = true)
    return nothing
end

end # module NumericalEarthArchGDALExt
