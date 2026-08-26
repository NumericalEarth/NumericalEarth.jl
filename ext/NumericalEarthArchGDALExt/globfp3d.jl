#####
##### The first global three-dimensional building footprint dataset (3D-GloBFP):
##### https://doi.org/10.5194/essd-16-5357-2024
#####
##### 3D-GloBFP building-footprint ingest: download the figshare tile shapefiles intersecting a
##### `BoundingBox` and rasterize their footprint `Height`s into a building-height NetCDF.
##### Requires GDAL_jll with the OGR shapefile driver.
#####

# Cached as a TSV so the figshare API calls happen once per machine.
function globfp3d_tile_catalog(cache_dir)
    catalog_path = joinpath(cache_dir, "tile_catalog.tsv")
    if isfile(catalog_path)
        entries = read_tile_catalog(catalog_path)
        isnothing(entries) || return entries
        rm(catalog_path; force = true)
    end

    mkpath(cache_dir)
    entries = NamedTuple[]
    # figshare file objects are flat JSON (no nested braces), so each object can be matched whole;
    # a link-only file, with a null `download_url`, is skipped.
    object_regex = r"\{[^{}]*\}"
    name_regex   = r"\"name\"\s*:\s*\"([^\"]+)\""
    url_regex    = r"\"download_url\"\s*:\s*\"([^\"]+)\""
    for id in GLOBFP3D_FIGSHARE_ARTICLE_IDS
        json = sprint() do io
            Downloads.download(figshare_article_url(id), io)
        end
        for object in eachmatch(object_regex, json)
            name = match(name_regex, object.match)
            url  = match(url_regex,  object.match)
            (isnothing(name) || isnothing(url)) && continue
            bounds = globfp3d_parse_tile_bounds(name[1])
            isnothing(bounds) && continue
            push!(entries, (; name = String(name[1]), url = String(url[1]),
                              west = bounds.west, south = bounds.south,
                              east = bounds.east, north = bounds.north))
        end
    end
    isempty(entries) && error("Could not build the 3D-GloBFP tile catalog from figshare.")

    staging = catalog_path * ".part"
    open(staging, "w") do io
        for e in entries
            println(io, join((e.name, e.url, e.west, e.south, e.east, e.north), '\t'))
        end
    end
    mv(staging, catalog_path; force = true)
    return entries
end

# Returns `nothing`, so the caller rebuilds, if a line is not six tab-separated fields with
# parseable coordinates — a truncated write, or a stale format.
function read_tile_catalog(catalog_path)
    entries = NamedTuple[]
    for line in eachline(catalog_path)
        isempty(line) && continue
        fields = split(line, '\t')
        length(fields) == 6 || return nothing
        name, url, W, S, E, N = fields
        coordinates = tryparse.(Float64, (W, S, E, N))
        any(isnothing, coordinates) && return nothing
        west, south, east, north = coordinates
        push!(entries, (; name = String(name), url = String(url), west, south, east, north))
    end
    return isempty(entries) ? nothing : entries
end

# Returns the `/vsizip/` path to the shapefile inside the tile archive, downloading it first if
# it is not already cached.
function globfp3d_download_tile(entry, cache_dir)
    zip_path = joinpath(cache_dir, entry.name)
    if !isfile(zip_path)
        staging = tempname(cache_dir)  # unique per process, so concurrent fetches don't collide
        try
            Downloads.download(entry.url, staging)
            mv(staging, zip_path; force = true)
        finally
            rm(staging; force = true)
        end
    end
    return globfp3d_vsi_shapefile(zip_path)
end

# The `.shp` basename can differ from the zip's and can sit in a subfolder, so list the archive.
function globfp3d_vsi_shapefile(zip_path)
    vsi_root = string("/vsizip/", zip_path)
    entries = vsireaddirrecursive(vsi_root)
    (isnothing(entries) || isempty(entries)) &&
        error("Could not list the 3D-GloBFP tile archive $zip_path.")
    index = findfirst(name -> endswith(lowercase(name), ".shp"), entries)
    isnothing(index) &&
        error("No shapefile (.shp) found inside 3D-GloBFP tile archive $zip_path.")
    return string(vsi_root, "/", entries[index])
end

# Burn one tile's footprint `Height`s onto the region raster. The window comes from the layer
# extent rather than the tile's nominal bounds, because footprints can overhang the tile edge.
function globfp3d_rasterize_tile!(height, vsi_path, grid)
    ArchGDAL.read(vsi_path) do dataset
        layer = ArchGDAL.getlayer(dataset, 0)
        envelope = Ref(OGREnvelope(0, 0, 0, 0))
        err = ogr_l_getextent(layer, envelope, true)
        err == 0 ||  # OGRERR_NONE; a nonzero code leaves envelope at (0,0,0,0) → an off-target window
            error("Failed to compute the extent of the 3D-GloBFP tile layer at $vsi_path (OGRErr $err).")
        extent = envelope[]

        i₁ = clamp(floor(Int, (extent.MinX - grid.west)  / grid.Δλ) + 1, 1, grid.Nx)
        i₂ = clamp(ceil( Int, (extent.MaxX - grid.west)  / grid.Δλ),     1, grid.Nx)
        j₁ = clamp(floor(Int, (extent.MinY - grid.south) / grid.Δφ) + 1, 1, grid.Ny)
        j₂ = clamp(ceil( Int, (extent.MaxY - grid.south) / grid.Δφ),     1, grid.Ny)
        (i₁ ≤ i₂ && j₁ ≤ j₂) || return nothing

        west  = grid.west  + (i₁ - 1) * grid.Δλ
        south = grid.south + (j₁ - 1) * grid.Δφ
        Nx = i₂ - i₁ + 1
        Ny = j₂ - j₁ + 1
        east  = west  + Nx * grid.Δλ
        north = south + Ny * grid.Δφ

        ArchGDAL.gdalrasterize(dataset,
            ["-a", "Height",
             "-init", "0", "-a_nodata", "0",
             "-te", string(west), string(south), string(east), string(north),
             "-ts", string(Nx), string(Ny),
             "-ot", "Float32"]) do raster
            tile_height = Float64.(ArchGDAL.read(raster, 1))
            tile_height = reverse(tile_height, dims = 2)  # GDAL writes y north→south
            window = view(height, i₁:i₂, j₁:j₂)
            @. window = max(window, tile_height)
        end
    end
    return nothing
end

function NumericalEarth.DataWrangling.GloBFP3D.globfp3d_rasterize_to_netcdf(
        metadatum::GlobalBuildingFootprints3DMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("globfp3d_rasterize_to_netcdf requires a BoundingBox region.")

    cache_dir = joinpath(dirname(nc_path), "tiles")
    catalog = globfp3d_tile_catalog(cache_dir)
    tiles = filter(e -> bounding_box_intersects(e, region), catalog)
    isempty(tiles) && return nothing   # no file: `Downloads.download` reports no intersecting tiles

    Δ = globfp3d_native_cell_size(dataset)
    grid = native_region_grid(region, Δ, Δ)
    height = zeros(Float64, grid.Nx, grid.Ny)
    for tile in tiles
        vsi_path = globfp3d_download_tile(tile, cache_dir)
        globfp3d_rasterize_tile!(height, vsi_path, grid)
    end

    longitude = [grid.west  + (i - 1/2) * grid.Δλ for i in 1:grid.Nx]
    latitude  = [grid.south + (j - 1/2) * grid.Δφ for j in 1:grid.Ny]

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
