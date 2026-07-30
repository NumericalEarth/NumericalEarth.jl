module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using GDAL: OGREnvelope, ogr_l_getextent, vsireaddirrecursive
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

const OpenLandMap = NumericalEarth.DataWrangling.OpenLandMap
const GloBFP3D = NumericalEarth.DataWrangling.GloBFP3D
const BoundingBox = NumericalEarth.DataWrangling.BoundingBox

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
##### OpenLandMap-soilDB windowed COG reader
#####

const vsicurl_configured = Ref(false)

function configure_vsicurl!()
    vsicurl_configured[] && return nothing
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    ArchGDAL.setconfigoption("GDAL_HTTP_MULTIRANGE", "YES")

    if !haskey(ENV, "CURL_CA_BUNDLE")
        ENV["CURL_CA_BUNDLE"] = NetworkOptions.ca_roots_path()
    end
    vsicurl_configured[] = true
    return nothing
end

# Decode raw COG integers to Float32 physical values. Order matters: mask nodata
# to NaN first, then apply the band scale/offset (a scaled fill is a spurious value).
function decode_cog_window(raw, scale, offset, nodata)
    decoded = Array{Float32}(undef, size(raw))
    @inbounds for idx in eachindex(raw)
        value = Float64(raw[idx])
        is_nodata = !isnothing(nodata) && isequal(value, nodata)
        decoded[idx] = is_nodata ? NaN32 : Float32(value * scale + offset)
    end
    return decoded
end

# The windowing math and the north→south row reversal below assume a north-up,
# axis-aligned geographic (EPSG:4326, degrees) grid.
function validate_geographic_northup(dataset, geotransform)
    _, dx, rx, _, ry, dy = geotransform
    (rx == 0 && ry == 0) ||
        error("Windowed COG reader requires an axis-aligned grid (no rotation/shear); " *
              "got geotransform $geotransform.")
    (dx > 0 && dy < 0) ||
        error("Windowed COG reader assumes west→east (Δλ > 0) and north→south (Δφ < 0) " *
              "pixel order; got Δλ = $dx, Δφ = $dy.")

    # If the source declares a CRS, require EPSG:4326 — the windowing is done in
    # degrees, so a projected grid would silently land the window in the wrong place.
    wkt = ArchGDAL.getproj(dataset)
    if !isempty(wkt)
        epsg = try
            ArchGDAL.toEPSG(ArchGDAL.importWKT(wkt))
        catch  # WKT without an EPSG authority tag: rely on the geometry checks above.
            nothing
        end
        isnothing(epsg) || epsg == 4326 ||
            error("Windowed COG reader expects EPSG:4326 lon/lat in degrees; " *
                  "the source declares EPSG:$epsg.")
    end
    return nothing
end

function OpenLandMap.read_cog_window(source, bbox::BoundingBox)
    configure_vsicurl!()

    W, E = bbox.longitude
    S, N = bbox.latitude

    return ArchGDAL.read(source) do ds
        geotransform = ArchGDAL.getgeotransform(ds)  # [x₀, Δλ, 0, y₀, 0, Δφ]
        validate_geographic_northup(ds, geotransform)
        x0, dx, _, y0, _, dy = geotransform
        width  = ArchGDAL.width(ds)
        height = ArchGDAL.height(ds)

        # Pad one native cell on each side so the window is a strict superset of the
        # framework's center-bracketed native grid; otherwise the grid can hold one
        # more cell than the file, forcing a clamped read that shifts the whole
        # window by a pixel and duplicates the outermost row/column.
        xoff  = clamp(floor(Int, (W - x0) / dx) - 1, 0, width - 1)
        yoff  = clamp(floor(Int, (N - y0) / dy) - 1, 0, height - 1)
        xsize = clamp(ceil(Int, (E - x0) / dx) + 1 - xoff, 1, width - xoff)
        ysize = clamp(ceil(Int, (S - y0) / dy) + 1 - yoff, 1, height - yoff)

        band   = ArchGDAL.getband(ds, 1)
        scale  = ArchGDAL.getscale(band)
        offset = ArchGDAL.getoffset(band)
        nodata = ArchGDAL.getnodatavalue(band)

        raw = ArchGDAL.read(ds, 1, xoff, yoff, xsize, ysize)  # (lon, lat), north-first
        # Pixel centers: x₀ is the corner of pixel 0, so the 0-based column
        # (xoff + i - 1) plus half a pixel (+0.5) gives the center: xoff + i - 0.5.
        longitude = [x0 + (xoff + i - 0.5) * dx for i in 1:xsize]
        # COGs store rows north-first (Δφ < 0); reverse latitude and data so both
        # come out ascending (south-to-north), per CF convention.
        latitude  = reverse([y0 + (yoff + j - 0.5) * dy for j in 1:ysize])
        data = reverse(decode_cog_window(raw, scale, offset, nodata), dims = 2)
        return (longitude, latitude, data)
    end
end

#####
##### 3D-GloBFP building-footprint ingest (per-tile shapefiles → per-cell morphometry)
#####
#####
##### Discovers the figshare tile `.zip`s intersecting a BoundingBox, downloads them, rasterizes
##### the EPSG:4326 footprint `Height`s onto a fine lat/lon raster, and writes a regional
##### building-height NetCDF. The shared `Field(::Metadatum, grid)` path reads and regrids it, and
##### `building_morphometry` reduces it to per-cell morphometry.
#####

# Fetch and cache the tile catalog: (name, download_url, west, south, east, north) for every
# tile across the ten figshare parts, parsed from the filenames. Cached as a TSV so the ten
# figshare API calls happen only once per machine.
function globfp3d_tile_catalog(cache_dir)
    catalog_path = joinpath(cache_dir, "tile_catalog.tsv")
    if isfile(catalog_path)
        entries = read_tile_catalog(catalog_path)
        isnothing(entries) || return entries
        # A malformed or partial catalog (e.g. an interrupted legacy write) is discarded and rebuilt.
        rm(catalog_path; force = true)
    end

    mkpath(cache_dir)
    entries = NamedTuple[]
    # figshare file objects are flat JSON (no nested braces), so match each object and read its
    # `name`/`download_url` independently: order-independent, and a link-only file (null
    # `download_url`) simply fails the url match and is skipped. Non-file objects are filtered by
    # `parse_tile_bounds`.
    object_regex = r"\{[^{}]*\}"
    name_regex   = r"\"name\"\s*:\s*\"([^\"]+)\""
    url_regex    = r"\"download_url\"\s*:\s*\"([^\"]+)\""
    for id in GloBFP3D.FIGSHARE_ARTICLE_IDS
        json = sprint() do io
            Downloads.download(GloBFP3D.figshare_article_url(id), io)
        end
        for object in eachmatch(object_regex, json)
            name = match(name_regex, object.match)
            url  = match(url_regex,  object.match)
            (isnothing(name) || isnothing(url)) && continue
            bounds = GloBFP3D.parse_tile_bounds(name[1])
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

# Read a cached tile catalog, returning `nothing` (so the caller rebuilds) if any line is not the
# expected six tab-separated fields with parseable coordinates — e.g. a truncated last line from an
# interrupted write, or a legacy format.
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

# Download a tile archive (idempotent, staged rename), returning the `/vsizip/` path to the
# shapefile inside it.
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

# The shapefile inside a tile archive can have a different basename than the zip, or sit in a
# subfolder, so find the `.shp` by listing the archive recursively instead of assuming its name.
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

# Burn the footprint `Height`s of one tile onto the region raster, over just the window the tile's
# features occupy — found from the layer extent, which bounds every footprint including any that
# overhangs the nominal tile edge — so cost scales with the tile rather than the whole region.
# Disjoint tiles combine by max (a plain union).
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

    Δ = GloBFP3D.native_cell_size(dataset)
    grid = GloBFP3D.native_region_grid(region, Δ, Δ)
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
end # module NumericalEarthArchGDALExt
