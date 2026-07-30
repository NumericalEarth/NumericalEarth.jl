module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

using Oceananigans: Center, CPU
using Oceananigans.Grids: λnodes, φnodes

using NumericalEarth.DataWrangling: BoundingBox, native_grid,
                                    cmr_granules, earthdata_download_cached
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name, asterged_version,
                                             asterged_decode_emissivity, asterged_decode_uncertainty,
                                             broadband_map, place_tile!,
                                             OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

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
##### Advanced Spaceborne Thermal Emission and Reflection Radiometer 
##### Global Emissivity Database version 3 (ASTER-GEDv3):
##### https://lpdaac.usgs.gov/products/ag100v003/
#####
##### ASTER GED emissivity ingest: resolve the 1°×1° HDF5 tiles intersecting the region
##### via NASA CMR, download them with Earthdata credentials, and write a regional NetCDF
##### of the broadband emissivity and uncertainty. Requires GDAL_jll with the HDF5 driver.
#####

# Open an HDF5 subdataset via GDAL's `HDF5:"file"://path` syntax and return the full
# raster array. Multi-band datasets come back as `(Nx, Ny, nbands)`.
function read_asterged_subdataset(h5_path, layer)
    name = string("HDF5:\"", h5_path, "\":", layer)
    return ArchGDAL.read(name) do dataset
        ArchGDAL.read(dataset)
    end
end

function NumericalEarth.DataWrangling.ASTERGED.asterged_tiles_to_netcdf(metadatum, nc_path::AbstractString)
    dataset = metadatum.dataset
    bbox = metadatum.region
    (bbox isa BoundingBox && !isnothing(bbox.longitude) && !isnothing(bbox.latitude)) ||
        error("asterged_tiles_to_netcdf requires a BoundingBox region.")

    # Write on the same native grid the Field will use, so tile cells land on file
    # cells by construction.
    grid = native_grid(metadatum, CPU())
    longitude = collect(λnodes(grid, Center()))
    latitude  = collect(φnodes(grid, Center()))
    Nx, Ny = length(longitude), length(latitude)
    Δλ = (longitude[end] - longitude[1]) / max(Nx - 1, 1)
    Δφ = (latitude[end]  - latitude[1])  / max(Ny - 1, 1)

    short_name = asterged_short_name(dataset)
    version = asterged_version(dataset)
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # Pad the query a cell so every file cell is covered. CMR needs [-180, 180]
    # longitudes, and reads a folded west > east as crossing the antimeridian.
    to_pm180(λ) = rem(λ, 360, RoundNearest)
    query = BoundingBox(longitude = (to_pm180(longitude[1] - Δλ), to_pm180(longitude[end] + Δλ)),
                        latitude  = (latitude[1]  - Δφ, latitude[end]  + Δφ))
    granule_urls = cmr_granules(short_name, version, query)
    isempty(granule_urls) &&
        error("CMR returned no $(short_name).$(version) tiles for region $(bbox).")

    emissivity  = fill(NaN32, Nx, Ny)
    uncertainty = fill(NaN32, Nx, Ny)
    # 0 = land, 1 = water, −9999 = fill (the GEE coding on the AG100/AG1KM tiles, not
    # the LP DAAC 1/2 coding).
    land_water_map = fill(NaN32, Nx, Ny)

    tile_cache = joinpath(dirname(nc_path), string(short_name, "_tiles"))
    mkpath(tile_cache)

    for url in granule_urls
        h5 = earthdata_download_cached(url, tile_cache)

        tile_longitude = Float64.(read_asterged_subdataset(h5, "//Geolocation/Longitude")[:, 1, 1])
        tile_latitude  = Float64.(read_asterged_subdataset(h5, "//Geolocation/Latitude")[1, :, 1])

        # GDAL returns (Nx, Ny, 5) band-last; permute to the band-first (5, Nx, Ny)
        # the broadband collapse expects.
        mean_bands = permutedims(asterged_decode_emissivity.(read_asterged_subdataset(h5, "//Emissivity/Mean")), (3, 1, 2))
        sdev_bands = permutedims(asterged_decode_uncertainty.(read_asterged_subdataset(h5, "//Emissivity/SDev")), (3, 1, 2))

        lwmap_tile = Float32.(read_asterged_subdataset(h5, "//Land_Water_Map/LWmap")[:, :, 1])

        place_tile!(emissivity,     broadband_map(mean_bands, coefficients), tile_longitude, tile_latitude, longitude, latitude)
        place_tile!(uncertainty,    broadband_map(sdev_bands, coefficients), tile_longitude, tile_latitude, longitude, latitude)
        place_tile!(land_water_map, lwmap_tile,                              tile_longitude, tile_latitude, longitude, latitude)
    end

    all(isnan, emissivity) &&
        error("No ASTER GED tile cells fell within region $(bbox).")

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defVar(ds, "lon", Float64, ("lon",); attrib = ["units" => "degrees_east", "long_name" => "longitude"])[:] = longitude
        defVar(ds, "lat", Float64, ("lat",); attrib = ["units" => "degrees_north", "long_name" => "latitude"])[:] = latitude
        defVar(ds, "emissivity", Float32, ("lon", "lat"))[:, :] = emissivity
        defVar(ds, "emissivity_uncertainty", Float32, ("lon", "lat"))[:, :] = uncertainty
        defVar(ds, "land_water_map", Float32, ("lon", "lat"))[:, :] = land_water_map
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

function NumericalEarth.DataWrangling.OpenLandMap.read_cog_window(source, bbox::BoundingBox)
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

end # module NumericalEarthArchGDALExt
