#####
##### ETH Sentinel-2 canopy-height COGs: window the 3° 10 m tiles served over the libdrive
##### WebDAV share and either write a regional NetCDF or drop the heights onto a model grid.
#####

# GDAL config for the ETH libdrive WebDAV read, scoped per-read by `with_gdal_config`: the
# public read-only share token as basic-auth, plus transient-failure retries. Scoping keeps
# the token from leaking into later unrelated reads. (`configure_vsicurl!` already suppresses
# /vsicurl directory listing session-wide.)
eth_http_config() =
    ["GDAL_HTTP_USERPWD"     => ETH_LIBDRIVE_TOKEN * ":",
     "GDAL_HTTP_MAX_RETRY"   => "3",
     "GDAL_HTTP_RETRY_DELAY" => "1"]

# Mosaic + window the intersecting COG tiles under `geometry` (the gdalwarp `-te`/`-tr`
# or `-te`/`-ts` options), returning `(; data, longitude, latitude)` — the height array in
# (Nx, Ny) order with latitude increasing south→north, and the cell-center coordinates read
# straight off the warped geotransform. A canopy product tiles only over land, so a 3° cell
# with no published tile is a legitimate miss (open ocean), not an error — skip the ones that
# fail to open and mosaic the rest, but surface every read error if *all* fail (an all-ocean
# region and a network/TLS/credential failure both leave nothing opened). `identity.` narrows
# the collected vector back to the concrete dataset type so `gdalwarp`'s
# `Vector{<:AbstractDataset}` method still dispatches.
function warp_canopy_sources(sources, geometry; resampling, nodata = nothing)
    configure_vsicurl!()
    opened = []
    read_errors = Pair{Any, Any}[]
    for source in sources
        try
            push!(opened, ArchGDAL.read(source))
        catch err
            err isa InterruptException && rethrow()
            push!(read_errors, source => err)
        end
    end
    isempty(opened) && error(
        "No canopy-height tiles could be read for the requested region. The product " *
        "publishes no tile over open ocean, so an all-ocean region legitimately yields " *
        "nothing — but a network, TLS, or credential failure produces the same empty " *
        "result. Underlying read errors:\n" *
        join(("  $source: $(sprint(showerror, err))" for (source, err) in read_errors), "\n"))
    # A land tile that fails transiently is dropped just like an absent ocean tile, so its
    # area silently returns as no-data/NaN — warn so the hole is at least visible.
    isempty(read_errors) || @warn string(
        length(read_errors), " of ", length(sources), " canopy-height tiles were dropped ",
        "and return no-data/NaN. Absent tiles over open ocean are expected, but a network, ",
        "TLS, or credential failure looks identical here. Dropped:\n",
        join(("  $source: $(sprint(showerror, err))" for (source, err) in read_errors), "\n"))
    datasets = identity.(opened)
    # Declare the categorical no-data byte so `-r average` drops it from cell means rather
    # than blending it in; all-no-data cells then come out as `nodata` for the caller to mask.
    nodata_options = isnothing(nodata) ? String[] :
                     String["-srcnodata", string(nodata), "-dstnodata", string(nodata)]
    options = vcat(String["-t_srs", "EPSG:4326"], geometry,
                   String["-r", resampling, "-ot", "Float32"], nodata_options)
    try
        return ArchGDAL.gdalwarp(datasets, options) do warped
            λ₀, Δλ, _, φ₀, _, Δφ = ArchGDAL.getgeotransform(warped)   # Δφ < 0, rows run north→south
            band = Float32.(ArchGDAL.read(warped, 1))
            Nx, Ny = size(band)
            # Cell centers off the geotransform stay exact even when `-tr` snaps the extent
            # to whole pixels. Reverse the rows and the latitudes together: north→south→north.
            data = reverse(band, dims = 2)
            longitude = [λ₀ + (i - 0.5) * Δλ for i in 1:Nx]
            latitude  = reverse([φ₀ + (j - 0.5) * Δφ for j in 1:Ny])
            return (; data, longitude, latitude)
        end
    finally
        for dataset in datasets
            ArchGDAL.destroy(dataset)
        end
    end
end

# Window the tiles onto an explicit (Nx, Ny) grid over the bbox. `-ts` pins the
# output to the grid's cell count (so it drops straight into a grid Field) and `-r average`
# coarse-grains the native pixels within each cell — not point interpolation. `nodata`
# excludes the product no-data byte from those means (see `warp_canopy_sources`).
warp_canopy_onto_grid(sources, longitude, latitude, Nx, Ny; resampling = "average", nodata = nothing) =
    warp_canopy_sources(sources,
        String["-te", string(longitude[1]), string(latitude[1]),
               string(longitude[2]), string(latitude[2]),
               "-ts", string(Nx), string(Ny)]; resampling, nodata)

# `longitude`/`latitude` are the cell-center coordinate vectors from `warp_canopy_sources`.
function write_canopy_netcdf(nc_path, longitude, latitude, layers)
    Nx = length(longitude)
    Ny = length(latitude)

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)

        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])

        lon_var[:] = longitude
        lat_var[:] = latitude

        for (name, data) in layers
            long_name = name == "SD" ? "canopy height standard deviation" : "canopy height"
            var = defVar(ds, name, Float32, ("lon", "lat");
                         attrib = ["long_name" => long_name, "units" => "m"])
            var[:, :] = data
        end
    end

    return nothing
end

# ETH: window the intersecting 3° 10 m COG tiles (libdrive WebDAV) for the requested layer,
# mask the no-data byte (255) to NaN — keeping non-forest zeros — and write one regional
# NetCDF. The WebDAV endpoint needs the public read-only share token as basic-auth credentials
# and honors HTTP range requests, so `/vsicurl/` fetches only the windowed COG blocks rather
# than whole 415 MB tiles. Nearest-neighbor resampling keeps the categorical 255 no-data byte
# exact so `mask_eth` catches it (bilinear would blend 255 into a valid neighbor).
function NumericalEarth.DataWrangling.ETHSentinel2Canopy.canopy_height_cog_to_netcdf(metadatum::ETHSentinel2CanopyHeightMetadatum, nc_path)
    raster = canopy_regional_raster(metadatum)
    sources = eth_tile_urls(raster.region, metadatum.name)

    warped = with_gdal_config(eth_http_config()) do
        warp_canopy_onto_grid(sources, raster.region.longitude, raster.region.latitude,
                              raster.Nx, raster.Ny; resampling = "near")
    end

    layer = dataset_variable_name(metadatum)   # "Map" or "SD"
    layers = Dict(layer => mask_eth.(warped.data, 255))
    write_canopy_netcdf(nc_path, raster.longitude, raster.latitude, layers)
    return nothing
end

# Drop a masked (Nx, Ny) canopy array straight into a grid Field; `-ts` guarantees the
# array matches the grid's cell count.
function canopy_field(grid, data)
    h = Field{Center, Center, Nothing}(grid)
    interior(h) .= on_architecture(architecture(grid), reshape(data, size(data, 1), size(data, 2), 1))
    return h
end

# Area-averaged read straight onto a model grid, coarse-graining the native pixels within
# each cell without going through a regional NetCDF.
function NumericalEarth.DataWrangling.ETHSentinel2Canopy.canopy_height_field(grid, ::ETHSentinel2CanopyHeight;
                                                                            name = :canopy_height,
                                                                            resampling = "average")
    region = BoundingBox(grid)
    sources = eth_tile_urls(region, name)

    warped = with_gdal_config(eth_http_config()) do
        warp_canopy_onto_grid(sources, region.longitude, region.latitude,
                              size(grid, 1), size(grid, 2); resampling, nodata = 255)
    end

    return canopy_field(grid, mask_eth.(warped.data, 255))
end
