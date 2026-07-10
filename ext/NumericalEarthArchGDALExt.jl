module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

const OpenLandMap = NumericalEarth.DataWrangling.OpenLandMap
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

function configure_vsicurl!()
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    ArchGDAL.setconfigoption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    ArchGDAL.setconfigoption("GDAL_HTTP_MULTIRANGE", "YES")

    # GDAL's bundled libcurl reads its CA bundle from CURL_CA_BUNDLE and does not
    # reliably fall back to a system trust store, which breaks https /vsicurl reads.
    # Point it at Julia's cross-platform CA roots unless the user already set it, so
    # an explicit override wins.
    if !haskey(ENV, "CURL_CA_BUNDLE")
        ENV["CURL_CA_BUNDLE"] = NetworkOptions.ca_roots_path()
    end
    return nothing
end

# Decode raw COG integers to Float32 physical values. Order matters: mask nodata
# to NaN first, then apply the band scale/offset (a scaled fill is a spurious value).
function decode_cog_window(raw, scale, offset, nodata)
    decoded = Array{Float32}(undef, size(raw))
    @inbounds for idx in eachindex(raw)
        value = Float64(raw[idx])
        is_nodata = !isnothing(nodata) && value == nodata
        decoded[idx] = is_nodata ? NaN32 : Float32(value * scale + offset)
    end
    return decoded
end

# Dispatch on `bbox::BoundingBox` (more specific than the module fallback) so this
# adds a specialization rather than overwriting a method during precompilation.
function OpenLandMap.cog_window_to_netcdf(sources, nc_path, variable_name, bbox::BoundingBox)
    configure_vsicurl!()

    W, E = bbox.longitude
    S, N = bbox.latitude

    layers = Matrix{Float32}[]
    longitude = Float64[]
    latitude  = Float64[]

    for source in sources
        ArchGDAL.read(source) do ds
            # EPSG:4326 geotransform: [x₀, Δλ, 0, y₀, 0, Δφ] with Δφ < 0 (north→south).
            x0, dx, _, y0, _, dy = ArchGDAL.getgeotransform(ds)
            width  = ArchGDAL.width(ds)
            height = ArchGDAL.height(ds)

            xoff  = clamp(floor(Int, (W - x0) / dx), 0, width - 1)
            yoff  = clamp(floor(Int, (N - y0) / dy), 0, height - 1)
            xsize = clamp(ceil(Int, (E - x0) / dx) - xoff, 1, width - xoff)
            ysize = clamp(ceil(Int, (S - y0) / dy) - yoff, 1, height - yoff)

            band   = ArchGDAL.getband(ds, 1)
            scale  = ArchGDAL.getscale(band)
            offset = ArchGDAL.getoffset(band)
            nodata = ArchGDAL.getnodatavalue(band)

            raw = ArchGDAL.read(ds, 1, xoff, yoff, xsize, ysize)  # (lon, lat), north-first
            decoded = reverse(decode_cog_window(raw, scale, offset, nodata), dims = 2)
            push!(layers, decoded)

            if isempty(longitude)
                longitude = [x0 + (xoff + i - 0.5) * dx for i in 1:xsize]
                latitude  = reverse([y0 + (yoff + j - 0.5) * dy for j in 1:ysize])
            end
        end
    end

    Nx = length(longitude)
    Ny = length(latitude)
    Nz = length(layers)
    data = Array{Float32}(undef, Nx, Ny, Nz)
    for (k, layer) in enumerate(layers)
        data[:, :, k] = layer
    end

    # 60–100 / 30–60 / 0–30 cm interval midpoints (m), deepest first.
    depth_centers = Nz == 3 ? [-0.8, -0.45, -0.15] : Float64.(1:Nz)

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

end # module NumericalEarthArchGDALExt
