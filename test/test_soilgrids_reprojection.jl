include("runtests_setup.jl")

using ArchGDAL
using NCDatasets: NCDataset

using NumericalEarth.DataWrangling: BoundingBox, native_grid
using NumericalEarth.DataWrangling.SoilGrids
using NumericalEarth.DataWrangling.SoilGrids: SoilGrids250m, soilgrids_raster_geometry

const NumericalEarthArchGDALExt = Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt)

const homolosine = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

# Homolosine (x, y) of a longitude/latitude point
function homolosine_coordinates(λ, φ)
    source = ArchGDAL.importPROJ4("+proj=longlat +datum=WGS84 +no_defs")
    target = ArchGDAL.importPROJ4(homolosine)
    return ArchGDAL.createcoordtrans(source, target) do transform
        point = ArchGDAL.createpoint(λ, φ)
        ArchGDAL.transform!(point, transform)
        return ArchGDAL.getx(point, 0), ArchGDAL.gety(point, 0)
    end
end

# A 250 m Homolosine GeoTIFF that covers `region` with a margin, holding `offset` plus a
# northward ramp from 0 to 10, mimicking one depth layer of an ISRIC VRT.
function write_homolosine_raster(path, region, offset; step = 250, margin = 2000)
    corners = [homolosine_coordinates(λ, φ) for λ in region.longitude, φ in region.latitude]
    west  = minimum(first, corners) - margin
    east  = maximum(first, corners) + margin
    south = minimum(last, corners) - margin
    north = maximum(last, corners) + margin
    Nx = ceil(Int, (east - west) / step)
    Ny = ceil(Int, (north - south) / step)

    # GDAL rows run north to south
    values = Float32[offset + 10 * (Ny - j) / (Ny - 1) for i in 1:Nx, j in 1:Ny]

    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"), width = Nx, height = Ny,
                    nbands = 1, dtype = Float32) do raster
        ArchGDAL.setgeotransform!(raster, Float64[west, step, 0, north, 0, -step])
        ArchGDAL.setproj!(raster, ArchGDAL.toWKT(ArchGDAL.importPROJ4(homolosine)))
        ArchGDAL.write!(ArchGDAL.getband(raster, 1), values)
    end
    return path
end

@testset "SoilGrids2 Homolosine rasters warp and stack onto the native grid" begin
    region = BoundingBox(longitude = (-0.35, -0.25), latitude = (51.20, 51.27))
    dataset = SoilGrids2(grid_type = SoilGrids250m)

    mktempdir() do dir
        metadatum = Metadatum(:clay_fraction; dataset, region, dir)
        raster = soilgrids_raster_geometry(metadatum)

        # Six depths, deepest first; depth k holds 100k g/kg (+ ramp)
        sources = [write_homolosine_raster(joinpath(dir, "clay_$k.tif"), region, 100k) for k in 1:6]
        NumericalEarthArchGDALExt.soilgrids_sources_to_netcdf(sources, "clay", raster, metadata_path(metadatum))

        NCDataset(metadata_path(metadatum)) do ds
            @test size(ds["clay"]) == (raster.Nx, raster.Ny, 6)
            @test ds["lon"][:] == raster.longitude
            @test ds["lat"][:] == raster.latitude

            data = Array(ds["clay"][:, :, :])
            @test all(isfinite, data)
            for k in 1:6
                layer = data[:, :, k]
                @test all(x -> 100k ≤ x ≤ 100k + 10, layer)
                # Stored south to north, so the ramp increases with the latitude index
                @test all(layer[:, end] .> layer[:, 1])
            end
        end

        # The read path picks up the stacked file and converts g/kg → kg/kg, deepest layer first
        field = Field(metadatum, CPU())
        data = Array(interior(field))
        @test size(data) == size(native_grid(metadatum))
        for k in 1:6
            @test all(x -> 0.1k ≤ x ≤ 0.1k + 0.01 + 1e-6, data[:, :, k])
        end
    end
end
