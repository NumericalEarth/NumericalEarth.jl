# Disable HDF5 file locking before any HDF5-backed package loads: this test
# writes a regional NetCDF and reopens it within the same session for several
# datasets, which can otherwise trip an HDF5 lock error (-101) on reopen.
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

include("runtests_setup.jl")
include("download_utils.jl")

using Zarr  # activates NumericalEarthZarrExt (registers the bitround filter, enables the read)

using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.DataWrangling.CopernicusDEM: GLO30, GLO90

# Requires a DestinE personal access token in DESTINE_ACCESS_TOKEN (free; see
# https://earthdatahub.destine.eu/account-settings#my-personal-access-tokens).
# Excluded from CI in runtests.jl; run manually with the token set.

# A small high-Alps window (Monte Rosa massif) — kept to 1°×1° so the GLO-30 read
# stays light on the shared downloading CI job, while still spanning peaks > 1 km.
const dem_longitude = (7.5, 8.5)
const dem_latitude  = (45.5, 46.5)
const dem_region = BoundingBox(longitude = dem_longitude, latitude = dem_latitude)

@testset "Downloading Copernicus DEM regional window" begin
    for dataset in (GLO30(), GLO90())
        metadatum = Metadatum(:bottom_height; dataset, region = dem_region)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force = true)
        download(metadatum)
        @test isfile(filepath)
    end
end

@testset "Regridding Copernicus DEM topography" begin
    for dataset in (GLO30(), GLO90())
        grid = LatitudeLongitudeGrid(CPU();
                                     size = (40, 40, 1),
                                     longitude = dem_longitude,
                                     latitude = dem_latitude,
                                     z = (-1, 0))

        metadatum = Metadatum(:bottom_height; dataset, region = dem_region)
        topography = regrid_topography(grid, metadatum; cache = false)
        z = Array(interior(topography, :, :, 1))

        @test all(isfinite, z)
        @test all(>=(0), z)        # land surface elevation; ocean clamped to 0
        @test maximum(z) > 1000    # the Alps reach well above 1 km
        @test maximum(z) < 5000    # but below the Mont Blanc ceiling
    end
end
