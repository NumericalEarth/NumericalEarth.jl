include("runtests_setup.jl")

using ArchGDAL  # activates NumericalEarthArchGDALExt (the HDF5 tile read)

using NCDatasets: NCDataset

using NumericalEarth.DataWrangling: BoundingBox, inpainted_metadata_path, native_grid

# Requires NASA Earthdata credentials in EARTHDATA_USERNAME / EARTHDATA_PASSWORD
# (register free at https://urs.earthdata.nasa.gov). Excluded from CI in runtests.jl;
# run manually with the credentials set.

# A small Grand Canyon window, as in examples/inspect_aster_ged_emissivity.jl. It
# straddles the 112°W/36°N tile corner, so the read mosaics four tiles.
const asterged_longitude = (-112.2, -111.8)
const asterged_latitude  = (35.8, 36.2)
const asterged_region = BoundingBox(longitude = asterged_longitude, latitude = asterged_latitude)

# 1 km: the 100 m tiles cover the same region at ~100× the download.
const asterged_dataset = ASTERGEDv3(resolution = :low_1km)

asterged_metadatum(name) = Metadatum(name; dataset = asterged_dataset, region = asterged_region)

@testset "Downloading ASTER GED tiles" begin
    metadatum = asterged_metadatum(:emissivity)
    path = metadata_path(metadatum)

    # Start from the tiles: drop the regional NetCDF and the inpainted caches keyed on
    # it, so a stale file cannot stand in for the download.
    for name in (:emissivity, :emissivity_uncertainty)
        rm(inpainted_metadata_path(asterged_metadatum(name)); force = true)
    end
    rm(path; force = true)

    download(metadatum)
    @test isfile(path)

    # One regional file, written on the native grid, serves both variables.
    @test metadata_path(asterged_metadatum(:emissivity_uncertainty)) == path

    grid = native_grid(metadatum, CPU())
    NCDataset(path) do ds
        @test size(ds["emissivity"]) == (size(grid, 1), size(grid, 2))
        @test haskey(ds, "emissivity_uncertainty")
        @test haskey(ds, "land_water_map")
    end
end

@testset "Regridding ASTER GED emissivity" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (20, 20),
                                 longitude = asterged_longitude,
                                 latitude = asterged_latitude,
                                 topology = (Bounded, Bounded, Flat))

    ε = Array(interior(Field(asterged_metadatum(:emissivity), grid), :, :, 1))
    σ = Array(interior(Field(asterged_metadatum(:emissivity_uncertainty), grid), :, :, 1))

    @test all(isfinite, ε)                  # clear-sky retrieval gaps are inpainted
    @test all(x -> 0.8 ≤ x ≤ 1, ε)
    @test length(unique(ε)) > 1              # a real window, not a constant fill

    @test all(isfinite, σ)
    @test all(x -> 0 ≤ x ≤ 0.1, σ)
end
