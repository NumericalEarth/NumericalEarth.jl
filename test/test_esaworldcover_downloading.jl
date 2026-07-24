include("runtests_setup.jl")

using ArchGDAL  # activates the anonymous-COG read path in NumericalEarthArchGDALExt

using NumericalEarth: ESAWorldCover
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.DataWrangling.WorldCover: ESA_WORLDCOVER_CLASS_CODES,
                                               ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES

# Reads the anonymous public `s3://esa-worldcover/` COG tiles (no credentials),
# so this needs `using ArchGDAL` and network access. Excluded from CI in
# runtests.jl; run manually.

# A small mixed land/water/urban window over the IJmeer near Amsterdam.
const wc_longitude = (4.8, 5.0)
const wc_latitude  = (52.3, 52.5)
const wc_region = BoundingBox(longitude = wc_longitude, latitude = wc_latitude)

@testset "Downloading ESA WorldCover regional window" begin
    dataset = ESAWorldCover()
    for name in (:vegetation_fraction, :landcover_class)
        metadatum = Metadatum(name; dataset, region = wc_region)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force = true)
        download(metadatum)
        @test isfile(filepath)
    end
end

@testset "ESA WorldCover materializes sane fields" begin
    dataset = ESAWorldCover()
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (40, 40),
                                 longitude = wc_longitude,
                                 latitude = wc_latitude,
                                 topology = (Bounded, Bounded, Flat))

    # Vegetation fraction is a continuous field in [0, 1] with no missing cells.
    f_veg = Field(Metadatum(:vegetation_fraction; dataset, region = wc_region), grid)
    v = Array(interior(f_veg))
    @test !any(isnan, v)
    @test all(0 .<= v .<= 1)

    # The majority class on its native grid is an exact legend code (never an
    # interpolated intermediate value).
    class_native = Field(Metadatum(:landcover_class; dataset, region = wc_region), CPU())
    codes = round.(Int, filter(!isnan, vec(Array(interior(class_native)))))
    @test !isempty(codes)
    @test all(c -> c in ESA_WORLDCOVER_CLASS_CODES, codes)

    # The eleven per-class fractions sum to ≈ 1 over every valid cell.
    fraction_sum = sum(Array(interior(Field(Metadatum(name; dataset, region = wc_region), grid)))
                       for name in ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES)
    valid = .!isnan.(fraction_sum)
    @test all(≈(1; atol = 1e-4), fraction_sum[valid])
end
