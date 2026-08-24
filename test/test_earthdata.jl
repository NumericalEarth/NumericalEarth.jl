include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, cmr_granules_url, earthdata_download_cached
using Dates: DateTime

# The granule query itself needs network access; the URL it is built from does not.
@testset "NASA CMR granule-search URL" begin
    region = BoundingBox(longitude = (-112.8, -111.2), latitude = (35.2, 36.8))
    url = cmr_granules_url("AG100", "003", region)

    @test startswith(url, "https://cmr.earthdata.nasa.gov/search/granules.json?")
    @test occursin("short_name=AG100", url)
    @test occursin("version=003", url)
    # Bounding box is encoded W,S,E,N.
    @test occursin("bounding_box=-112.8,35.2,-111.2,36.8", url)
    @test occursin("page_num=1", url)
    @test occursin("page_num=3", cmr_granules_url("AG1KM", "003", region; page_num = 3))

    # A static-epoch product searches the whole record; a dated one narrows to the day.
    @test !occursin("temporal=", url)
    dated = cmr_granules_url("MCD15A2H", "061", region; date = DateTime(2020, 7, 3))
    @test occursin("temporal=2020-07-03T00:00:00Z,2020-07-04T00:00:00Z", dated)

    @test_throws ArgumentError cmr_granules_url("AG100", "003", BoundingBox())
end

@testset "Cached Earthdata granules" begin
    cache = mktempdir()
    url = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/" *
          "MCD15A2H.061/MCD15A2H.A2020185.h10v05.061.2020340132006.hdf"
    path = joinpath(cache, basename(url))
    write(path, "granule")

    # A granule already in the cache is returned as it stands, so a repeated ingestion
    # needs neither the network nor credentials.
    @test earthdata_download_cached(url, cache) == path
    @test read(path, String) == "granule"
end
