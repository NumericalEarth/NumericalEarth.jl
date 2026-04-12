include("runtests_setup.jl")
include("download_utils.jl")

@testset "JRA55 data downloading" begin
    @info "Testing JRA55 download infrastructure..."
    # Test a small subset of variables to verify download works
    test_variables = (:temperature, :eastward_velocity, :downwelling_shortwave_radiation)

    for name in test_variables
        datum = Metadatum(name; dataset=JRA55.RepeatYearJRA55())
        filepath = metadata_path(datum)

        fts = download_dataset_with_fallback(filepath; dataset_name="JRA55 $name") do
            NumericalEarth.JRA55.JRA55FieldTimeSeries(name; backend=NumericalEarth.JRA55.JRA55NetCDFBackend(2))
        end
        @test isfile(fts.path)
        rm(fts.path; force=true)
    end
end

@testset "ETOPO2022 Bathymetry downloading" begin
    @info "Testing bathymetry download..."
    metadata = Metadatum(:bottom_height, dataset=ETOPO2022())
    filepath = metadata_path(metadata)
    isfile(filepath) && rm(filepath; force=true)

    download_dataset_with_fallback(filepath; dataset_name="ETOPO2022") do
        NumericalEarth.DataWrangling.download_dataset(metadata)
    end
    @test isfile(filepath)
end
