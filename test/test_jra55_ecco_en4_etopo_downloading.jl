include("runtests_setup.jl")
include("download_utils.jl")

@testset "Availability of JRA55 data" begin
    @info "Testing that we can download all the JRA55 data..."
    for name in NumericalEarth.DataWrangling.JRA55.JRA55_variable_names
        # Build a Metadatum to determine the expected file path
        datum = Metadatum(name; dataset=JRA55.RepeatYearJRA55())
        filepath = metadata_path(datum)

        fts = download_dataset_with_fallback(filepath; dataset_name="JRA55 $name") do
            NumericalEarth.JRA55.JRA55FieldTimeSeries(name; backend=NumericalEarth.JRA55.JRA55NetCDFBackend(2))
        end
        @test isfile(fts.path)
        rm(fts.path; force=true)
    end
end

@testset "Availability of ECCO/EN4 data" begin
    for dataset in test_datasets

        @info "Testing that we can download $(typeof(dataset)) data..."

        variables = dataset isa ECCO2Daily ?         keys(NumericalEarth.ECCO.ECCO2_dataset_variable_names) :
                    dataset isa ECCO2Monthly ?       keys(NumericalEarth.ECCO.ECCO2_dataset_variable_names) :
                    dataset isa ECCO4Monthly ?       keys(NumericalEarth.ECCO.ECCO4_dataset_variable_names) :
                    dataset isa ECCO4DarwinMonthly ? keys(NumericalEarth.ECCO.ECCO_darwin_dataset_variable_names) :
                    dataset isa ECCO2DarwinMonthly ? keys(NumericalEarth.ECCO.ECCO_darwin_dataset_variable_names) :
                    dataset isa EN4Monthly ?         keys(NumericalEarth.EN4.EN4_dataset_variable_names) :
                    error("what am I supposed to download?")

        for variable in variables
            metadata = Metadata(variable; dates=DateTimeProlepticGregorian(1993, 1, 1), dataset)
            filepath = metadata_path(metadata)
            isfile(filepath) && rm(filepath; force=true)

            download_dataset_with_fallback(filepath; dataset_name="$(typeof(dataset)) $variable") do
                NumericalEarth.DataWrangling.download_dataset(metadata)
            end
            @test isfile(filepath)
            rm(filepath; force=true)
        end
    end
end

@testset "Availability of the ETOPO2022 Bathymetry" begin
    @info "Testing that we can download the bathymetry..."
    metadata = Metadatum(:bottom_height, dataset=ETOPO2022())
    filepath = metadata_path(metadata)
    isfile(filepath) && rm(filepath; force=true)

    download_dataset_with_fallback(filepath; dataset_name="ETOPO2022") do
        NumericalEarth.DataWrangling.download_dataset(metadata)
    end
    @test isfile(filepath)
end
