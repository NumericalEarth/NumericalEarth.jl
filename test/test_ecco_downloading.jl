include("runtests_setup.jl")
include("download_utils.jl")

@testset "ECCO/EN4 data downloading" begin
    # Test a small subset of variables per dataset to verify download infrastructure
    test_variables = Dict(
        ECCO2Monthly()       => (:u_velocity, :free_surface),
        ECCO2Daily()         => (:u_velocity,),
        ECCO4Monthly()       => (:u_velocity, :sea_ice_thickness),
        ECCO2DarwinMonthly() => (:dissolved_inorganic_carbon,),
        ECCO4DarwinMonthly() => (:dissolved_inorganic_carbon,),
        EN4Monthly()         => (:temperature,),
    )

    for (dataset, variables) in test_variables
        @testset "$(typeof(dataset))" begin
            @info "Testing download for $(typeof(dataset))..."
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
end
