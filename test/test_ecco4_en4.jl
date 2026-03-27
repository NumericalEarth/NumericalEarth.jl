include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.ECCO
using NumericalEarth.EN4
using NumericalEarth.DataWrangling: NearestNeighborInpainting, metadata_path, native_times, download_dataset

using Dates
using Oceananigans.Grids: topology
using Oceananigans.OutputReaders: time_indices

using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units

using CUDA: @allowscalar

# Inpaint only the first ten cells inside the missing mask
inpainting = NearestNeighborInpainting(10)
start_date = DateTime(1993, 1, 1)

test_ecco_en4_datasets = tuple((ds for ds in test_datasets if occursin(r"^E.*4",string(typeof(ds)),))...)

for arch in test_architectures, dataset in test_ecco_en4_datasets
    A = typeof(arch)
    D = typeof(dataset)
    @testset "$A metadata tests for $D" begin
        @info "Running Metadata tests for $D on $A..."

        time_resolution = dataset isa ECCO2Daily ? Day(1) : Month(1)
        # ECCO4 fallback artifacts only cover January 1993,
        # so limit to 1 date for all ECCO4 variants to avoid download failures.
        # TODO: when ecco.jpl.nasa.gov is reliable again, revert
        # ECCO4DarwinMonthly to n_dates=4 so it tests multiple dates.
        n_dates = dataset isa Union{ECCO4Monthly, ECCO4DarwinMonthly} ? 0 : 4
        end_date = start_date + n_dates * time_resolution
        dates = start_date : time_resolution : end_date

        @testset "Fields utilities" begin
            for name in test_names[dataset]
                metadata = Metadata(name; dates, dataset)

                download_dataset(metadata) # just in case is not downloaded
                for datum in metadata
                    @test isfile(metadata_path(datum))
                end

                datum = first(metadata)
                ψ = Field(datum, arch, inpainting=NearestNeighborInpainting(2))
                @test ψ isa Field
                datapath = NumericalEarth.DataWrangling.inpainted_metadata_path(datum)
                @test isfile(datapath)
            end
        end

        @testset "Setting a field from a dataset" begin
            test_setting_from_metadata(arch, dataset, start_date, inpainting, 
                                       varnames=test_names[dataset])
        end

        @testset "Timestepping with fields from Dataset" begin
            test_timestepping_with_dataset(arch, dataset, start_date, inpainting, 
                                           varnames=test_names[dataset], 
                                           fldnames=test_fields[dataset])
        end

        # Multi-date tests require ≥3 dates in artifacts.
        # ECCO4 artifacts only have January 1993, so skip multi-date tests for ECCO4 variants.
        if length(dates) >= 3
            @testset "Field utilities" begin
                test_ocean_metadata_utilities(arch, dataset, dates, inpainting,
                                              varnames=test_names[dataset])
            end

            @testset "DatasetRestoring with LinearlyTaperedPolarMask" begin
                test_dataset_restoring(arch, dataset, dates, inpainting,
                                       varnames=test_names[dataset],
                                       fldnames=test_fields[dataset])
            end

            @testset "Timestepping with DatasetRestoring" begin
                test_timestepping_with_dataset_restoring(arch, dataset, dates, inpainting,
                                                         varnames=test_names[dataset],
                                                         fldnames=test_fields[dataset])
            end

            @testset "Dataset cycling boundaries" begin
                test_cycling_dataset_restoring(arch, dataset, dates, inpainting,
                                               varnames=test_names[dataset],
                                               fldnames=test_fields[dataset])
            end
        end

        @testset "Inpainting algorithm" begin
            test_inpainting_algorithm(arch, dataset, start_date, inpainting, 
                                      varnames=test_names[dataset])
        end
    end
end
