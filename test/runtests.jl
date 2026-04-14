# Common test setup file to make stand-alone tests easy
include("runtests_setup.jl")
include("download_utils.jl")

using CUDA
using Scratch
using NumericalEarth.DataWrangling: download_dataset
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

# download_utils and runtests_setup are not tests!
delete!(testsuite, "runtests_setup")
delete!(testsuite, "download_utils")
delete!(testsuite, "test_distributed_utils")

gpu_test = parse(Bool, get(ENV, "GPU_TEST", "false"))

if filter_tests!(testsuite, args)
    # Always remove tests that are treated separately
    delete!(testsuite, "test_downloading")
    delete!(testsuite, "test_cds_downloading")
    delete!(testsuite, "test_distributed_utils")
    delete!(testsuite, "test_reactant")

    # Remove CPU-only tests when
    # testing on GPUs
    if gpu_test
        delete!(testsuite, "test_veros")
        delete!(testsuite, "test_speedy_coupling")
    end
end

function delete_inpainted_files(dir)
    @info "Cleaning inpainted files..."
    for (root, _, files) in walkdir(dir)
        for file in files
            if endswith(file, "_inpainted.jld2")
                filepath = joinpath(root, file)
                rm(filepath; force=true)
                @info "    Deleted: $filepath"
            end
        end
    end
end

function __init__()
    #####
    ##### Delete inpainted files
    #####

    delete_inpainted_files(@get_scratch!("."))

    #####
    ##### Download bathymetry data
    #####

    ETOPOmetadata = Metadatum(:bottom_height, dataset=NumericalEarth.ETOPO.ETOPO2022())
    download_dataset_with_fallback(metadata_path(ETOPOmetadata); dataset_name="ETOPO2022") do
        NumericalEarth.DataWrangling.download_dataset(ETOPOmetadata)
    end

    #####
    ##### Download JRA55 data
    #####

    try
        atmosphere = JRA55PrescribedAtmosphere(backend=JRA55NetCDFBackend(2))
    catch e
        @warn "Original JRA55 download failed, trying NumericalEarthArtifacts fallback..." exception=(e, catch_backtrace())
        emit_ci_warning("Broken JRA55 download", "Original source failed during init")
        for name in NumericalEarth.DataWrangling.JRA55.JRA55_variable_names
            datum = Metadatum(name; dataset=JRA55.RepeatYearJRA55())
            download_from_artifacts(metadata_path(datum))
        end
        atmosphere = JRA55PrescribedAtmosphere(backend=JRA55NetCDFBackend(2))
    end

    #####
    ##### Download Dataset data
    #####

    # Download few datasets for tests
    for dataset in test_datasets
        time_resolution = dataset isa ECCO2Daily ? Day(1) : Month(1)
        end_date = start_date + 2 * time_resolution
        dates = start_date:time_resolution:end_date

        temperature_metadata = Metadata(:temperature; dataset, dates)
        salinity_metadata    = Metadata(:salinity; dataset, dates)

        for md in (temperature_metadata, salinity_metadata)
            download_dataset_with_fallback(metadata_path(md); dataset_name="$(typeof(dataset)) $(md.name)") do
                download_dataset(md)
            end
        end

        if dataset isa Union{ECCO2DarwinMonthly, ECCO4DarwinMonthly}
            PO₄_metadata = Metadata(:phosphate; dataset, dates)
            download_dataset_with_fallback(metadata_path(PO₄_metadata); dataset_name="$(typeof(dataset)) phosphate") do
                download_dataset(PO₄_metadata)
            end
        end
    end
end

# Initialize and download required datasets
__init__()

runtests(NumericalEarth, args; testsuite)

delete_inpainted_files(@get_scratch!("."))
