# Common test setup file to make stand-alone tests easy
include("runtests_setup.jl")
include("download_utils.jl")

using CUDA
using Scratch
using NumericalEarth.DataWrangling: download
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

# download_utils and runtests_setup are not tests!
delete!(testsuite, "runtests_setup")
delete!(testsuite, "download_utils")
delete!(testsuite, "test_distributed_utils")
delete!(testsuite, "test_ospapa")

gpu_test = parse(Bool, get(ENV, "GPU_TEST", "false"))

if filter_tests!(testsuite, args)
    # Always remove tests that are treated separately
    delete!(testsuite, "test_jra55_ecco_en4_etopo_downloading")
    delete!(testsuite, "test_cds_downloading")
    delete!(testsuite, "test_glorys_downloading")
    delete!(testsuite, "test_distributed_utils")
    delete!(testsuite, "test_reactant")
    delete!(testsuite, "test_veros") # Veros seems to have introduce a pypi conflict issue; temporarily removing from CI

    if gpu_test
        # Remove CPU-only tests when testing on GPUs
        delete!(testsuite, "test_veros")
        delete!(testsuite, "test_speedy_coupling")
    else
        # Remove the slowest tests from CPU CI to keep total runtime
        # manageable; GPU CI still runs them. See issue #193.
        delete!(testsuite, "test_ocean_only_model")
        delete!(testsuite, "test_ocean_sea_ice_model")
        delete!(testsuite, "test_diagnostics_1")
        delete!(testsuite, "test_ecco2_daily")
        delete!(testsuite, "test_orca_grid")
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
        NumericalEarth.DataWrangling.download(ETOPOmetadata)
    end

    #####
    ##### Download JRA55 data
    #####

    try
        atmosphere = JRA55PrescribedAtmosphere(time_indices_in_memory=2)
        land       = JRA55PrescribedLand(time_indices_in_memory=2)
        # Touch the radiation variables (rlds/rsds) too, so a corrupted cached
        # download is caught by the same fallback path.
        radiation = JRA55PrescribedRadiation(time_indices_in_memory=2)
    catch e
        @warn "Original JRA55 download failed, trying NumericalEarthArtifacts fallback..." exception=(e, catch_backtrace())
        emit_ci_warning("Broken JRA55 download", "Original source failed during init")
        for name in NumericalEarth.DataWrangling.JRA55.JRA55_variable_names
            datum = Metadatum(name; dataset=JRA55.RepeatYearJRA55())
            download_from_artifacts(metadata_path(datum))
        end
        atmosphere = JRA55PrescribedAtmosphere(time_indices_in_memory=2)
        land       = JRA55PrescribedLand(time_indices_in_memory=2)
        radiation  = JRA55PrescribedRadiation(time_indices_in_memory=2)
    end

    #####
    ##### Download Dataset data
    #####

    # Download few datasets for tests
    for dataset in test_datasets
        time_resolution = dataset isa ECCO2Daily ? Day(1) : Month(1)
        end_date = start_date + 1 * time_resolution
        dates = start_date:time_resolution:end_date

        ts_set = MetadataSet(:temperature, :salinity; dataset, dates)

        for md in ts_set
            download_dataset_with_fallback(metadata_path(md); dataset_name="$(typeof(dataset)) $(md.name)") do
                download(md)
            end
        end

        if dataset isa Union{ECCO2DarwinMonthly, ECCO4DarwinMonthly}
            PO₄_metadata = Metadata(:phosphate; dataset, dates)
            download_dataset_with_fallback(metadata_path(PO₄_metadata); dataset_name="$(typeof(dataset)) phosphate") do
                download(PO₄_metadata)
            end
        end
    end
end

# Initialize and download required datasets
__init__()

runtests(NumericalEarth, args; testsuite)

delete_inpainted_files(@get_scratch!("."))
