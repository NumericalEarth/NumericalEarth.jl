# The default run is offline: the data directory is a fresh temporary directory, and any file
# that lands there is an accidental download. Set `NUMERICALEARTH_TEST_REMOTE_DATA=true` to
# also run the tests that need real remote datasets, in the regular data directory.
remote_data = parse(Bool, get(ENV, "NUMERICALEARTH_TEST_REMOTE_DATA", "false"))
remote_data || (ENV["NUMERICALEARTH_DATA_DIRECTORY"] = mktempdir())

# Common test setup file to make stand-alone tests easy
include("runtests_setup.jl")
include("download_utils.jl")

using CUDA
using Scratch
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

# download_utils, runtests_setup and synthetic_datasets are not tests!
delete!(testsuite, "runtests_setup")
delete!(testsuite, "download_utils")
delete!(testsuite, "synthetic_datasets")
delete!(testsuite, "test_distributed_utils")

gpu_test = parse(Bool, get(ENV, "GPU_TEST", "false"))

# Tests that need real remote datasets
remote_data_tests = ["test_bathymetry",
                     "test_polar_bathymetry",
                     "test_jra55",
                     "test_jra55_region",
                     "test_ecco2_monthly",
                     "test_ecco2_daily",
                     "test_ecco4_en4",
                     "test_ecco_atmosphere",
                     "test_woa",
                     "test_orca_grid",
                     "test_soilgrids",
                     "test_dataset_region",
                     "test_diagnostics_1",
                     "test_mangling"]

if filter_tests!(testsuite, args)
    # Network probes run only from the DataDownload workflow, which names them explicitly
    for name in filter(endswith("_downloading"), collect(keys(testsuite)))
        delete!(testsuite, name)
    end

    if !remote_data
        for name in remote_data_tests
            delete!(testsuite, name)
        end
    end

    delete!(testsuite, "test_reactant")
    delete!(testsuite, "test_veros") # Veros seems to have introduce a pypi conflict issue; temporarily removing from CI

    if gpu_test
        # Remove CPU-only tests when testing on GPUs
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

if remote_data
    delete_inpainted_files(@get_scratch!("."))
    download_test_data()
end

runtests(NumericalEarth, args; testsuite)

if remote_data
    delete_inpainted_files(@get_scratch!("."))
else
    # The regridding cache lives here too, so only a dataset file is an accidental download
    downloaded = [joinpath(root, file) for (root, _, files) in walkdir(ENV["NUMERICALEARTH_DATA_DIRECTORY"])
                  for file in files if basename(root) != "field_cache"]
    isempty(downloaded) || error("The offline test run downloaded data: ", join(downloaded, ", "))
end
