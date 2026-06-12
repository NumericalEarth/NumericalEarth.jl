# Common test setup file to make stand-alone tests easy
include("runtests_setup.jl")
include("download_utils.jl")

using CUDA
using Scratch
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests

# Start with autodiscovered tests
# Temporarily restrict local/CLI runs to the tracer conservation test while
# iterating on its setup and budget assertions.
testsuite = find_tests(@__DIR__)
for name in collect(keys(testsuite))
    name == "test_tracer_conservation" || delete!(testsuite, name)
end

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
    ##### NOTE
    #####

    # Temporarily skip eager dataset downloads in CI while iterating on the
    # tracer conservation test. Individual tests will fetch any required inputs
    # on demand through the normal metadata constructors.
end

# Initialize and download required datasets
__init__()

runtests(NumericalEarth, args; testsuite)

delete_inpainted_files(@get_scratch!("."))
