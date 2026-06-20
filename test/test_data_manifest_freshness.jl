include("runtests_setup.jl")

using NumericalEarth.DataWrangling.DataModes: DataModes, pregenerate_dataset_manifest, manifest_path_in
using TOML

# Regenerate the test-folder `NumericalEarthDataManifest.toml` by tracing every `test_*.jl`
# (excluding self) in pregenerate mode, and compare the result against the committed manifest.
# If they differ the manifest is stale — re-run the pregenerate command below and commit the
# resulting `test/NumericalEarthDataManifest.toml`.
function regenerate_manifest_in(out_dir)
    test_dir = @__DIR__
    self     = @__FILE__
    for f in sort(readdir(test_dir; join=true))
        endswith(f, ".jl") && startswith(basename(f), "test_") || continue
        abspath(f) == abspath(self) && continue
        try
            pregenerate_dataset_manifest(f; dir = out_dir, overwrite_existing = false)
        catch
        end
    end
    out = manifest_path_in(out_dir)
    return isfile(out) ? TOML.parsefile(out) : Dict{String, Any}()
end

@testset "DataManifest freshness" begin
    # This test self-invokes `pregenerate_dataset_manifest` on every other `test_*.jl`. If
    # we're already running inside a pregenerate trace (i.e. some outer loop is tracing this
    # very file), recursing here both wastes work and corrupts per-process state — most
    # notably MPI, which gets re-initialised across nested sandbox boundaries.
    if DataModes.DATA_MODE[] === :pregenerate
        @info "Skipping DataManifest freshness test inside a pregenerate trace"
        return
    end

    committed_path = manifest_path_in(@__DIR__)
    @test isfile(committed_path)

    committed   = TOML.parsefile(committed_path)
    regenerated = mktempdir(regenerate_manifest_in)

    if committed != regenerated
        added   = sort(collect(setdiff(keys(regenerated), keys(committed))))
        removed = sort(collect(setdiff(keys(committed),  keys(regenerated))))
        isempty(added)   || @info  "Datasets added to the regenerated manifest" datasets=added
        isempty(removed) || @info  "Datasets missing from the regenerated manifest" datasets=removed
        for k in sort(collect(intersect(keys(committed), keys(regenerated))))
            committed[k] == regenerated[k] && continue
            @info "Entries differ for dataset" dataset=k committed=committed[k] regenerated=regenerated[k]
        end
        @info "Manifest is stale. To regenerate, run from the repo root:\n  " *
              "julia --project -e 'using NumericalEarth.DataWrangling.DataModes: " *
              "pregenerate_dataset_manifest, manifest_path_in; dir = abspath(\"test\"); " *
              "rm(manifest_path_in(dir); force=true); for f in sort(readdir(dir; join=true)); " *
              "endswith(f, \".jl\") && startswith(basename(f), \"test_\") && " *
              "basename(f) != \"test_data_manifest_freshness.jl\" || continue; " *
              "try; pregenerate_dataset_manifest(f; dir, overwrite_existing=false); catch; end; end'"
    end

    @test committed == regenerated
end
