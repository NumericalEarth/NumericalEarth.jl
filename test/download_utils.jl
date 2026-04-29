using Downloads
using NumericalEarth.DataWrangling: metadata_path

const ARTIFACTS_BASE_URL = "https://github.com/NumericalEarth/NumericalEarthArtifacts/releases/download/data-v1/"

function emit_ci_warning(title, message)
    if haskey(ENV, "GITHUB_ACTIONS")
        println(stderr, "::warning title=$(title)::$(message)")
    end
end

function download_from_artifacts(filepath::AbstractString)
    if !isfile(filepath)
        filename = basename(filepath)
        fallback_url = ARTIFACTS_BASE_URL * filename
        @info "Downloading $filename from NumericalEarthArtifacts fallback..."
        Downloads.download(fallback_url, filepath)
    end
end

function download_from_artifacts(filepaths::AbstractVector)
    for filepath in unique(filepaths)
        download_from_artifacts(filepath)
    end
end

"""
    download_dataset_with_fallback(download_fn, filepaths; dataset_name="dataset")

Try `download_fn()`. If it throws, download the required files from
NumericalEarthArtifacts and retry. Emits a CI warning when the fallback is used.

Returns `true` when the data is available (either path succeeded), `false` when
neither path could obtain it (CDS unavailable AND artifact 404). Callers should
treat `false` as a signal to skip the rest of the testset.
"""
function download_dataset_with_fallback(download_fn, filepaths; dataset_name="dataset")
    try
        download_fn()
        return true
    catch e
        @warn "Original download failed for $dataset_name, trying NumericalEarthArtifacts fallback..." exception=(e, catch_backtrace())
        emit_ci_warning("Broken $dataset_name download", "Original source failed: $(sprint(showerror, e))")
    end

    try
        download_from_artifacts(filepaths)
        download_fn()
        return true
    catch e
        @info "Artifact fallback also failed for $dataset_name; skipping test" exception=e
        emit_ci_warning("Missing $dataset_name artifact", "Both CDS and artifact fallback failed")
        return false
    end
end
