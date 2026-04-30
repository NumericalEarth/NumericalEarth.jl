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

Returns the result of `download_fn()`.
"""
function download_dataset_with_fallback(download_fn, filepaths; dataset_name="dataset")
    try
        return download_fn()
    catch e
        @warn "Original download failed for $dataset_name, trying NumericalEarthArtifacts fallback..." exception=(e, catch_backtrace())
        emit_ci_warning("Broken $dataset_name download", "Original source failed: $(sprint(showerror, e))")
        download_from_artifacts(filepaths)
        return download_fn()
    end
end

"""
    try_download_or_skip(download_fn, filepaths; dataset_name="dataset")

Like `download_dataset_with_fallback`, but if both the original source and the
artifact fallback fail, emit an `@info` and return `false` instead of throwing.
Returns `true` on success. Use as `try_download_or_skip(...) do ... end || return`
at testset scope to skip cleanly when test data is unobtainable.
"""
function try_download_or_skip(download_fn, filepaths; dataset_name="dataset")
    try
        download_dataset_with_fallback(download_fn, filepaths; dataset_name)
        return true
    catch e
        @info "Skipping $dataset_name test: both CDS and artifact fallback failed" exception=e
        emit_ci_warning("Missing $dataset_name", "Both CDS and artifact fallback failed")
        return false
    end
end
