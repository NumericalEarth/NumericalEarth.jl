using Downloads
using NCDatasets
using NumericalEarth.DataWrangling: metadata_path

const ARTIFACTS_BASE_URL = "https://github.com/NumericalEarth/NumericalEarthArtifacts/releases/download/data-v1/"

function emit_ci_warning(title, message)
    if haskey(ENV, "GITHUB_ACTIONS")
        println(stderr, "::warning title=$(title)::$(message)")
    end
end

"""
    validate_netcdf(filepath)

Return `true` if `filepath` is a valid NetCDF file that can be opened.
"""
function validate_netcdf(filepath)
    try
        ds = NCDataset(filepath)
        close(ds)
        return true
    catch
        return false
    end
end

function download_from_artifacts(filepath::AbstractString)
    # Delete corrupt files so they get re-downloaded
    if isfile(filepath) && endswith(filepath, ".nc") && !validate_netcdf(filepath)
        @warn "Deleting corrupt file: $(basename(filepath))"
        rm(filepath; force=true)
    end

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
    download_dataset_with_fallback(download_fn; dataset_name="dataset")

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
