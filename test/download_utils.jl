using Downloads: Downloads
using NumericalEarth.DataWrangling: metadata_path

const ARTIFACTS_BASE_URL = "https://github.com/NumericalEarth/NumericalEarthArtifacts/releases/download/data-v1/"
const TEST_FIXTURES_BASE_URL = "https://github.com/NumericalEarth/NumericalEarthArtifacts/releases/download/test-fixtures-v1/"

"""
    download_test_fixtures()

Populate the JRA55 download cache with cropped RYF fixtures, replacing an 11.6 GB download with roughly 100 MB. 
Returns `true` when fixtures were used. A no-op unless `NUMERICALEARTH_TEST_FIXTURES == "true"`. 
"""
function download_test_fixtures()
    get(ENV, "NUMERICALEARTH_TEST_FIXTURES", "false") == "true" || return false

    for name in NumericalEarth.DataWrangling.JRA55.JRA55_variable_names
        filepath = metadata_path(Metadatum(name; dataset=NumericalEarth.JRA55.RepeatYearJRA55()))
        isfile(filepath) && continue

        filename = basename(filepath)
        mkpath(dirname(filepath))

        try
            @info "Fetching cropped JRA55 fixture $(filename)..."
            mktemp(dirname(filepath)) do tmppath, tmpio
                close(tmpio)
                Downloads.download(TEST_FIXTURES_BASE_URL * filename, tmppath)
                mv(tmppath, filepath; force=true)
            end
        catch e
            @warn "Could not fetch JRA55 fixture $(filename); falling back to the full file." exception=(e, catch_backtrace())
            emit_ci_warning("Missing JRA55 test fixture", "$(filename): $(sprint(showerror, e))")
        end
    end

    return true
end

function emit_ci_warning(title, message)
    if haskey(ENV, "GITHUB_ACTIONS")
        println(stderr, "::warning title=$(title)::$(message)")
    end
end

function download_from_artifacts(filepath::AbstractString; max_retries=3)
    filename = basename(filepath)
    fallback_url = ARTIFACTS_BASE_URL * filename
    @info "Downloading $filename from NumericalEarthArtifacts fallback..."
    for attempt in 1:max_retries
        try
            mktemp(dirname(filepath)) do tmppath, tmpio
                close(tmpio)
                Downloads.download(fallback_url, tmppath)
                mv(tmppath, filepath; force=true)
            end
            return
        catch e
            attempt < max_retries || rethrow(e)
            @warn "Artifact download attempt $attempt/$max_retries failed for $filename; retrying..." exception=(e, catch_backtrace())
            sleep(2.0 * attempt)  # linear backoff: 2s, 4s, ...
        end
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
