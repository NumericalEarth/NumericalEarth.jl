using Downloads: Downloads
using NumericalEarth.DataWrangling: metadata_path

const ARTIFACTS_BASE_URL = "https://github.com/NumericalEarth/NumericalEarthArtifacts/releases/download/data-v1/"

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

# Download everything the remote-data tests share before the test files run in parallel, so
# no two workers fetch the same file at once.
function download_test_data()
    #####
    ##### Download bathymetry data
    #####

    ETOPOmetadata = Metadatum(:bottom_height, dataset=NumericalEarth.ETOPO.ETOPO2022())
    download_dataset_with_fallback(metadata_path(ETOPOmetadata); dataset_name="ETOPO2022") do
        download(ETOPOmetadata)
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
