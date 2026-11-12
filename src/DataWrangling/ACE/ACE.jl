module ACE

export ACE2ERA5

using ..DataWrangling: DataWrangling, download_with_retries

"""
    ACE2ERA5

Ai2 Climate Emulator version 2 trained on the ERA5 reanalysis: a 1° atmosphere emulator with a 6-hour time step,
distributed with its initial conditions and forcing data.

Data source: https://huggingface.co/allenai/ACE2-ERA5
"""
struct ACE2ERA5 end

download_ACE_cache::String = ""

function __init__()
    global download_ACE_cache = DataWrangling.download_cache("ACE")
    return nothing
end

DataWrangling.default_download_directory(::ACE2ERA5) = download_ACE_cache

const ace2_era5_repository = "https://huggingface.co/allenai/ACE2-ERA5/resolve/main"
const ace2_era5_initial_condition_years = (1940, 1950, 1979, 2001, 2020)

function ace_file(dataset::ACE2ERA5, path)
    local_path = joinpath(DataWrangling.default_download_directory(dataset), path)
    mkpath(dirname(local_path))
    isfile(local_path) || download_with_retries("$ace2_era5_repository/$path", local_path; description = "Download of $path")
    return local_path
end

end # module ACE
