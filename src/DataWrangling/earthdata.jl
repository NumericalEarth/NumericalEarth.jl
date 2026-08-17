#####
##### NASA Earthdata access: granule discovery through the Common Metadata Repository
##### (CMR), plus authenticated granule download. Product-specific only through the
##### `short_name` / `version` passed in, so any Earthdata dataset module can use it.
#####

# Retry on transient failures, discarding a partial file between attempts. A long job is
# hundreds of sequential granules, and the archive throttles, so a single stalled transfer
# would otherwise abort it. An interrupt is never retried.
function download_with_retries(url, path; attempts = 3, downloader = nothing, description = "Download")
    for attempt in 1:attempts
        try
            if isnothing(downloader)
                Downloads.download(url, path)
            else
                Downloads.download(url, path; downloader)
            end
            return path
        catch error
            error isa InterruptException && rethrow()
            rm(path, force = true)
            attempt == attempts && rethrow()
            @warn "$description failed (attempt $attempt of $attempts); retrying..." url error
            sleep(2attempt)
        end
    end
end

"""
    earthdata_download(url, path; attempts = 3)

Download the NASA Earthdata granule at `url` to `path`, authenticating with the
`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` environment variables (register free at
https://urs.earthdata.nasa.gov).
"""
function earthdata_download(url, path; attempts = 3)
    username = get(ENV, "EARTHDATA_USERNAME", nothing)
    password = get(ENV, "EARTHDATA_PASSWORD", nothing)

    if isnothing(username)
        error("NASA Earthdata credentials not found: EARTHDATA_USERNAME is not set. " *
              "Register free at https://urs.earthdata.nasa.gov.")
    elseif isnothing(password)
        error("NASA Earthdata credentials not found: EARTHDATA_PASSWORD is not set. " *
              "Register free at https://urs.earthdata.nasa.gov.")
    end

    mktempdir() do tmp
        downloader = netrc_downloader(username, password, "urs.earthdata.nasa.gov", tmp)
        download_with_retries(url, path; attempts, downloader, description = "Earthdata granule download")
    end

    return path
end

"""
    earthdata_download_cached(url, cache_dir; attempts = 3)

Download the granule at `url` into `cache_dir` unless it is already there, and return
its path. Keyed on the granule name, so overlapping regions and sibling variables
reuse a granule instead of re-downloading it.
"""
function earthdata_download_cached(url, cache_dir; attempts = 3)
    path = joinpath(cache_dir, basename(url))
    isfile(path) && return path
    return earthdata_download(url, path; attempts)
end

"""
    cmr_granules_url(short_name, version, bbox; date = nothing, page_size = 2000, page_num = 1)

Build the NASA CMR granule-search URL for page `page_num` of the product `short_name` /
`version` whose granules intersect the `bbox` `BoundingBox` (encoded `W,S,E,N`,
longitudes in `[-180, 180]`). CMR search is anonymous; only the granule download itself
needs Earthdata credentials.

A `date` narrows the search to the day it opens, for a product whose granules are dated;
`nothing` searches the whole record, for one whose tiles are a single static epoch.
"""
function cmr_granules_url(short_name, version, bbox::BoundingBox;
                          date = nothing, page_size = 2000, page_num = 1)

    (!isnothing(bbox.longitude) && !isnothing(bbox.latitude)) ||
        throw(ArgumentError("cmr_granules_url requires a bounded (longitude, latitude) BoundingBox."))
    west, east = bbox.longitude
    south, north = bbox.latitude
    return string("https://cmr.earthdata.nasa.gov/search/granules.json",
                  "?short_name=", short_name,
                  "&version=", version,
                  "&bounding_box=", west, ",", south, ",", east, ",", north,
                  cmr_temporal_query(date),
                  "&page_size=", page_size,
                  "&page_num=", page_num)
end

cmr_temporal_query(::Nothing) = ""

cmr_temporal_query(date) =
    string("&temporal=", cmr_time(date), ",", cmr_time(DateTime(date) + Dates.Day(1)))

cmr_time(date) = string(Dates.format(DateTime(date), "yyyy-mm-ddTHH:MM:SS"), "Z")

"""
    cmr_granules(short_name, version, bbox; extension = "h5", page_size = 2000, attempts = 3)

Return the download URLs of the `short_name` / `version` granules whose footprints
intersect `bbox`, querying NASA CMR page by page until a short page signals the last
one. Only URLs ending in `extension` are collected, and one URL is kept per granule
name (preferring a protected `data`-host endpoint).
"""
function cmr_granules(short_name, version, bbox::BoundingBox;
                      extension = "h5", page_size = 2000, attempts = 3)

    granule_url_pattern = Regex(string("https://[^\"]+\\.", extension))
    by_granule = Dict{String, String}()

    mktempdir() do tmp
        page_num = 1
        while true
            url = cmr_granules_url(short_name, version, bbox; page_size, page_num)
            json_path = joinpath(tmp, string("cmr_granules_", page_num, ".json"))
            download_with_retries(url, json_path; attempts, description = "CMR granule query")
            text = read(json_path, String)

            granules_on_page = Set{String}()
            for match in eachmatch(granule_url_pattern, text)
                granule_url = match.match
                name = basename(granule_url)
                push!(granules_on_page, name)
                if !haskey(by_granule, name) || occursin("protected", granule_url)
                    by_granule[name] = granule_url
                end
            end

            length(granules_on_page) < page_size && break
            page_num += 1
        end
    end

    return collect(values(by_granule))
end
