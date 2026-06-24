# Pure Julia CDS (Copernicus Climate Data Store) API client for ERA5 data
# Replaces Python era5cli dependency

# Note: HTTP and JSON3 are lazy-loaded in functions to avoid precompilation issues
# Downloads and Dates are stdlib so safe to load at top level
using Downloads
using Dates

struct CDSCredentials
    url::String
    key::String
end

"""
    read_cds_credentials(config_path=nothing)

Read CDS API credentials from:
1. Explicit config_path
2. Environment variables: CDSAPI_URL, CDSAPI_KEY
3. ~/.cdsapirc (standard location)
4. ~/.config/era5cli/cds_key.txt (era5cli location)
"""
function read_cds_credentials(config_path=nothing)
    # Priority 1: Explicit config path
    if !isnothing(config_path) && isfile(config_path)
        return parse_cdsapi_rc(config_path)
    end

    # Priority 2: Environment variables
    if haskey(ENV, "CDSAPI_URL") && haskey(ENV, "CDSAPI_KEY")
        return CDSCredentials(ENV["CDSAPI_URL"], ENV["CDSAPI_KEY"])
    end

    # Priority 3: ~/.cdsapirc (standard)
    default_rc = joinpath(homedir(), ".cdsapirc")
    if isfile(default_rc)
        return parse_cdsapi_rc(default_rc)
    end

    # Priority 4: era5cli config
    era5cli_config = joinpath(homedir(), ".config", "era5cli", "cds_key.txt")
    if isfile(era5cli_config)
        return parse_cdsapi_rc(era5cli_config)
    end

    error("""
    CDS credentials not found. Please create ~/.cdsapirc with:
    url: https://cds.climate.copernicus.eu/api
    key: <your-api-key>

    Or set environment variables CDSAPI_URL and CDSAPI_KEY.
    Register at: https://cds.climate.copernicus.eu/
    """)
end

function parse_cdsapi_rc(path::String)
    lines = readlines(path)
    url = ""
    key = ""

    for line in lines
        line = strip(line)
        isempty(line) && continue
        startswith(line, '#') && continue

        if contains(line, ':')
            parts = split(line, ':', limit=2)
            key_part = strip(parts[1])
            val_part = strip(parts[2])

            if key_part == "url"
                url = val_part
            elseif key_part == "key"
                key = val_part
            end
        end
    end

    isempty(url) && error("No 'url' found in $path")
    isempty(key) && error("No 'key' found in $path")

    return CDSCredentials(url, key)
end

"""
    submit_cds_request(credentials, dataset, params)

Submit download request to CDS API. Returns request ID.
"""
function submit_cds_request(credentials::CDSCredentials, dataset::String, params::Dict)
    import HTTP
    import JSON3

    endpoint = "$(credentials.url)/resources/$(dataset)"

    headers = [
        "Content-Type" => "application/json",
        "Accept" => "application/json",
        "Authorization" => "Basic $(base64encode(credentials.key))"
    ]

    body = JSON3.write(params)

    response = HTTP.post(endpoint, headers, body)
    result = JSON3.read(String(response.body))

    return result.request_id
end

"""
    poll_request_status(credentials, request_id; max_wait=3600, poll_interval=5, verbose=true)

Poll CDS request status until completion or timeout.
Returns download URL when ready.
"""
function poll_request_status(credentials::CDSCredentials, request_id::String;
                             max_wait=3600, poll_interval=5, verbose=true)
    import HTTP
    import JSON3

    endpoint = "$(credentials.url)/tasks/$(request_id)"
    headers = ["Authorization" => "Basic $(base64encode(credentials.key))"]

    start_time = time()
    last_status = ""

    while time() - start_time < max_wait
        response = HTTP.get(endpoint, headers)
        result = JSON3.read(String(response.body))

        status = result.state

        if verbose && status != last_status
            @info "CDS request $request_id: $status"
            last_status = status
        end

        if status == "completed"
            return result.location  # Download URL
        elseif status == "failed"
            error("CDS request failed: $(get(result, :error, "Unknown error"))")
        end

        sleep(poll_interval)
    end

    error("Request $request_id timed out after $(max_wait)s")
end

"""
    download_cds_file(url, output_path; credentials=nothing)

Download file from CDS result URL.
"""
function download_cds_file(url::String, output_path::String; credentials=nothing)
    mkpath(dirname(output_path))

    if !isnothing(credentials)
        # Add authentication if needed
        headers = ["Authorization" => "Basic $(base64encode(credentials.key))"]
        Downloads.download(url, output_path; headers)
    else
        Downloads.download(url, output_path)
    end

    @info "Downloaded: $output_path ($(filesize(output_path)) bytes)"
    return output_path
end

"""
    download_era5(variables, dates, area; dataset="reanalysis-era5-single-levels", output_path, kwargs...)

High-level function to download ERA5 data via CDS API.

# Arguments
- `variables`: Vector of CDS variable names (strings)
- `dates`: Vector of DateTime objects
- `area`: NamedTuple with (north, west, south, east) in degrees, or nothing for global
- `dataset`: CDS dataset name (default: "reanalysis-era5-single-levels")
- `output_path`: Where to save the downloaded NetCDF file
- `credentials`: CDSCredentials object, or nothing to auto-detect
- `product_type`: "reanalysis" or "ensemble_members"
- `time`: Hours to download (default: all 24 hours)
- `format`: "netcdf" or "grib"
- `max_wait`: Maximum wait time for request (seconds)
- `poll_interval`: How often to check request status (seconds)
- `verbose`: Print progress messages

# Returns
Path to downloaded file
"""
function download_era5(variables::Vector{String},
                       dates::AbstractVector{DateTime},
                       area;
                       dataset="reanalysis-era5-single-levels",
                       output_path::String,
                       credentials=nothing,
                       product_type="reanalysis",
                       time=collect(0:23),
                       format="netcdf",
                       max_wait=3600,
                       poll_interval=10,
                       verbose=true)

    # Get credentials
    creds = isnothing(credentials) ? read_cds_credentials() : credentials

    # Build request parameters
    params = build_cds_params(variables, dates, area;
                              product_type, time, format)

    if verbose
        @info "Submitting CDS request for $(length(variables)) variables, $(length(dates)) days"
    end

    # Submit request
    request_id = submit_cds_request(creds, dataset, params)

    if verbose
        @info "Request submitted: $request_id"
    end

    # Poll until ready
    download_url = poll_request_status(creds, request_id;
                                       max_wait, poll_interval, verbose)

    # Download file
    download_cds_file(download_url, output_path; credentials=creds)

    return output_path
end

"""
    build_cds_params(variables, dates, area; product_type, time, format)

Build CDS API request parameters dictionary.
"""
function build_cds_params(variables::Vector{String},
                          dates::AbstractVector{DateTime},
                          area;
                          product_type="reanalysis",
                          time=collect(0:23),
                          format="netcdf")

    # Extract unique years, months, days
    years = unique(year.(dates))
    months = unique(month.(dates))
    days = unique(day.(dates))

    # Format time as strings
    time_strs = [lpad(string(h), 2, '0') * ":00" for h in time]

    params = Dict(
        "product_type" => product_type,
        "variable" => variables,
        "year" => string.(years),
        "month" => [lpad(string(m), 2, '0') for m in months],
        "day" => [lpad(string(d), 2, '0') for d in days],
        "time" => time_strs,
        "format" => format
    )

    # Add area if specified (regional download)
    if !isnothing(area)
        params["area"] = [area.north, area.west, area.south, area.east]
    end

    return params
end

# Note: Metadata system integration can be added via extension later
# For now, use download_era5() function directly for ERA5 downloads
