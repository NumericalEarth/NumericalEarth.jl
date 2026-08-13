#####
##### Copernicus land surface albedo (C3S `satellite-albedo` catalog entry)
#####
##### One CDS request per calendar month fetches the black-sky (`albb_dh`) and
##### white-sky (`albb_bh`) products for all of that month's dekads; each dekad's
##### pair is repacked into one compact local file by the CopernicusLandAlbedo module.
#####

const ALBEDO_CDS_PRODUCT = "satellite-albedo"

const AlbedoMetadata = NumericalEarth.DataWrangling.Metadata{<:CopernicusAlbedo}

"""
    build_albedo_request(name, dates) -> Dict{String, Any}

Construct the CDS request for the 1 km v2 black-sky/white-sky albedo pair covering
`dates`, which must share a `(year, month)` (CDS interprets `year`/`month`/
`nominal_day` as a Cartesian product, and the valid nominal days differ by month).
"""
function build_albedo_request(name, dates)
    dts = dates isa AbstractVector ? dates : [dates]

    length(unique((Dates.year(dt), Dates.month(dt)) for dt in dts)) == 1 ||
        error("build_albedo_request expects dates within one calendar month; got $(dts).")

    years  = unique(string.(Dates.year.(dts)))
    months = unique(lpad.(string.(Dates.month.(dts)), 2, '0'))
    days   = unique(lpad.(string.(Dates.day.(dts)), 2, '0'))
    satellites = unique([albedo_satellite(dt) for dt in dts])

    return Dict{String, Any}(
        "variable"              => collect(albedo_cds_request_variables[name]),
        "satellite"             => satellites,
        "sensor"                => ["vgt"],
        "product_version"       => ["v2"],
        "horizontal_resolution" => ["1km"],
        "year"                  => years,
        "month"                 => months,
        "nominal_day"           => days,
    )
end

# The delivery is either a ZIP of per-variable NetCDF files or a single NetCDF.
function extract_albedo_files(download_path, extraction_dir)
    if is_zip(download_path)
        run(`unzip -qo $download_path -d $extraction_dir`)
    else
        cp(download_path, joinpath(extraction_dir, "albedo.nc"); force=true)
    end
    return filter(p -> endswith(p, ".nc"), readdir(extraction_dir; join=true))
end

# The dekad a delivered file belongs to, from its time coordinate (authoritative)
# or the timestamp in its filename.
function albedo_file_date(path)
    date = NCDatasets.Dataset(path) do ds
        haskey(ds, "time") || return nothing
        t = ds["time"][1]
        return Dates.DateTime(Dates.year(t), Dates.month(t), Dates.day(t))
    end
    isnothing(date) || return date

    stamp = match(r"_(\d{8})\d{0,6}_", basename(path))
    isnothing(stamp) &&
        error("Cannot determine the date of the delivered albedo file $(basename(path)).")
    return Dates.DateTime(stamp[1], Dates.dateformat"yyyymmdd")
end

function repack_albedo_batch(nc_files, batch, path_of, destination_names, expected_size)
    members = Dict{Dates.DateTime, Dict{Symbol, Tuple{String, String}}}()
    for path in nc_files
        date = albedo_file_date(path)
        entry = get!(members, date, Dict{Symbol, Tuple{String, String}}())
        blacksky_name = find_albedo_variable(path, albedo_source_variable_candidates.blacksky)
        whitesky_name = find_albedo_variable(path, albedo_source_variable_candidates.whitesky)
        isnothing(blacksky_name) || (entry[:blacksky] = (path, blacksky_name))
        isnothing(whitesky_name) || (entry[:whitesky] = (path, whitesky_name))
    end

    for date in batch
        entry = get(members, date, nothing)
        if isnothing(entry) || !haskey(entry, :blacksky) || !haskey(entry, :whitesky)
            error("The CDS delivery is missing the black-sky/white-sky albedo pair for $date; ",
                  "it contained dates $(sort!(collect(keys(members)))).")
        end
        repack_albedo_pair(entry[:blacksky], entry[:whitesky], destination_names,
                           path_of[date], expected_size)
    end
    return nothing
end

"""
    download(metadata::Metadata{<:CopernicusAlbedo}; skip_existing=true, cleanup=true)

Download the dekadal black-sky and white-sky broadband albedo for every date of
`metadata` from the C3S `satellite-albedo` catalog entry, one CDS request per
calendar month, and repack each dekad's pair into a single compact local NetCDF.
Requires `~/.cdsapirc` credentials and acceptance of the Copernicus Global Land
product license on the CDS portal.
"""
function Downloads.download(metadata::AlbedoMetadata; skip_existing=true, cleanup=true)
    meta_filename = NumericalEarth.DataWrangling.metadata_filename
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    dir = metadata.dir
    mkpath(dir)

    dt_path_pairs = [(dt, joinpath(dir, meta_filename(metadata.dataset, metadata.name, dt, metadata.region)))
                     for dt in dates]
    pending = skip_existing ? filter(dt_path -> !isfile(dt_path[2]), dt_path_pairs) : dt_path_pairs
    isempty(pending) && return metadata_path(metadata)

    expected_size = size(metadata.dataset, metadata.name)[1:2]
    destination_names = copernicus_albedo_variables[metadata.name]
    path_of = Dict(dt => path for (dt, path) in pending)
    monthly = group_by_calendar_month([dt for (dt, _) in pending])

    @root for key in sort(collect(keys(monthly)))
        batch = sort(unique(monthly[key]))
        request = build_albedo_request(metadata.name, batch)
        tmp_download = joinpath(dir, "_tmp_albedo_$(key[1])$(lpad(key[2], 2, '0')).download")
        extraction_dir = mktempdir(dir)
        try
            retrieve_with_retries(ALBEDO_CDS_PRODUCT, request, tmp_download)
            nc_files = extract_albedo_files(tmp_download, extraction_dir)
            repack_albedo_batch(nc_files, batch, path_of, destination_names, expected_size)
        finally
            # Gate both on `cleanup` so `cleanup=false` keeps the raw download and the
            # extracted NetCDFs for debugging.
            cleanup && rm(tmp_download; force=true)
            cleanup && rm(extraction_dir; recursive=true, force=true)
        end
    end

    return metadata_path(metadata)
end
