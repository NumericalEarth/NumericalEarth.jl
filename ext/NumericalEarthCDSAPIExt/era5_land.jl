#####
##### Dispatch helpers and request hooks for ERA5-Land
#####

cds_product(::ERA5HourlyLand)  = "reanalysis-era5-land"
cds_product(::ERA5MonthlyLand) = "reanalysis-era5-land-monthly-means"

cds_varnames(::ERA5LandDataset) = ERA5Land_dataset_variable_names
nc_varnames(::ERA5LandDataset)  = ERA5Land_netcdf_variable_names

# The `reanalysis-era5-land` catalogue entry is not keyed by product type;
# sending a `product_type` key is rejected. The monthly-means entry is keyed
# by its averaging flavor instead of "reanalysis".
request_product_type!(request, ::ERA5LandDataset) = nothing
request_product_type!(request, ::ERA5MonthlyLand) = request["product_type"] = ["monthly_averaged_reanalysis"]

# Monthly means carry one field per (year, month); `day` is not a request key.
function request_dates!(request, ::ERA5MonthlyLand, dts)
    request["year"]  = unique(string.(Dates.year.(dts)))
    request["month"] = unique(lpad.(string.(Dates.month.(dts)), 2, '0'))
    request["time"]  = ["00:00"]
    return request
end

#####
##### ERA5-Land yearly-file download
#####
##### ERA5-Land is stored as one file per (variable, year) — the `ERA5YearlySingleLevel` /
##### `MultiYearJRA55` layout — with timesteps located by the file's own time axis at read
##### time. A whole year of monthly means fits in one CDS request (12 fields), fetched for
##### all requested variables at once; a year of hourly data exceeds the ERA5-Land
##### per-request cost limit, so it is fetched one variable and calendar month at a time.
##### Either way the chunks are concatenated locally into the per-variable yearly files.
#####

# All native dates of `year`, defining the request(s) that fill that year's file.
function era5_land_year_dates(dataset, name, year)
    dates = NumericalEarth.DataWrangling.all_dates(dataset, name)
    return filter(dt -> Dates.year(dt) == year, dates)
end

# Date batches per CDS request: a monthly-means year in one, hourly by calendar month.
era5_land_year_batches(::ERA5MonthlyLand, dates) = [dates]

function era5_land_year_batches(::ERA5HourlyLand, dates)
    monthly = group_by_calendar_month(dates)
    return [sort(monthly[key]) for key in sort(collect(keys(monthly)))]
end

# Variables per CDS request: monthly means are cheap enough to bundle; an hourly month
# is already ~744 fields per variable, so hourly variables get their own requests.
era5_land_variable_groups(::ERA5MonthlyLand, names) = [names]
era5_land_variable_groups(::ERA5HourlyLand, names) = [[name] for name in names]

function download_era5_land_year(name_path_pairs, dataset, year; region, dir, cleanup)
    names = [name for (name, _) in name_path_pairs]
    dates = era5_land_year_dates(dataset, first(names), year)
    batches = era5_land_year_batches(dataset, dates)

    mkpath(dir)
    chunk_paths = [joinpath(dir, "_tmp_$(names[1])_$(year)_$(lpad(b, 2, '0')).nc")
                   for b in eachindex(batches)]
    nc_name_path_pairs = [(nc_varnames(dataset)[name], path) for (name, path) in name_path_pairs]

    @root begin
        for (batch, chunk_path) in zip(batches, chunk_paths)
            request = build_era5_request(names, dataset, batch; region)
            retrieve_with_retries(cds_product(dataset), request, chunk_path)
            # Instantaneous-only variables share one step type, so no ZIP is expected.
            is_zip(chunk_path) &&
                error("The CDS returned a ZIP archive for the ERA5-Land request $(request).")
        end
        concatenate_era5_nc(chunk_paths, nc_name_path_pairs, coord_vars(dataset),
                            Set(["time", "valid_time"]))
        cleanup && foreach(chunk_path -> rm(chunk_path; force=true), chunk_paths)
    end

    return map(last, name_path_pairs)
end

"""
    Downloads.download(names, dataset::ERA5LandDataset, datetimes::AbstractVector; ...)

Download one or more ERA5-Land variables covering `datetimes` into per-variable
yearly files. Every year touched by `datetimes` is fetched whole, so all dates of a
year share one canonical file per variable.
"""
function Downloads.download(names::Vector{Symbol}, dataset::ERA5LandDataset, datetimes::AbstractVector;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            skip_existing = true,
                            cleanup = true)
    meta_filename = NumericalEarth.DataWrangling.metadata_filename

    paths = String[]
    for year in sort(unique(Dates.year.(datetimes)))
        for group in era5_land_variable_groups(dataset, names)
            name_path_pairs = [(name, joinpath(dir, meta_filename(dataset, name, Dates.DateTime(year), region)))
                               for name in group]
            append!(paths, map(last, name_path_pairs))

            pending = skip_existing ? filter(name_path -> !isfile(name_path[2]), name_path_pairs) :
                                      name_path_pairs
            isempty(pending) && continue

            download_era5_land_year(pending, dataset, year; region, dir, cleanup)
        end
    end

    return paths
end

function Downloads.download(metadata::ERA5LandMetadata; skip_existing=true, cleanup=true)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    return Downloads.download([metadata.name], metadata.dataset, dates;
                              region = metadata.region, dir = metadata.dir, skip_existing, cleanup)
end

function Downloads.download(meta::ERA5LandMetadatum; skip_existing=true, cleanup=true)
    paths = Downloads.download([meta.name], meta.dataset, [meta.dates];
                               region = meta.region, dir = meta.dir, skip_existing, cleanup)
    return first(paths)
end
