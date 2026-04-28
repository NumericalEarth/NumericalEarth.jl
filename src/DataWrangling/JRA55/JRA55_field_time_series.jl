using NumericalEarth.DataWrangling: all_dates, native_times
using NumericalEarth.DataWrangling: compute_native_date_range
using NumericalEarth.DataWrangling: set_region_data!
using Oceananigans.Grids: AbstractGrid
using Oceananigans.OutputReaders: PartlyInMemory
using Adapt

import NumericalEarth.DataWrangling: retrieve_data


const JRA55NetCDFFTS              = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{<:Any, <:Any, <:Any, <:JRA55Metadata}}
const JRA55NetCDFFTSRepeatYear    = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:RepeatYearJRA55}}}
const JRA55NetCDFFTSMultipleYears = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:MultiYearJRA55}}}

"""
    retrieve_data(metadatum::JRA55Metadatum)

Read the 2D slice from the JRA55 NetCDF file corresponding to `metadatum`'s
single date. JRA55 files chunk the series by calendar year and use a
`DateTimeNoLeap` (365-day) calendar internally, so the file-local index
must be resolved against either the file's own time axis (the safe path
for `MultiYearJRA55`, which spans real leap years) or the position within
the year's `all_dates` (which is unambiguous for `RepeatYearJRA55`,
because the repeat year — 1990 — is itself non-leap and the file holds
exactly 2920 entries that align 1:1 with `all_dates`).
"""
function retrieve_data(metadatum::RepeatYearJRA55Metadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    dates = all_dates(metadatum.dataset, metadatum.name)
    file_idx = findfirst(==(metadatum.dates), dates)

    if isnothing(file_idx)
        throw(ArgumentError("Date $(metadatum.dates) not found in $(metadatum.dataset) :$(metadatum.name) all_dates."))
    end

    ds = Dataset(path)
    data = ds[name][:, :, file_idx]
    close(ds)
    return data
end

function retrieve_data(metadatum::MultiYearJRA55Metadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    ds = Dataset(path)
    file_dates = ds["time"][:]
    file_idx = jra55_no_leap_file_index(file_dates, metadatum.dates)

    if isnothing(file_idx)
        close(ds)
        throw(ArgumentError(string("Date ", metadatum.dates,
                                   " not found in JRA55 multi-year file ", path,
                                   " (note: JRA55 multi-year files use a no-leap calendar; ",
                                   "Feb 29 of leap years has no corresponding file entry).")))
    end

    data = ds[name][:, :, file_idx]
    close(ds)
    return data
end

# Find the file-time index whose calendar components (Y/M/D/H/min) match
# the target date. Calendar-component matching avoids the
# `DateTimeNoLeap` ↔ `DateTime` epoch / leap-day mismatch that would
# otherwise break naive arithmetic-based lookup.
function jra55_no_leap_file_index(file_dates, target)
    return findfirst(file_dates) do d
        !ismissing(d) &&
        Dates.year(d)   == Dates.year(target)   &&
        Dates.month(d)  == Dates.month(target)  &&
        Dates.day(d)    == Dates.day(target)    &&
        Dates.hour(d)   == Dates.hour(target)   &&
        Dates.minute(d) == Dates.minute(target)
    end
end

# Note that each file should have the variables
#   - ds["time"]:     time coordinate
#   - ds["lon"]:      longitude at the location of the variable
#   - ds["lat"]:      latitude at the location of the variable
#   - ds["lon_bnds"]: bounding longitudes between which variables are averaged
#   - ds["lat_bnds"]: bounding latitudes between which variables are averaged
#   - ds[shortname]:  the variable data

# Split at the wrap point if `nn` cycles past 1 — DiskArrays requires sorted indices.
function jra55_read_data(ds, name, i, j, nn)
    if issorted(nn)
        return ds[name][i, j, nn]
    else
        m = findfirst(==(1), nn)
        d1 = ds[name][i, j, nn[1:m-1]]
        d2 = ds[name][i, j, nn[m:end]]
        return cat(d1, d2; dims=3)
    end
end

function set!(fts::JRA55NetCDFFTSRepeatYear, backend=fts.backend)
    metadata = backend.metadata
    ds = Dataset(joinpath(metadata.dir, metadata.filename))

    λc = ds["lon"][:]
    φc = ds["lat"][:]
    nn = collect(time_indices(fts))
    name = dataset_variable_name(metadata)

    raw = jra55_read_data(ds, name, :, :, nn)
    close(ds)
    full_data = reshape(raw, length(λc), length(φc), 1, length(nn))

    set_region_data!(fts, full_data, λc, φc, metadata)
    fill_halo_regions!(fts)
    return nothing
end

# JRA55 multi-year files use the no-leap calendar; matching by date components
# sidesteps the seconds-since-start drift across leap days.
function set!(fts::JRA55NetCDFFTSMultipleYears, backend=fts.backend)
    metadata = backend.metadata
    name     = dataset_variable_name(metadata)

    ftsn       = collect(time_indices(fts))
    slot_dates = metadata.dates[ftsn]
    needed_files = unique(getfilename(metadata.filename, n) for n in ftsn)

    for file in needed_files
        ds = Dataset(joinpath(metadata.dir, file))
        file_dates = ds["time"][:]

        nn       = Int[]
        ftsn_loc = Int[]
        for (loc, slot_date) in enumerate(slot_dates)
            file_idx = jra55_no_leap_file_index(file_dates, slot_date)
            if !isnothing(file_idx)
                push!(nn, file_idx)
                push!(ftsn_loc, loc)
            end
        end

        if !isempty(nn)
            λc = ds["lon"][:]
            φc = ds["lat"][:]
            raw = jra55_read_data(ds, name, :, :, nn)
            full_data = reshape(raw, length(λc), length(φc), 1, length(nn))
            set_region_data!(fts, full_data, λc, φc, metadata; slot_indices = ftsn_loc)
        end
        close(ds)
    end

    fill_halo_regions!(fts)
    return nothing
end

