using NumericalEarth.DataWrangling: all_dates, native_times
using NumericalEarth.DataWrangling: compute_native_date_range
using Oceananigans.Grids: AbstractGrid
using Oceananigans.OutputReaders: PartlyInMemory
using Adapt

import NumericalEarth.DataWrangling: retrieve_data

compute_bounding_nodes(::Nothing, ::Nothing, LH, hnodes) = nothing
compute_bounding_nodes(bounds, ::Nothing, LH, hnodes) = bounds

function compute_bounding_nodes(x::Number, ::Nothing, LH, hnodes)
    ϵ = convert(typeof(x), 0.001) # arbitrary?
    return (x - ϵ, x + ϵ)
end

# TODO: remove the allowscalar
function compute_bounding_nodes(::Nothing, grid, LH, hnodes)
    hg = hnodes(grid, LH())
    h₁ = @allowscalar minimum(hg)
    h₂ = @allowscalar maximum(hg)
    return h₁, h₂
end

function compute_bounding_indices(::Nothing, hc)
    Nh = length(hc)
    return 1, Nh
end

function compute_bounding_indices(bounds::Tuple, hc)
    h₁, h₂ = bounds
    Nh = length(hc)

    # The following should work. If ᵒ are the extrema of nodes we want to
    # interpolate to, and the following is a sketch of the JRA55 native grid,
    #
    #      1         2         3         4         5
    # |         |         |         |         |         |
    # |    x  ᵒ |    x    |    x    |    x  ᵒ |    x    |
    # |         |         |         |         |         |
    # 1         2         3         4         5         6
    #
    # then for example, we should find that (iᵢ, i₂) = (1, 5).
    # So we want to reduce the first index by one, and limit them
    # both by the available data. There could be some mismatch due
    # to the use of different coordinate systems (ie whether λ ∈ (0, 360)
    # which we may also need to handle separately.
    i₁ = searchsortedfirst(hc, h₁)
    i₂ = searchsortedfirst(hc, h₂)
    i₁ = max(1, i₁ - 1)
    i₂ = min(Nh, i₂)

    return i₁, i₂
end

infer_longitudinal_topology(::Nothing) = Periodic

function infer_longitudinal_topology(λbounds)
    λ₁, λ₂ = λbounds
    TX = λ₂ - λ₁ ≈ 360 ? Periodic : Bounded
    return TX
end

function compute_bounding_indices(longitude, latitude, grid, LX, LY, λc, φc)
    λbounds = compute_bounding_nodes(longitude, grid, LX, λnodes)
    φbounds = compute_bounding_nodes(latitude, grid, LY, φnodes)

    i₁, i₂ = compute_bounding_indices(λbounds, λc)
    j₁, j₂ = compute_bounding_indices(φbounds, φc)
    TX = infer_longitudinal_topology(λbounds)

    return i₁, i₂, j₁, j₂, TX
end

"""
    JRA55NetCDFBackend(length [, metadata])
    JRA55NetCDFBackend(start, length, metadata)

Backwards-compatible shorthand for a `DatasetBackend` configured for
JRA55-style chunked NetCDF input: multiple time instances per file and no
inpainting (`inpainting = nothing`). Returns a `DatasetBackend` whose
metadata-driven `set!` dispatches to the JRA55 multi-year or repeat-year
methods defined below.
"""
JRA55NetCDFBackend(length) = DatasetBackend(length, nothing; inpainting=nothing)
JRA55NetCDFBackend(length, metadata::Metadata) = DatasetBackend(length, metadata; inpainting=nothing)
JRA55NetCDFBackend(start::Integer, length::Integer) = DatasetBackend(start, length, nothing; inpainting=nothing)
JRA55NetCDFBackend(start::Integer, length::Integer, metadata) = DatasetBackend(start, length, metadata; inpainting=nothing)

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

# Simple case, only one file per variable, no need to deal with multiple files
function set!(fts::JRA55NetCDFFTSRepeatYear, backend=fts.backend)

    metadata = backend.metadata

    filename = metadata.filename
    path = joinpath(metadata.dir, filename)
    ds = Dataset(path)

    # Nodes at the variable location

    λc = ds["lon"][:]
    φc = ds["lat"][:]
    LX, LY, LZ = location(fts)
    i₁, i₂, j₁, j₂, TX = compute_bounding_indices(nothing, nothing, fts.grid, LX, LY, λc, φc)

    nn   = time_indices(fts)
    nn   = collect(nn)
    name = dataset_variable_name(fts.backend.metadata)

    if issorted(nn)
        data = ds[name][i₁:i₂, j₁:j₂, nn]
    else
        # The time indices may be cycling past 1; eg ti = [6, 7, 8, 1].
        # However, DiskArrays does not seem to support loading data with unsorted
        # indices. So to handle this, we load the data in chunks, where each chunk's
        # indices are sorted, and then glue the data together.
        m = findfirst(n -> n == 1, nn)
        n1 = nn[1:m-1]
        n2 = nn[m:end]

        data1 = ds[name][i₁:i₂, j₁:j₂, n1]
        data2 = ds[name][i₁:i₂, j₁:j₂, n2]
        data = cat(data1, data2, dims=3)
    end

    close(ds)

    copyto!(interior(fts, :, :, 1, :), data)
    fill_halo_regions!(fts)

    return nothing
end

# Tricky case: multiple files per variable -- one file per year --
# we need to infer the file name from the metadata and split the data loading
function set!(fts::JRA55NetCDFFTSMultipleYears, backend=fts.backend)

    metadata   = backend.metadata
    name       = dataset_variable_name(metadata)
    start_date = first_date(metadata.dataset, metadata.name)

    ftsn = collect(time_indices(fts))

    # Only open files that actually contain needed time indices
    # (metadata.filename maps each time index to its yearly file)
    needed_files = unique(getfilename(metadata.filename, n) for n in ftsn)

    for file in needed_files

        path = joinpath(metadata.dir, file)
        ds = Dataset(path)

        # This can be simplified once we start supporting a
        # datetime `Clock` in Oceananigans
        file_dates = ds["time"][:]
        file_indices = 1:length(file_dates)
        file_times = zeros(length(file_dates))
        for (t, date) in enumerate(file_dates)
            delta = date - start_date
            delta = Second(delta).value
            file_times[t] = delta
        end

        # Intersect the time indices with the file times
        nn       = findall(n -> file_times[n] ∈ fts.times[ftsn], file_indices)
        ftsn_loc = findall(n -> fts.times[n] ∈ file_times[nn], ftsn)

        if !isempty(nn)
            # Nodes at the variable location
            λc = ds["lon"][:]
            φc = ds["lat"][:]
            LX, LY, LZ = location(fts)
            i₁, i₂, j₁, j₂, TX = compute_bounding_indices(nothing, nothing, fts.grid, LX, LY, λc, φc)

            if issorted(nn)
                data = ds[name][i₁:i₂, j₁:j₂, nn]
            else
                # At the cyclical wrap (end of year 60 → start of year 1),
                # file-local indices may be unsorted. DiskArrays requires
                # sorted indices, so we load in two sorted chunks.
                m = findfirst(n -> n == 1, nn)
                n1 = nn[1:m-1]
                n2 = nn[m:end]

                data1 = ds[name][i₁:i₂, j₁:j₂, n1]
                data2 = ds[name][i₁:i₂, j₁:j₂, n2]
                data = cat(data1, data2, dims=3)
            end

            close(ds)

            for n in 1:length(nn)
                copyto!(interior(fts, :, :, 1, ftsn_loc[n]), data[:, :, n])
            end
        else
            close(ds)
        end
    end

    fill_halo_regions!(fts)

    return nothing
end
