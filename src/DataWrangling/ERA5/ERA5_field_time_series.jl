using NCDatasets

#####
##### Type aliases for yearly ERA5 FieldTimeSeries
#####

const ERA5YearlyFileDataset = Union{ERA5YearlySingleLevel, ERA5HourlyLand, ERA5MonthlyLand}
const ERA5YearlySingleLevelBackend = DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:ERA5YearlyFileDataset}}
const ERA5NetCDFFTSMultipleYears = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:ERA5YearlySingleLevelBackend}

#####
##### Single timestep retrieval from yearly files
#####

"""
    retrieve_data(metadatum::Metadatum{<:ERA5YearlyFileDataset})

Read a 2D slice from the yearly ERA5 NetCDF file corresponding to the metadatum's date.
Opens the yearly file, finds the time index matching the date, and extracts that timestep.

The yearly file contains all 8760-8784 hours for one year. This function indexes into
the time dimension to extract just the requested hour.
"""
function DataWrangling.retrieve_data(metadatum::Metadatum{<:ERA5YearlyFileDataset})
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    ds = NCDatasets.Dataset(path)

    # Find time index for this specific datetime
    # ERA5 CDS files use "valid_time" as the time dimension name
    time_var = haskey(ds, "time") ? "time" : "valid_time"
    file_dates = ds[time_var][:]
    file_idx = findfirst(==(metadatum.dates), file_dates)

    if isnothing(file_idx)
        close(ds)
        error("Date $(metadatum.dates) not found in ERA5 yearly file $path. " *
              "File contains $(length(file_dates)) timesteps from $(first(file_dates)) to $(last(file_dates))")
    end

    # Extract 2D slice at this timestep
    # ERA5 is (lon, lat, time)
    data_2d = ds[name][:, :, file_idx]
    close(ds)

    # Latitude is stored from 90°N → 90°S, reverse it
    data_2d = reverse(data_2d, dims=2)

    # Add singleton z-dimension for 3D field compatibility
    # Return as (Nx, Ny, 1)
    return reshape(data_2d, size(data_2d, 1), size(data_2d, 2), 1)
end

#####
##### Multiple timestep loading for FieldTimeSeries
#####

"""
    read_era5_yearly_series(paths, requested_times, name)

Read `requested_times` for variable `name` from `paths`, which is either a single
file path (all requested times live in one file) or a `Vector` with one path per
requested time (consecutive equal paths — e.g. every hour of the same year — are
read together in a single, contiguous-when-possible slice, so each underlying
file is opened at most once).

Returns `(raw, λc, φc)` where `raw` has shape `(Nx, Ny, Nt)` with latitude still
in on-disk order (not yet reversed to -90→90).
"""
function read_era5_yearly_series(paths, requested_times, name)
    K = length(requested_times)
    getpath(k) = paths isa AbstractVector ? paths[k] : paths

    λc = φc = nothing
    chunks = Any[]

    k = 1
    while k <= K
        path = getpath(k)
        k_end = k
        while k_end < K && getpath(k_end + 1) == path
            k_end += 1
        end

        ds = NCDatasets.Dataset(path)

        if isnothing(λc)
            λc = ds["longitude"][:]
            φc = ds["latitude"][:]
        end

        # ERA5 CDS files use "valid_time" as the time dimension name
        time_var = haskey(ds, "time") ? "time" : "valid_time"
        file_times = ds[time_var][:]

        group_times = requested_times[k:k_end]
        file_indices = Vector{Int}(undef, length(group_times))
        for (m, t) in enumerate(group_times)
            idx = findfirst(==(t), file_times)
            if isnothing(idx)
                close(ds)
                error("Time $t not found in ERA5 file $path. File contains $(length(file_times)) " *
                      "timesteps from $(first(file_times)) to $(last(file_times))")
            end
            file_indices[m] = idx
        end

        # Check if indices are contiguous to use efficient range indexing
        if length(file_indices) > 1 && all(file_indices[m+1] == file_indices[m] + 1 for m in 1:length(file_indices)-1)
            raw = ds[name][:, :, file_indices[1]:file_indices[end]]
        elseif length(file_indices) == 1
            raw = ds[name][:, :, file_indices[1]:file_indices[1]]
        else
            raw = cat([ds[name][:, :, m] for m in file_indices]..., dims=3)
        end
        close(ds)

        push!(chunks, raw)
        k = k_end + 1
    end

    full = length(chunks) == 1 ? chunks[1] : cat(chunks...; dims=3)

    return full, λc, φc
end

"""
    set!(fts::ERA5NetCDFFTSMultipleYears, backend=fts.backend)

Load multiple timesteps from the yearly ERA5 file(s) into the FieldTimeSeries.
Reads a time range from each yearly file in one operation (efficient!); a
`FieldTimeSeries` spanning more than one calendar year reads from the
corresponding number of yearly files (see `read_era5_yearly_series`).

This is called by Oceananigans when a FieldTimeSeries needs to load new data from disk.
"""
function Oceananigans.Fields.set!(fts::ERA5NetCDFFTSMultipleYears, backend=fts.backend)
    metadata = backend.metadata
    paths = metadata_path(metadata)

    # Get time indices relative to the FieldTimeSeries
    nn = collect(time_indices(fts))

    # metadata.dates is a StepRange{DateTime} covering the simulation period
    dates_vec = collect(metadata.dates)
    requested_times = dates_vec[nn]
    requested_paths = paths isa AbstractVector ? paths[nn] : paths

    name = dataset_variable_name(metadata)
    raw, λc, φc = read_era5_yearly_series(requested_paths, requested_times, name)

    # Reverse latitude dimension (ERA5 stores 90→-90, we want -90→90)
    raw = reverse(raw, dims=2)

    # Reshape to FieldTimeSeries format: (Nx, Ny, 1, Nt)
    full_data = reshape(raw, length(λc), length(φc), 1, length(nn))

    # Set data in the FieldTimeSeries (handles interpolation/regridding)
    DataWrangling.set_region_data!(fts, full_data, λc, φc, metadata)

    # Fill halo regions for GPU computations
    fill_halo_regions!(fts)

    return nothing
end
