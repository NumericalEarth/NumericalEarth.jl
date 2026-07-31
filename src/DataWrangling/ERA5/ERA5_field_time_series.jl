using NCDatasets

#####
##### Type aliases for yearly ERA5 FieldTimeSeries
#####

const ERA5YearlySingleLevelBackend = DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:ERA5YearlySingleLevel}}
const ERA5NetCDFFTSMultipleYears = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:ERA5YearlySingleLevelBackend}

const ERA5LandBackend = DatasetBackend{<:Any, <:Any, <:Any, <:ERA5LandMetadata}
const ERA5LandNetCDFFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:ERA5LandBackend}

#####
##### Single timestep retrieval from yearly files
#####

"""
    retrieve_data(metadatum::Metadatum{<:ERA5YearlySingleLevel})

Read a 2D slice from the yearly ERA5 NetCDF file corresponding to the metadatum's date.
Opens the yearly file, finds the time index matching the date, and extracts that timestep.

The yearly file contains all 8760-8784 hours for one year. This function indexes into
the time dimension to extract just the requested hour.
"""
function DataWrangling.retrieve_data(metadatum::Metadatum{<:ERA5YearlySingleLevel})
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
    set!(fts::ERA5NetCDFFTSMultipleYears, backend=fts.backend)

Load multiple timesteps from the yearly ERA5 file into the FieldTimeSeries.
Reads a time range from the yearly file in one operation (efficient!).

This is called by Oceananigans when a FieldTimeSeries needs to load new data from disk.
Instead of reading files one by one, we read a chunk of timesteps from the yearly file.
"""
function Oceananigans.Fields.set!(fts::ERA5NetCDFFTSMultipleYears, backend=fts.backend)
    metadata = backend.metadata
    ds = NCDatasets.Dataset(metadata_path(metadata))

    # Get coordinate arrays
    λc = ds["longitude"][:]
    φc = ds["latitude"][:]

    # Get time indices relative to the FieldTimeSeries
    nn = collect(time_indices(fts))

    # Map FTS time indices to file time indices
    # metadata.dates contains the DateTime range for the simulation
    # We need to find where those dates are in the yearly file
    # ERA5 CDS files use "valid_time" as the time dimension name
    time_var = haskey(ds, "time") ? "time" : "valid_time"
    file_times = ds[time_var][:]  # All times in the file (8784 for full year)

    # Get the DateTime values for the requested indices from metadata
    # metadata.dates is a StepRange{DateTime} covering the simulation period
    dates_vec = collect(metadata.dates)
    requested_times = dates_vec[nn]

    # Find corresponding indices in the file
    file_indices = Vector{Int}(undef, length(requested_times))
    for (i, t) in enumerate(requested_times)
        idx = findfirst(==(t), file_times)
        if isnothing(idx)
            close(ds)
            error("Time $t not found in yearly file. File contains $(length(file_times)) timesteps from $(first(file_times)) to $(last(file_times))")
        end
        file_indices[i] = idx
    end

    # Get variable name
    name = dataset_variable_name(metadata)

    # Read all requested timesteps at once using FILE indices
    # Check if indices are contiguous to use efficient range indexing
    if length(file_indices) > 1 && all(file_indices[i+1] == file_indices[i] + 1 for i in 1:length(file_indices)-1)
        # Contiguous indices: use range (efficient)
        raw = ds[name][:, :, file_indices[1]:file_indices[end]]
    elseif length(file_indices) == 1
        # Single index
        raw = ds[name][:, :, file_indices[1]:file_indices[1]]
    else
        # Non-contiguous: read individually and stack
        raw = cat([ds[name][:, :, i] for i in file_indices]..., dims=3)
    end
    close(ds)

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

#####
##### Multi-year loading from yearly ERA5-Land files (cf. `MultiYearJRA55`)
#####

"""
    set!(fts::ERA5LandNetCDFFTS, backend=fts.backend)

Load the in-memory timesteps of an ERA5-Land `FieldTimeSeries` from its yearly files.
The window may straddle year boundaries: the needed files are resolved per slot through
the metadata's datewise filenames, and each file's timesteps are located by matching its
own time axis.
"""
function Oceananigans.Fields.set!(fts::ERA5LandNetCDFFTS, backend=fts.backend)
    metadata = backend.metadata
    name = dataset_variable_name(metadata)

    ftsn = collect(time_indices(fts))
    slot_dates = metadata.dates[ftsn]
    needed_files = unique(DataWrangling.getfilename(metadata.filename, n) for n in ftsn)

    filled_slots = Int[]

    for file in needed_files
        ds = NCDatasets.Dataset(joinpath(metadata.dir, file))
        time_name = haskey(ds, "valid_time") ? "valid_time" : "time"
        file_dates = ds[time_name][:]

        nn       = Int[]
        ftsn_loc = Int[]
        for (loc, slot_date) in enumerate(slot_dates)
            file_idx = findfirst(==(slot_date), file_dates)
            if !isnothing(file_idx)
                push!(nn, file_idx)
                push!(ftsn_loc, loc)
            end
        end

        if !isempty(nn)
            # DiskArrays requires sorted indices; a cyclic window can wrap past a year boundary.
            p = sortperm(nn)
            nn, ftsn_loc = nn[p], ftsn_loc[p]

            λc = ds["longitude"][:]
            φc = ds["latitude"][:]
            raw = ds[name][:, :, nn]

            # Latitude is stored from 90°N → 90°S
            raw = reverse(raw, dims=2)
            reverse!(φc)

            full_data = reshape(raw, length(λc), length(φc), 1, length(nn))
            DataWrangling.set_region_data!(fts, full_data, λc, φc, metadata; slot_indices = ftsn_loc)
            append!(filled_slots, ftsn_loc)
        end
        close(ds)
    end

    # An unmatched slot would otherwise keep its initial zeros and read back as valid data.
    if length(filled_slots) != length(slot_dates)
        missing_dates = slot_dates[setdiff(eachindex(slot_dates), filled_slots)]
        error("$(length(missing_dates)) of $(length(slot_dates)) requested $name timestamps are absent " *
              "from $(join(needed_files, ", ")); the first is $(first(missing_dates)).")
    end

    fill_halo_regions!(fts)

    return nothing
end
