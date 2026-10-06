const JRA55NetCDFFTSRepeatYear    = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:RepeatYearJRA55}}}
const JRA55NetCDFFTSMultipleYears = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:MultiYearJRA55}}}

"""
    retrieve_data(metadatum::JRA55Metadatum)

Read the 2D slice from the JRA55 NetCDF file corresponding to `metadatum`'s single date.
`RepeatYearJRA55` resolves the index from the position within `all_dates` (the file holds
exactly 2920 entries for 1990 and aligns 1:1 with `all_dates`); `MultiYearJRA55` resolves
the index against the file's own `time` axis (one Gregorian-calendar file per year).
"""
function DataWrangling.retrieve_data(metadatum::RepeatYearJRA55Metadatum)
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

function DataWrangling.retrieve_data(metadatum::MultiYearJRA55Metadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    ds = Dataset(path)
    file_dates = ds["time"][:]
    file_idx = findfirst(==(metadatum.dates), file_dates)

    if isnothing(file_idx)
        close(ds)
        throw(ArgumentError(string("Date ", metadatum.dates, " not found in JRA55 multi-year file ", path, ".")))
    end

    data = ds[name][:, :, file_idx]
    close(ds)
    return data
end

# Read the window one time slice at a time through two slice-sized host buffers. Reading it whole
# (`ds[name][:, :, nn]`) allocates ~14 bytes per element (137 MiB for a 640×320×50 window), and with every
# atmospheric variable reloading at once that forces full GC sweeps. The files are chunked one time slice
# per chunk, so slice reads cost the same I/O.
function set_jra55_slices!(fts, ds, name, file_indices, slots, metadata)
    λc  = ds["lon"][:]
    φc  = ds["lat"][:]
    var = ds[name]

    Nx, Ny = length(λc), length(φc)
    data   = Array{eltype(var)}(undef, Nx, Ny, 1, 1)
    buffer = Array{eltype(parent(var))}(undef, Nx, Ny, 1, 1)

    for (n, slot) in zip(file_indices, slots)
        NCDatasets.load!(var, data, buffer, :, :, n)
        set_region_data!(fts, data, λc, φc, metadata; slot_indices = slot:slot)
    end

    return nothing
end

function Oceananigans.Fields.set!(fts::JRA55NetCDFFTSRepeatYear, backend=fts.backend)
    metadata = backend.metadata
    ds = Dataset(joinpath(metadata.dir, metadata.filename))
    nn = time_indices(fts)
    set_jra55_slices!(fts, ds, dataset_variable_name(metadata), nn, eachindex(nn), metadata)
    close(ds)
    fill_halo_regions!(fts)
    return nothing
end

function Oceananigans.Fields.set!(fts::JRA55NetCDFFTSMultipleYears, backend=fts.backend)
    metadata = backend.metadata
    name     = dataset_variable_name(metadata)

    ftsn       = collect(time_indices(fts))
    slot_dates = metadata.dates[ftsn]
    needed_files = unique(getfilename(metadata.filename, n) for n in ftsn)

    filled_slots = Int[]

    for file in needed_files
        ds = Dataset(joinpath(metadata.dir, file))
        file_dates = ds["time"][:]

        nn       = Int[]
        ftsn_loc = Int[]
        for (loc, slot_date) in enumerate(slot_dates)
            file_idx = findfirst(==(slot_date), file_dates)
            if !isnothing(file_idx)
                push!(nn, file_idx)
                push!(ftsn_loc, loc)
            end
        end

        set_jra55_slices!(fts, ds, name, nn, ftsn_loc, metadata)
        append!(filled_slots, ftsn_loc)
        close(ds)
    end

    # An unmatched slot would otherwise keep its initial zeros and read back as valid data.
    if length(filled_slots) != length(slot_dates)
        missing_dates = slot_dates[setdiff(eachindex(slot_dates), filled_slots)]
        error("$(length(missing_dates)) of $(length(slot_dates)) requested $name timestamps are absent " *
              "from $(join(needed_files, ", ")); the first is $(first(missing_dates)). The dates that " *
              "`all_dates` reports for this variable must match the file's `time` axis exactly.")
    end

    fill_halo_regions!(fts)
    return nothing
end
