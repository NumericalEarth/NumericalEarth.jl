const ERA5SingleLevelDataset = Union{ERA5HourlySingleLevel, ERA5MonthlySingleLevel,
                                    ERA5YearlySingleLevel, ERA5HourlyLand, ERA5MonthlyLand}
const NativeERA5SingleLevelFTS =
    FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{true, <:Any, Nothing, <:Metadata{<:ERA5SingleLevelDataset}}}

"""
$(TYPEDSIGNATURES)

Load the in-memory window of a native-grid ERA5 single-level time series, opening each
file once and reading consecutive timesteps together. Latitude coordinates and data
are both reversed into south-to-north order.
"""
function Oceananigans.Fields.set!(fts::NativeERA5SingleLevelFTS, backend=fts.backend)
    metadata = backend.metadata
    indices = collect(time_indices(fts))
    dates = metadata.dates[indices]
    paths = [metadata_path(metadata[n]) for n in indices]
    name = dataset_variable_name(metadata)

    for path in unique(paths)
        slots = findall(==(path), paths)
        NCDatasets.Dataset(path) do ds
            variable = ds[name]
            if ndims(variable) == 2
                raw = repeat(variable[:, :], 1, 1, length(slots))
            else
                file_dates = ds[haskey(ds, "valid_time") ? "valid_time" : "time"][:]
                date_indices = Dict(date => n for (n, date) in enumerate(file_dates))
                file_indices = [date_indices[date] for date in dates[slots]]
                order = sortperm(file_indices)
                file_indices = file_indices[order]
                slots = slots[order]
                selection = all(diff(file_indices) .== 1) ? (first(file_indices):last(file_indices)) : file_indices
                raw = variable[:, :, selection]
            end

            longitude = ds["longitude"][:]
            latitude = reverse(ds["latitude"][:])
            data = reshape(reverse(raw; dims=2), size(raw, 1), size(raw, 2), 1, length(slots))
            DataWrangling.set_region_data!(fts, data, longitude, latitude, metadata; slot_indices=slots)
        end
    end

    fill_halo_regions!(fts)
    return nothing
end
