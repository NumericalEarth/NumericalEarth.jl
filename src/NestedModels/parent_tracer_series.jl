#####
##### parent_tracer_series: the parent of further tracers of a nested ocean, read from a dataset
#####

using Oceananigans.OutputReaders: Clamp, FieldTimeSeries
using ..DataWrangling: Metadata, bracketing_dates

"""
$(TYPEDEF)

A parent `FieldTimeSeries` multiplied by a constant wherever a nested ocean samples it.

$(TYPEDFIELDS)
"""
struct ScaledParentSeries{S, F}
    "the parent `FieldTimeSeries`"
    series :: S
    "the factor every sample is multiplied by"
    factor :: F
end

function sample_parent!(field, loc, source::ScaledParentSeries, time)
    sample_parent!(field, loc, source.series, time)
    Base.parent(field) .*= convert(eltype(field), source.factor)
    return field
end

"""
$(TYPEDSIGNATURES)

Return `parent_tracers` for a [`nested_ocean_model`](@ref) on `grid`, read from `dataset` over `dates` (as for
`nested_ocean_model`): a `NamedTuple` with one `FieldTimeSeries` per entry of `tracers`, on the parent's clock.
Each entry of `tracers` names the variable of `dataset` a tracer of the child takes, for example `NO₃ = :nitrate`,
or pairs it with the factor that converts the dataset's units into the tracer's, for example
`P = :phytoplankton => 16/106` for a phytoplankton concentration in carbon taken as nitrogen.

Each series covers `grid` padded by `padding` degrees on the dataset's native grid, holding
`time_indices_in_memory` samples at a time and using `time_indexing` beyond its samples.
"""
function parent_tracer_series(grid, dataset; dates, tracers,
                              dir = default_download_directory(dataset),
                              padding = default_horizontal_padding(dataset),
                              time_indices_in_memory = 2,
                              time_indexing = Clamp())

    region = BoundingBox(grid; padding)
    arch = architecture(grid)
    start_time = first(dates)

    function series(name)
        metadata = Metadata(name; dataset, dates = bracketing_dates(dataset, name, dates), region, dir)
        return FieldTimeSeries(metadata, arch; time_indices_in_memory, time_indexing, start_time)
    end

    parent_series(name::Symbol) = series(name)
    parent_series((name, factor)::Pair) = ScaledParentSeries(series(name), factor)

    return map(parent_series, tracers)
end
