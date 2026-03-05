"""
    Streamfunction(grid;
                   regridder = nothing,
                   coordinate_field,
                   transport_field,
                   retained_dims = 2,
                   x_bins = 1020:0.1:1037,
                   method = :integral,
                   cumulative = true,
                   reverse = true,
                   in_sverdrups = true)

Construct a histogram-based streamfunction diagnostic using a coordinate-like field
(for example density) and a transport-like weight field.

`retained_dims` can be:
- an `Int` in `1:3`,
- or a tuple of retained dimensions in `1:3`.

When `cumulative=true`, the histogram is cumulatively integrated along the bin axis
(`dims=1` in histogram space), which yields the density-latitude style overturning
streamfunction used in common diagnostics.

`regridder` is an optional callable hook for pre-processing fields before histogramming.
If provided, it must return `(coordinate_field_regridded, transport_field_regridded)`
when called as `regridder(coordinate_field, transport_field)`.
"""
function Streamfunction(grid;
                        regridder = nothing,
                        coordinate_field,
                        transport_field,
                        retained_dims = 2,
                        x_bins = 1020:0.1:1037,
                        method = :integral,
                        cumulative = true,
                        reverse = true,
                        in_sverdrups = true)

    coordinate_field, transport_field = maybe_regrid_streamfunction_fields(regridder, coordinate_field, transport_field)

    coordinate_field.grid === grid || throw(ArgumentError("`coordinate_field.grid` does not match the supplied `grid`."))
    transport_field.grid === grid || throw(ArgumentError("`transport_field.grid` does not match the supplied `grid`."))

    histogram_dims = reduction_dims_from_retained(retained_dims)
    bins = (x = collect(x_bins),)

    ψ = Field(Histogram(coordinate_field;
                        bins,
                        weights = transport_field,
                        dims = histogram_dims,
                        method))

    ψ = Field(ψ / 1e6)

    return ψ
end

"""
    Streamfunction(; coordinate_field, transport_field, kwargs...)

Convenience overload that infers `grid` from `coordinate_field.grid`.
"""
function Streamfunction(; coordinate_field,
                          transport_field,
                          kwargs...)
    coordinate_field.grid === transport_field.grid ||
        throw(ArgumentError("`coordinate_field` and `transport_field` must live on the same grid when `grid` is omitted."))
    return Streamfunction(coordinate_field.grid; coordinate_field, transport_field, kwargs...)
end

@inline function reduction_dims_from_retained(retained_dims::Int)
    1 <= retained_dims <= 3 ||
        throw(ArgumentError("Integer `retained_dims` must be in 1:3, got retained_dims=$retained_dims."))
    return Tuple(d for d in (1, 2, 3) if d != retained_dims)
end

function reduction_dims_from_retained(retained_dims::Tuple)
    isempty(retained_dims) && throw(ArgumentError("`retained_dims` tuple cannot be empty."))
    all(d -> d isa Int && 1 <= d <= 3, retained_dims) ||
        throw(ArgumentError("Tuple `retained_dims` entries must all be integers in 1:3, got retained_dims=$retained_dims."))
    unique_retained = unique(retained_dims)
    return Tuple(d for d in (1, 2, 3) if !(d in unique_retained))
end

reduction_dims_from_retained(retained_dims) =
    throw(ArgumentError("Unsupported `retained_dims` type $(typeof(retained_dims)). Use an Int or tuple of retained dimensions."))

@inline maybe_regrid_streamfunction_fields(::Nothing, coordinate_field, transport_field) = (coordinate_field, transport_field)

function maybe_regrid_streamfunction_fields(regridder, coordinate_field, transport_field)
    regridded = regridder(coordinate_field, transport_field)
    regridded isa Tuple && length(regridded) == 2 ||
        throw(ArgumentError("`regridder` must return a 2-tuple `(coordinate_field, transport_field)`."))
    return regridded
end
