"""
    Streamfunction(grid;
                   regridder = nothing,
                   x_field,
                   vel_field,
                   dims = 2,
                   x_bins = 1020:0.1:1037,
                   method = :integral,
                   cumulative = true,
                   reverse = true,
                   in_sverdrups = true)

Construct a histogram-based streamfunction diagnostic using an `x_field` coordinate
(for example density) and a velocity/transport-like weight field (`vel_field`).

`dims` can be:
- an `Int` in `1:3`, interpreted as the retained dimension (default `2`, latitude),
- or an explicit reduction tuple passed directly to `Histogram`.

When `cumulative=true`, the histogram is cumulatively integrated along the bin axis
(`dims=1` in histogram space), which yields the density-latitude style overturning
streamfunction used in common diagnostics.

`regridder` is an optional callable hook for pre-processing fields before histogramming.
If provided, it must return `(x_field_regridded, vel_field_regridded)` when called as
`regridder(x_field, vel_field)`.
"""
function Streamfunction(grid;
                        regridder = nothing,
                        x_field,
                        vel_field,
                        dims = 2,
                        x_bins = 1020:0.1:1037,
                        method = :integral,
                        cumulative = true,
                        reverse = true,
                        in_sverdrups = true)

    xf, vf = maybe_regrid_streamfunction_fields(regridder, x_field, vel_field)

    xf.grid === grid || throw(ArgumentError("`x_field.grid` does not match the supplied `grid`."))
    vf.grid === grid || throw(ArgumentError("`vel_field.grid` does not match the supplied `grid`."))

    histogram_dims = streamfunction_histogram_dims(dims)
    bins = (x = collect(x_bins),)

    ψ = Field(Histogram(xf;
                        bins,
                        weights = vf,
                        dims = histogram_dims,
                        method))

    if cumulative
        ψ = Field(CumulativeIntegral(ψ; dims = 1, reverse))
    end

    if in_sverdrups
        ψ = Field(ψ / 1e6)
    end

    return ψ
end

"""
    Streamfunction(; x_field, vel_field, kwargs...)

Convenience overload that infers `grid` from `x_field.grid`.
"""
function Streamfunction(; x_field, vel_field, kwargs...)
    x_field.grid === vel_field.grid ||
        throw(ArgumentError("`x_field` and `vel_field` must live on the same grid when `grid` is omitted."))
    return Streamfunction(x_field.grid; x_field, vel_field, kwargs...)
end

@inline function streamfunction_histogram_dims(dims::Int)
    1 <= dims <= 3 || throw(ArgumentError("Integer `dims` must be in 1:3, got dims=$dims."))
    return Tuple(d for d in (1, 2, 3) if d != dims)
end

function streamfunction_histogram_dims(dims::Tuple)
    all(d -> d isa Int && 1 <= d <= 3, dims) ||
        throw(ArgumentError("Tuple `dims` entries must all be integers in 1:3, got dims=$dims."))
    return dims
end

streamfunction_histogram_dims(dims) =
    throw(ArgumentError("Unsupported `dims` type $(typeof(dims)). Use an Int (retained dimension) or a tuple of reduced dimensions."))

@inline maybe_regrid_streamfunction_fields(::Nothing, x_field, vel_field) = (x_field, vel_field)

function maybe_regrid_streamfunction_fields(regridder, x_field, vel_field)
    regridded = regridder(x_field, vel_field)
    regridded isa Tuple && length(regridded) == 2 ||
        throw(ArgumentError("`regridder` must return a 2-tuple `(x_field, vel_field)`."))
    return regridded
end
