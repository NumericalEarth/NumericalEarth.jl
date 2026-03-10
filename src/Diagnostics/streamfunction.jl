"""
    retained_dims(dim::Int)

Validate and return a retained streamfunction dimension `dim` in `1:3`.
"""
@inline function retained_dims(dim::Int)
    1 <= dim <= 3 || throw(ArgumentError("`dim` must be in 1:3, got dim=$dim."))
    return dim
end

"""
    reduced_dims(dim::Int)

Return reduced dimensions for `Histogram` given retained dimension `dim`.
"""
@inline function reduced_dims(dim::Int)
    d = retained_dims(dim)
    return Tuple(n for n in (1, 2, 3) if n != d)
end

"""
    retained_velocity(velocities, y_field)

Return the velocity component corresponding to retained streamfunction axis `y_field`.
Velocity components are interpolated to cell centers so they are compatible with
center-located tracer-like `x_field` inputs.
"""
function retained_velocity(velocities, y_field::Int)
    dim = retained_dims(y_field)
    grid = velocities.u.grid

    if dim == 1
        return Field(Oceananigans.AbstractOperations.KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.ℑxᶜᵃᵃ, grid, velocities.u))
    elseif dim == 2
        return Field(Oceananigans.AbstractOperations.KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.ℑyᵃᶜᵃ, grid, velocities.v))
    else
        return Field(Oceananigans.AbstractOperations.KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.ℑzᵃᵃᶜ, grid, velocities.w))
    end
end

"""
    Streamfunction(coupled_model;
                   x_field = coupled_model.ocean.model.tracers.T,
                   y_field = retained_dims(2),
                   bins = (x_field = -2:1:1027,),
                   in_sverdrups = true)

Compute a histogram-based streamfunction diagnostic from `coupled_model`.

The streamfunction is formed by histogramming `x_field` using weights from the retained
velocity component implied by `y_field` (default: meridional velocity for `y_field=2`),
with reduced dimensions given by `reduced_dims(y_field)`.

`bins` must only provide bins for `x_field`. If bins for `y_field` are supplied,
an `ArgumentError` is thrown because this API retains the native y-axis surfaces.
"""
function Streamfunction(coupled_model;
                        x_field = coupled_model.ocean.model.tracers.T,
                        y_field = retained_dims(2),
                        bins = (x_field = -2:1:1027,),
                        in_sverdrups = true)

    ydim = retained_dims(y_field)
    xbins = validate_streamfunction_bins(bins)
    weights = retained_velocity(coupled_model.ocean.model.velocities, ydim)

    ψ = Field(Histogram(x_field;
                        bins = (x_field = xbins,),
                        weights = weights,
                        dims = reduced_dims(ydim),
                        method = :integral))

    if in_sverdrups
        ψ = Field(ψ / 1e6)
    end

    return ψ
end

function validate_streamfunction_bins(bins::NamedTuple)
    haskey(bins, :y_field) && throw(ArgumentError("`bins` for `y_field` are not supported in this API. " *
                                                  "Use native retained y-axis surfaces by only supplying `x_field` bins."))

    haskey(bins, :x_field) || throw(ArgumentError("`bins` must contain `x_field` bins."))
    return collect(bins.x_field)
end

validate_streamfunction_bins(bins) =
    throw(ArgumentError("`bins` must be a NamedTuple like `(x_field = 1020:0.1:1037,)`, got $(typeof(bins))."))
