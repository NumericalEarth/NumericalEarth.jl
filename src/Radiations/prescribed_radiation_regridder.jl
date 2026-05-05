function ComponentExchanger(radiation::PrescribedRadiation, grid)

    regridder = radiation_regridder(radiation, grid)

    state = (; ℐꜜˢʷ = Field{Center, Center, Nothing}(grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(grid))

    return ComponentExchanger(state, regridder)
end

function radiation_regridder(radiation::PrescribedRadiation, exchange_grid)
    rad_grid = radiation.grid
    arch = architecture(exchange_grid)

    FT = eltype(rad_grid)
    TX, TY, TZ = topology(exchange_grid)
    fi = TX() isa Flat ? nothing : Field{Center, Center, Nothing}(exchange_grid, FT)
    fj = TY() isa Flat ? nothing : Field{Center, Center, Nothing}(exchange_grid, FT)
    return (i = fi, j = fj)
end

function initialize!(exchanger::ComponentExchanger, grid, radiation::PrescribedRadiation)
    frac_indices = exchanger.regridder
    # Skip horizontal regridding when both fractional-index buffers are
    # absent (purely Flat horizontal exchange grid).
    if isnothing(frac_indices.i) && isnothing(frac_indices.j)
        return nothing
    end
    rad_grid = radiation.grid
    kernel_parameters = interface_kernel_parameters(grid)
    launch!(architecture(grid), grid, kernel_parameters,
            _compute_radiation_fractional_indices!, frac_indices, grid, rad_grid)
    return nothing
end

@kernel function _compute_radiation_fractional_indices!(indices_tuple, exchange_grid, rad_grid)
    i, j = @index(Global, NTuple)
    kᴺ = size(exchange_grid, 3)
    X = _node(i, j, kᴺ + 1, exchange_grid, Center(), Center(), Face())
    fractional_indices_ij = FractionalIndices(X, rad_grid, Center(), Center(), Center())
    fi = indices_tuple.i
    fj = indices_tuple.j
    @inbounds begin
        if !isnothing(fi)
            fi[i, j, 1] = fractional_indices_ij.i
        end
        if !isnothing(fj)
            fj[i, j, 1] = fractional_indices_ij.j
        end
    end
end
