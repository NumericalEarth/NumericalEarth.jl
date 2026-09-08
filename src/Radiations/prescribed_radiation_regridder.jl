function EarthSystemModels.InterfaceComputations.ComponentExchanger(radiation::PrescribedRadiation, grid)

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

function EarthSystemModels.InterfaceComputations.initialize!(exchanger::ComponentExchanger, grid, radiation::PrescribedRadiation)
    frac_indices = exchanger.regridder
    # Skip horizontal regridding when both fractional-index buffers are
    # absent (purely Flat horizontal exchange grid).
    if isnothing(frac_indices.i) && isnothing(frac_indices.j)
        return nothing
    end
    rad_grid = radiation.grid
    kernel_parameters = interface_kernel_parameters(grid)
    launch!(architecture(grid), grid, kernel_parameters,
            _compute_fractional_indices!, frac_indices, grid, rad_grid)
    return nothing
end
