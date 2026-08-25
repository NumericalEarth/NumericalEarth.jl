function EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::PrescribedAtmosphere, grid;
                                                                    correction = nothing)
    regridder = atmosphere_regridder(atmosphere, grid)

    velocity_bcs = velocity_boundary_conditions(grid, (Center(), Center(), nothing))

    state = (; u   = Field{Center, Center, Nothing}(grid; boundary_conditions = velocity_bcs),
               v   = Field{Center, Center, Nothing}(grid; boundary_conditions = velocity_bcs),
               T   = Field{Center, Center, Nothing}(grid),
               p   = Field{Center, Center, Nothing}(grid),
               q   = Field{Center, Center, Nothing}(grid),
               Jʳⁿ = Field{Center, Center, Nothing}(grid),
               Jˢⁿ = Field{Center, Center, Nothing}(grid))

    correction = EarthSystemModels.InterfaceComputations.materialize_correction(correction, grid, atmosphere)
    return ComponentExchanger(state, regridder, correction)
end

function atmosphere_regridder(atmosphere::PrescribedAtmosphere, exchange_grid)
    atmos_grid = atmosphere.grid
    arch = architecture(exchange_grid)
    Nx, Ny, Nz = size(exchange_grid)

    # Make a NamedTuple of fractional indices
    # Note: we could use an array of FractionalIndices. Instead, for compatbility
    # with Reactant we construct FractionalIndices on the fly in `interpolate_atmospheric_state`.
    FT = eltype(atmos_grid)
    TX, TY, TZ = topology(exchange_grid)
    fi = TX() isa Flat ? nothing : Field{Center, Center, Nothing}(exchange_grid, FT)
    fj = TY() isa Flat ? nothing : Field{Center, Center, Nothing}(exchange_grid, FT)
    frac_indices = (i=fi, j=fj) # no k needed, only horizontal interpolation

    return frac_indices
end

function EarthSystemModels.InterfaceComputations.initialize!(exchanger::ComponentExchanger, grid, atmosphere::PrescribedAtmosphere)

    frac_indices = exchanger.regridder
    atmos_grid = atmosphere.grid
    kernel_parameters = interface_kernel_parameters(grid)
    launch!(architecture(grid), grid, kernel_parameters, _compute_fractional_indices!, frac_indices, grid, atmos_grid)

    return nothing
end
