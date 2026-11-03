using Oceananigans: instantiated_location
using Oceananigans.Fields: ConstantField
using Oceananigans.OutputReaders: FTS0

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

    state = merge(state, map(tracer -> similar_surface_field(grid, tracer), atmosphere.tracers))

    correction = EarthSystemModels.InterfaceComputations.materialize_correction(correction, grid, atmosphere)
    return ComponentExchanger(state, regridder, correction)
end

similar_surface_field(grid, tracer) = Field{Center, Center, Nothing}(grid)
similar_surface_field(grid, cf::ConstantField) = ConstantField(cf.constant)
similar_surface_field(grid, ::FTS0) = Field{Nothing, Nothing, Nothing}(grid)

function atmosphere_regridder(atmosphere::PrescribedAtmosphere, exchange_grid)
    # Note: we could use an array of FractionalIndices. Instead, for compatbility
    # with Reactant we construct FractionalIndices on the fly in `interpolate_atmospheric_state`.
    FT = eltype(atmosphere.grid)
    fi, fj = fractional_index_fields(exchange_grid, FT, Center(), Center())

    # Each tracer is interpolated from its own grid and location
    tracers = map(atmosphere.tracers) do tracer
        ℓx, ℓy, _ = instantiated_location(tracer)
        fractional_index_fields(exchange_grid, FT, ℓx, ℓy)
    end

    return (i=fi, j=fj, tracers) # no k needed, only horizontal interpolation
end

function fractional_index_fields(exchange_grid, FT, ℓx, ℓy)
    TX, TY, _ = topology(exchange_grid)
    fi = TX() isa Flat || isnothing(ℓx) ? nothing : Field{Center, Center, Nothing}(exchange_grid, FT)
    fj = TY() isa Flat || isnothing(ℓy) ? nothing : Field{Center, Center, Nothing}(exchange_grid, FT)
    return (i=fi, j=fj)
end

function EarthSystemModels.InterfaceComputations.initialize!(exchanger::ComponentExchanger, grid, atmosphere::PrescribedAtmosphere)
    arch = architecture(grid)
    frac_indices = exchanger.regridder
    kernel_parameters = interface_kernel_parameters(grid)
    launch!(arch, grid, kernel_parameters, _compute_fractional_indices!, frac_indices, grid, atmosphere.grid, Center(), Center())

    for (name, tracer) in pairs(atmosphere.tracers)
        tracer_indices = frac_indices.tracers[name]
        isnothing(tracer_indices.i) && isnothing(tracer_indices.j) && continue
        ℓx, ℓy, _ = instantiated_location(tracer)
        launch!(arch, grid, kernel_parameters, _compute_fractional_indices!, tracer_indices, grid, tracer.grid, ℓx, ℓy)
    end

    return nothing
end
