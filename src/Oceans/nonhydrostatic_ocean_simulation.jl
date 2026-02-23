using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

"""
    nonhydrostatic_ocean_simulation(grid;
                                    Δt = 1,
                                    closure = nothing,
                                    tracers = (:T, :S),
                                    coriolis = nothing,
                                    reference_density = 1020,
                                    gravitational_acceleration = default_gravitational_acceleration,
                                    equation_of_state = TEOS10EquationOfState(; reference_density),
                                    advection = WENO(order=9),
                                    forcing = NamedTuple(),
                                    boundary_conditions::NamedTuple = NamedTuple(),
                                    verbose = false)

Construct and return a nonhydrostatic ocean simulation suitable for Large Eddy Simulation (LES).

Uses Oceananigans' `NonhydrostaticModel`, which resolves the full 3D pressure field
(no hydrostatic approximation). No free surface, barotropic forcing, or split-explicit
timestepping is needed.

## Keyword Arguments

- `Δt`: Timestep used by the `Simulation`. Defaults to `1` second.
- `closure`: Turbulence closure. Defaults to `nothing` (implicit LES via WENO).
- `tracers`: Tuple of tracer names. Defaults to `(:T, :S)`.
- `coriolis`: Coriolis parameter. Defaults to `nothing`.
- `reference_density`: Reference seawater density for the equation of state.
- `gravitational_acceleration`: Gravitational acceleration, passed to buoyancy.
- `equation_of_state`: Equation of state. Defaults to TEOS-10.
- `advection`: Advection scheme. Defaults to `WENO(order=9)`.
- `forcing`: Named tuple of additional forcing(s).
- `boundary_conditions`: User-supplied boundary conditions; merged with defaults.
- `verbose`: If `true`, prints additional setup information.
"""
function nonhydrostatic_ocean_simulation(grid;
                                         Δt = 1,
                                         closure = nothing,
                                         tracers = (:T, :S),
                                         coriolis = nothing,
                                         reference_density = 1020,
                                         gravitational_acceleration = default_gravitational_acceleration,
                                         equation_of_state = TEOS10EquationOfState(; reference_density),
                                         advection = WENO(order=9),
                                         forcing = NamedTuple(),
                                         boundary_conditions::NamedTuple = NamedTuple(),
                                         verbose = false)

    # Set up boundary conditions using Field
    top_zonal_momentum_flux      = τx = Field{Face, Center, Nothing}(grid)
    top_meridional_momentum_flux = τy = Field{Center, Face, Nothing}(grid)
    top_ocean_heat_flux          = Jᵀ = Field{Center, Center, Nothing}(grid)
    top_salt_flux                = Jˢ = Field{Center, Center, Nothing}(grid)

    u_top_bc = FluxBoundaryCondition(τx)
    v_top_bc = FluxBoundaryCondition(τy)
    T_top_bc = FluxBoundaryCondition(Jᵀ)
    S_top_bc = FluxBoundaryCondition(Jˢ)

    default_boundary_conditions = (u = FieldBoundaryConditions(top=u_top_bc),
                                   v = FieldBoundaryConditions(top=v_top_bc),
                                   T = FieldBoundaryConditions(top=T_top_bc),
                                   S = FieldBoundaryConditions(top=S_top_bc))

    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    buoyancy = SeawaterBuoyancy(; gravitational_acceleration, equation_of_state)

    ocean_model = NonhydrostaticModel(; grid,
                                      buoyancy,
                                      closure,
                                      advection,
                                      tracers,
                                      coriolis,
                                      forcing,
                                      boundary_conditions)

    ocean = Simulation(ocean_model; Δt, verbose)

    return ocean
end
