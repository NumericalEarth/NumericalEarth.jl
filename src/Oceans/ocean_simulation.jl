using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition
using Oceananigans.DistributedComputations: DistributedGrid, all_reduce
using Oceananigans.Grids: inactive_node
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity,
                                                                     CATKEMixingLength,
                                                                     CATKEEquation
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Statistics: mean

#####
##### Utilities
#####

keep_user_boundary_condition(user, default) = user isa DefaultBoundaryCondition ? default : user

merge_boundary_conditions(user, default) = user

"""
$(TYPEDSIGNATURES)

Merge `user` and `default` boundary conditions side-by-side: every side the user left
unspecified (a `DefaultBoundaryCondition`) inherits the corresponding default side. This
allows users to prescribe, for example, only the lateral boundary conditions of a field
while retaining the default surface fluxes, bottom drag, and immersed boundary condition.
"""
function merge_boundary_conditions(user::FieldBoundaryConditions, default::FieldBoundaryConditions)
    return FieldBoundaryConditions(keep_user_boundary_condition(user.west,     default.west),
                                   keep_user_boundary_condition(user.east,     default.east),
                                   keep_user_boundary_condition(user.south,    default.south),
                                   keep_user_boundary_condition(user.north,    default.north),
                                   keep_user_boundary_condition(user.bottom,   default.bottom),
                                   keep_user_boundary_condition(user.top,      default.top),
                                   keep_user_boundary_condition(user.immersed, default.immersed))
end

@inline ϕ²(i, j, k, grid, ϕ)    = @inbounds ϕ[i, j, k]^2
@inline spᶠᶜᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², Φ.v))
@inline spᶜᶠᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², Φ.u))

@inline u_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.u[i, j, 1] * spᶠᶜᶜ(i, j, 1, grid, Φ)
@inline v_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.v[i, j, 1] * spᶜᶠᶜ(i, j, 1, grid, Φ)

# Keep a constant linear drag parameter independent on vertical level
@inline u_immersed_bottom_drag(i, j, k, grid, clock, Φ, μ) = @inbounds - μ * Φ.u[i, j, k] * spᶠᶜᶜ(i, j, k, grid, Φ)
@inline v_immersed_bottom_drag(i, j, k, grid, clock, Φ, μ) = @inbounds - μ * Φ.v[i, j, k] * spᶜᶠᶜ(i, j, k, grid, Φ)

# With or without additional fluxes
@inline build_top_bc(flux_field, ::Nothing) = FluxBoundaryCondition(flux_field)
@inline build_top_bc(flux_field, additional) = FluxBoundaryCondition(MultipleFluxes(flux_field, additional); discrete_form=true)

#####
##### Defaults
#####

default_free_surface(grid) = SplitExplicitFreeSurface(grid; cfl=0.7)
default_tracer_advection() = WENO(order=5)

estimate_maximum_Δt(grid::RectilinearGrid) = 30minutes # ?

function estimate_maximum_Δt(grid)
    arch = architecture(grid)
    Δx = mean(xspacings(grid))
    Δy = mean(yspacings(grid))
    Δθ = rad2deg(mean([Δx, Δy])) / grid.radius

    # The maximum Δt is roughly 1hours * Δθ, giving:
    # - 60 minutes for a 1 degree ocean
    # - 30 minutes for a 0.5 degree ocean
    # - 15 minutes for a 1/4 degree ocean
    # - 7.5 minutes for a 1/8 degree ocean
    # - 3.75 minutes for a 1/16 degree ocean
    # - 1.875 minutes for a 1/32 degree ocean

    # We set the maximum Δt to 1 hour
    Δt = min(1hours, 1hours * Δθ)

    return all_reduce(min, Δt, arch)
end

const TripolarOfSomeKind = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}

function default_free_surface(grid::TripolarOfSomeKind;
                              fixed_Δt = estimate_maximum_Δt(grid),
                              cfl = 0.7)
    free_surface = SplitExplicitFreeSurface(grid; cfl, fixed_Δt)
    return free_surface
end

function default_free_surface(grid::DistributedGrid;
                              fixed_Δt = estimate_maximum_Δt(grid),
                              cfl = 0.7)

    free_surface = SplitExplicitFreeSurface(grid; cfl, fixed_Δt)
    substeps = length(free_surface.substepping.averaging_weights)
    substeps = all_reduce(max, substeps, architecture(grid))
    free_surface = SplitExplicitFreeSurface(grid; substeps)
    @info "Using a $(free_surface)"
    return free_surface
end

function default_ocean_closure(FT=Oceananigans.defaults.FloatType)
    mixing_length = CATKEMixingLength(Cᵇ=0.01)
    turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ=1.0)
    return CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; mixing_length, turbulent_kinetic_energy_equation)
end

# Two-band shortwave penetration in the Paulson & Simpson (1977) form,
# Defaults are Jerlov Type I (clearest open-ocean water)
function default_radiative_forcing(grid)
    surface_fraction = 0.58  # Paulson & Simpson 1977, Table 2, Type I
    surface_scale    = 0.35  # [m]
    deep_scale       = 23    # [m]
    forcing = TwoColorRadiation(grid;
                                first_color_fraction          = surface_fraction,
                                first_absorption_coefficient  = 1 / surface_scale,
                                second_absorption_coefficient = 1 / deep_scale)
    return forcing
end

# TODO: Specify the grid to a grid on the sphere; otherwise we can provide a different
# function that requires latitude and longitude etc for computing coriolis=FPlane...
"""
    ocean_simulation(grid; model = :hydrostatic, kwargs...)

Construct and return an ocean simulation tailored to `grid`. The `model` keyword
selects the underlying Oceananigans model formulation:

- `:hydrostatic` (default) — builds a `HydrostaticFreeSurfaceModel`-based simulation
  with a free surface, CATKE vertical mixing, quadratic bottom drag, and a
  TEOS-10 equation of state. See [`hydrostatic_ocean_simulation`](@ref) for the full kwarg list.

- `:nonhydrostatic` — builds a `NonhydrostaticModel`-based simulation suitable
  for LES (full 3D pressure, no free surface, no barotropic forcing). See
  [`nonhydrostatic_ocean_simulation`](@ref) for the full kwarg list.

Remaining `kwargs` are forwarded to the per-model builder; an unknown kwarg for
the selected model raises the usual `MethodError`.

# Examples

```jldoctest
julia> using NumericalEarth, Oceananigans

julia> grid = RectilinearGrid(size = (10, 10, 6), extent = (1, 1, 1), halo=(6, 6, 5));

julia> ocean = ocean_simulation(grid)
Simulation of HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── Next time step: 30 minutes
├── run_wall_time: 0 seconds
├── run_wall_time / iteration: NaN days
├── stop_time: Inf days
├── stop_iteration: Inf
├── wall_time_limit: Inf
├── minimum_relative_step: 0.0
├── callbacks: OrderedDict with 4 entries:
│   ├── stop_time_exceeded => Callback of stop_time_exceeded on IterationInterval(1)
│   ├── stop_iteration_exceeded => Callback of stop_iteration_exceeded on IterationInterval(1)
│   ├── wall_time_limit_exceeded => Callback of wall_time_limit_exceeded on IterationInterval(1)
│   └── nan_checker => Callback of NaNChecker for u on IterationInterval(100)
└── output_writers: OrderedDict with no entries

julia> les = ocean_simulation(grid; model = :nonhydrostatic, Δt = 2)
Simulation of NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── Next time step: 2 seconds
├── run_wall_time: 0 seconds
├── run_wall_time / iteration: NaN days
├── stop_time: Inf days
├── stop_iteration: Inf
├── wall_time_limit: Inf
├── minimum_relative_step: 0.0
├── callbacks: OrderedDict with 4 entries:
│   ├── stop_time_exceeded => Callback of stop_time_exceeded on IterationInterval(1)
│   ├── stop_iteration_exceeded => Callback of stop_iteration_exceeded on IterationInterval(1)
│   ├── wall_time_limit_exceeded => Callback of wall_time_limit_exceeded on IterationInterval(1)
│   └── nan_checker => Callback of NaNChecker for u on IterationInterval(100)
└── output_writers: OrderedDict with no entries
```
"""
function ocean_simulation(grid; model::Symbol = :hydrostatic, kwargs...)
    if model === :hydrostatic
        return hydrostatic_ocean_simulation(grid; kwargs...)
    elseif model === :nonhydrostatic
        return nonhydrostatic_ocean_simulation(grid; kwargs...)
    else
        throw(ArgumentError("ocean_simulation: unknown model $(repr(model)); " *
                            "use :hydrostatic (default) or :nonhydrostatic."))
    end
end

"""
    hydrostatic_ocean_simulation(grid;
                                 Δt = estimate_maximum_Δt(grid),
                                 closure = default_ocean_closure(),
                                 tracers = (:T, :S),
                                 free_surface = default_free_surface(grid),
                                 reference_density = 1020,
                                 rotation_rate = default_planet_rotation_rate,
                                 gravitational_acceleration = default_gravitational_acceleration,
                                 bottom_drag_coefficient = Default(0.003),
                                 forcing = NamedTuple(),
                                 additional_surface_fluxes = NamedTuple(),
                                 biogeochemistry = nothing,
                                 timestepper = :SplitRungeKutta3,
                                 coriolis = Default(HydrostaticSphericalCoriolis(; rotation_rate)),
                                 momentum_advection = WENOVectorInvariant(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)),
                                 tracer_advection = WENO(order=7, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)),
                                 equation_of_state = TEOS10EquationOfState(; reference_density),
                                 boundary_conditions::NamedTuple = NamedTuple(),
                                 radiative_forcing = default_radiative_forcing(grid),
                                 clock = nothing,
                                 warn = true,
                                 verbose = false)

Construct and return a hydrostatic ocean simulation tailored to `grid`. Called
by `ocean_simulation(grid; model=:hydrostatic, ...)`.

This function assembles an Oceananigans's `HydrostaticFreeSurfaceModel` with physically
consistent defaults for advection, closures, the equation of state, surface fluxes, Coriolis,
barotropic pressure–gradient forcing, boundary conditions, and optional biogeochemistry.
It then wraps the model into an Oceananigans's `Simulation` with the specified timestepping options.


## Behaviour and automatic configuration

### Coriolis
- On spherical grids, an `Oceananigans.Coriolis.HydrostaticSphericalCoriolis` object
  is used by default.
- On rectilinear grids, Coriolis force is disabled unless explicitly provided.

### Single-column grids (`grid.Nx == 1 && grid.Ny == 1`)
- Advection is turned off (`momentum_advection = nothing`, `tracer_advection = nothing`).
- Users may override `bottom_drag_coefficient`, but its default is `0`.
- Immersed boundaries are ignored.

### Bottom drag and immersed boundaries
For multi-column grids:
- Quadratic bottom drag is automatically applied to both `u` and `v`.
- Immersed-boundary bottom drag conditions are constructed for both velocity components.
- Barotropic potential forcings for `u` and `v` are also added automatically, and
  user forcing tuples (e.g. `forcing = (u = ..., v = ...)`) are appended if provided.

### Radiative forcing
By default, `radiative_forcing` is `TwoColorRadiation` scheme.

### Tracers and closures
- `tracers` defaults to `(:T, :S)`.
- If the closure requires turbulent kinetic energy (e.g. `CATKEVerticalDiffusivity`),
  the turbulent kinetic energy `:e` tracer is automatically added while its advection is disabled.

### Boundary conditions
Default boundary conditions are constructed for `u`, `v`, `T`, and `S`, including
surface fluxes and bottom drag. User-provided boundary conditions override the
defaults on a per-field basis.

## Keyword Arguments

- `Δt`: Timestep used by the `Simulation`. Defaults to the maximum stable timestep estimated from the `grid`.
- `closure`: A turbulence or mixing closure. Defaults to `default_ocean_closure()`.
- `tracers`: Tuple of tracer names. Defaults to `(:T, :S)`.
- `free_surface`: Free–surface solver. Defaults to `default_free_surface(grid)`.
- `reference_density`: Reference seawater density used by the equation of state.
- `rotation_rate`: Planetary rotation rate used for Coriolis forcing.
- `gravitational_acceleration`: Gravitational acceleration, passed to buoyancy.
- `bottom_drag_coefficient`: Bottom drag coefficient. May be a `Default` wrapper.
- `forcing`: Named tuple of additional forcing(s) for individual fields.
- `additional_surface_fluxes`: Named tuple of additional top boundary flux conditions (e.g. `(; S=SurfaceFluxRestoring(...))`) for any field (`u`, `v`, `T`, `S`).
- `biogeochemistry`: A biogeochemical model or `nothing`.
- `timestepper`: Time-stepping scheme; options are `:SplitRungeKutta3` (default), or `:QuasiAdamsBashforth2`.
- `coriolis`: Coriolis object or `Default(...)` wrapper.
- `momentum_advection`: Momentum advection scheme. Defaults to `WENOVectorInvariant(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))`.
- `tracer_advection`: Tracer advection scheme or named tuple of schemes. Defaults to `WENO(order=7, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))`.
- `equation_of_state`: Equation of state object. Defaults to TEOS-10 (`TEOS10EquationOfState`).
- `boundary_conditions`: User-supplied boundary conditions; merged with defaults.
- `radiative_forcing`: Additional temperature forcing; merged into `forcing`.
- `clock`: Clock for the underlying model. Defaults to `nothing`, in which case the
  model builds its own default clock. Pass a `Clock` (e.g. `Clock{Float64}(time=0)` or
  a `DateTime`-based clock) to control the time type, for instance when coupling.
- `warn`: If `true`, warnings are emitted for potentially unintended setups.
- `verbose`: If `true`, prints additional setup information.
"""
function hydrostatic_ocean_simulation(grid;
                                      Δt = estimate_maximum_Δt(grid),
                                      closure = default_ocean_closure(),
                                      tracers = (:T, :S),
                                      free_surface = default_free_surface(grid),
                                      reference_density = 1020,
                                      rotation_rate = default_planet_rotation_rate,
                                      gravitational_acceleration = default_gravitational_acceleration,
                                      bottom_drag_coefficient = Default(0.003),
                                      forcing = NamedTuple(),
                                      additional_surface_fluxes = NamedTuple(),
                                      biogeochemistry = nothing,
                                      timestepper = :SplitRungeKutta3,
                                      coriolis = Default(HydrostaticSphericalCoriolis(; rotation_rate)),
                                      momentum_advection = WENOVectorInvariant(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)),
                                      tracer_advection = WENO(order=7, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)),
                                      equation_of_state = TEOS10EquationOfState(; reference_density),
                                      boundary_conditions::NamedTuple = NamedTuple(),
                                      radiative_forcing = default_radiative_forcing(grid),
                                      clock = nothing,
                                      warn = true,
                                      verbose = false)

    FT = eltype(grid)

    if grid isa RectilinearGrid # turn off Coriolis unless user-supplied
        coriolis = default_or_override(coriolis, nothing)
    else
        coriolis = default_or_override(coriolis)
    end

    # Detect whether we are on a single column grid
    Nx, Ny, _ = size(grid)
    single_column_simulation = Nx == 1 && Ny == 1

    if single_column_simulation
        # Let users put a bottom drag if they want
        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient, zero(grid))

        # Don't let users use advection in a single column model
        tracer_advection = nothing
        momentum_advection = nothing

        # No immersed boundaries in a single column grid
        u_immersed_bc = DefaultBoundaryCondition()
        v_immersed_bc = DefaultBoundaryCondition()
    else
        if warn && !(grid isa ImmersedBoundaryGrid) && verbose
            msg = """Are you totally, 100% sure that you want to build a simulation on

                   $(summary(grid))

                   rather than on an ImmersedBoundaryGrid?
                   """
            @warn msg
        end

        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient)

        u_immersed_drag = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
        v_immersed_drag = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

        u_immersed_bc = ImmersedBoundaryCondition(bottom=u_immersed_drag)
        v_immersed_bc = ImmersedBoundaryCondition(bottom=v_immersed_drag)

        # Forcing for u, v
        barotropic_potential = Field{Center, Center, Nothing}(grid)
        u_forcing = BarotropicPotentialForcing(XDirection(), barotropic_potential)
        v_forcing = BarotropicPotentialForcing(YDirection(), barotropic_potential)

        :u ∈ keys(forcing) && (u_forcing = (u_forcing, forcing[:u]))
        :v ∈ keys(forcing) && (v_forcing = (v_forcing, forcing[:v]))
        forcing = merge(forcing, (u=u_forcing, v=v_forcing))
    end

    if !isnothing(radiative_forcing)
        if :T ∈ keys(forcing)
            T_forcing = (forcing.T, radiative_forcing)
        else
            T_forcing = radiative_forcing
        end
        forcing = merge(forcing, (; T=T_forcing))
    end

    bottom_drag_coefficient = convert(FT, bottom_drag_coefficient)

    # Set up boundary conditions using Field
    top_zonal_momentum_flux      = τˣ = Field{Face, Center, Nothing}(grid)
    top_meridional_momentum_flux = τʸ = Field{Center, Face, Nothing}(grid)
    top_ocean_heat_flux          = Jᵀ = Field{Center, Center, Nothing}(grid)
    top_salt_flux                = Jˢ = Field{Center, Center, Nothing}(grid)
    top_freshwater_volume_flux   = Jʷ = Field{Center, Center, Nothing}(grid)
    
    if grid isa MutableGridOfSomeKind
        if :η ∈ keys(forcing)
            forcing = merge(forcing, (η = (Jʷ, forcing.η),))
        else
            forcing = merge(forcing, (η = Jʷ,))
        end
    end

    # Merge user-supplied additional fluxes with defaults
    default_additional_fluxes = (u=nothing, v=nothing, T=nothing, S=nothing)
    additional = merge(default_additional_fluxes, additional_surface_fluxes)

    # Construct ocean boundary conditions including surface forcing and bottom drag
    u_top_bc = build_top_bc(τˣ, additional.u)
    v_top_bc = build_top_bc(τʸ, additional.v)
    T_top_bc = build_top_bc(Jᵀ, additional.T)
    S_top_bc = build_top_bc(Jˢ, additional.S)

    u_bot_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
    v_bot_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

    default_boundary_conditions = (u = FieldBoundaryConditions(top=u_top_bc, bottom=u_bot_bc, immersed=u_immersed_bc),
                                   v = FieldBoundaryConditions(top=v_top_bc, bottom=v_bot_bc, immersed=v_immersed_bc),
                                   T = FieldBoundaryConditions(top=T_top_bc),
                                   S = FieldBoundaryConditions(top=S_top_bc))

    # Merge boundary conditions side-by-side with preference to user
    merged_boundary_conditions = NamedTuple(name => haskey(default_boundary_conditions, name) ?
                                            merge_boundary_conditions(boundary_conditions[name], default_boundary_conditions[name]) :
                                            boundary_conditions[name]
                                            for name in keys(boundary_conditions))

    boundary_conditions = merge(default_boundary_conditions, merged_boundary_conditions)
    buoyancy = SeawaterBuoyancy(; gravitational_acceleration, equation_of_state)

    if tracer_advection isa NamedTuple
        tracer_advection = with_tracers(tracers, tracer_advection, default_tracer_advection())
    else
        tracer_advection = NamedTuple(name => tracer_advection for name in tracers)
    end

    if hasclosure(closure, CATKEVerticalDiffusivity)
        # Turn off CATKE tracer advection
        tke_advection = (; e=nothing)
        tracer_advection = merge(tracer_advection, tke_advection)
    end

    # Only forward `clock` when supplied so the model keeps its own default otherwise.
    clock_kw = isnothing(clock) ? NamedTuple() : (; clock)

    ocean_model = HydrostaticFreeSurfaceModel(grid;
                                              buoyancy,
                                              closure,
                                              biogeochemistry,
                                              tracer_advection,
                                              momentum_advection,
                                              tracers,
                                              timestepper,
                                              free_surface,
                                              coriolis,
                                              forcing,
                                              boundary_conditions,
                                              clock_kw...)

    ocean = Simulation(ocean_model; Δt, verbose)

    return ocean
end

hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

const OceananigansModelSimulations = Union{
    Simulation{<:HydrostaticFreeSurfaceModel},
    Simulation{<:NonhydrostaticModel}
}

Grids.grid(ocean::OceananigansModelSimulations) = ocean.model.grid

#####
##### Extending NumericalEarth interface
#####

EarthSystemModels.reference_density(eos::TEOS10EquationOfState) = eos.reference_density
EarthSystemModels.reference_density(buoyancy_formulation::SeawaterBuoyancy) = EarthSystemModels.reference_density(buoyancy_formulation.equation_of_state)
EarthSystemModels.reference_density(ocean::OceananigansModelSimulations) = EarthSystemModels.reference_density(ocean.model.buoyancy.formulation)

EarthSystemModels.heat_capacity(ocean::OceananigansModelSimulations) = heat_capacity(ocean.model.buoyancy.formulation)
EarthSystemModels.heat_capacity(buoyancy_formulation::SeawaterBuoyancy) = heat_capacity(buoyancy_formulation.equation_of_state)

function EarthSystemModels.heat_capacity(::TEOS10EquationOfState{FT}) where FT
    cₚ⁰ = SeawaterPolynomials.TEOS10.teos10_reference_heat_capacity
    return convert(FT, cₚ⁰)
end
