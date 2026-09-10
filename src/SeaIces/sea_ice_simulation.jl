using ClimaSeaIce: ClimaSeaIce, SeaIceModel, PhaseTransitions, ConductiveFlux,
                   sea_ice_slab_thermodynamics, snow_slab_thermodynamics,
                   default_sea_ice_boundary_conditions
using ClimaSeaIce.SeaIceThermodynamics.HeatBoundaryConditions: PrescribedTemperature
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium, IceSnowConductiveFlux, LinearLiquidus,
                                        ThicknessDependentConductivity
using ClimaSeaIce.SeaIceDynamics: SplitExplicitSolver, SemiImplicitStress, SeaIceMomentumEquation, StressBalanceFreeDrift,
                                  LandfastBasalStress, maybe_extended_grid
using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology

using Oceananigans.Fields: ZeroField
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGridOfSomeKind
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper

using ..EarthSystemModels: ocean_surface_salinity, ocean_surface_velocities, ocean_surface_height,
                           surface_layer_velocities, reference_density
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations, SkinTemperature

default_rotation_rate = Oceananigans.defaults.planet_rotation_rate

# The ocean carries Conservative Temperature and the liquidus is compared against it, so the slope is
# a least-squares fit to the TEOS-10 freezing point expressed in Θ rather than in situ, over
# S = 28-35.5 psu: 0.013 K there, against 0.032 K for `LinearLiquidus`'s own 0.054, which is too warm
# at every salinity and so biases the ice-ocean heat flux `ρ cᵖ αₕ u★ (Θ - Tₘ)` one way everywhere.
#
# ⚠⚠ THE INTERCEPT MUST STAY AT ZERO. Fresh water freezes at 0 ᵒC, so the true liquidus passes through
# (0, 0) exactly and a fitted intercept is unphysical below S ≈ 1.9 psu — it would reach 0.0012 K in
# the 28-35.5 band and be WRONG at low salinity, which the model reaches at river mouths and, more
# dangerously, in cells inside the bathymetry, which are masked to S = 0. A fitted intercept of
# +0.107 turned the entire seafloor into a frazil source and killed two runs on 2026-09-03.
#
# ⚠ TWO freezing relations coexist, and the split is FORCED, not a preference:
#
#   * this linear one, used by every INTERFACE term — the ice base, the ice-ocean heat flux, the
#     atmosphere-ice flux. `solve_interface_conditions` dispatches on `::LinearLiquidus` and derives a
#     closed-form quadratic from (λ₁, λ₂) = (-slope, intercept), so a non-linear liquidus cannot be
#     substituted there without replacing the solve with an iteration.
#   * the exact UNESCO-in-situ-converted-to-Θ relation WITH pressure, in
#     `InterfaceComputations.melting_temperature_at_depth`, used by the frazil clamp.
#
# They agree to 0.011 K at S = 34.9, p = 0, which is where the interface terms all live — the ice base
# is at p ≈ 0, so the missing pressure term costs nothing there. That is exactly why the error hid for
# so long: it only bites in the frazil clamp, which scans the WHOLE column and reaches 0.5 K of error
# at 660 m and 2.4 K at 3000 m.
conservative_temperature_liquidus(FT) = LinearLiquidus(FT; slope = 0.054523)

ocean_reference_density(ocean::Simulation, FT) = convert(FT, reference_density(ocean))
ocean_reference_density(::Nothing, FT) = convert(FT, 1026.0)

# No slip is spelled exactly as on a domain boundary: a zero-value condition on the immersed
# boundary, which ClimaSeaIce's rheology reads to reflect the tangential velocity into the land and
# so double the wall contribution to the shear strain rate. Free slip leaves the wall stress-free.
velocity_boundary_conditions(grid, location, ::Val{:free_slip}) =
    correct_tripolar_bcs(grid, FieldBoundaryConditions(grid, location))

velocity_boundary_conditions(grid, location, ::Val{:no_slip}) =
    correct_tripolar_bcs(grid, FieldBoundaryConditions(grid, location; immersed = ValueBoundaryCondition(0)))

function sea_ice_velocity_boundary_conditions(grid, lateral_boundary_condition)
    slip = Val(lateral_boundary_condition)
    u = velocity_boundary_conditions(grid, (Face(), Center(), nothing), slip)
    v = velocity_boundary_conditions(grid, (Center(), Face(), nothing), slip)
    return (; u, v)
end

# `thickness_categories` requires a ClimaSeaIce that supports the sub-grid effective conductivity, so
# it is forwarded only when it differs from the mean-thickness default.
subgrid_conductivity_keyword(thickness_categories) =
    thickness_categories == 1 ? NamedTuple() : (; thickness_categories)

function default_snow_thermodynamics(grid; thickness_categories = 1)
    FT = eltype(grid)
    snow_conductivity = FT(0.31)
    snow_surface_temperature = Field{Center, Center, Nothing}(grid)
    top_heat_boundary_condition = PrescribedTemperature(snow_surface_temperature.data)
    return snow_slab_thermodynamics(grid; conductivity = snow_conductivity,
                                    subgrid_conductivity_keyword(thickness_categories)...,
                                    top_heat_boundary_condition)
end

correct_tripolar_bcs(grid, bcs) = bcs

function correct_tripolar_bcs(grid::TripolarGridOfSomeKind, bcs)
    if bcs.north isa BoundaryCondition && bcs.north.classification isa Zipper
        north = BoundaryCondition(bcs.north.classification, - bcs.north.condition)
        bcs = FieldBoundaryConditions(bcs.west, bcs.east, bcs.south, north, bcs.bottom, bcs.top, bcs.immersed)
    end
    return bcs
end

"""
    sea_ice_simulation(grid, ocean=nothing;
                       clock = Clock(grid),
                       stop_time = default_stop_time(grid, clock),
                       Δt = 5minutes,
                       ice_salinity = 4, # psu
                       advection = nothing,
                       tracers = (),
                       ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                       ice_consolidation_thickness = 0.05, # m
                       sea_ice_density = 900, # kg m⁻³
                       snow_density = 330, # kg m⁻³
                       lateral_boundary_condition = :no_slip,
                       dynamics = sea_ice_dynamics(grid, ocean),
                       bottom_heat_boundary_condition = nothing,
                       top_heat_boundary_condition = nothing,
                       timestepper = :SplitRungeKutta3,
                       phase_transitions = PhaseTransitions(eltype(grid);
                                                            heat_capacity=ice_heat_capacity,
                                                            density=sea_ice_density),
                       conductivity = 2, # W m⁻¹ K⁻¹
                       thickness_categories = 1,
                       snow_thickness_categories = thickness_categories,
                       itd_shape = nothing,
                       internal_heat_flux = ConductiveFlux(; conductivity, itd_shape,
                                                           subgrid_conductivity_keyword(thickness_categories)...),
                       snow_thermodynamics = default_snow_thermodynamics(grid; thickness_categories = snow_thickness_categories))

Construct a sea ice simulation with the given grid and optional ocean simulation.
The sea ice model is configured with a slab thermodynamics, Elasto-Visco-Plastic rheology,
and a SplitExplicit Runge-Kutta 3rd order time stepper by default. The thermodynamics
include conductive internal heat flux, and the option to specify top and bottom heat
boundary conditions. The dynamics include a semi-implicit ocean stress formulation,
with the option to specify a free drift velocity.

Arguments
=========
- `grid`: the grid on which to build the sea ice model
- `ocean`: optional ocean simulation to provide surface velocities and salinity for the sea ice

Keyword Arguments
=================
- `clock`: Clock for the underlying model. Defaults to `Clock(grid)`, a numeric clock starting at `time = 0`. 
  Pass a `DateTime`-based clock to step the simulation in calendar time (e.g. when coupling).
- `stop_time`: Stop time for the simulation. Defaults to `Inf` for numeric clocks, or 
  `DateTime(9999, 12, 31, 23, 59, 59)` for `DateTime` clocks. On Reactant architectures it defaults to `nothing`, since 
  Reactant does not support `stop_time`.
- `Δt`: time step for the sea ice simulation
- `ice_salinity`: salinity of the sea ice (psu)
- `advection`: optional advection scheme for the sea ice model; if `nothing` (default), no advection
               is applied and only thermodynamics evolve the sea ice state
- `tracers`: optional tracers to include in the sea ice model
- `ice_heat_capacity`: heat capacity of the sea ice (J kg⁻¹ K⁻¹)
- `ice_consolidation_thickness`: thickness threshold for sea ice consolidation (m)
- `sea_ice_density`: density of the sea ice (kg m⁻³)
- `snow_density`: density of the snow (kg m⁻³)
- `lateral_boundary_condition`: `:no_slip` (default) sets the ice velocity to zero on immersed
                                lateral boundaries, arresting ice against coastlines and through
                                narrow channels; `:free_slip` leaves them stress-free
- `dynamics`: sea ice dynamics model to use (default is `sea_ice_dynamics(grid, ocean)`)
- `bottom_heat_boundary_condition`: heat boundary condition at the ice-ocean interface (default
                                    is `IceWaterThermalEquilibrium` with ocean surface salinity)
- `top_heat_boundary_condition`: heat boundary condition at the ice-atmosphere interface (default
                                 is a prescribed temperature calculated in the flux computation)
- `timestepper`: time stepper to use for the sea ice model (default is `:SplitRungeKutta3`)
- `phase_transitions`: phase transition properties for the sea ice (default is a `PhaseTransitions`
                       with specified heat capacity and density)
- `conductivity`: thermal conductivity for the internal heat flux (W m⁻¹ K⁻¹)
- `internal_heat_flux`: internal heat flux formulation for the sea ice (default is a
                        `ConductiveFlux` with specified conductivity)
- `snow_thermodynamics`: thermodynamics for the snow layer (default is a slab thermodynamics with
                         specified conductivity and prescribed temperature)
"""
function sea_ice_simulation(grid, ocean=nothing;
                            clock = Clock(grid),
                            stop_time = default_stop_time(grid, clock),
                            Δt = 5minutes,
                            ice_salinity = 4, # psu
                            advection = nothing,
                            tracers = (),
                            ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                            ice_consolidation_thickness = 0.05, # m
                            sea_ice_density = 900, # kg m⁻³
                            snow_density = 330, # kg m⁻³
                            lateral_boundary_condition = :no_slip,
                            dynamics = sea_ice_dynamics(grid, ocean),
                            bottom_heat_boundary_condition = nothing,
                            top_heat_boundary_condition = nothing,
                            timestepper = :ForwardEuler,
                            liquidus = conservative_temperature_liquidus(eltype(grid)),
                            phase_transitions = PhaseTransitions(eltype(grid);
                                                                 heat_capacity=ice_heat_capacity,
                                                                 density=sea_ice_density,
                                                                 liquidus),
                            conductivity = 2, # W m⁻¹ K⁻¹
                            thickness_categories = 1,
                            snow_thickness_categories = thickness_categories,
                            itd_shape = nothing,
                            internal_heat_flux = ConductiveFlux(; conductivity, itd_shape,
                                                                subgrid_conductivity_keyword(thickness_categories)...),
                            snow_thermodynamics = default_snow_thermodynamics(grid; thickness_categories = snow_thickness_categories))

    # Build consistent boundary conditions for the ice model:
    # - bottom -> flux boundary condition
    # - top -> prescribed temperature boundary condition (calculated in the flux computation)

    if isnothing(top_heat_boundary_condition)
        top_surface_temperature = Field{Center, Center, Nothing}(grid)
        top_heat_boundary_condition = PrescribedTemperature(top_surface_temperature.data)
    end

    if isnothing(bottom_heat_boundary_condition)
        if isnothing(ocean)
            surface_ocean_salinity = 0
        else
            surface_ocean_salinity = ocean_surface_salinity(ocean)
        end
        bottom_heat_boundary_condition = IceWaterThermalEquilibrium(surface_ocean_salinity)
    end

    ice_thermodynamics = sea_ice_slab_thermodynamics(grid;
                                                     internal_heat_flux,
                                                     top_heat_boundary_condition,
                                                     bottom_heat_boundary_condition)

    bottom_heat_flux = Field{Center, Center, Nothing}(grid)
    top_heat_flux    = Field{Center, Center, Nothing}(grid)
    snowfall         = Field{Center, Center, Nothing}(grid)

    velocity_bcs = sea_ice_velocity_boundary_conditions(grid, lateral_boundary_condition)

    # Build the sea ice model
    sea_ice_model = SeaIceModel(grid;
                                clock,
                                ice_salinity,
                                advection,
                                tracers,
                                ice_consolidation_thickness,
                                sea_ice_density,
                                snow_density,
                                phase_transitions,
                                ice_thermodynamics,
                                snow_thermodynamics,
                                snowfall,
                                dynamics,
                                timestepper,
                                bottom_heat_flux,
                                boundary_conditions = velocity_bcs,
                                top_heat_flux)

    verbose = false
    sea_ice = Simulation(sea_ice_model; Δt, stop_time, verbose)

    return sea_ice
end

default_coriolis(ocean::Simulation) = ocean.model.coriolis
default_coriolis(ocean::Nothing) = HydrostaticSphericalCoriolis(; rotation_rate=default_rotation_rate)

# `ocean_surface_height` needs a ClimaSeaIce that carries the free-surface term, so it is forwarded
# only when the tilt is switched on.
#
# ⚠ When it is OFF we still pass an explicitly typed `ZeroField(eltype(grid))`. Leaving it to
# `SeaIceMomentumEquation`'s own default gives `ZeroField()`, which is `ZeroField{Int64}` — an
# Int-typed field inside a Float64 GPU kernel that evaluates `g * ∂xᶠᶜᶜ(…, η)`. That combination
# faulted with an illegal memory access on the first time step of every run with ice dynamics on
# 2026-09-03, while `_icetilt` (which passes a real Field) and `orca_noicedyn` (no dynamics) were
# unaffected. On a ClimaSeaIce without the free-surface term the keyword is simply not forwarded.
function ocean_surface_tilt_keyword(with_tilt, ocean, grid, gravitational_acceleration)
    :free_surface in fieldnames(SeaIceMomentumEquation) || return NamedTuple()
    η = with_tilt ? ocean_surface_height(ocean) : ZeroField(eltype(grid))
    return (; ocean_surface_height = η, gravitational_acceleration)
end

function sea_ice_dynamics(grid, ocean=nothing;
                          sea_ice_ocean_drag_coefficient = 5.5e-3,
                          sea_ice_ocean_drag_reference_depth = 6,
                          basal_stress = LandfastBasalStress(eltype(grid)),
                          rheology = ElastoViscoPlasticRheology(),
                          coriolis = default_coriolis(ocean),
                          free_drift = nothing,
                          with_ocean_surface_tilt = false,
                          gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                          solver = SplitExplicitSolver(grid; substeps=150))

    SSU, SSV = surface_layer_velocities(ocean, sea_ice_ocean_drag_reference_depth)
    FT = eltype(grid)
    sea_ice_ocean_drag_coefficient = convert(FT, sea_ice_ocean_drag_coefficient)
    ρₑ = ocean_reference_density(ocean, FT)

    τo = SemiImplicitStress(uₑ=SSU, vₑ=SSV, Cᴰ=sea_ice_ocean_drag_coefficient, ρₑ=ρₑ)

    velocity_grid = maybe_extended_grid(solver, grid)

    τua = Field{Face, Center, Nothing}(velocity_grid; boundary_conditions = default_sea_ice_boundary_conditions(velocity_grid, :u))
    τva = Field{Center, Face, Nothing}(velocity_grid; boundary_conditions = default_sea_ice_boundary_conditions(velocity_grid, :v))

    if isnothing(free_drift)
        free_drift = StressBalanceFreeDrift((u=τua, v=τva), τo)
    end

    return SeaIceMomentumEquation(velocity_grid;
                                  coriolis,
                                  top_momentum_stress = (u=τua, v=τva),
                                  bottom_momentum_stress = τo,
                                  basal_stress,
                                  rheology,
                                  free_drift,
                                  ocean_surface_tilt_keyword(with_ocean_surface_tilt, ocean, velocity_grid,
                                                             gravitational_acceleration)...,
                                  solver)
end

#####
##### Extending EarthSystemModels interface
#####

EarthSystemModels.sea_ice_thickness(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.ice_thickness
EarthSystemModels.sea_ice_concentration(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.ice_concentration
EarthSystemModels.intercepted_snowfall(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.mass_fluxes.intercepted_snowfall

EarthSystemModels.heat_capacity(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.phase_transitions.heat_capacity
# `sea_ice.model.sea_ice_density` is wrapped as a `ConstantField` by `SeaIceModel`;
# the scalar value lives on `phase_transitions.density`.
EarthSystemModels.reference_density(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.phase_transitions.density

function InterfaceComputations.net_fluxes(sea_ice::Simulation{<:SeaIceModel})
    net_momentum_fluxes = if isnothing(sea_ice.model.dynamics)
        u = Field{Face, Center, Nothing}(sea_ice.model.grid)
        v = Field{Center, Face, Nothing}(sea_ice.model.grid)
        (; u, v)
    else
        u = sea_ice.model.dynamics.external_momentum_stresses.top.u
        v = sea_ice.model.dynamics.external_momentum_stresses.top.v
        (; u, v)
    end

    net_top_sea_ice_fluxes = merge((; heat=sea_ice.model.external_heat_fluxes.top, snowfall=sea_ice.model.snowfall), net_momentum_fluxes)
    net_bottom_sea_ice_fluxes = (; heat=sea_ice.model.external_heat_fluxes.bottom)

    return (; bottom = net_bottom_sea_ice_fluxes, top = net_top_sea_ice_fluxes)
end

function InterfaceComputations.default_ai_temperature(sea_ice::Simulation{<:SeaIceModel})
    ice_flux = sea_ice.model.ice_thermodynamics.internal_heat_flux
    snow_thermo = sea_ice.model.snow_thermodynamics
    internal_flux = if isnothing(snow_thermo)
        ice_flux
    else
        IceSnowConductiveFlux(snow_thermo.internal_heat_flux.conductivity, ice_flux.conductivity,
                              ice_flux.itd_shape)
    end
    return SkinTemperature(internal_flux)
end

# Constructor that accepts the sea-ice model
function InterfaceComputations.ThreeEquationHeatFlux(sea_ice::Simulation{<:SeaIceModel}, FT::DataType = Oceananigans.defaults.FloatType;
                                                     heat_transfer_coefficient = 0.0095,
                                                     salt_transfer_coefficient = heat_transfer_coefficient / 35,
                                                     friction_velocity = convert(FT, 0.002))

    conductive_flux = sea_ice.model.ice_thermodynamics.internal_heat_flux
    ice_temperature = sea_ice.model.ice_thermodynamics.top_surface_temperature

    return ThreeEquationHeatFlux(conductive_flux,
                                 ice_temperature,
                                 convert(FT, heat_transfer_coefficient),
                                 convert(FT, salt_transfer_coefficient),
                                 friction_velocity)
end
