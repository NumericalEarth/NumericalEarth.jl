using ClimaSeaIce: ClimaSeaIce, SeaIceModel, PhaseTransitions, ConductiveFlux,
                   sea_ice_slab_thermodynamics, snow_slab_thermodynamics
using ClimaSeaIce.SeaIceThermodynamics.HeatBoundaryConditions: PrescribedTemperature
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium, IceSnowConductiveFlux
using ClimaSeaIce.SeaIceDynamics: SplitExplicitSolver, SemiImplicitStress, SeaIceMomentumEquation, StressBalanceFreeDrift
using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology

using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper

using ..EarthSystemModels: ocean_surface_salinity, ocean_surface_velocities, reference_density
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations, SkinTemperature

default_rotation_rate = Oceananigans.defaults.planet_rotation_rate

ocean_reference_density(ocean::Simulation, FT) = convert(FT, reference_density(ocean))
ocean_reference_density(::Nothing, FT) = convert(FT, 1026.0)

function default_snow_thermodynamics(grid)
    FT = eltype(grid)
    snow_conductivity = FT(0.31)
    snow_surface_temperature = Field{Center, Center, Nothing}(grid)
    top_heat_boundary_condition = PrescribedTemperature(snow_surface_temperature.data)
    return snow_slab_thermodynamics(grid; conductivity = snow_conductivity, top_heat_boundary_condition)
end

"""
    sea_ice_simulation(grid, ocean=nothing;
                       clock = Clock(grid),
                       stop_time = clock.time isa Number ? Inf : Dates.DateTime(9999, 12, 31, 23, 59, 59),
                       Δt = 5minutes,
                       ice_salinity = 4, # psu
                       advection = nothing,
                       tracers = (),
                       ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                       ice_consolidation_thickness = 0.05, # m
                       sea_ice_density = 900, # kg m⁻³
                       snow_density = 330, # kg m⁻³
                       dynamics = sea_ice_dynamics(grid, ocean),
                       bottom_heat_boundary_condition = nothing,
                       top_heat_boundary_condition = nothing,
                       timestepper = :SplitRungeKutta3,
                       phase_transitions = PhaseTransitions(eltype(grid);
                                                            heat_capacity=ice_heat_capacity,
                                                            density=sea_ice_density),
                       conductivity = 2, # W m⁻¹ K⁻¹
                       internal_heat_flux = ConductiveFlux(; conductivity),
                       snow_thermodynamics = default_snow_thermodynamics(grid))

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
- `clock`: Clock for the underlying model. Defaults to `Clock(grid)`, a numeric clock starting at `time = 0`. Pass a `DateTime`-based clock to step the simulation in calendar time (e.g. when coupling).
- `stop_time`: Stop time for the simulation. Defaults to `Inf` for numeric clocks, or `DateTime(9999, 12, 31, 23, 59, 59)` for `DateTime` clocks.
- `Δt`: time step for the sea ice simulation
- `ice_salinity`: salinity of the sea ice (psu)
- `advection`: optional advection scheme for the sea ice model; if `nothing` (default), no advection
               is applied and only thermodynamics evolve the sea ice state
- `tracers`: optional tracers to include in the sea ice model
- `ice_heat_capacity`: heat capacity of the sea ice (J kg⁻¹ K⁻¹)
- `ice_consolidation_thickness`: thickness threshold for sea ice consolidation (m)
- `sea_ice_density`: density of the sea ice (kg m⁻³)
- `snow_density`: density of the snow (kg m⁻³)
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
                            stop_time = clock.time isa Number ? Inf : Dates.DateTime(9999, 12, 31, 23, 59, 59),
                            Δt = 5minutes,
                            ice_salinity = 4, # psu
                            advection = nothing,
                            tracers = (),
                            ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                            ice_consolidation_thickness = 0.05, # m
                            sea_ice_density = 900, # kg m⁻³
                            snow_density = 330, # kg m⁻³
                            dynamics = sea_ice_dynamics(grid, ocean),
                            bottom_heat_boundary_condition = nothing,
                            top_heat_boundary_condition = nothing,
                            timestepper = :SplitRungeKutta3,
                            phase_transitions = PhaseTransitions(eltype(grid);
                                                                 heat_capacity=ice_heat_capacity,
                                                                 density=sea_ice_density),
                            conductivity = 2, # W m⁻¹ K⁻¹
                            internal_heat_flux = ConductiveFlux(; conductivity),
                            snow_thermodynamics = default_snow_thermodynamics(grid))

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
                                top_heat_flux)

    verbose = false
    sea_ice = Simulation(sea_ice_model; Δt, stop_time, verbose)

    return sea_ice
end

default_coriolis(ocean::Simulation) = ocean.model.coriolis
default_coriolis(ocean::Nothing) = HydrostaticSphericalCoriolis(; rotation_rate=default_rotation_rate)

default_solver(grid, ocean) = SplitExplicitSolver(grid; substeps=120)

# We assume RK3 has a larger timestep
function default_solver(grid, ocean::Simulation)
    substeps = if ocean.model.timestepper isa SplitRungeKuttaTimeStepper
        240
    else
        120
    end
    return SplitExplicitSolver(grid; substeps)
end

function sea_ice_dynamics(grid, ocean=nothing;
                          sea_ice_ocean_drag_coefficient = 3.24e-3,
                          rheology = ElastoViscoPlasticRheology(),
                          coriolis = default_coriolis(ocean),
                          free_drift = nothing,
                          solver = default_solver(grid, ocean))

    SSU, SSV = ocean_surface_velocities(ocean)
    FT = eltype(grid)
    sea_ice_ocean_drag_coefficient = convert(FT, sea_ice_ocean_drag_coefficient)
    ρₑ = ocean_reference_density(ocean, FT)

    τo  = SemiImplicitStress(uₑ=SSU, vₑ=SSV, Cᴰ=sea_ice_ocean_drag_coefficient, ρₑ=ρₑ)
    τua = Field{Face, Center, Nothing}(grid)
    τva = Field{Center, Face, Nothing}(grid)

    if isnothing(free_drift)
        free_drift = StressBalanceFreeDrift((u=τua, v=τva), τo)
    end

    return SeaIceMomentumEquation(grid;
                                  coriolis,
                                  top_momentum_stress = (u=τua, v=τva),
                                  bottom_momentum_stress = τo,
                                  rheology,
                                  free_drift,
                                  solver)
end

#####
##### Extending EarthSystemModels interface
#####

EarthSystemModels.sea_ice_thickness(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.ice_thickness
EarthSystemModels.sea_ice_concentration(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.ice_concentration

EarthSystemModels.heat_capacity(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.phase_transitions.heat_capacity
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
        IceSnowConductiveFlux(snow_thermo.internal_heat_flux.conductivity, ice_flux.conductivity)
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
