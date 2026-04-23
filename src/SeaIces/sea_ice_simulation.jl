using ClimaSeaIce
using ClimaSeaIce: SeaIceModel, PhaseTransitions, ConductiveFlux, sea_ice_slab_thermodynamics, snow_slab_thermodynamics
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium, IceSnowConductiveFlux
using ClimaSeaIce.SeaIceDynamics: SplitExplicitSolver, SemiImplicitStress, SeaIceMomentumEquation, StressBalanceFreeDrift
using ClimaSeaIce.Rheologies: IceStrength, ElastoViscoPlasticRheology

using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper

using NumericalEarth.EarthSystemModels: ocean_surface_salinity, ocean_surface_velocities
using NumericalEarth.Oceans: Default, reference_density

default_rotation_rate = Oceananigans.defaults.planet_rotation_rate

ocean_reference_density(ocean::Simulation, FT) = convert(FT, reference_density(ocean))
ocean_reference_density(::Nothing, FT) = convert(FT, 1026.0)

function default_snow_thermodynamics(grid)
    FT = eltype(grid)
    snow_conductivity = FT(0.31)
    # Use PrescribedTemperature so ClimaSeaIce does NOT run its own surface solve;
    # the coupled flux solver in NumericalEarth handles the snow surface temperature.
    snow_surface_temperature = Field{Center, Center, Nothing}(grid)
    top_heat_boundary_condition = PrescribedTemperature(snow_surface_temperature.data)
    return snow_slab_thermodynamics(grid; conductivity = snow_conductivity, top_heat_boundary_condition)
end

function sea_ice_simulation(grid, ocean=nothing;
                            Δt = 5minutes,
                            ice_salinity = 4, # psu
                            advection = nothing,
                            tracers = (),
                            ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                            ice_consolidation_thickness = 0.05, # m
                            sea_ice_density = 900, # kg m⁻³
                            snow_density = 330, # kg m⁻³
                            dynamics = sea_ice_dynamics(grid, ocean),
                            timestepper = :SplitRungeKutta3,
                            bottom_heat_boundary_condition = nothing,
                            top_heat_boundary_condition = nothing,
                            timestepper = :SplitRungeKutta3,
                            phase_transitions = PhaseTransitions(eltype(grid); heat_capacity=ice_heat_capacity, density=sea_ice_density),
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
                                top_heat_flux,
                                timestepper)

    verbose = false
    sea_ice = Simulation(sea_ice_model; Δt, verbose)

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

sea_ice_thickness(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.ice_thickness
sea_ice_concentration(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.ice_concentration

heat_capacity(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.phase_transitions.heat_capacity
# `sea_ice.model.sea_ice_density` is wrapped as a `ConstantField` by `SeaIceModel`;
# the scalar value lives on `phase_transitions.density`.
reference_density(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.phase_transitions.density

function net_fluxes(sea_ice::Simulation{<:SeaIceModel})
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

function default_ai_temperature(sea_ice::Simulation{<:SeaIceModel})
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
function ThreeEquationHeatFlux(sea_ice::Simulation{<:SeaIceModel}, FT::DataType = Oceananigans.defaults.FloatType;
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
