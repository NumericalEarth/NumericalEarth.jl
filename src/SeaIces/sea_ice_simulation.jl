using ClimaSeaIce
using ClimaSeaIce: SeaIceModel, SlabThermodynamics, PhaseTransitions, ConductiveFlux
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using ClimaSeaIce.SeaIceDynamics: SplitExplicitSolver, SemiImplicitStress, SeaIceMomentumEquation, StressBalanceFreeDrift
using ClimaSeaIce.Rheologies: IceStrength, ElastoViscoPlasticRheology

using NumericalEarth.EarthSystemModels: ocean_surface_salinity, ocean_surface_velocities
using NumericalEarth.Oceans: Default

default_rotation_rate = Oceananigans.defaults.planet_rotation_rate

function sea_ice_simulation(grid, ocean=nothing;
                            Δt = 5minutes,
                            ice_salinity = 4, # psu
                            advection = nothing, # for the moment
                            tracers = (),
                            ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                            ice_consolidation_thickness = 0.05, # m
                            sea_ice_density = 900, # kg m⁻³
                            ice_density = nothing, # deprecated name for sea_ice_density
                            dynamics = sea_ice_dynamics(grid, ocean),
                            bottom_heat_boundary_condition = nothing,
                            top_heat_boundary_condition = nothing,
                            phase_transitions = PhaseTransitions(; heat_capacity = ice_heat_capacity),
                            conductivity = 2, # kg m s⁻³ K⁻¹
                            internal_heat_flux = ConductiveFlux(; conductivity))

    if !isnothing(ice_density)
        sea_ice_density = ice_density
    end

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
            kᴺ = size(grid, 3)
            surface_ocean_salinity = ocean_surface_salinity(ocean)
        end
        bottom_heat_boundary_condition = IceWaterThermalEquilibrium(surface_ocean_salinity)
    end

    ice_thermodynamics = SlabThermodynamics(grid;
                                                  internal_heat_flux,
                                                  top_heat_boundary_condition,
                                                  bottom_heat_boundary_condition)

    bottom_heat_flux = Field{Center, Center, Nothing}(grid)
    top_heat_flux    = Field{Center, Center, Nothing}(grid)

    # Build the sea ice model
    sea_ice_model = SeaIceModel(grid;
                                ice_salinity,
                                advection,
                                tracers,
                                ice_consolidation_thickness,
                                sea_ice_density,
                                phase_transitions,
                                ice_thermodynamics,
                                dynamics,
                                bottom_heat_flux,
                                top_heat_flux)

    verbose = false

    # Build the simulation
    sea_ice = Simulation(sea_ice_model; Δt, verbose)

    return sea_ice
end

function sea_ice_dynamics(grid, ocean=nothing;
                          sea_ice_ocean_drag_coefficient = 5.5e-3,
                          rheology = ElastoViscoPlasticRheology(),
                          coriolis = HydrostaticSphericalCoriolis(; rotation_rate=default_rotation_rate),
                          free_drift = nothing,
                          solver = SplitExplicitSolver(grid; substeps=120))

    SSU, SSV = ocean_surface_velocities(ocean)
    sea_ice_ocean_drag_coefficient = convert(eltype(grid), sea_ice_ocean_drag_coefficient)

    τo  = SemiImplicitStress(uₑ=SSU, vₑ=SSV, Cᴰ=sea_ice_ocean_drag_coefficient)
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

sea_ice_model(sea_ice::Simulation) = sea_ice_model(sea_ice.model)
sea_ice_model(model::SeaIceModel) = model

sea_ice_model(model) =
    throw(ArgumentError("Expected a ClimaSeaIce.SeaIceModel or sea-ice Simulation, got $(typeof(model))"))

sea_ice_thickness(sea_ice) = sea_ice_model(sea_ice).ice_thickness
sea_ice_concentration(sea_ice) = sea_ice_model(sea_ice).ice_concentration

heat_capacity(sea_ice) = sea_ice_model(sea_ice).phase_transitions.heat_capacity
reference_density(sea_ice) = sea_ice_model(sea_ice).phase_transitions.density

function net_fluxes(sea_ice)
    model = sea_ice_model(sea_ice)

    net_momentum_fluxes = if isnothing(model.dynamics)
        u = Field{Face, Center, Nothing}(model.grid)
        v = Field{Center, Face, Nothing}(model.grid)
        (; u, v)
    else
        u = model.dynamics.external_momentum_stresses.top.u
        v = model.dynamics.external_momentum_stresses.top.v
        (; u, v)
    end

    net_top_sea_ice_fluxes = merge((; heat=model.external_heat_fluxes.top), net_momentum_fluxes)
    net_bottom_sea_ice_fluxes = (; heat=model.external_heat_fluxes.bottom)

    return (; bottom = net_bottom_sea_ice_fluxes, top = net_top_sea_ice_fluxes)
end

function default_ai_temperature(sea_ice)
    conductive_flux = internal_heat_flux_coefficient(sea_ice)
    return SkinTemperature(conductive_flux)
end

internal_heat_flux_coefficient(sea_ice::Simulation) =
    internal_heat_flux_coefficient(sea_ice_model(sea_ice))

internal_heat_flux_coefficient(model::SeaIceModel) =
    internal_heat_flux_coefficient(model.ice_thermodynamics.internal_heat_flux)

internal_heat_flux_coefficient(flux) = flux
internal_heat_flux_coefficient(flux::FluxFunction) = flux.parameters.flux

# Constructor that accepts the sea-ice model or a sea-ice Simulation.
function ThreeEquationHeatFlux(sea_ice, FT::DataType = Oceananigans.defaults.FloatType;
                               heat_transfer_coefficient = 0.0095,
                               salt_transfer_coefficient = heat_transfer_coefficient / 35,
                               friction_velocity = convert(FT, 0.002))

    model = sea_ice_model(sea_ice)
    conductive_flux = internal_heat_flux_coefficient(model)
    ice_temperature = model.ice_thermodynamics.top_surface_temperature

    return ThreeEquationHeatFlux(conductive_flux,
                                 ice_temperature,
                                 convert(FT, heat_transfer_coefficient),
                                 convert(FT, salt_transfer_coefficient),
                                 friction_velocity)
end
