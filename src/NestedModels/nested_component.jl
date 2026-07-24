#####
##### ESM component interface: a NestedModel couples like its child
#####
#
# The parent enters only through the child's lateral BCs and relaxation, so every
# surface-coupling call forwards (explicitly, not via getproperty) to the child.

const NestedModelSimulation = Simulation{<:NestedModel}

EarthSystemModels.thermodynamics_parameters(nm::NestedModel) =
    EarthSystemModels.thermodynamics_parameters(nm.child)

EarthSystemModels.surface_layer_height(nm::NestedModel) =
    EarthSystemModels.surface_layer_height(nm.child)

EarthSystemModels.boundary_layer_height(nm::NestedModel) =
    EarthSystemModels.boundary_layer_height(nm.child)

InterfaceComputations.ComponentExchanger(nm::NestedModel, exchange_grid; kw...) =
    ComponentExchanger(nm.child, exchange_grid; kw...)

EarthSystemModels.interpolate_state!(exchanger, exchange_grid, nm::NestedModel, coupled_model) =
    EarthSystemModels.interpolate_state!(exchanger, exchange_grid, nm.child, coupled_model)

InterfaceComputations.net_fluxes(nm::NestedModel) = InterfaceComputations.net_fluxes(nm.child)

EarthSystemModels.update_net_fluxes!(coupled_model, nm::NestedModel) =
    EarthSystemModels.update_net_fluxes!(coupled_model, nm.child)

# Simulation-wrapped delegates: unwrap once to the NestedModel.

EarthSystemModels.thermodynamics_parameters(sim::NestedModelSimulation) =
    EarthSystemModels.thermodynamics_parameters(component_model(sim))

EarthSystemModels.surface_layer_height(sim::NestedModelSimulation) =
    EarthSystemModels.surface_layer_height(component_model(sim))

EarthSystemModels.boundary_layer_height(sim::NestedModelSimulation) =
    EarthSystemModels.boundary_layer_height(component_model(sim))

InterfaceComputations.ComponentExchanger(sim::NestedModelSimulation, exchange_grid; kw...) =
    ComponentExchanger(component_model(sim), exchange_grid; kw...)

EarthSystemModels.interpolate_state!(exchanger, exchange_grid, sim::NestedModelSimulation, coupled_model) =
    EarthSystemModels.interpolate_state!(exchanger, exchange_grid, component_model(sim), coupled_model)

InterfaceComputations.net_fluxes(sim::NestedModelSimulation) =
    InterfaceComputations.net_fluxes(component_model(sim))

EarthSystemModels.update_net_fluxes!(coupled_model, sim::NestedModelSimulation) =
    EarthSystemModels.update_net_fluxes!(coupled_model, component_model(sim))

# The advective CFL belongs to the innermost prognostic model (the child).
Oceananigans.Advection.cell_advection_timescale(esm::EarthSystemModels.EarthSystemModel{<:Any, <:NestedModelSimulation}) =
    Oceananigans.Advection.cell_advection_timescale(component_model(esm.atmosphere))
