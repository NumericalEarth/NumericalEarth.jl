using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.Oceans: SlabOcean
using NumericalEarth.EarthSystemModels.InterfaceComputations: DegreesKelvin

const BreezeAtmosphere = Breeze.AtmosphereModel

#####
##### Thermodynamics parameters
#####

# This is a _hack_: the parameters should ideally be derived from Breeze.ThermodynamicConstants,
# but the ESM similarity theory expects CliMA Thermodynamics parameters.
thermodynamics_parameters(::BreezeAtmosphere) = AtmosphereThermodynamicsParameters(Float64)

#####
##### Surface layer and boundary layer height
#####

# Height of the lowest atmospheric cell center (the "surface layer")
function surface_layer_height(atmosphere::BreezeAtmosphere)
    grid = atmosphere.grid
    return grid.Lz / grid.Nz / 2
end

boundary_layer_height(::BreezeAtmosphere) = 600

#####
##### ComponentExchanger: state fields for flux computations
#####

function ComponentExchanger(atmosphere::BreezeAtmosphere, exchange_grid)
    state = (; u  = Oceananigans.CenterField(exchange_grid),
               v  = Oceananigans.CenterField(exchange_grid),
               T  = Oceananigans.CenterField(exchange_grid),
               p  = Oceananigans.CenterField(exchange_grid),
               q  = Oceananigans.CenterField(exchange_grid),
               Qs = Oceananigans.CenterField(exchange_grid),
               Qℓ = Oceananigans.CenterField(exchange_grid),
               Mp = Oceananigans.CenterField(exchange_grid))

    return ComponentExchanger(state, nothing)
end

#####
##### Interpolate atmospheric state onto exchange grid
#####

function interpolate_state!(exchanger, exchange_grid, atmosphere::BreezeAtmosphere, coupled_model)
    state = exchanger.state
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    qᵗ = atmosphere.specific_moisture

    # Extract surface layer (k=1) values and put on exchange grid
    # Note: u is at Face locations in x, so this introduces a half-cell shift.
    # For flux computations, this approximation is acceptable.
    Oceananigans.interior(state.u, :, :, 1) .= @views Oceananigans.interior(u, :, :, 1)
    Oceananigans.interior(state.v, :, :, 1) .= @views Oceananigans.interior(v, :, :, 1)
    Oceananigans.interior(state.T, :, :, 1) .= @views Oceananigans.interior(T, :, :, 1)
    Oceananigans.interior(state.q, :, :, 1) .= @views Oceananigans.interior(qᵗ, :, :, 1)

    # Surface pressure from reference state (constant for anelastic dynamics)
    p₀ = atmosphere.dynamics.reference_state.surface_pressure
    Oceananigans.set!(state.p, p₀)

    # Radiation and precipitation: not computed by Breeze LES
    Oceananigans.set!(state.Qs, 0)
    Oceananigans.set!(state.Qℓ, 0)
    Oceananigans.set!(state.Mp, 0)

    return nothing
end

#####
##### Net fluxes: Breeze atmosphere handles its own BCs
#####

net_fluxes(::BreezeAtmosphere) = nothing
update_net_fluxes!(coupled_model, ::BreezeAtmosphere) = nothing

#####
##### AtmosphereOceanModel constructor for Breeze
#####

"""
    AtmosphereOceanModel(atmosphere::Breeze.AtmosphereModel, ocean::SlabOcean; kw...)

Construct an `EarthSystemModel` coupling a Breeze `AtmosphereModel` with a `SlabOcean`.

The exchange grid is set to the slab ocean's grid, and the ocean temperature units
are set to Kelvin (consistent with Breeze's temperature convention).

The slab ocean SST field should be the same field used in the atmosphere's boundary
conditions (if any), so that updates to SST are automatically seen by the atmosphere
on the next time step.
"""
function AtmosphereOceanModel(atmosphere::BreezeAtmosphere, ocean::SlabOcean; kw...)
    exchange_grid = ocean.grid

    interfaces = ComponentInterfaces(atmosphere, ocean, nothing;
                                     exchange_grid,
                                     ocean_temperature_units = DegreesKelvin(),
                                     ocean_reference_density = ocean.density,
                                     ocean_heat_capacity = ocean.heat_capacity,
                                     sea_ice_reference_density = 0.0,
                                     sea_ice_heat_capacity = 0.0)

    return NumericalEarth.EarthSystemModel(atmosphere, ocean, nothing;
                                           interfaces,
                                           ocean_reference_density = ocean.density,
                                           ocean_heat_capacity = ocean.heat_capacity,
                                           sea_ice_reference_density = 0.0,
                                           sea_ice_heat_capacity = 0.0,
                                           kw...)
end

#####
##### CFL wizard support
#####

cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphere}) =
    cell_advection_timescale(model.atmosphere)
