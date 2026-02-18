using Oceananigans.Grids: Center
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
    return Oceananigans.zspacing(1, 1, 1, grid, Center(), Center(), Center()) / 2
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

@kernel function _interpolate_breeze_state!(state, u, v, T, qᵗ, p₀, Mp)
    i, j = @index(Global, NTuple)

    @inbounds begin
        state.u[i, j, 1]  = u[i, j, 1]
        state.v[i, j, 1]  = v[i, j, 1]
        state.T[i, j, 1]  = T[i, j, 1]
        state.q[i, j, 1]  = qᵗ[i, j, 1]
        state.p[i, j, 1]  = p₀
        state.Qs[i, j, 1] = 0
        state.Qℓ[i, j, 1] = 0
        state.Mp[i, j, 1] = Mp[i, j, 1]
    end
end

function interpolate_state!(exchanger, exchange_grid, atmosphere::BreezeAtmosphere, coupled_model)
    state = exchanger.state
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    qᵗ = atmosphere.specific_moisture

    # Surface pressure from reference state (constant for anelastic dynamics)
    p₀ = atmosphere.dynamics.reference_state.surface_pressure

    # Surface precipitation rate from microphysics (if available)
    μ = atmosphere.microphysical_fields
    Mp = hasproperty(μ, :precipitation_rate) ? μ.precipitation_rate : ZeroField()

    arch = architecture(exchange_grid)
    launch!(arch, exchange_grid, :xy,
            _interpolate_breeze_state!,
            state, u, v, T, qᵗ, p₀, Mp)

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
                                     sea_ice_reference_density = 0,
                                     sea_ice_heat_capacity = 0)

    return NumericalEarth.EarthSystemModel(atmosphere, ocean, nothing;
                                           interfaces,
                                           ocean_reference_density = ocean.density,
                                           ocean_heat_capacity = ocean.heat_capacity,
                                           sea_ice_reference_density = 0,
                                           sea_ice_heat_capacity = 0,
                                           kw...)
end

#####
##### CFL wizard support
#####

cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphere}) =
    cell_advection_timescale(model.atmosphere)
