using Oceananigans.Grids: Center
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField, Field, compute!
using Breeze.AtmosphereModels: thermodynamic_density, dynamics_pressure,
                               specific_humidity, bottom_precipitation_flux
using Breeze.TerrainFollowingDiscretization: TerrainFollowingGrid
using GPUArraysCore: @allowscalar
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.EarthSystemModels: component_model
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters, exchange_regridder, regrid_surface_field!,
                                                              rotate_to_geographic_frame!,
                                                              surface_flux_fields, vector_component_boundary_conditions

const BreezeAtmosphere    = Breeze.AtmosphereModel
const BreezeAtmosphereSim = Simulation{<:Breeze.AtmosphereModel}

# Callers in this file work in terms of the underlying `Breeze.AtmosphereModel`,
# and Simulation-typed entry points delegate to the Model-typed methods via the
# generic `component_model` unwrap defined in `EarthSystemModels`.

#####
##### Thermodynamics parameters
#####

# This is a _hack_: the parameters should ideally be derived from Breeze.ThermodynamicConstants,
# but the ESM similarity theory expects CliMA Thermodynamics parameters.
NumericalEarth.EarthSystemModels.thermodynamics_parameters(::BreezeAtmosphere) = AtmosphereThermodynamicsParameters(Float64)
NumericalEarth.EarthSystemModels.thermodynamics_parameters(atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.thermodynamics_parameters(component_model(atmos))

#####
##### Surface layer and boundary layer height
#####

# The MOST reference height is the lowest cell-center elevation above ground, ½·Δz(i,j,1).
# On a terrain-following grid this varies column-to-column, so fill it on-device; on any
# other grid it is horizontally uniform. Built once and cached in `interfaces.properties`.
@kernel function _fill_surface_layer_height!(z₁, atmos_grid)
    i, j = @index(Global, NTuple)
    @inbounds z₁[i, j, 1] = Oceananigans.zspacing(i, j, 1, atmos_grid, Center(), Center(), Center()) / 2
end

function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::BreezeAtmosphere, exchange_grid)
    grid = Oceananigans.Grids.grid(atmosphere)
    if grid isa TerrainFollowingGrid
        # Terrain makes the AGL first-cell height vary per column → a 2-D field.
        z₁ = Field{Center, Center, Nothing}(grid)
        launch!(architecture(grid), grid, interface_kernel_parameters(grid), _fill_surface_layer_height!, z₁, grid)
        regridder = exchange_regridder(on_architecture(CPU(), grid), exchange_grid)
        isnothing(regridder) && return z₁
        exchange_grid_height = Field{Center, Center, Nothing}(exchange_grid)
        regrid_surface_field!(exchange_grid_height, regridder.from_component, z₁)
        fill_halo_regions!(exchange_grid_height)
        return exchange_grid_height
    else
        # Horizontally uniform → one scalar. Read once here (cached, not per step), so the
        # single host index into a possibly-stretched device Δz array is fine under @allowscalar.
        return @allowscalar Oceananigans.zspacing(1, 1, 1, grid, Center(), Center(), Center()) / 2
    end
end

NumericalEarth.EarthSystemModels.surface_layer_height(atmos::BreezeAtmosphereSim, exchange_grid) =
    NumericalEarth.EarthSystemModels.surface_layer_height(component_model(atmos), exchange_grid)

# Fallback boundary-layer height for the surface-flux convective gustiness, used when the
# atmosphere's turbulence closure diagnoses no z_i (e.g. closure = nothing).
const default_boundary_layer_height = 600 # m

# The boundary-layer height is the per-column z_i diagnosed by the turbulence closure when
# it provides one (e.g. Breeze's ScaleAdaptiveTKE writes it into its `zi` closure field).
NumericalEarth.EarthSystemModels.boundary_layer_height(atmosphere::BreezeAtmosphere) =
    hasproperty(atmosphere.closure_fields, :zi) ? atmosphere.closure_fields.zi :
                                                  default_boundary_layer_height

NumericalEarth.EarthSystemModels.boundary_layer_height(atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.boundary_layer_height(component_model(atmos))

#####
##### ComponentExchanger: state fields for flux computations
#####

function NumericalEarth.EarthSystemModels.surface_state(atmosphere::BreezeAtmosphere)
    grid = Oceananigans.Grids.grid(atmosphere)
    velocity_bcs = vector_component_boundary_conditions(grid, (Center(), Center(), nothing))

    # Breeze's bottom flux of every sedimenting condensate, liquid and frozen (positive down,
    # kg m⁻² s⁻¹), enters as rain; Jˢⁿ stays zero.
    state = (; u    = Field{Center, Center, Nothing}(grid; boundary_conditions = velocity_bcs),
               v    = Field{Center, Center, Nothing}(grid; boundary_conditions = velocity_bcs),
               T    = Field{Center, Center, Nothing}(grid),
               p    = Field{Center, Center, Nothing}(grid),
               q    = Field{Center, Center, Nothing}(grid),
               ℐꜜˢʷ = Field{Center, Center, Nothing}(grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(grid),
               Jʳⁿ  = bottom_precipitation_flux(atmosphere),
               Jˢⁿ  = Field{Center, Center, Nothing}(grid))

    h_bℓ = NumericalEarth.EarthSystemModels.boundary_layer_height(atmosphere)

    return h_bℓ isa AbstractField ? merge(state, (; h_bℓ)) : state
end

#####
##### Interpolate atmospheric state onto exchange grid
#####

@kernel function _interpolate_breeze_state!(state, grid, u, v, T, qᵛ, p)
    i, j = @index(Global, NTuple)

    @inbounds begin
        state.u[i, j, 1]    = ℑxᶜᵃᵃ(i, j, 1, grid, u)
        state.v[i, j, 1]    = ℑyᵃᶜᵃ(i, j, 1, grid, v)
        state.T[i, j, 1]    = T[i, j, 1]
        state.q[i, j, 1]    = qᵛ[i, j, 1]
        state.p[i, j, 1]    = p[i, j, 1]
        state.ℐꜜˢʷ[i, j, 1] = 0
        state.ℐꜜˡʷ[i, j, 1] = 0
        state.Jˢⁿ[i, j, 1]  = 0
    end
end

function NumericalEarth.EarthSystemModels.update_surface_state!(state, atmosphere::BreezeAtmosphere)
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    qᵛ = specific_humidity(atmosphere)

    # Breeze's diagnosed vapor mass fraction, not the scheme-dependent moisture prognostic;
    # `dynamics_pressure` gives the per-column pressure (a single scalar `base_pressure`
    # would bias fluxes over terrain).
    p = dynamics_pressure(atmosphere.dynamics)

    grid = Oceananigans.Grids.grid(atmosphere)
    launch!(architecture(grid), grid, interface_kernel_parameters(grid),
            _interpolate_breeze_state!,
            state, grid, u, v, T, qᵛ, p)

    compute!(state.Jʳⁿ)   # refresh the precipitation diagnostic

    return nothing
end

#####
##### Net fluxes: extract coupling flux fields from Breeze boundary conditions
#####

function NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmosphere::BreezeAtmosphere)
    # Momentum flux fields (direct FluxBoundaryCondition on ρu, ρv)
    ρu = atmosphere.momentum.ρu.boundary_conditions.bottom.condition
    ρv = atmosphere.momentum.ρv.boundary_conditions.bottom.condition

    # BulkDrag bottoms would break the extraction below and double-count the coupler's stress.
    ρu isa Oceananigans.Field || throw(ArgumentError(
        "the atmosphere's bottom momentum boundary condition is $(summary(ρu)), not a coupling " *
        "flux field, so its surface stress cannot come from the coupler. Build the atmosphere " *
        "without `bottom_drag_coefficient` (Breeze `BulkDrag`) when coupling to land or ocean."))

    ρE = thermodynamic_density(atmosphere.formulation).boundary_conditions.bottom.condition.condition

    # Moisture flux field
    ρqᵛᵉ = atmosphere.moisture_density.boundary_conditions.bottom.condition

    return (; ρu, ρv, ρE, ρqᵛᵉ, surface_fluxes = surface_flux_fields(Oceananigans.Grids.grid(atmosphere)))
end

#####
##### Assemble the partition-weighted surface fluxes into Breeze bottom BCs
#####

regrid_surface_fluxes!(::Nothing, atmosphere_fluxes, surface_fluxes, exchange_grid) = surface_fluxes

function regrid_surface_fluxes!(regridder, atmosphere_fluxes, surface_fluxes, exchange_grid)
    to_atmosphere = regridder.exchange.to_component

    rotate_to_geographic_frame!(surface_fluxes.x_momentum, surface_fluxes.y_momentum, exchange_grid)

    regrid_surface_field!(atmosphere_fluxes.sensible_heat, to_atmosphere, surface_fluxes.sensible_heat)
    regrid_surface_field!(atmosphere_fluxes.water_vapor,   to_atmosphere, surface_fluxes.water_vapor)
    regrid_surface_field!(atmosphere_fluxes.x_momentum,    to_atmosphere, surface_fluxes.x_momentum)
    regrid_surface_field!(atmosphere_fluxes.y_momentum,    to_atmosphere, surface_fluxes.y_momentum)

    fill_halo_regions!((atmosphere_fluxes.x_momentum, atmosphere_fluxes.y_momentum))

    return atmosphere_fluxes
end

@kernel function _assemble_net_atmosphere_fluxes!(net, surface_fluxes, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        net.ρu[i, j, 1] = ℑxᶠᵃᵃ(i, j, 1, grid, surface_fluxes.x_momentum)
        net.ρv[i, j, 1] = ℑyᵃᶠᵃ(i, j, 1, grid, surface_fluxes.y_momentum)
        net.ρE[i, j, 1] = surface_fluxes.sensible_heat[i, j, 1] # latent heat is handled by the moisture flux
        net.ρqᵛᵉ[i, j, 1] = surface_fluxes.water_vapor[i, j, 1]
    end
end

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::BreezeAtmosphere)
    net = coupled_model.interfaces.net_fluxes.atmosphere
    isnothing(net) && return nothing

    grid = Oceananigans.Grids.grid(atmosphere)
    exchanger = coupled_model.interfaces.exchanger
    surface_fluxes = regrid_surface_fluxes!(exchanger.atmosphere.regridder,
                                            net.surface_fluxes,
                                            coupled_model.interfaces.surface_partition.fluxes,
                                            exchanger.grid)

    launch!(architecture(grid), grid, interface_kernel_parameters(grid), _assemble_net_atmosphere_fluxes!,
            net, surface_fluxes, grid)

    return nothing
end

#####
##### CFL wizard support
#####

Oceananigans.Advection.cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphere}) =
    cell_advection_timescale(model.atmosphere)

Oceananigans.Advection.cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphereSim}) =
    cell_advection_timescale(component_model(model.atmosphere))
