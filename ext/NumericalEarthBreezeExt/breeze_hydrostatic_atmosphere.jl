#####
##### Surface exchange with Breeze's `HydrostaticDynamics`
#####
##### The grid of a hydrostatic atmosphere labels coordinate surfaces: the pressure on the z-face `k` is
##### `p = A[k] + B[k] pˢ`, and the grid's `z` is a label η, not a height. Three things follow for the coupling.
#####
##### 1. The height of the lowest level above the ground is `(Φ₁ - Φˢ) / g`, with `Φ₁` the geopotential that
#####    the dynamical core diagnoses at the lowest level (where the Exner function is the mass-weighted mean
#####    of the layer) and `Φˢ` the geopotential of the ground. It moves with the surface pressure and the
#####    potential temperature of the column, so it is diagnosed with the rest of the exchanged state.
##### 2. The prognostic variables are the velocities `u` and `v` and the potential temperature density `ρθ`,
#####    where `ρ = Δp / (g Δz)` is the mass of a layer per unit area and unit of `z`. A surface stress `τ`
#####    accelerates the lowest layer at `τ / (Δp / g)` and an energy flux `𝒬` changes its `ρθ` at
#####    `𝒬 / (cᵖᵈ Π Δz)`. The density of the air in the similarity theory stays `p / (Rᵐ T)`.
##### 3. The equations are those of dry air. The atmosphere has no boundary condition that could take a
#####    vapor flux (Breeze rejects one), and a vapor flux from the surfaces stops the model.
#####

using Oceananigans.BoundaryConditions: BoundaryCondition, Flux
using Oceananigans.Grids: AbstractGrid
using Breeze: HydrostaticDynamics

const HydrostaticAtmosphere = Breeze.AtmosphereModel{<:HydrostaticDynamics}

#####
##### Defaults of `atmosphere_model`
#####

default_microphysics(::HydrostaticDynamics) = nothing
default_momentum_advection(::HydrostaticDynamics) = Oceananigans.VectorInvariant()

# Breeze rejects a hydrostatic model that is built with radiation: see `HydrostaticCoupledRadiation`.
default_radiation(::HydrostaticDynamics) = nothing

# A resting atmosphere of uniform potential temperature, which has no pressure-gradient force over any
# terrain and whose lowest level is at a positive height.
initialize_resting_atmosphere!(model, ::HydrostaticDynamics, potential_temperature) =
    set!(model; θ = potential_temperature)

#####
##### Flux insertion: boundary conditions of the lowest layer
#####

# Bottom boundary condition of a velocity of `HydrostaticDynamics`: the surface stress (N m⁻², the upward flux
# of momentum density through the ground, at the velocity point) as a flux of velocity,
#
#     stress / ρ,        ρ = Δp / (g Δz),
#
# with `ρ` the mass of the lowest layer per unit area and unit of the grid's vertical coordinate, brought to
# the velocity point. Breeze divides the flux by the layer's extent `Δz` in that coordinate, so the layer
# accelerates at `stress / (Δp / g)`: the stress over the mass of the layer per unit area, whatever the
# coordinate labels are.
struct LowestLayerStress{S, I}
    stress :: S               # surface stress at the velocity point, filled by the coupler
    interpolate_density :: I  # ℑxᶠᵃᵃ for u, ℑyᵃᶠᵃ for v
end

Adapt.adapt_structure(to, bc::LowestLayerStress) =
    LowestLayerStress(Adapt.adapt(to, bc.stress), bc.interpolate_density)

Base.summary(::LowestLayerStress) = "LowestLayerStress"

@inline function Oceananigans.BoundaryConditions.getbc(bc::LowestLayerStress, i::Integer, j::Integer,
                                                       grid::AbstractGrid, clock, fields, dynamics_fields)
    ρ = bc.interpolate_density(i, j, 1, grid, dynamics_fields.ρ)
    return @inbounds bc.stress[i, j, 1] / ρ
end

# Bottom boundary condition of the potential temperature density of `HydrostaticDynamics`: the energy flux
# (W m⁻², upward through the ground) as a flux of `ρθ` in dry air,
#
#     energy_flux / (cᵖᵈ Π),        Π = (p / pˢᵗ)^κ,
#
# with `p` the pressure of the lowest layer, at which its Exner function is the layer mean that the
# dynamical core relates temperature and potential temperature with. An energy input `𝒬 Δt` per unit area
# then raises the enthalpy `cᵖᵈ T Δp / g` of the lowest layer by `𝒬 Δt`.
struct DryAirEnergyFlux{F, FT}
    energy_flux :: F          # energy flux into the atmosphere, filled by the coupler
    heat_capacity :: FT       # cᵖᵈ
    adiabatic_exponent :: FT  # κ = Rᵈ / cᵖᵈ
    standard_pressure :: FT   # pˢᵗ
end

Adapt.adapt_structure(to, bc::DryAirEnergyFlux) =
    DryAirEnergyFlux(Adapt.adapt(to, bc.energy_flux), bc.heat_capacity, bc.adiabatic_exponent, bc.standard_pressure)

Base.summary(::DryAirEnergyFlux) = "DryAirEnergyFlux"

@inline function Oceananigans.BoundaryConditions.getbc(bc::DryAirEnergyFlux, i::Integer, j::Integer,
                                                       grid::AbstractGrid, clock, fields, dynamics_fields)
    Π = @inbounds (dynamics_fields.p[i, j, 1] / bc.standard_pressure)^bc.adiabatic_exponent
    return @inbounds bc.energy_flux[i, j, 1] / (bc.heat_capacity * Π)
end

# TODO: Breeze's energy key `ρE` fails for HydrostaticDynamics (its conversion reads a dry density that
# these dynamics do not carry). When it works, use it and remove `DryAirEnergyFlux`.
#
# The surface stress and the energy flux of the coupler enter `u`, `v` and `ρθ` through the mass of the
# lowest layer. There is no boundary condition for the vapor flux: the model is dry. The flux divergence of
# a coupled radiative transfer model heats the air as a forcing of the energy (see `HydrostaticCoupledRadiation`).
function surface_coupling(dynamics::HydrostaticDynamics, grid, microphysics, thermodynamic_constants)
    FT = eltype(grid)
    τˣ = Field{Face, Center, Nothing}(grid)
    τʸ = Field{Center, Face, Nothing}(grid)
    𝒬  = Field{Center, Center, Nothing}(grid)

    cᵖᵈ = convert(FT, thermodynamic_constants.dry_air.heat_capacity)
    κ = convert(FT, dry_air_gas_constant(thermodynamic_constants) / thermodynamic_constants.dry_air.heat_capacity)
    pˢᵗ = convert(FT, dynamics.standard_pressure)

    bottom(condition) = FieldBoundaryConditions(bottom = BoundaryCondition(Flux(), condition))

    boundary_conditions = (u  = bottom(LowestLayerStress(τˣ, ℑxᶠᵃᵃ)),
                           v  = bottom(LowestLayerStress(τʸ, ℑyᵃᶠᵃ)),
                           ρθ = bottom(DryAirEnergyFlux(𝒬, cᵖᵈ, κ, pˢᵗ)))

    radiative_heating = Oceananigans.CenterField(grid)

    return (; boundary_conditions, forcing = (; ρE = radiative_heating), radiative_heating)
end

#####
##### Radiation
#####

# The radiation of a hydrostatic Breeze atmosphere in a coupled model, in the role `CoupledRadiation` has for
# the other dynamics. `HydrostaticDynamics` has no radiative term in its tendency and refuses to be built
# with radiation, so the heating reaches the air as a forcing of the energy instead: `radiative_heating` is
# the field that forces `ρE` (W per m² and unit of the grid's vertical coordinate), which Breeze converts to
# a tendency of `ρθ` with `cᵖᵈ Π`. Each time Breeze updates the radiation, the radiative transfer model is
# solved on its schedule and its flux divergence, which is a difference of fluxes over the same coordinate
# extent, is copied into `radiative_heating`.
#
# `atmosphere_model` attaches the skeleton `HydrostaticCoupledRadiation(radiative_heating, nothing)` after
# Breeze has built the model, which is after Breeze's check that a hydrostatic model has no radiation, and
# `materialize_earth_system_radiation!` gives it the radiative transfer model of the coupled model. Without
# one, the heating stays zero.
#
# TODO: when Breeze's HydrostaticDynamics adds the radiative flux divergence to its tendency and accepts
# radiation, use `CoupledRadiation` and remove this type, the forcing and `attach_coupled_radiation`.
struct HydrostaticCoupledRadiation{F, R}
    radiative_heating        :: F  # forces ρE of the atmosphere
    radiative_transfer_model :: R  # nothing in the skeleton
end

Base.summary(::HydrostaticCoupledRadiation{<:Any, Nothing}) = "HydrostaticCoupledRadiation (skeleton)"
Base.summary(::HydrostaticCoupledRadiation) = "HydrostaticCoupledRadiation (materialized)"

attach_coupled_radiation(model::HydrostaticAtmosphere, coupling) =
    @set model.radiation = HydrostaticCoupledRadiation(coupling.radiative_heating, nothing)

Breeze.AtmosphereModels.update_radiation!(::HydrostaticCoupledRadiation{<:Any, Nothing}, model) = nothing

function Breeze.AtmosphereModels.update_radiation!(radiation::HydrostaticCoupledRadiation, model)
    radiative_transfer_model = radiation.radiative_transfer_model
    Breeze.AtmosphereModels.update_radiation!(radiative_transfer_model, model)
    parent(radiation.radiative_heating) .= parent(radiative_transfer_model.flux_divergence)
    return nothing
end

# The heating is diagnosed from the radiative transfer model, which is checkpointed with the coupled model.
Oceananigans.prognostic_state(::HydrostaticCoupledRadiation) = nothing

NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:HydrostaticAtmosphere}, ::Nothing) = atmosphere

function NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(
        atmosphere               :: Simulation{<:HydrostaticAtmosphere},
        radiative_transfer_model :: Breeze.RadiativeTransferModel)
    radiative_heating = atmosphere.model.radiation.radiative_heating
    return @set atmosphere.model.radiation = HydrostaticCoupledRadiation(radiative_heating, radiative_transfer_model)
end

#####
##### Surface-layer height
#####

@kernel function _compute_geopotential_height_above_ground!(z₁, Φ, Φˢ, g)
    i, j = @index(Global, NTuple)
    @inbounds z₁[i, j, 1] = (Φ[i, j, 1] - Φˢ[i, j, 1]) / g
end

"""
$(TYPEDSIGNATURES)

Diagnose `height`, the height above the ground (m) of the lowest level of the hydrostatic `atmosphere`,
`(Φ₁ - Φˢ) / g`, from the geopotential `Φ` of the dynamical core's last `update_state!` and the geopotential
`Φˢ` of the ground. Throw a `DomainError` unless it is positive in every column.
"""
function compute_surface_layer_height!(height, atmosphere::HydrostaticAtmosphere)
    grid = height.grid
    dynamics = atmosphere.dynamics
    g = atmosphere.thermodynamic_constants.gravitational_acceleration

    launch!(architecture(grid), grid, interface_kernel_parameters(grid),
            _compute_geopotential_height_above_ground!,
            height, dynamics.geopotential, dynamics.surface_geopotential, g)

    lowest = minimum(height)
    lowest > 0 ||
        throw(DomainError(lowest, "the lowest level of the hydrostatic atmosphere is not above the ground: " *
                                  "its height (Φ₁ - Φˢ) / g must be positive in every column. Set the " *
                                  "potential temperature of the atmosphere before coupling it."))

    return nothing
end

# A field on the exchange grid, diagnosed here and again each time the state is exchanged.
function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::HydrostaticAtmosphere, exchange_grid)
    height = Field{Center, Center, Nothing}(exchange_grid)
    compute_surface_layer_height!(height, atmosphere)
    return height
end

update_surface_layer_height!(coupled_model, atmosphere::HydrostaticAtmosphere) =
    compute_surface_layer_height!(coupled_model.interfaces.properties.surface_layer_height, atmosphere)

#####
##### Net fluxes
#####

function NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmosphere::HydrostaticAtmosphere)
    u_condition = atmosphere.momentum.u.boundary_conditions.bottom.condition
    v_condition = atmosphere.momentum.v.boundary_conditions.bottom.condition
    ρθ_condition = thermodynamic_density(atmosphere.formulation).boundary_conditions.bottom.condition

    (u_condition isa LowestLayerStress && v_condition isa LowestLayerStress && ρθ_condition isa DryAirEnergyFlux) ||
        throw(ArgumentError("the bottom boundary conditions of u, v and ρθ of the hydrostatic atmosphere are " *
                            "$(summary(u_condition)), $(summary(v_condition)) and $(summary(ρθ_condition)), not " *
                            "the coupling conditions that `atmosphere_model` builds, so its surface fluxes " *
                            "cannot come from the coupler."))

    # The vapor flux that the surfaces computed. Nothing in the atmosphere takes it: `update_net_fluxes!`
    # stops the model unless it is zero.
    ρqᵛᵉ = Field{Center, Center, Nothing}(atmosphere.grid)

    return (; ρu = u_condition.stress, ρv = v_condition.stress, ρE = ρθ_condition.energy_flux, ρqᵛᵉ)
end

# The dry guard of the coupling. Breeze stops a hydrostatic model whose moisture density is not zero and
# rejects every source of it, so water that left a surface would go nowhere, while the surface lost its mass
# and its latent heat.
function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::HydrostaticAtmosphere)
    assemble_net_atmosphere_fluxes!(coupled_model, atmosphere)

    vapor_flux = coupled_model.interfaces.net_fluxes.atmosphere.ρqᵛᵉ
    lowest, highest = minimum(vapor_flux), maximum(vapor_flux)

    (iszero(lowest) && iszero(highest)) ||
        throw(ArgumentError("the surfaces exchange water vapor with the atmosphere (net flux between $lowest and " *
                            "$highest kg m⁻² s⁻¹), which Breeze's HydrostaticDynamics cannot take: its equations " *
                            "are those of dry air. Build the coupled model with surfaces that do not evaporate, " *
                            "for example `ImpureSaturationSpecificHumidity(phase, 0)` over the ocean and the sea " *
                            "ice and `DryLand()` hydrology."))

    return nothing
end
