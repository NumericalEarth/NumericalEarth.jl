using Oceananigans.Operators: Δzᶜᶜᶜ
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using ClimaSeaIce.SeaIceDynamics: x_momentum_stress, y_momentum_stress,
                                  explicit_τx, explicit_τy,
                                  implicit_τx_coefficient, implicit_τy_coefficient

using ..EarthSystemModels: ocean_temperature, ocean_salinity

#####
##### How the ice-ocean mass exchange reaches the ocean
#####

"""
    ConservativeIceFreshwater()

Hand the ocean the mass the sea ice actually exchanged: the volume flux `−(Eᵢ + Eₛ)/ρᵒᶜ` and the salt
held in the ice, `Eᵢ Sˢⁱ/ρᵒᶜ`. The `Sᴺ`-weighted dilution rides on the volume flux in the salinity
boundary condition, so melting ice of salinity `Sˢⁱ` adds exactly its own volume and its own salt.
"""
struct ConservativeIceFreshwater end

"""
    ScaledIceFreshwater(fraction)

Deliver `fraction` of the exchange, volume and salt alike. The withheld water leaves the ocean-ice-snow
total, which `normalize_freshwater` returns globally by moving the free surface, so the global budget
still closes while the *local* delivery is scaled. A diagnostic knob for how sensitive a basin is to
the freshwater the ice puts into it; `fraction = 1` is [`ConservativeIceFreshwater`](@ref).
"""
struct ScaledIceFreshwater{FT}
    fraction :: FT
end

"""
    VirtualSaltFluxIceFreshwater()

Deliver the exchange as a salt flux at fixed ocean volume — the classical virtual salt flux,
`Jˢ = Jʷ (Sᴺ − Sˢⁱ)` with `Jʷ = 0` — instead of as a real volume flux. Isolates the volume pathway
from the freshwater amount. ⚠ This is an approximation, exact only for `Sᴺ` uniform over the column,
and it does not conserve total salt; the drift is measurable in the ocean + ice + snow budget.
"""
struct VirtualSaltFluxIceFreshwater end

"""
$(TYPEDSIGNATURES)

The ocean-side volume and salt fluxes `(Jʷ, Jˢ)` for an ice-ocean mass exchange of `Eᵢ` ice and `Eₛ`
snow, against ocean surface salinity `Sᴺ` and ice salinity `Sˢⁱ`.
"""
@inline ice_freshwater_and_salt(::ConservativeIceFreshwater, Eᵢ, Eₛ, Sᴺ, Sˢⁱ, ρᵒᶜ) =
    (- (Eᵢ + Eₛ) / ρᵒᶜ, Eᵢ * Sˢⁱ / ρᵒᶜ)

@inline function ice_freshwater_and_salt(delivery::ScaledIceFreshwater, Eᵢ, Eₛ, Sᴺ, Sˢⁱ, ρᵒᶜ)
    α = delivery.fraction
    return (- α * (Eᵢ + Eₛ) / ρᵒᶜ, α * Eᵢ * Sˢⁱ / ρᵒᶜ)
end

@inline function ice_freshwater_and_salt(::VirtualSaltFluxIceFreshwater, Eᵢ, Eₛ, Sᴺ, Sˢⁱ, ρᵒᶜ)
    Jʷ = - (Eᵢ + Eₛ) / ρᵒᶜ
    return (zero(Jʷ), Jʷ * (Sᴺ - Sˢⁱ))
end

"""
    compute_sea_ice_ocean_fluxes!(coupled_model)

Compute heat, salt, and momentum fluxes at the sea ice-ocean interface.

This function computes:
- Frazil heat flux: heat released when ocean temperature drops below freezing (all formulations)
- Interface heat flux: heat flux from ocean to ice, computed using the specified formulation
- Salt flux: salt exchange due to ice growth/melt
- Momentum stresses: ice-ocean momentum transfer

The interface heat flux formulation is determined by `coupled_model.interfaces.sea_ice_ocean_interface.flux_formulation`.
"""
function compute_sea_ice_ocean_fluxes!(coupled_model)
    interface = coupled_model.interfaces.sea_ice_ocean_interface
    isnothing(interface) && return nothing

    ocean = coupled_model.ocean
    sea_ice = coupled_model.sea_ice
    ocean_properties = coupled_model.interfaces.ocean_properties

    compute_sea_ice_ocean_fluxes!(interface, ocean, sea_ice, ocean_properties)

    return nothing
end

function compute_sea_ice_ocean_fluxes!(interface, ocean, sea_ice, ocean_properties)
    Δt = sea_ice.Δt
    Tᵒᶜ = ocean_temperature(ocean)
    Sᵒᶜ = ocean_salinity(ocean)
    Sⁱ = sea_ice.model.tracers.S
    ℵ = sea_ice.model.ice_concentration
    hˢⁱ = sea_ice.model.ice_thickness
    hc = sea_ice.model.ice_consolidation_thickness

    phase_transitions = sea_ice.model.phase_transitions
    liquidus = phase_transitions.liquidus
    L = phase_transitions.reference_latent_heat

    grid = sea_ice.model.grid
    clock = sea_ice.model.clock
    arch = architecture(grid)

    uˢⁱ, vˢⁱ = sea_ice.model.velocities
    dynamics = sea_ice.model.dynamics

    # Get interface data
    fluxes = interface.fluxes
    flux_formulation = interface.flux_formulation
    Tˢⁱ = interface.temperature
    Sˢⁱ = interface.salinity

    # Mass the ice/snow exchanged with the ocean during the previous sea-ice step
    mass_fluxes = sea_ice.model.mass_fluxes.thermodynamics

    if !isnothing(dynamics)
        kernel_parameters = interface_kernel_parameters(grid)
        τₛ = dynamics.external_momentum_stresses.bottom
        launch!(arch, grid, kernel_parameters, _compute_sea_ice_ocean_stress!,
                fluxes, grid, clock, hˢⁱ, ℵ, uˢⁱ, vˢⁱ, τₛ)
    else
        τₛ = nothing
    end

    launch!(arch, grid, :xy, _compute_sea_ice_ocean_fluxes!,
            flux_formulation, fluxes, Tˢⁱ, Sˢⁱ, grid, clock,
            hˢⁱ, hc, ℵ, Sⁱ, Tᵒᶜ, Sᵒᶜ, uˢⁱ, vˢⁱ, τₛ,
            liquidus, ocean_properties, L, Δt, mass_fluxes.ice, mass_fluxes.snow,
            interface.freshwater_delivery)

    return nothing
end

@kernel function _compute_sea_ice_ocean_stress!(fluxes,
                                                grid,
                                                clock,
                                                ice_thickness,
                                                ice_concentration,
                                                sea_ice_u_velocity,
                                                sea_ice_v_velocity,
                                                sea_ice_ocean_stress)
    i, j = @index(Global, NTuple)

    τˣ = fluxes.x_momentum
    τʸ = fluxes.y_momentum
    λˣ = fluxes.x_momentum_coefficient
    λʸ = fluxes.y_momentum_coefficient
    Nz = size(grid, 3)

    uˢⁱ = sea_ice_u_velocity
    vˢⁱ = sea_ice_v_velocity
    hˢⁱ = ice_thickness
    ℵ = ice_concentration
    sea_ice_fields = (; u = uˢⁱ, v = vˢⁱ, h = hˢⁱ, ℵ = ℵ)
    τₛ = sea_ice_ocean_stress

    # The drag ρₑ Cᴰ |Δu| (uᵒ - uⁱ) is split so the ocean can treat it semi-implicitly: λ is embedded
    # in the ocean's vertical solver diagonal and Fₑ enters the tendency. Since `x_momentum_stress`
    # is `explicit_τx - λ uⁱ`, subtracting `explicit_τx` leaves exactly Fₑ = -λ uⁱ.
    @inbounds begin
        λˣ[i, j, 1] = implicit_τx_coefficient(i, j, Nz, grid, τₛ, clock, sea_ice_fields)
        λʸ[i, j, 1] = implicit_τy_coefficient(i, j, Nz, grid, τₛ, clock, sea_ice_fields)
        τˣ[i, j, 1] = x_momentum_stress(i, j, Nz, grid, τₛ, clock, sea_ice_fields) -
                      explicit_τx(i, j, Nz, grid, τₛ, clock, sea_ice_fields)
        τʸ[i, j, 1] = y_momentum_stress(i, j, Nz, grid, τₛ, clock, sea_ice_fields) -
                      explicit_τy(i, j, Nz, grid, τₛ, clock, sea_ice_fields)
    end
end

@kernel function _compute_sea_ice_ocean_fluxes!(flux_formulation,
                                                fluxes,
                                                interface_temperature,
                                                interface_salinity,
                                                grid,
                                                clock,
                                                ice_thickness,
                                                ice_consolidation_thickness,
                                                ice_concentration,
                                                ice_salinity,
                                                ocean_temperature,
                                                ocean_salinity,
                                                sea_ice_u_velocity,
                                                sea_ice_v_velocity,
                                                sea_ice_ocean_stresses,
                                                liquidus,
                                                ocean_properties,
                                                latent_heat,
                                                Δt,
                                                ice_ocean_mass_flux,
                                                snow_ocean_mass_flux,
                                                freshwater_delivery)

    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    𝒬ᶠʳᶻ = fluxes.frazil_heat
    𝒬ⁱⁿ = fluxes.interface_heat
    Jˢ = fluxes.salt
    Jʷ = fluxes.freshwater
    τˣ = fluxes.x_momentum
    τʸ = fluxes.y_momentum
    T★ = interface_temperature
    S★ = interface_salinity
    Tᵒᶜ = ocean_temperature
    Sᵒᶜ = ocean_salinity
    hc = ice_consolidation_thickness
    ℰ  = latent_heat

    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity

    # =============================================
    # Part 1: Frazil ice formation (all formulations)
    # =============================================
    # When ocean temperature drops below freezing, frazil ice forms
    # and heat is released to the ice component.

    δ𝒬ᶠʳᶻ = zero(grid)

    for k = Nz:-1:1
        @inbounds begin
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            Tᵏ = Tᵒᶜ[i, j, k]
            Sᵏ = Sᵒᶜ[i, j, k]
        end

        # Melting/freezing temperature at this depth
        Tₘ = melting_temperature(liquidus, Sᵏ)
        freezing = Tᵏ < Tₘ

        # Compute change in ocean heat energy due to freezing.
        # When Tᵏ < Tₘ, we heat the ocean back to melting temperature
        # by extracting heat from the ice.
        δE = freezing * ρᵒᶜ * cᵒᶜ * (Tₘ - Tᵏ)

        # Perform temperature adjustment
        @inbounds Tᵒᶜ[i, j, k] = ifelse(freezing, Tₘ, Tᵏ)

        # Compute the heat flux from ocean into ice during frazil formation.
        # A negative value δ𝒬ᶠʳᶻ < 0 implies heat is fluxed from the ice into
        # the ocean (frazil ice formation).
        δ𝒬ᶠʳᶻ -= δE * Δz / Δt
    end

    # Store frazil heat flux
    @inbounds 𝒬ᶠʳᶻ[i, j, 1] = δ𝒬ᶠʳᶻ

    @inbounds begin
        Tᴺ  = Tᵒᶜ[i, j, Nz]
        Sᴺ  = Sᵒᶜ[i, j, Nz]
        Sˢⁱ = ice_salinity[i, j, 1]
        hˢⁱ = ice_thickness[i, j, 1]
        ℵᵢ  = ice_concentration[i, j, 1]
        hc  = ice_consolidation_thickness[i, j, 1]
    end

    # Extract internal temperature (for ConductiveFluxTEF, zero otherwise)
    Tˢⁱ = extract_internal_temperature(flux_formulation, i, j)

    # Package states
    ocean_surface_state = (; T = Tᴺ, S = Sᴺ)
    ice_state = (; S = Sˢⁱ, h = hˢⁱ, hc = hc, ℵ = ℵᵢ, T = Tˢⁱ)

    # Compute friction velocity
    u★ = get_friction_velocity(flux_formulation.friction_velocity, i, j, grid, τˣ, τʸ, ρᵒᶜ)

    # =============================================
    # Part 3: Interface heat flux (formulation-specific)
    # =============================================
    # Returns interfacial heat flux and interface T, S
    𝒬ⁱᵒ, Tᵦ, Sᵦ = compute_interface_heat_flux(flux_formulation,
                                              ocean_surface_state, ice_state,
                                              liquidus, ocean_properties, ℰ, u★)

    # Store interface values and heat flux
    @inbounds 𝒬ⁱⁿ[i, j, 1] = 𝒬ⁱᵒ
    store_interface_state!(flux_formulation, T★, S★, i, j, Tᵦ, Sᵦ)

    # =============================================
    # Part 4: Freshwater and salt fluxes
    # =============================================
    # Derived from the mass the sea-ice model actually exchanged with the ocean during its last step.
    # Jˢ carries only the salt held in the ice itself (Eᵢ Sˢⁱ); the Sᴺ-weighted dilution from the
    # freshwater volume Jʷ is applied live in the ocean salinity boundary condition.
    @inbounds begin
        Eᵢ = ice_ocean_mass_flux[i, j, 1]
        Eₛ = snow_ocean_mass_flux[i, j, 1]
        # the snow term Sˢⁿ * Eₛ drops from the salt flux since Sˢⁿ == 0
        Jʷⁱᵒ, Jˢⁱᵒ = ice_freshwater_and_salt(freshwater_delivery, Eᵢ, Eₛ, Sᴺ, Sˢⁱ, ρᵒᶜ)
        Jʷ[i, j, 1] = Jʷⁱᵒ
        Jˢ[i, j, 1] = Jˢⁱᵒ
    end
end
