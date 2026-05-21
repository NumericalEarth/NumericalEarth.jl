using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using ClimaSeaIce.SeaIceDynamics: x_momentum_stress, y_momentum_stress

using ..EarthSystemModels: ocean_temperature, ocean_salinity

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
    ocean = coupled_model.ocean
    sea_ice = coupled_model.sea_ice
    interface = coupled_model.interfaces.sea_ice_ocean_interface
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

    if !isnothing(dynamics)
        kernel_parameters = interface_kernel_parameters(grid)
        τₛ = dynamics.external_momentum_stresses.bottom
        launch!(arch, grid, kernel_parameters, _compute_sea_ice_ocean_stress!,
                fluxes, grid, clock, hˢⁱ, ℵ, uˢⁱ, vˢⁱ, τₛ)

        # Regularize vector/stress halo values before they are used to compute
        # friction velocity in the scalar ice-ocean flux kernel. The field's own
        # tripolar-appropriate zipper boundary condition determines the fold treatment.
        fill_halo_regions!((fluxes.x_momentum,
                            fluxes.y_momentum))
    else
        τₛ = nothing
    end

    # Regularize the source ocean tracer state before the coupling kernel reads
    # across fold-adjacent cells. Constant and zero fields are no-ops here.
    fill_halo_regions!((Tᵒᶜ, Sᵒᶜ))

    launch!(arch, grid, :xy, _compute_sea_ice_ocean_fluxes!,
            flux_formulation, fluxes, Tˢⁱ, Sˢⁱ, grid, clock,
            hˢⁱ, hc, ℵ, Sⁱ, Tᵒᶜ, Sᵒᶜ, uˢⁱ, vˢⁱ, τₛ,
            liquidus, ocean_properties, L, Δt)

    # Regularize scalar halo and fold values before ocean/sea-ice assemblers
    # consume these surface fields on a tripolar grid. This relies on each field's
    # own grid-appropriate zipper boundary condition (for example, UPivot on a
    # RightCenterFolded grid), rather than hard-coding a pivot choice here.
    fill_halo_regions!((fluxes.interface_heat,
                        fluxes.frazil_heat,
                        fluxes.salt))

    if flux_formulation isa ThreeEquationHeatFlux
        fill_halo_regions!((Tˢⁱ, Sˢⁱ))
    end

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
    Nz = size(grid, 3)

    uˢⁱ = sea_ice_u_velocity
    vˢⁱ = sea_ice_v_velocity
    hˢⁱ = ice_thickness
    ℵ = ice_concentration
    sea_ice_fields = (; u = uˢⁱ, v = vˢⁱ, h = hˢⁱ, ℵ = ℵ)

    # Momentum stresses
    @inbounds begin
        τˣ[i, j, 1] = x_momentum_stress(i, j, Nz, grid, sea_ice_ocean_stress, clock, sea_ice_fields)
        τʸ[i, j, 1] = y_momentum_stress(i, j, Nz, grid, sea_ice_ocean_stress, clock, sea_ice_fields)
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
                                                Δt)

    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    𝒬ᶠʳᶻ = fluxes.frazil_heat
    𝒬ⁱⁿᵗ = fluxes.interface_heat
    Jˢ = fluxes.salt
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

    # Freezing rate
    qᶠ = δ𝒬ᶠʳᶻ / ℰ

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
    # Returns interfacial heat flux, melt rate qᵐ, and interface T, S
    𝒬ⁱᵒ, qᵐ, Tᵦ, Sᵦ = compute_interface_heat_flux(flux_formulation,
                                                   ocean_surface_state, ice_state,
                                                   liquidus, ocean_properties, ℰ, u★)

    # Store interface values and heat flux
    @inbounds 𝒬ⁱⁿᵗ[i, j, 1] = 𝒬ⁱᵒ
    store_interface_state!(flux_formulation, T★, S★, i, j, Tᵦ, Sᵦ)

    # =============================================
    # Part 4: Salt flux
    # =============================================
    # Salt flux from melting/freezing:
    # - during ice melt   (qᵐ > 0), fresh meltwater dilutes the ocean
    # - during ice growth (qᶠ < 0), brine rejection adds salt to ocean
    @inbounds Jˢ[i, j, 1] = (qᵐ + qᶠ) / ρᵒᶜ * (Sᴺ - Sˢⁱ)
end
