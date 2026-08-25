#####
##### Shared pieces of the atmosphere-ocean, atmosphere-sea-ice and atmosphere-land flux kernels
#####

"""
$(TYPEDSIGNATURES)

Atmospheric state at cell `(i, j)`: the surface-layer velocities, temperature, pressure and specific
humidity, with the elevation `z` at which they are taken and the boundary-layer height `h_bℓ`.
"""
@inline function local_atmosphere_state(i, j, atmosphere_state, atmosphere_properties)
    @inbounds begin
        uᵃᵗ = atmosphere_state.u[i, j, 1]
        vᵃᵗ = atmosphere_state.v[i, j, 1]
        Tᵃᵗ = atmosphere_state.T[i, j, 1]
        pᵃᵗ = atmosphere_state.p[i, j, 1]
        qᵃᵗ = atmosphere_state.q[i, j, 1]
    end

    return (z = state2dindex(atmosphere_properties.surface_layer_height, i, j),
            u = uᵃᵗ,
            v = vᵃᵗ,
            T = Tᵃᵗ,
            p = pᵃᵗ,
            q = qᵃᵗ,
            h_bℓ = state2dindex(atmosphere_state.h_bℓ, i, j))
end

"""
$(TYPEDSIGNATURES)

Write the turbulent fluxes carried by the converged interface state `Ψₛ` into `interface_fluxes` at cell
`(i, j)`, and the interface temperature `Tₛ` into `interface_temperature`. `ℒ` is the latent heat of the
phase change at this interface — vaporization over water and land, sublimation over ice.

The sign convention is `+` for cooling of the interface and `-` for heating.
"""
@inline function store_interface_fluxes!(interface_fluxes, interface_temperature, i, j,
                                         Ψₛ, Ψₐ, ℂᵃᵗ, ℒ, Tₛ, interface_properties)

    u★ = Ψₛ.fluxes.u★
    θ★ = Ψₛ.fluxes.θ★
    q★ = Ψₛ.fluxes.q★

    Δu, Δv = velocity_difference(interface_properties.velocity_formulation, Ψₐ, Ψₛ)
    ΔU = sqrt(Δu^2 + Δv^2)

    τˣ = ifelse(ΔU == 0, zero(ΔU), - u★^2 * Δu / ΔU)
    τʸ = ifelse(ΔU == 0, zero(ΔU), - u★^2 * Δv / ΔU)

    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, Ψₐ.q) # moist heat capacity

    @inbounds begin
        interface_fluxes.latent_heat[i, j, 1]   = - ρᵃᵗ * ℒ * u★ * q★
        interface_fluxes.sensible_heat[i, j, 1] = - ρᵃᵗ * cᵖᵐ * u★ * θ★
        interface_fluxes.water_vapor[i, j, 1]   = - ρᵃᵗ * u★ * q★
        interface_fluxes.x_momentum[i, j, 1]    = + ρᵃᵗ * τˣ
        interface_fluxes.y_momentum[i, j, 1]    = + ρᵃᵗ * τʸ
        interface_temperature[i, j, 1]          = Tₛ
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Write the characteristic scales of the converged interface state `Ψₛ` into the diagnostic slots of
`interface_fluxes`.
"""
@inline function store_interface_scales!(interface_fluxes, i, j, Ψₛ)
    @inbounds begin
        interface_fluxes.friction_velocity[i, j, 1] = Ψₛ.fluxes.u★
        interface_fluxes.temperature_scale[i, j, 1] = Ψₛ.fluxes.θ★
        interface_fluxes.water_vapor_scale[i, j, 1] = Ψₛ.fluxes.q★
    end

    return nothing
end
