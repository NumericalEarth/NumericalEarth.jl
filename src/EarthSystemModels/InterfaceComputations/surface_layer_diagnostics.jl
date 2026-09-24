using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

using ..EarthSystemModels: DegreesKelvin

# Per-cell scales re-derived from what the flux solve stored: the characteristic scales,
# the Monin--Obukhov length, and the resolved roughness lengths.
@inline function local_similarity_scales(i, j, interface_state)
    ℱ = local_flux_formulation(interface_state.flux_formulation, i, j)
    ℂᵃᵗ = interface_state.thermodynamics_parameters

    @inbounds begin
        Tₛ = convert_to_kelvin(interface_state.temperature_units, interface_state.temperature[i, j, 1])
        qₛ = interface_state.specific_humidity[i, j, 1]
        u★ = interface_state.fluxes.friction_velocity[i, j, 1]
        θ★ = interface_state.fluxes.temperature_scale[i, j, 1]
        q★ = interface_state.fluxes.water_vapor_scale[i, j, 1]
    end

    b★ = buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, interface_state.gravitational_acceleration)
    L★ = monin_obukhov_length(u★, b★, ℱ.von_karman_constant)

    # The admissible roughness formulations do not read the bulk velocity.
    U = zero(u★)
    ℓu₀ = roughness_length(ℱ.roughness_lengths.momentum, u★, U, ℂᵃᵗ, Tₛ)
    ℓθ₀ = roughness_length(ℱ.roughness_lengths.temperature, ℓu₀, u★, U, ℂᵃᵗ, Tₛ)
    ℓq₀ = roughness_length(ℱ.roughness_lengths.water_vapor, ℓu₀, u★, U, ℂᵃᵗ, Tₛ)

    return (; formulation = ℱ, L★, Tₛ, qₛ, u★, θ★, q★, ℓu₀, ℓθ₀, ℓq₀)
end

# The similarity profile `Π` at the geometric height `z`, undefined inside the
# roughness sublayer. `max` keeps the logarithm's argument positive there so the
# `NaN` comes from this branch rather than from a domain error.
@inline function displaced_similarity_profile(ℱ, ψ, z, ℓ₀, L★)
    zᵈ = z - ℱ.zero_plane_displacement
    Π = similarity_profile(ℱ.similarity_form, ψ, max(zᵈ, ℓ₀), ℓ₀, L★)
    return ifelse(zᵈ ≤ ℓ₀, convert(typeof(Π), NaN), Π)
end

# A zero scale leaves the profile at the interface value: inactive cells store zero
# scales alongside a zero roughness length, whose profile is infinite.
@inline function similarity_theory_state(φₛ, φ★, ϰ, Π)
    φ = φₛ + φ★ * Π / ϰ
    return ifelse(φ★ == 0, φₛ, φ)
end

# The velocity relative to the interface: the profile speed along `- ρτ / |ρτ|`, the unit
# vector of the velocity difference the flux solve saw.
@inline function surface_layer_velocity_difference(i, j, interface_state, z)
    scales = local_similarity_scales(i, j, interface_state)
    ℱ = scales.formulation
    Πu = displaced_similarity_profile(ℱ, ℱ.stability_functions.momentum, z, scales.ℓu₀, scales.L★)
    U = similarity_theory_state(zero(scales.u★), scales.u★, ℱ.von_karman_constant, Πu)

    @inbounds ρτˣ = interface_state.fluxes.x_momentum[i, j, 1]
    @inbounds ρτʸ = interface_state.fluxes.y_momentum[i, j, 1]
    ρτ = sqrt(ρτˣ^2 + ρτʸ^2)

    return ifelse(ρτ == 0, zero(U), -U * ρτˣ / ρτ),
           ifelse(ρτ == 0, zero(U), -U * ρτʸ / ρτ)
end

@inline function surface_layer_x_velocity(i, j, k, grid, interface_state, z)
    Δu, Δv = surface_layer_velocity_difference(i, j, interface_state, z)
    return ℑxᶜᵃᵃ(i, j, size(grid, 3), grid, interface_state.surface_velocities.u) + Δu
end

@inline function surface_layer_y_velocity(i, j, k, grid, interface_state, z)
    Δu, Δv = surface_layer_velocity_difference(i, j, interface_state, z)
    return ℑyᵃᶜᵃ(i, j, size(grid, 3), grid, interface_state.surface_velocities.v) + Δv
end

@inline function surface_layer_temperature(i, j, k, grid, interface_state, z)
    scales = local_similarity_scales(i, j, interface_state)
    ℱ = scales.formulation
    Πθ = displaced_similarity_profile(ℱ, ℱ.stability_functions.temperature, z, scales.ℓθ₀, scales.L★)
    θ = similarity_theory_state(scales.Tₛ, scales.θ★, ℱ.von_karman_constant, Πθ)

    # `θ★` scales a potential temperature referenced to z = 0; the diagnostic is in situ.
    @inbounds qᵃᵗ = interface_state.atmosphere_specific_humidity[i, j, 1]
    cᵖᵐ = AtmosphericThermodynamics.cp_m(interface_state.thermodynamics_parameters, qᵃᵗ)
    return θ - interface_state.gravitational_acceleration * z / cᵖᵐ
end

@inline function surface_layer_specific_humidity(i, j, k, grid, interface_state, z)
    scales = local_similarity_scales(i, j, interface_state)
    ℱ = scales.formulation
    Πq = displaced_similarity_profile(ℱ, ℱ.stability_functions.water_vapor, z, scales.ℓq₀, scales.L★)
    return similarity_theory_state(scales.qₛ, scales.q★, ℱ.von_karman_constant, Πq)
end

"""
$(SIGNATURES)

Return the named tuple `(; u, v, T, q)` of lazy `KernelFunctionOperation`s that evaluate the
Monin--Obukhov profile of `interface` at the requested heights above the interface: the wind
components at `velocity_height`, the in-situ temperature at `temperature_height`, and the
specific humidity at `specific_humidity_height`, all in meters. The heights default to the WMO
observation heights, `velocity_height = 10` and `temperature_height = specific_humidity_height
= 2`, and `interface` defaults to `model.interfaces.atmosphere_land_interface`.

Each variable is the similarity profile evaluated with the scales that `interface` stored at
the last flux solve,

```math
φ(z) = φₛ + \\frac{φ_★}{ϰ} Π(z - d, ℓ_φ, L_★) ,
```

with the wind direction taken from the stored stress and, over the ocean, the surface current
added back. The profiles are `NaN` inside the roughness sublayer `z - d ≤ ℓ_φ`, and evaluating
them at or below the zero-plane displacement `d` is an error.

The temperature is in situ: `θ_★` scales a potential temperature referenced to `z = 0`, so the
adiabatic increment `g z / cᵖᵐ` is subtracted at the geometric height `z`.

Where the flux solve enhances the bulk velocity with subgrid velocities (convective gustiness,
a mesoscale scale), that enhancement is carried by `u_★` and is not undone here, so `u` and `v`
exceed the resolved wind. `T` and `q` are unaffected: their scales are built from the resolved
increments.
"""
function surface_layer_diagnostics(model, interface = model.interfaces.atmosphere_land_interface;
                                   velocity_height = 10,
                                   temperature_height = 2,
                                   specific_humidity_height = 2)

    isnothing(interface) &&
        throw(ArgumentError("surface_layer_diagnostics requires an atmosphere interface, found nothing"))

    flux_formulation = interface.formulation.turbulent_fluxes

    flux_formulation isa SimilarityTheoryFluxes ||
        throw(ArgumentError("surface_layer_diagnostics requires SimilarityTheoryFluxes, found $(summary(flux_formulation))"))

    ℓu = flux_formulation.roughness_lengths.momentum
    ℓu isa MomentumRoughnessLength && ℓu.wave_formulation isa WindDependentWaveFormulation &&
        throw(ArgumentError("surface_layer_diagnostics cannot resolve a WindDependentWaveFormulation roughness length from the stored scales"))

    for z in (velocity_height, temperature_height, specific_humidity_height)
        validate_zero_plane_displacement(flux_formulation, z)
    end

    interfaces = model.interfaces
    over_ocean = interface === interfaces.atmosphere_ocean_interface

    temperature_units = if over_ocean
        interfaces.ocean_properties.temperature_units
    elseif interface === interfaces.atmosphere_sea_ice_interface
        interfaces.sea_ice_properties.temperature_units
    else
        DegreesKelvin()
    end

    # The stress is built from the velocity difference the solve saw, so the profile is
    # relative to the same interface velocity.
    surface_velocities = if over_ocean && interface.formulation.velocity_difference isa RelativeVelocity
        ocean_state = interfaces.exchanger.ocean.state
        (u = ocean_state.u, v = ocean_state.v)
    else
        (u = ZeroField(), v = ZeroField())
    end

    grid = interfaces.exchanger.grid
    FT = eltype(grid)

    interface_state = (fluxes = interface.fluxes,
                       temperature = interface.temperature,
                       specific_humidity = interface.specific_humidity,
                       atmosphere_specific_humidity = interfaces.exchanger.atmosphere.state.q,
                       surface_velocities = surface_velocities,
                       flux_formulation = flux_formulation,
                       temperature_units = temperature_units,
                       thermodynamics_parameters = thermodynamics_parameters(model.atmosphere),
                       gravitational_acceleration = convert(FT, interfaces.properties.gravitational_acceleration))

    SurfaceLayerOperation = KernelFunctionOperation{Center, Center, Nothing}

    return (u = SurfaceLayerOperation(surface_layer_x_velocity, grid, interface_state, convert(FT, velocity_height)),
            v = SurfaceLayerOperation(surface_layer_y_velocity, grid, interface_state, convert(FT, velocity_height)),
            T = SurfaceLayerOperation(surface_layer_temperature, grid, interface_state, convert(FT, temperature_height)),
            q = SurfaceLayerOperation(surface_layer_specific_humidity, grid, interface_state, convert(FT, specific_humidity_height)))
end
