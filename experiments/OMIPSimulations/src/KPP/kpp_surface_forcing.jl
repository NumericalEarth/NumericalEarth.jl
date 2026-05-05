# Surface-forcing primitives for KPP: friction velocity, non-solar buoyancy
# flux Bo, two-band SW penetration, and the helper that integrates the total
# buoyancy forcing absorbed above a depth d.
#
# Sign convention (KPP / Large 1994): positive Bo = stabilizing. Oceananigans'
# `top_buoyancy_flux` uses positive = destabilizing, so we negate it here.

"""
    KPPTopBoundaryConditions{V, T}

Carrier for the velocity and tracer surface BCs that the KPP closure needs
at compute time. Built CPU-side once per `compute_closure_fields!` call,
then passed to the kernel.
"""
struct KPPTopBoundaryConditions{V, T}
    velocities :: V
    tracers    :: T
end

Adapt.adapt_structure(to, b::KPPTopBoundaryConditions) =
    KPPTopBoundaryConditions(adapt(to, b.velocities), adapt(to, b.tracers))

#####
##### Friction velocity
#####

@inline function friction_velocity(i, j, grid, clock, fields, top_velocity_bcs, params)
    τx = getbc(top_velocity_bcs.u, i, j, grid, clock, fields)
    τy = getbc(top_velocity_bcs.v, i, j, grid, clock, fields)
    return max(sqrt(sqrt(τx^2 + τy^2)), params.minimum_friction_velocity)
end

#####
##### Non-solar surface buoyancy flux Bo (KPP-sign: stabilizing positive)
#####

@inline non_solar_buoyancy(i, j, grid, clock, fields, buoyancy, top_tracer_bcs) =
    - top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, fields)

#####
##### Two-band SW penetration: fraction remaining at depth d, and absorbed
##### buoyancy gain integrated from surface to d.
#####

@inline shortwave_fraction(d, ::Nothing) = zero(d)

@inline function shortwave_fraction(d, radiation)
    FT = typeof(d)
    ϵ₁ = radiation.first_color_fraction
    κ₁ = radiation.first_absorption_coefficient
    κ₂ = radiation.second_absorption_coefficient
    return ϵ₁ * exp(- κ₁ * d) + (one(FT) - ϵ₁) * exp(- κ₂ * d)
end

@inline solar_buoyancy_above(i, j, d, ::Nothing, α, g) = zero(d)

@inline function solar_buoyancy_above(i, j, d, radiation, α, g)
    FT = typeof(d)
    J₀ = @inbounds radiation.surface_flux[i, j, 1]
    return - g * α * J₀ * (one(FT) - shortwave_fraction(d, radiation))
end

# Total KPP buoyancy forcing absorbed above depth d.
@inline buoyancy_forcing_above(i, j, d, Bo, radiation, α, g) =
    Bo + solar_buoyancy_above(i, j, d, radiation, α, g)

#####
##### Surface-cell EOS coefficients
#####

@inline αᶜᶜᶜ(i, j, grid, buoyancy, tracers) =
    thermal_expansionᶜᶜᶜ(i, j, grid.Nz, grid, buoyancy.equation_of_state, tracers.T, tracers.S)

@inline βᶜᶜᶜ(i, j, grid, buoyancy, tracers) =
    haline_contractionᶜᶜᶜ(i, j, grid.Nz, grid, buoyancy.equation_of_state, tracers.T, tracers.S)
