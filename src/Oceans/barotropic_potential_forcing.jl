using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: node
using Oceananigans.TimeSteppers: kernel_time_type

struct XDirection end
struct YDirection end

struct BarotropicPotentialForcing{D, P}
    direction :: D
    potential :: P
end

Adapt.@adapt_structure BarotropicPotentialForcing

const XDirectionBPF = BarotropicPotentialForcing{<:XDirection}
const YDirectionBPF = BarotropicPotentialForcing{<:YDirection}

@inline (bpf::XDirectionBPF)(i, j, k, grid, clock, fields) = - ∂xᶠᶜᶜ(i, j, k, grid, barotropic_potentialᶜᶜᵃ, clock, bpf.potential)
@inline (bpf::YDirectionBPF)(i, j, k, grid, clock, fields) = - ∂yᶜᶠᶜ(i, j, k, grid, barotropic_potentialᶜᶜᵃ, clock, bpf.potential)

struct TimeInterpolatedPotential{P, T}
    previous :: P
    next :: P
    times :: T
end

"""
$(TYPEDSIGNATURES)

Return a barotropic potential (m² s⁻²) that varies linearly in time from `previous`
at `times[1]` to `next` at `times[2]`, evaluated at the time of `clock`.
A coupled model fills both snapshots and `times` from the atmospheric surface
pressure before each ocean time step.
"""
function TimeInterpolatedPotential(grid, clock=Clock(grid))
    previous = Field{Center, Center, Nothing}(grid)
    next = Field{Center, Center, Nothing}(grid)
    t = convert(kernel_time_type(clock), clock.time)
    times = on_architecture(architecture(grid), [t, t])
    return TimeInterpolatedPotential(previous, next, times)
end

Adapt.@adapt_structure TimeInterpolatedPotential

@inline barotropic_potentialᶜᶜᵃ(i, j, k, grid, clock, Φ::AbstractArray) = @inbounds Φ[i, j, 1]
@inline barotropic_potentialᶜᶜᵃ(i, j, k, grid, clock, Φ::Function) = Φ(node(i, j, 1, grid, Center(), Center(), nothing)..., clock.time)
@inline barotropic_potentialᶜᶜᵃ(i, j, k, grid, clock, Φ::Tuple) = +(map(ϕ -> barotropic_potentialᶜᶜᵃ(i, j, k, grid, clock, ϕ), Φ)...)

@inline function barotropic_potentialᶜᶜᵃ(i, j, k, grid, clock, Φ::TimeInterpolatedPotential)
    t₁ = @inbounds Φ.times[1]
    t₂ = @inbounds Φ.times[2]
    χ = (clock.time - t₁) / (t₂ - t₁)
    χ = ifelse(t₂ == t₁, zero(χ), χ)
    return @inbounds (1 - χ) * Φ.previous[i, j, 1] + χ * Φ.next[i, j, 1]
end

forcing_barotropic_potential(something) = nothing
forcing_barotropic_potential(Φ::TimeInterpolatedPotential) = Φ
forcing_barotropic_potential(f::BarotropicPotentialForcing) = forcing_barotropic_potential(f.potential)
forcing_barotropic_potential(mf::MultipleForcings) = forcing_barotropic_potential(mf.forcings)

function forcing_barotropic_potential(forcings::Tuple)
    n = findfirst(f -> !isnothing(forcing_barotropic_potential(f)), forcings)
    return isnothing(n) ? nothing : forcing_barotropic_potential(forcings[n])
end

forcing_barotropic_potential(sim::Simulation) = forcing_barotropic_potential(sim.model)

function forcing_barotropic_potential(model::HydrostaticFreeSurfaceModel)
    u_potential = forcing_barotropic_potential(model.forcing.u)
    v_potential = forcing_barotropic_potential(model.forcing.v)
    @assert u_potential === v_potential
    return u_potential
end
