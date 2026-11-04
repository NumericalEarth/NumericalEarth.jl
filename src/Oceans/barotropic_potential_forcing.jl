using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans: UpdateStateCallsite
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: node
using Oceananigans.Simulations: Callback
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

#####
##### Pointwise potentials, evaluated once per state update
#####

# A potential that is a function of (λ, φ, t), such as Tidejinks.TidalPotential, would otherwise be evaluated inside
# the momentum tendency kernels at every level and for both cells of each derivative. It is evaluated instead into a
# two-dimensional field each time the model state is updated, and kernels read that field.
struct EvaluatedPotential{F, S}
    field :: F
    source :: S
end

# Kernels see only the field, which the AbstractArray method of barotropic_potentialᶜᶜᵃ reads.
Adapt.adapt_structure(to, Φ::EvaluatedPotential) = Adapt.adapt(to, Φ.field)

evaluate_pointwise_potentials(grid, Φ) = Φ
evaluate_pointwise_potentials(grid, Φ::Function) = EvaluatedPotential(Field{Center, Center, Nothing}(grid), Φ)
evaluate_pointwise_potentials(grid, Φ::Tuple) = map(ϕ -> evaluate_pointwise_potentials(grid, ϕ), Φ)

@kernel function _evaluate_potential!(Φ, grid, source, time)
    i, j = @index(Global, NTuple)
    @inbounds Φ[i, j, 1] = source(node(i, j, 1, grid, Center(), Center(), nothing)..., time)
end

evaluate_potentials!(model, Φ) = nothing
evaluate_potentials!(model, Φ::Tuple) = foreach(ϕ -> evaluate_potentials!(model, ϕ), Φ)

function evaluate_potentials!(model, Φ::EvaluatedPotential)
    grid = model.grid
    launch!(architecture(grid), grid, :xy, _evaluate_potential!, Φ.field, grid, Φ.source, model.clock.time)
    fill_halo_regions!(Φ.field)
    return nothing
end

has_evaluated_potential(Φ) = false
has_evaluated_potential(Φ::EvaluatedPotential) = true
has_evaluated_potential(Φ::Tuple) = any(has_evaluated_potential, Φ)

function add_evaluated_potential_callback!(simulation, Φ)
    has_evaluated_potential(Φ) || return nothing
    evaluate_potentials!(simulation.model, Φ) # for the first stage, before any state update
    simulation.callbacks[:evaluate_barotropic_potential] =
        Callback(evaluate_potentials!; parameters = Φ, callsite = UpdateStateCallsite())
    return nothing
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
