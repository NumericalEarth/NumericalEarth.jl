#####
##### Child prognostics computed on the parent grid: the "combine-then-interpolate" state exchange
#####
#
# The child (Breeze `CompressibleDynamics`) prognostic variables — dry density `ρᵈ`, momentum densities
# `ρu`/`ρv`, potential-temperature density `ρθ`, and vapor density `ρqᵛ` — are computed from the raw
# parent (ERA5) specific state *on the parent grid* and stored as `FieldTimeSeries`. A downstream child
# boundary condition / forcing then interpolates these precomputed prognostics in space + time. Computing
# the nonlinear combines on the dense parent grid first (then interpolating) is both cheaper — once per
# parent time level rather than per child node per RK stage — and more faithful than interpolating the
# raw fields and combining afterward.
#
# Density weighting matches Breeze's `establish_densities!`/`set!` (dry density is the prognostic):
#   ρ   = p / (Rᵈ Tᵥ)            (total moist density),   Tᵥ = T (1 + (Rᵛ/Rᵈ − 1) qᵛ)
#   qᵗ  = qᵛ + qᶜˡ + qᶜⁱ
#   ρᵈ  = ρ (1 − qᵗ)                                    ← the prognostic (dry) density
#   ρθ  = ρᵈ · θˡⁱ,   ρu = ρᵈ · u,   ρv = ρᵈ · v         ← DRY-weighted (energy + momentum)
#   ρqᵛ = ρ · qᵛ                                         ← TOTAL-weighted (moisture mass density)

using Oceananigans.Fields: Center, ZeroField, AbstractField, fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical

@kernel function _compute_child_prognostics!(ρᵈ, ρu, ρv, ρθ, ρqᵛ,
                                             T, qᵛ, qᶜˡ, qᶜⁱ, p, u, v,
                                             pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, ℒˡ, ℒⁱ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Tᵢ  = T[i, j, k]
        qᵛᵢ = qᵛ[i, j, k]
        qˡ  = qᶜˡ[i, j, k]
        qⁱ  = qᶜⁱ[i, j, k]
        pᵢ  = p[i, j, k]

        ρ  = air_density(Tᵢ, qᵛᵢ, pᵢ, Rᵈ, Rᵛ)
        qᵗ = qᵛᵢ + qˡ + qⁱ
        ρd = ρ * (1 - qᵗ)
        θ  = liquid_ice_potential_temperature(Tᵢ, qˡ, qⁱ, pᵢ, pˢᵗ, Rᵈ, cᵖᵈ, ℒˡ, ℒⁱ)

        ρᵈ[i, j, k]  = ρd
        ρθ[i, j, k]  = ρd * θ
        ρu[i, j, k]  = ρd * u[i, j, k]
        ρv[i, j, k]  = ρd * v[i, j, k]
        ρqᵛ[i, j, k] = ρ * qᵛᵢ
    end
end

# A per-variable input accessor at time level `n`: a `FieldTimeSeries` yields its `n`-th snapshot, a
# static `AbstractField` (e.g. the pressure-level coordinate) is time-constant, and `nothing` means the
# variable is absent (a `ZeroField` — e.g. omitted cloud condensate, so `qᵗ = qᵛ`).
@inline source_snapshot(fts::FieldTimeSeries, n) = fts[n]
@inline source_snapshot(field::AbstractField, n) = field
@inline source_snapshot(::Nothing, n) = ZeroField()

# Allocate the child-prognostic `FieldTimeSeries` NamedTuple on the *parent* grid (Center-located,
# sharing the parent's time axis + indexing) and fill it from the raw parent state.
function child_prognostic_field_time_series(parent_atmosphere, pˢᵗ, constants;
                                            qᶜˡ_source = parent_atmosphere.microphysical_variables.qᶜˡ,
                                            qᶜⁱ_source = parent_atmosphere.microphysical_variables.qᶜⁱ)

    grid  = parent_atmosphere.temperature.grid
    times = parent_atmosphere.temperature.times
    build() = FieldTimeSeries{Center, Center, Center}(grid, times; time_indexing = Cyclical())

    prognostic = (ρᵈ = build(), ρu = build(), ρv = build(), ρθ = build(), ρqᵛ = build())
    compute_child_prognostics!(prognostic, parent_atmosphere, pˢᵗ, constants, qᶜˡ_source, qᶜⁱ_source)
    return prognostic
end

# Fill every resident time level of the derived FTS with one fused `launch!` per level.
function compute_child_prognostics!(prognostic, parent_atmosphere, pˢᵗ, constants, qᶜˡ_source, qᶜⁱ_source)
    grid = parent_atmosphere.temperature.grid
    arch = architecture(grid)

    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    ℒˡ  = constants.liquid.reference_latent_heat
    ℒⁱ  = constants.ice.reference_latent_heat

    for n in eachindex(parent_atmosphere.temperature.times)
        launch!(arch, grid, :xyz, _compute_child_prognostics!,
                prognostic.ρᵈ[n], prognostic.ρu[n], prognostic.ρv[n], prognostic.ρθ[n], prognostic.ρqᵛ[n],
                parent_atmosphere.temperature[n], parent_atmosphere.specific_humidity[n],
                source_snapshot(qᶜˡ_source, n), source_snapshot(qᶜⁱ_source, n),
                parent_atmosphere.pressure,
                parent_atmosphere.velocities.u[n], parent_atmosphere.velocities.v[n],
                pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, ℒˡ, ℒⁱ)
    end

    for fts in prognostic
        fill_halo_regions!(fts)
    end

    return prognostic
end

#####
##### StateExchanger: owns the derived child-prognostic FTS and refreshes them from the parent.
#####
#
# Held by `NestedModel` (as `nested.exchanger`). `NestedModel.time_step!`/`update_state!` call
# `exchange_state!` before the child steps, so the child's FTS-driven boundary conditions / forcings see
# current parent-derived prognostics. The name is direction-neutral for eventual two-way nesting.

struct StateExchanger{P, Pr, C, S, QL, QI}
    parent       :: P    # the parent PrescribedAtmosphere (raw ERA5 state)
    prognostic   :: Pr   # NamedTuple of derived child-prognostic FTS on the parent grid
    constants    :: C
    pˢᵗ          :: S
    qᶜˡ_source   :: QL
    qᶜⁱ_source   :: QI
end

function state_exchanger(parent_atmosphere, pˢᵗ, constants;
                         qᶜˡ_source = parent_atmosphere.microphysical_variables.qᶜˡ,
                         qᶜⁱ_source = parent_atmosphere.microphysical_variables.qᶜⁱ)

    prognostic = child_prognostic_field_time_series(parent_atmosphere, pˢᵗ, constants; qᶜˡ_source, qᶜⁱ_source)
    return StateExchanger(parent_atmosphere, prognostic, constants, pˢᵗ, qᶜˡ_source, qᶜⁱ_source)
end

# TODO: recomputes every resident level each call; a 2-level windowed cycle (recompute only when the
# clock crosses a parent interval) is the memory-O(1) optimization for long runs.
function NumericalEarth.EarthSystemModels.NestedSimulations.exchange_state!(ex::StateExchanger, time)
    compute_child_prognostics!(ex.prognostic, ex.parent, ex.pˢᵗ, ex.constants, ex.qᶜˡ_source, ex.qᶜⁱ_source)
    return nothing
end
