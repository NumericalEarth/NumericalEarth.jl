#####
##### Child prognostics computed on the parent grid: the "combine-then-interpolate" state exchange
#####
#
# The child (Breeze `CompressibleDynamics`) prognostic variables — dry density `ρᵈ`, momentum densities
# `ρu`/`ρv`, potential-temperature density `ρθ`, and vapor density `ρqᵛ` — are computed from the raw
# parent (ERA5) specific state *on the parent grid* and stored as a `FieldTimeSeries` holding just the
# **two time levels bracketing the child's clock** (memory-O(1) in time). A downstream child boundary
# condition / forcing then interpolates these precomputed prognostics in space + time. Computing the
# nonlinear combines on the dense parent grid first (then interpolating) is both cheaper — once per
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
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical, AbstractInMemoryBackend, FlavorOfFTS,
                                  time_indices, interpolating_time_indices, extract_field_time_series
using Oceananigans.Units: Time
using Adapt: Adapt
import Oceananigans.OutputReaders: new_backend, update_field_time_series!
import NumericalEarth.NestedModels: exchange_state!

#####
##### A 2-level in-memory backend whose resident window is filled by the StateExchanger (not by `set!`).
##### `update_field_time_series!` is a no-op so the child's `update_model_field_time_series!` never cycles
##### it — the exchanger is the sole owner of the window (advancing it as the child clock crosses a parent
##### interval). The backend is isbits, so it survives `Adapt` to the device unchanged.
#####

struct PrognosticStateBackend <: AbstractInMemoryBackend{Int}
    start  :: Int
    length :: Int
end

Base.length(backend::PrognosticStateBackend) = backend.length
new_backend(::PrognosticStateBackend, start, length) = PrognosticStateBackend(start, length)

# No-op the auto-update: `update_model_field_time_series!` calls the `Time` form, so short-circuiting it
# keeps the child from cycling these FTS — the StateExchanger owns their window.
const PrognosticStateFTS = FieldTimeSeries{<:Any, <:Any, <:Any, <:Any, <:PrognosticStateBackend}
update_field_time_series!(::PrognosticStateFTS, ::Time) = nothing

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

# Allocate the child-prognostic `FieldTimeSeries` NamedTuple on the *parent* grid: Center-located, over
# the parent's time axis + indexing, but holding only 2 resident levels (`PrognosticStateBackend`).
function child_prognostic_field_time_series(parent_atmosphere)
    grid  = parent_atmosphere.temperature.grid
    times = parent_atmosphere.temperature.times
    build() = FieldTimeSeries{Center, Center, Center}(grid, times;
                                                      backend = PrognosticStateBackend(1, 2),
                                                      time_indexing = Cyclical())
    return (ρᵈ = build(), ρu = build(), ρv = build(), ρθ = build(), ρqᵛ = build())
end

# Fill the derived FTS's resident window (the 2 levels bracketing the child clock) with one fused
# `launch!` per level, reading the parent at the matching resident time index.
function compute_child_prognostics!(prognostic, parent_atmosphere, pˢᵗ, constants, condensates)
    grid = parent_atmosphere.temperature.grid
    arch = architecture(grid)

    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    ℒˡ  = constants.liquid.reference_latent_heat
    ℒⁱ  = constants.ice.reference_latent_heat

    for n in time_indices(prognostic.ρᵈ)   # the 2 resident bracketing indices
        launch!(arch, grid, :xyz, _compute_child_prognostics!,
                prognostic.ρᵈ[n], prognostic.ρu[n], prognostic.ρv[n], prognostic.ρθ[n], prognostic.ρqᵛ[n],
                parent_atmosphere.temperature[n], parent_atmosphere.specific_humidity[n],
                source_snapshot(condensates.qᶜˡ, n), source_snapshot(condensates.qᶜⁱ, n),
                source_snapshot(parent_atmosphere.pressure, n),   # static Field (ERA5) or FTS: both handled
                parent_atmosphere.velocities.u[n], parent_atmosphere.velocities.v[n],
                pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, ℒˡ, ℒⁱ)
    end

    for fts in prognostic
        fill_halo_regions!(fts)
    end

    return prognostic
end

#####
##### StateExchanger: owns the 2-level derived FTS and refreshes/cycles them from the parent.
#####
#
# Held by `NestedModel` (as `nested.exchanger`). `NestedModel.time_step!`/`update_state!` call
# `exchange_state!` before the child steps: it advances the parent's own FTS windows to bracket the
# child clock, and — when the child clock has crossed into a new parent interval — cycles the derived
# 2-level window forward and recomputes it. The name is direction-neutral for eventual two-way nesting.

struct StateExchanger{P, Pr, C, S, Q}
    parent       :: P    # the parent PrescribedAtmosphere (raw ERA5 state)
    prognostic   :: Pr   # NamedTuple of derived child-prognostic FTS on the parent grid (2 resident levels)
    constants    :: C
    pˢᵗ          :: S
    condensates  :: Q    # NamedTuple (qᶜˡ, qᶜⁱ); entries may be `nothing` (⇒ `ZeroField`)
end

function state_exchanger(parent_atmosphere, pˢᵗ, constants;
                         condensates = (qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ,
                                        qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ))

    prognostic = child_prognostic_field_time_series(parent_atmosphere)
    exchanger  = StateExchanger(parent_atmosphere, prognostic, constants, pˢᵗ, condensates)
    exchange_state!(exchanger, first(parent_atmosphere.temperature.times))   # fill the initial window
    return exchanger
end

# Advance the derived 2-level window (and the parent's own FTS windows) to bracket `time`, recomputing
# the derived prognostics only when the bracket moves.
function exchange_state!(ex::StateExchanger, time)
    parent = ex.parent
    p = ex.prognostic

    # Advance the parent's own (possibly limited-memory) FTS windows to bracket `time`.
    for fts in extract_field_time_series(parent)
        update_field_time_series!(fts, Time(time))
    end

    # Bracketing indices for `time` on the parent's time axis; cycle the derived window if it moved.
    _, n₁, _ = interpolating_time_indices(p.ρᵈ.time_indexing, p.ρᵈ.times, time)
    if p.ρᵈ.backend.start != n₁
        for fts in p
            fts.backend = new_backend(fts.backend, n₁, length(fts.backend))
        end
    end

    compute_child_prognostics!(p, parent, ex.pˢᵗ, ex.constants, ex.condensates)
    return nothing
end
