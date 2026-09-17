#####
##### Child prognostics computed on the parent grid: the "combine-then-interpolate" state exchange
#####
#
# The child (Breeze `CompressibleDynamics`) prognostic variables — dry density `ρᵈ`, momentum densities
# `ρu`/`ρv`, potential-temperature density `ρθ`, and moisture density `ρqᵛᵉ` — are computed from the raw
# parent (ERA5) specific state *on the parent grid* and stored as a `FieldTimeSeries` holding the
# resident time window that brackets the child's clock (memory-O(1) in time for streaming parents, full
# memory when the parent is full-memory). A downstream child boundary condition / forcing then
# interpolates these precomputed prognostics in space + time. Computing the
# nonlinear combines on the dense parent grid first (then interpolating) is both cheaper — once per
# parent time level rather than per child node per RK stage — and more faithful than interpolating the
# raw fields and combining afterward.
#
# Density weighting matches Breeze's `establish_densities!`/`set!` (dry density is the prognostic):
#   ρ   = p / (Rᵐ T)             (total moist density),   Rᵐ = (1 − qᵗ) Rᵈ + qᵛ Rᵛ  (condensate loads the mixture)
#   qᵗ  = qᵛ + qˡ + qⁱ,   qˡ = qᶜˡ + qʳ (all liquid), qⁱ = qᶜⁱ + qˢ (all ice)
#   ρᵈ  = ρ (1 − qᵗ)                                    ← the prognostic (dry) density
#   ρθ  = ρᵈ · θˡⁱ,   ρu = ρᵈ · u,   ρv = ρᵈ · v         ← DRY-weighted (energy + momentum)
#                                                        θˡⁱ removes latent heat only for the condensate
#                                                        the moisture slot carries
#   ρqᵛᵉ = ρ · qᵛᵉ                                      ← TOTAL-weighted (moisture mass density)
#
# `qᵛᵉ` is the scheme-dependent moisture the child binds under `moisture_prognostic_name`: true vapor
# for `:ρqᵛ`, vapor + cloud condensate for `:ρqᵉ` (saturation adjustment).

using Oceananigans.Fields: Center, ZeroField, AbstractField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical, AbstractInMemoryBackend,
                                  time_indices, interpolating_time_indices, extract_field_time_series
using Oceananigans.Units: Time
import Oceananigans.OutputReaders: new_backend, update_field_time_series!
import NumericalEarth.NestedModels: exchange_state!, total_density, reconstruct_parent_state

#####
##### An in-memory backend whose resident window is filled by the StateExchanger (not by `set!`).
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

# Runtime BC/forcing kernels interpolate FTS values elementwise under `@inbounds`; unlike whole-field
# `fts[Time(t)]`, that path cannot ask the exchanger to move the window. Keep any off-window query on a
# resident edge slot rather than inheriting the generic cyclic mapping, which can point past the 4th
# dimension of this backend's storage on GPU.
@inline function Oceananigans.OutputReaders.memory_index(backend::PrognosticStateBackend, ::Cyclical, Nt, n)
    window = length(backend)
    raw_index = n - backend.start + 1
    wrapped_index = raw_index + Nt
    window_wraps = backend.start + window - 1 > Nt

    below_window_index = ifelse(wrapped_index <= window, wrapped_index,
                                ifelse(window_wraps, window, 1))

    return ifelse(raw_index < 1, below_window_index,
                  ifelse(raw_index > window, window, raw_index))
end

# No-op the auto-update: `update_model_field_time_series!` calls the `Time` form, so short-circuiting it
# keeps the child from cycling these FTS — the StateExchanger owns their window.
const PrognosticStateFTS = FieldTimeSeries{<:Any, <:Any, <:Any, <:Any, <:PrognosticStateBackend}
update_field_time_series!(::PrognosticStateFTS, ::Time) = nothing

@kernel function _compute_child_prognostics!(ρᵈ, ρu, ρv, ρθ, ρqᵛᵉ, θ, u, v,
                                             T, qᵛ, qᶜˡ, qʳ, qᶜⁱ, qˢ, p, uₚ, vₚ,
                                             pˢᵗ, constants, equilibrium_moisture)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Tᵢ   = T[i, j, k]
        qᵛᵢ  = qᵛ[i, j, k]
        qᶜˡᵢ = qᶜˡ[i, j, k]
        qᶜⁱᵢ = qᶜⁱ[i, j, k]
        qˡ   = qᶜˡᵢ + qʳ[i, j, k]
        qⁱ   = qᶜⁱᵢ + qˢ[i, j, k]
        pᵢ   = p[i, j, k]

        Rᵈ = dry_air_gas_constant(constants)
        Rᵛ = vapor_gas_constant(constants)

        # Density and dry density carry every species: all condensate is mass that is not dry gas.
        ρ  = air_density(Tᵢ, qᵛᵢ, qˡ, qⁱ, pᵢ, Rᵈ, Rᵛ)
        qᵗ = qᵛᵢ + qˡ + qⁱ

        # The child is handed precipitation-free air: it makes its own precipitation, at its own
        # resolution and dynamics, so the parent's `qʳ`/`qˢ` cross neither channel. The moisture slot is
        # `moisture_prognostic_name` at face value — `qᵉ = qᵗ − qʳ − qˢ` for an equilibrium scheme, true
        # vapor otherwise — and θˡⁱ gives up latent heat for exactly the condensate that slot carries, so
        # the child recovers the parent's temperature from the pair it was actually given.
        qᶜˡᶜ = ifelse(equilibrium_moisture, qᶜˡᵢ, zero(qᶜˡᵢ))
        qᶜⁱᶜ = ifelse(equilibrium_moisture, qᶜⁱᵢ, zero(qᶜⁱᵢ))
        qᵛᵉ  = qᵛᵢ + qᶜˡᶜ + qᶜⁱᶜ

        θᵢ = liquid_ice_potential_temperature(Tᵢ, qᵛᵢ, qᶜˡᶜ, qᶜⁱᶜ, pᵢ, pˢᵗ, constants)

        ρᵈ[i, j, k]   = ρ * (1 - qᵗ)
        ρθ[i, j, k]   = ρᵈ[i, j, k] * θᵢ
        ρu[i, j, k]   = ρᵈ[i, j, k] * uₚ[i, j, k]
        ρv[i, j, k]   = ρᵈ[i, j, k] * vₚ[i, j, k]
        ρqᵛᵉ[i, j, k] = ρ * qᵛᵉ

        # When performing Davies `Relaxation` to a `PrescribedAtmosphere` parent, we construct
        # `SpecificForcing`s with these intensive quantities
        #
        # Note: we can do a memory optimization here for `parent_atmosphere.velocities.*`
        θ[i, j, k] = θᵢ
        u[i, j, k] = uₚ[i, j, k]
        v[i, j, k] = vₚ[i, j, k]
    end
end

# A per-variable input accessor at time level `n`: a `FieldTimeSeries` yields its `n`-th snapshot, a
# static `AbstractField` (e.g. the pressure-level coordinate) is time-constant, and `nothing` means the
# variable is absent (a `ZeroField` — e.g. omitted cloud condensate, so `qᵗ = qᵛ`).
@inline source_snapshot(fts::FieldTimeSeries, n) = fts[n]
@inline source_snapshot(field::AbstractField, n) = field
@inline source_snapshot(::Nothing, n) = ZeroField()

# Full-memory snapshot at `Time(time)` for the parity reconstruction (`reconstruct_parent_state`): an FTS
# yields its time-interpolated snapshot over its *whole* series (not a 3-level window), a static
# `AbstractField` (the pressure-level coordinate) is time-constant, and `nothing` ⇒ a literal `0` (an
# absent condensate). Unlike `source_snapshot`'s `ZeroField` (indexed inside a kernel), this feeds
# `breeze_prognostic_state`'s AbstractOperations, where a scalar `0` composes but a grid-less `ZeroField`
# cannot.
@inline full_snapshot(fts::FieldTimeSeries, time) = fts[Time(time)]
@inline full_snapshot(field::AbstractField, time) = field
# A bare `0`, not a `ZeroField`: these compose into AbstractOperations, whose `validate_grid` needs a
# grid that a gridless `ZeroField` cannot supply. The kernel-side `source_snapshot` only ever indexes,
# so it can return one.
@inline full_snapshot(::Nothing, time) = 0

# Allocate the child-prognostic `FieldTimeSeries` NamedTuple on the *parent* grid: Center-located, over
# the parent's time axis + indexing, holding `time_indices_in_memory` resident levels (bounded to
# `[3, length(times)]`). The 3-level default is the minimal moving window that still spans a child step
# crossing a parent node: one level below the end-of-step interpolation bracket, the bracket itself, and
# one level above it.
function child_prognostic_field_time_series(parent_atmosphere; time_indices_in_memory = 3)
    grid  = parent_atmosphere.temperature.grid
    times = parent_atmosphere.temperature.times
    window = clamp(time_indices_in_memory, 3, length(times))
    build() = FieldTimeSeries{Center, Center, Center}(grid, times;
                                                      backend = PrognosticStateBackend(1, window),
                                                      time_indexing = Cyclical())
    return (ρᵈ = build(), ρu = build(), ρv = build(), ρθ = build(), ρqᵛᵉ = build(), θ = build(), u = build(), v = build())
end

# Fill the derived FTS's resident window with one fused
# `launch!` per level, reading the parent at the matching resident time index.
function compute_child_prognostics!(prognostic, parent_atmosphere, pˢᵗ, constants, condensates,
                                   moisture_name)
    grid = parent_atmosphere.temperature.grid
    arch = architecture(grid)

    equilibrium_moisture = moisture_name === :ρqᵉ   # host-side, so the kernel takes an isbits `Bool`

    for n in time_indices(prognostic.ρᵈ)
        launch!(arch, grid, :xyz, _compute_child_prognostics!,
                prognostic.ρᵈ[n], prognostic.ρu[n], prognostic.ρv[n], prognostic.ρθ[n], prognostic.ρqᵛᵉ[n], prognostic.θ[n], prognostic.u[n], prognostic.v[n],
                parent_atmosphere.temperature[n], parent_atmosphere.specific_humidity[n],
                source_snapshot(condensates.qᶜˡ, n), source_snapshot(condensates.qʳ, n),
                source_snapshot(condensates.qᶜⁱ, n), source_snapshot(condensates.qˢ, n),
                source_snapshot(parent_atmosphere.pressure, n),   # static Field (ERA5) or FTS: both handled
                parent_atmosphere.velocities.u[n], parent_atmosphere.velocities.v[n],
                pˢᵗ, constants, equilibrium_moisture)
    end

    for fts in prognostic
        fill_halo_regions!(fts)
    end

    return prognostic
end

#####
##### StateExchanger: owns the derived prognostic FTS and refreshes/cycles them from the parent.
#####
#
# Held by `NestedModel` (as `nested.exchanger`). `NestedModel.time_step!`/`update_state!` call
# `exchange_state!` before the child steps: it advances the parent's own FTS windows to bracket the
# child clock, and — when the child clock has crossed into a new parent interval — cycles the derived
# window forward and recomputes it. The name is direction-neutral for eventual two-way nesting.

struct StateExchanger{P, Pr, C, S, Q}
    parent        :: P    # the parent PrescribedAtmosphere (raw ERA5 state)
    prognostic    :: Pr   # NamedTuple of derived child-prognostic FTS on the parent grid
    constants     :: C
    pˢᵗ           :: S
    condensates   :: Q    # NamedTuple (qᶜˡ, qʳ, qᶜⁱ, qˢ); entries may be `nothing` (⇒ `ZeroField`)
    moisture_name :: Symbol  # the child's `moisture_prognostic_name`: what `ρqᵛᵉ` must hold
end

# Diagnostics on the exchanger's density-weighted prognostics at time index `n`, as lazy operations.
# Momentum/energy are dry-weighted (recover ÷ρᵈ); moisture is a partial density (recover ÷ρ). The
# total density ρ = ρᵈ + Σρqˣ sums the dry density with every moisture/condensate partial density the
# exchanger carries — currently just ρqᵛᵉ, so it stays correct when condensate densities are added.
# It omits the parent's precipitation, which the exchanger does not transfer.
total_density(ex::StateExchanger, n=1) = ex.prognostic.ρᵈ[n] + ex.prognostic.ρqᵛᵉ[n]

# Reconstruct the child prognostic state (ρ, θˡⁱ, qᵗ) at `time` from the parent's FULL-memory raw fields
# via `breeze_prognostic_state`, using the exchanger's stored constants / pˢᵗ / condensate sources. Unlike
# indexing the derived resident window, this is exact for any `time` (no residency aliasing) — the intended
# path for post-run diagnostics/animation that query arbitrary times.
function reconstruct_parent_state(ex::StateExchanger, time)
    parent = ex.parent
    return NumericalEarth.Atmospheres.breeze_prognostic_state(ex.constants, ex.pˢᵗ,
               full_snapshot(parent.temperature, time),
               full_snapshot(parent.specific_humidity, time),
               full_snapshot(ex.condensates.qᶜˡ, time) + full_snapshot(ex.condensates.qʳ, time),
               full_snapshot(ex.condensates.qᶜⁱ, time) + full_snapshot(ex.condensates.qˢ, time),
               full_snapshot(parent.pressure, time))
end

function state_exchanger(parent_atmosphere, pˢᵗ, constants;
                         condensates = (qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ,
                                        qʳ  = parent_atmosphere.microphysical_variables.qʳ,
                                        qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ,
                                        qˢ  = parent_atmosphere.microphysical_variables.qˢ),
                         time_indices_in_memory = 3,
                         moisture_name = :ρqᵛ)

    # Fill any hydrometeor a caller-supplied `condensates` omits with `nothing` (⇒ ZeroField), so the
    # 4-species contract (qᶜˡ, qʳ, qᶜⁱ, qˢ) holds regardless of how many species the source carries.
    condensates = merge((qᶜˡ = nothing, qʳ = nothing, qᶜⁱ = nothing, qˢ = nothing), condensates)

    prognostic = child_prognostic_field_time_series(parent_atmosphere; time_indices_in_memory)
    exchanger  = StateExchanger(parent_atmosphere, prognostic, constants, pˢᵗ, condensates, moisture_name)
    exchange_state!(exchanger, first(parent_atmosphere.temperature.times); force=true)   # fill the initial window
    return exchanger
end

# Advance the derived resident window (and the parent's own FTS windows) to bracket `time`, recomputing
# the derived prognostics only when the bracket moves (`force` fills it once at construction).
function exchange_state!(ex::StateExchanger, time; force=false)
    parent = ex.parent
    p = ex.prognostic

    # Position the 3-level window one level BELOW the bracket of `time` (= t + Δt from `time_step!`): the
    # step's start t can sit in the previous interval [n₁-1, n₁] while `time` sits in [n₁, n₁+1], so a
    # window spanning [n₁-1, n₁, n₁+1] keeps EVERY sub-stage query resident across a node crossing.
    # A 2-level window cannot span the crossing — its start-side query returns a stale/wrong boundary target
    # (the hourly-seam kick that tips the child at every ERA5 crossing).
    # A full window (memory ≥ the whole time axis) holds every level, so it never moves — pinned at
    # start = 1. Only a limited (streaming) window slides to bracket `time` one level below `t + Δt`.
    _, n₁, _ = interpolating_time_indices(p.ρᵈ.time_indexing, p.ρᵈ.times, time)
    N = length(p.ρᵈ.times)
    window = length(p.ρᵈ.backend)
    start = window >= N ? 1 : clamp(n₁ - 1, 1, max(1, N - window + 1))

    # Advance the parent's own (possibly limited-memory) FTS windows to bracket the child window's LOWER
    # edge `times[start]`, NOT `time` (= t + Δt). A parent bracketed on t+Δt holds a forward window from
    # n₁ that EXCLUDES the n₁-1 (= start) level the 3-level child window needs, so on a window-move recompute
    # `parent[start]` would be non-resident → garbage → a discrete NaN at the crossing. Bracketing
    # `times[start]` keeps parent levels [start, start+1, start+2] resident (needs time_indices_in_memory ≥ 3).
    for fts in extract_field_time_series(parent)
        update_field_time_series!(fts, Time(p.ρᵈ.times[start]))
    end

    moved = p.ρᵈ.backend.start != start
    if moved
        for fts in p
            fts.backend = new_backend(fts.backend, start, length(fts.backend))
        end
    end

    # The window's values are a pure function of the resident parent levels, so they change only
    # when the bracket moves; on every intra-interval child step the recompute would reproduce identical
    # values. Skip it unless the bracket moved (or this is the initial fill) to spare the hot path two
    # parent-grid kernels + halo fills per step.
    (moved || force) && compute_child_prognostics!(p, parent, ex.pˢᵗ, ex.constants, ex.condensates,
                                                   ex.moisture_name)
    return nothing
end
