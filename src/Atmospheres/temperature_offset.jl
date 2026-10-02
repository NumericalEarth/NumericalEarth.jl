#####
##### Air-temperature offsets applied to the regridded exchange state.
#####
##### An offset ΔT (in K; negative cools) is added to the atmosphere temperature
##### after `interpolate_state!`, with specific humidity rescaled to keep the
##### relative humidity fixed, and the downwelling longwave is shifted by a
##### linear sensitivity times the same ΔT. Like `AltitudeCorrection`, the
##### corrections are constructed grid-free and materialized onto the exchange
##### grid by the component's `ComponentExchanger` constructor. They hold no
##### state beyond a reference to the component clock, so checkpoint restarts
##### need nothing extra.
#####

#####
##### Offset sources: what a user-supplied ΔT becomes on the exchange grid.
#####

# `f(t)`: evaluated on the host each step and handed to the kernel as a scalar.
struct TimeDependentOffset{F}
    func :: F
end

# `f(λ, φ, t)`: evaluated in the kernel at the exchange-grid cell centers.
struct SpaceTimeDependentOffset{F}
    func :: F
end

# A `FieldTimeSeries` on its own grid, interpolated in space and time onto the
# exchange grid with fractional indices computed once at materialization.
struct TimeSeriesOffset{F, I}
    time_series :: F
    fractional_indices :: I
end

"""
    materialize_offset(ΔT, exchange_grid, clock)

Turn a user-supplied air-temperature offset `ΔT` (K) into an offset source on
`exchange_grid`:

- `Number`: a spatially uniform, constant offset;
- `FieldTimeSeries`: interpolated in space and time onto the exchange grid;
- `AbstractArray` (horizontal exchange-grid array) or `Field`: a static pattern;
- callable `f(λ, φ, t)` (`λ`, `φ` in degrees, `t` the clock time in seconds):
  evaluated in the kernel at the exchange-grid cell centers;
- callable `f(t)`: a spatially uniform offset evaluated on the host each step;
- anything else `set!` accepts (e.g. a function `f(λ, φ)`): a static pattern.
"""
materialize_offset(ΔT::Number, grid, clock) = convert(eltype(grid), ΔT)

function materialize_offset(ΔT::AbstractArray, grid, clock)
    cpu_grid = on_architecture(CPU(), grid)
    cpu_offset = Field{Center, Center, Nothing}(cpu_grid)
    Oceananigans.interior(cpu_offset, :, :, 1) .= ΔT
    return static_offset_field(cpu_offset, grid)
end

materialize_offset(ΔT::AbstractField, grid, clock) = static_offset_field(ΔT, grid)

function materialize_offset(ΔT::FieldTimeSeries, grid, clock)
    arch = architecture(grid)
    FT = eltype(ΔT.grid)
    ℓx, ℓy, _ = instantiated_location(ΔT)
    fractional_indices = fractional_index_fields(grid, FT, ℓx, ℓy)

    if !(isnothing(fractional_indices.i) && isnothing(fractional_indices.j))
        launch!(arch, grid, interface_kernel_parameters(grid),
                _compute_fractional_indices!, fractional_indices, grid, ΔT.grid, ℓx, ℓy)
    end

    return TimeSeriesOffset(ΔT, fractional_indices)
end

function materialize_offset(ΔT, grid, clock)
    t = clock.time
    z = zero(eltype(grid))
    applicable(ΔT, z, z, t) && return SpaceTimeDependentOffset(ΔT)
    applicable(ΔT, t) && return TimeDependentOffset(ΔT)
    return static_offset_field(ΔT, grid)
end

# Already-materialized sources (e.g. a correction reused for a second model)
materialize_offset(source::Union{TimeDependentOffset, SpaceTimeDependentOffset}, grid, clock) = source
materialize_offset(source::TimeSeriesOffset, grid, clock) = materialize_offset(source.time_series, grid, clock)

function static_offset_field(ΔT, grid)
    offset = Field{Center, Center, Nothing}(grid)
    Oceananigans.set!(offset, ΔT)
    fill_halo_regions!(offset)
    return offset
end

#####
##### Host-side evaluation: what the kernel receives each step.
#####

struct SpaceTimeOffsetKernelArgument{F, G, T}
    func :: F
    grid :: G
    time :: T
end

Adapt.adapt_structure(to, a::SpaceTimeOffsetKernelArgument) =
    SpaceTimeOffsetKernelArgument(adapt(to, a.func), adapt(to, a.grid), a.time)

struct TimeSeriesOffsetKernelArgument{D, I, TI, B, X}
    data :: D
    fractional_indices :: I
    time_interpolator :: TI
    backend :: B
    time_indexing :: X
end

Adapt.adapt_structure(to, a::TimeSeriesOffsetKernelArgument) =
    TimeSeriesOffsetKernelArgument(adapt(to, a.data),
                                   adapt(to, a.fractional_indices),
                                   adapt(to, a.time_interpolator),
                                   adapt(to, a.backend),
                                   adapt(to, a.time_indexing))

kernel_offset(source::Number, clock, grid) = source
kernel_offset(source::Field, clock, grid) = source
kernel_offset(source::TimeDependentOffset, clock, grid) = convert(eltype(grid), source.func(clock.time))
kernel_offset(source::SpaceTimeDependentOffset, clock, grid) =
    SpaceTimeOffsetKernelArgument(source.func, grid, clock.time)

function kernel_offset(source::TimeSeriesOffset, clock, grid)
    fts = source.time_series
    t = clock.time
    update_field_time_series!(fts, Time(t))
    time_interpolator, backend, time_indexing = time_arguments(architecture(grid), fts, t)
    return TimeSeriesOffsetKernelArgument(fts.data, source.fractional_indices,
                                          time_interpolator, backend, time_indexing)
end

# A scalar offset of exactly zero needs no kernel launch.
skip_offset(δT::Number) = iszero(δT)
skip_offset(δT) = false

#####
##### Kernel-side evaluation of δT at exchange-grid column (i, j).
#####

@inline offset_value(δT::Number, i, j) = δT
@inline offset_value(δT::AbstractArray, i, j) = @inbounds δT[i, j, 1]

@inline function offset_value(δT::SpaceTimeOffsetKernelArgument, i, j)
    λ = λnode(i, j, 1, δT.grid, Center(), Center(), Center())
    φ = φnode(i, j, 1, δT.grid, Center(), Center(), Center())
    return δT.func(λ, φ, δT.time)
end

@inline function offset_value(δT::TimeSeriesOffsetKernelArgument, i, j)
    fi = get_fractional_index(i, j, δT.fractional_indices.i)
    fj = get_fractional_index(i, j, δT.fractional_indices.j)
    X = FractionalIndices(fi, fj, nothing)
    return interp_atmos_time_series(δT.data, X, δT.time_interpolator, δT.backend, δT.time_indexing)
end

# The clock the offset is evaluated against: the component's own clock, which the
# `EarthSystemModel` keeps synchronized with (and of the same type as) the model clock.
component_clock(component) = component.clock
component_clock(simulation::Simulation) = simulation.model.clock

#####
##### Atmosphere temperature offset at fixed relative humidity
#####

"""
    AtmosphereTemperatureOffset(ΔT)

A post-regrid correction that adds the air-temperature offset `ΔT` (in K;
negative values cool) to the atmosphere exchange state while keeping the
relative humidity fixed. `ΔT` may be a `Number`, a function of time `f(t)` (`t`
the clock time in seconds), a function of space and time `f(λ, φ, t)` (`λ`, `φ`
in degrees), a `FieldTimeSeries` on any grid (interpolated in space and time
onto the exchange grid), or a static pattern (a `Field`, a horizontal
exchange-grid array, or anything else `set!` accepts).

The clock (the atmosphere's) and the thermodynamics parameters `ℂ` used for the
saturation specific humidity are pulled from the atmosphere when the atmosphere
`ComponentExchanger` materializes the correction. Pass it to the model via the
`exchanger_correction` keyword of `ComponentInterfaces` (or of the model
constructors, which forward it).

Applied in place to the regridded exchange state each step,

```math
q ← q \\, \\frac{qᵛ⁺(T + δT, p)}{qᵛ⁺(T, p)}, \\qquad T ← T + δT,
```

where `δT` is the offset at the exchange-grid column and `qᵛ⁺` is the
saturation specific humidity over liquid water at pressure `p` (both
saturation values are evaluated with the original `T`, before the shift).
Pressure is unchanged.
"""
struct AtmosphereTemperatureOffset{O, C, P}
    offset :: O
    clock :: C                     # `nothing` until materialized
    thermodynamics_parameters :: P # `nothing` until materialized
end

AtmosphereTemperatureOffset(ΔT) = AtmosphereTemperatureOffset(ΔT, nothing, nothing)

function EarthSystemModels.InterfaceComputations.materialize_correction(c::AtmosphereTemperatureOffset, grid, atmosphere)
    clock = component_clock(atmosphere)
    offset = materialize_offset(c.offset, grid, clock)
    ℂ = thermodynamics_parameters(atmosphere)
    return AtmosphereTemperatureOffset(offset, clock, ℂ)
end

function EarthSystemModels.InterfaceComputations.correct_state!(correction::AtmosphereTemperatureOffset, exchanger, grid)
    δT = kernel_offset(correction.offset, correction.clock, grid)
    skip_offset(δT) && return nothing

    state = exchanger.state
    launch!(architecture(grid), grid, interface_kernel_parameters(grid),
            _offset_atmosphere_temperature!,
            state.T, state.q, state.p, δT,
            correction.thermodynamics_parameters)

    return nothing
end

@kernel function _offset_atmosphere_temperature!(T, q, p, δT_source, ℂ)
    i, j = @index(Global, NTuple)
    FT = eltype(T)
    phase = Thermodynamics.Liquid()

    @inbounds begin
        δT = convert(FT, offset_value(δT_source, i, j))
        T₀ = T[i, j, 1]
        p₀ = p[i, j, 1]

        qᵛ⁺₀ = saturation_specific_humidity(ℂ, T₀, p₀, phase)
        qᵛ⁺₁ = saturation_specific_humidity(ℂ, T₀ + δT, p₀, phase)

        # Guard against unfilled (zero) columns, where qᵛ⁺₀ = 0
        ratio = ifelse(qᵛ⁺₀ > 0, qᵛ⁺₁ / qᵛ⁺₀, one(qᵛ⁺₀))

        q[i, j, 1] = q[i, j, 1] * convert(FT, ratio)
        T[i, j, 1] = T₀ + δT
    end
end

#####
##### Downwelling longwave offset
#####

"""
    DownwellingLongwaveOffset(ΔT; sensitivity)

A post-regrid correction that shifts the downwelling longwave radiation of the
radiation exchange state by `sensitivity * ΔT`, where `ΔT` is an air-temperature
offset (K) accepting the same forms as for [`AtmosphereTemperatureOffset`](@ref)
and `sensitivity` is in W m⁻² K⁻¹. The radiation's clock is attached when the
radiation `ComponentExchanger` materializes the correction. Pass it to the model
via the `radiation_correction` keyword of `ComponentInterfaces` (or of the model
constructors, which forward it).

Applied in place to the regridded exchange state each step,

```math
ℐꜜˡʷ ← \\max(ℐꜜˡʷ + s \\, δT, 0),
```

with `s` the `sensitivity` and `δT` the offset at the exchange-grid column.
"""
struct DownwellingLongwaveOffset{O, FT, C}
    offset :: O
    sensitivity :: FT
    clock :: C # `nothing` until materialized
end

DownwellingLongwaveOffset(ΔT; sensitivity) = DownwellingLongwaveOffset(ΔT, sensitivity, nothing)

function EarthSystemModels.InterfaceComputations.materialize_correction(c::DownwellingLongwaveOffset, grid, radiation)
    clock = component_clock(radiation)
    offset = materialize_offset(c.offset, grid, clock)
    sensitivity = convert(eltype(grid), c.sensitivity)
    return DownwellingLongwaveOffset(offset, sensitivity, clock)
end

function EarthSystemModels.InterfaceComputations.correct_state!(correction::DownwellingLongwaveOffset, exchanger, grid)
    δT = kernel_offset(correction.offset, correction.clock, grid)
    skip_offset(δT) && return nothing

    launch!(architecture(grid), grid, interface_kernel_parameters(grid),
            _offset_downwelling_longwave!,
            exchanger.state.ℐꜜˡʷ, δT, correction.sensitivity)

    return nothing
end

@kernel function _offset_downwelling_longwave!(ℐꜜˡʷ, δT_source, sensitivity)
    i, j = @index(Global, NTuple)
    FT = eltype(ℐꜜˡʷ)
    @inbounds begin
        δT = convert(FT, offset_value(δT_source, i, j))
        ℐꜜˡʷ[i, j, 1] = max(0, ℐꜜˡʷ[i, j, 1] + sensitivity * δT)
    end
end

#####
##### Convenience
#####

"""
    temperature_offset_corrections(ΔT; longwave_sensitivity = 0.7 / 0.133)

Return the named tuple `(; atmosphere, radiation)` of corrections that together
apply the air-temperature offset `ΔT` (K) to prescribed forcing: `atmosphere` is
`AtmosphereTemperatureOffset(ΔT)` and `radiation` is
`DownwellingLongwaveOffset(ΔT; sensitivity = longwave_sensitivity)`, with
`longwave_sensitivity` in W m⁻² K⁻¹. Pass `atmosphere` as
`exchanger_correction` and `radiation` as `radiation_correction` to
`ComponentInterfaces` or to the model constructor.
"""
temperature_offset_corrections(ΔT; longwave_sensitivity = 0.7 / 0.133) =
    (; atmosphere = AtmosphereTemperatureOffset(ΔT),
       radiation  = DownwellingLongwaveOffset(ΔT; sensitivity = longwave_sensitivity))

"""
    linear_ramp(t; from, to, start = 0, duration)

Return `from` for `t ≤ start`, then ramp linearly to `to` over `duration`
(same units as `t`, e.g. seconds), and return `to` afterwards. Useful as a
time-dependent offset, e.g. `ΔT(t) = linear_ramp(t; from = 0, to = -2, start = 0, duration = 10 * 365days)`.
"""
@inline function linear_ramp(t; from, to, start = 0, duration)
    fraction = ifelse(t <= start, zero(t), clamp((t - start) / duration, 0, 1))
    return from + (to - from) * fraction
end

Base.summary(c::AtmosphereTemperatureOffset) = string("AtmosphereTemperatureOffset(", prettysummary(c.offset), ")")
Base.summary(c::DownwellingLongwaveOffset) = string("DownwellingLongwaveOffset(", prettysummary(c.offset),
                                                    "; sensitivity=", prettysummary(c.sensitivity), ")")
Base.show(io::IO, c::Union{AtmosphereTemperatureOffset, DownwellingLongwaveOffset}) = print(io, summary(c))
