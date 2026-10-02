#####
##### Air-temperature offsets applied to the regridded exchange state.
#####
##### The offset ΔT (K; negative cools) is any condition Oceananigans' `getbc` can
##### evaluate on a top boundary: a `Number`, a horizontal exchange-grid `Field` or
##### array, an exchange-grid `FieldTimeSeries`, or a function `f(λ, φ, t)`
##### (`discrete_form = true` for `f(i, j, grid, clock, model_fields)`).
#####

offset_condition(ΔT; kw...) = ΔT
offset_condition(ΔT::Function; parameters = nothing, discrete_form = false) =
    BoundaryCondition(Value(), ΔT; parameters, discrete_form).condition

regularize_offset(condition, grid) =
    regularize_boundary_condition(condition, grid, (Center(), Center(), nothing), 3, RightBoundary, ())

@inline offset_value(ΔT, i, j, grid, clock) = getbc(ΔT, i, j, grid, clock, NamedTuple())

"""
    AtmosphereTemperatureOffset(ΔT; parameters = nothing, discrete_form = false)

Post-regrid correction that adds the air-temperature offset `ΔT` (K; negative cools) to
the atmosphere exchange state at fixed relative humidity. `ΔT` takes the same forms as a
boundary condition: a `Number`, a horizontal `Field` or `FieldTimeSeries` on the exchange
grid, or a function `ΔT(λ, φ, t)` (with `parameters` / `discrete_form` as for
`BoundaryCondition`). Pass it as `exchanger_correction` to `ComponentInterfaces`.

```math
q ← q \\, \\frac{qᵛ⁺(T + ΔT, p)}{qᵛ⁺(T, p)}, \\qquad T ← T + ΔT,
```

with `qᵛ⁺` the saturation specific humidity over liquid water.
"""
struct AtmosphereTemperatureOffset{O, C, P}
    offset :: O
    clock :: C                     # `nothing` until materialized
    thermodynamics_parameters :: P # `nothing` until materialized
end

AtmosphereTemperatureOffset(ΔT; kw...) = AtmosphereTemperatureOffset(offset_condition(ΔT; kw...), nothing, nothing)

function EarthSystemModels.InterfaceComputations.materialize_correction(c::AtmosphereTemperatureOffset, grid, atmosphere)
    return AtmosphereTemperatureOffset(regularize_offset(c.offset, grid), atmosphere.clock,
                                       thermodynamics_parameters(atmosphere))
end

function EarthSystemModels.InterfaceComputations.correct_state!(c::AtmosphereTemperatureOffset, exchanger, grid)
    update_field_time_series!(c.offset, Time(c.clock.time))
    state = exchanger.state
    launch!(architecture(grid), grid, interface_kernel_parameters(grid), _offset_atmosphere_temperature!,
            state.T, state.q, state.p, grid, c.clock, c.offset, c.thermodynamics_parameters)
    return nothing
end

@kernel function _offset_atmosphere_temperature!(T, q, p, grid, clock, ΔT, ℂ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        δT = convert(eltype(T), offset_value(ΔT, i, j, grid, clock))
        qᵛ⁺₀ = saturation_specific_humidity(ℂ, T[i, j, 1], p[i, j, 1], Thermodynamics.Liquid())
        qᵛ⁺₁ = saturation_specific_humidity(ℂ, T[i, j, 1] + δT, p[i, j, 1], Thermodynamics.Liquid())
        q[i, j, 1] = q[i, j, 1] * ifelse(qᵛ⁺₀ > 0, qᵛ⁺₁ / qᵛ⁺₀, one(qᵛ⁺₀)) # guard unfilled (zero) cells
        T[i, j, 1] = T[i, j, 1] + δT
    end
end

"""
    DownwellingLongwaveOffset(ΔT; sensitivity, parameters = nothing, discrete_form = false)

Post-regrid correction that shifts the downwelling longwave of the radiation exchange state
by `sensitivity * ΔT` (`sensitivity` in W m⁻² K⁻¹; `ΔT` as for
[`AtmosphereTemperatureOffset`](@ref)), `ℐꜜˡʷ ← max(ℐꜜˡʷ + sensitivity ΔT, 0)`.
Pass it as `radiation_correction` to `ComponentInterfaces`.
"""
struct DownwellingLongwaveOffset{O, FT, C}
    offset :: O
    sensitivity :: FT
    clock :: C # `nothing` until materialized
end

DownwellingLongwaveOffset(ΔT; sensitivity, kw...) = DownwellingLongwaveOffset(offset_condition(ΔT; kw...), sensitivity, nothing)

function EarthSystemModels.InterfaceComputations.materialize_correction(c::DownwellingLongwaveOffset, grid, radiation)
    return DownwellingLongwaveOffset(regularize_offset(c.offset, grid),
                                     convert(eltype(grid), c.sensitivity), radiation.clock)
end

function EarthSystemModels.InterfaceComputations.correct_state!(c::DownwellingLongwaveOffset, exchanger, grid)
    update_field_time_series!(c.offset, Time(c.clock.time))
    launch!(architecture(grid), grid, interface_kernel_parameters(grid), _offset_downwelling_longwave!,
            exchanger.state.ℐꜜˡʷ, grid, c.clock, c.offset, c.sensitivity)
    return nothing
end

@kernel function _offset_downwelling_longwave!(ℐꜜˡʷ, grid, clock, ΔT, sensitivity)
    i, j = @index(Global, NTuple)
    @inbounds ℐꜜˡʷ[i, j, 1] = max(0, ℐꜜˡʷ[i, j, 1] + sensitivity * offset_value(ΔT, i, j, grid, clock))
end

"""
    temperature_offset_corrections(ΔT; longwave_sensitivity = 0.7 / 0.133, kw...)

Return `(; atmosphere, radiation)`: an [`AtmosphereTemperatureOffset`](@ref) and a
[`DownwellingLongwaveOffset`](@ref) sharing the offset `ΔT` (K), with `longwave_sensitivity`
in W m⁻² K⁻¹. Pass them as `exchanger_correction` and `radiation_correction`.
"""
temperature_offset_corrections(ΔT; longwave_sensitivity = 0.7 / 0.133, kw...) =
    (; atmosphere = AtmosphereTemperatureOffset(ΔT; kw...),
       radiation  = DownwellingLongwaveOffset(ΔT; sensitivity = longwave_sensitivity, kw...))
