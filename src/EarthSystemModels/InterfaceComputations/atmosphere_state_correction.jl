#####
##### Corrections applied to the regridded atmosphere exchange state.
#####
##### A correction transforms the atmosphere state *after* it is brought onto the
##### exchange grid (`interpolate_state!`), so it applies identically whether the
##### atmosphere is prescribed (reanalysis) or a live coupled model. `nothing` is
##### a no-op (the default).
#####

"""
    ElevationCorrection(elevation_difference; lapse_rate = 6.5e-3,
                        gravitational_acceleration = 9.81,
                        dry_air_gas_constant = 287.052)

A moist-environmental lapse-rate correction of the near-surface atmosphere state
for the elevation difference `Δz = z_surface − z_atmosphere` between the exchange
grid and the atmosphere's effective surface elevation. Applied in place to the
regridded exchange state each step,

```math
T ← T − Γ\\,Δz, \\qquad p ← p\\,\\exp\\!\\left(\\frac{-g\\,Δz}{Rᵈ\\,\\bar{T}}\\right),
\\qquad \\bar{T} = T − Γ\\,Δz/2,
```

with specific humidity `q` conserved (adiabatic lifting conserves `q`).
`elevation_difference` is a per-cell exchange-grid `Field` (or a `Number`); it is
naturally ≈ 0 over ocean / sea-ice (sea level), so a single correction is correct
across surfaces. The `Field` must live on the exchange grid.
"""
struct ElevationCorrection{Z, FT}
    elevation_difference :: Z
    lapse_rate :: FT
    gravitational_acceleration :: FT
    dry_air_gas_constant :: FT
end

function ElevationCorrection(elevation_difference;
                             lapse_rate = 6.5e-3,
                             gravitational_acceleration = 9.81,
                             dry_air_gas_constant = 287.052)

    FT = promote_type(typeof(lapse_rate),
                      typeof(gravitational_acceleration),
                      typeof(dry_air_gas_constant))

    return ElevationCorrection(elevation_difference,
                               convert(FT, lapse_rate),
                               convert(FT, gravitational_acceleration),
                               convert(FT, dry_air_gas_constant))
end

# Per-cell elevation offset from a `Field` or a uniform `Number`.
@inline elevation_offset(Δz::Number, i, j) = Δz
@inline elevation_offset(Δz, i, j) = @inbounds Δz[i, j, 1]

# Apply the correction to the atmosphere exchange state. `nothing` and absent
# atmospheres are no-ops.
@inline correct_atmosphere_state!(::Nothing, atmosphere_exchanger, grid) = nothing
@inline correct_atmosphere_state!(correction, ::Nothing, grid) = nothing

function correct_atmosphere_state!(correction::ElevationCorrection, atmosphere_exchanger, grid)
    arch  = architecture(grid)
    state = atmosphere_exchanger.state
    launch!(arch, grid, interface_kernel_parameters(grid),
            _correct_atmosphere_elevation!,
            state.T, state.p,
            correction.elevation_difference,
            correction.lapse_rate,
            correction.gravitational_acceleration,
            correction.dry_air_gas_constant)
    return nothing
end

@kernel function _correct_atmosphere_elevation!(T, p, Δz_field, Γ, g, Rᵈ)
    i, j = @index(Global, NTuple)
    FT = eltype(T)
    @inbounds begin
        Δz = convert(FT, elevation_offset(Δz_field, i, j))
        ΔT = convert(FT, Γ) * Δz
        T₀ = T[i, j, 1]
        T̄  = T₀ - ΔT / 2 # layer-mean temperature for the hydrostatic integral
        p[i, j, 1] = p[i, j, 1] * exp(- convert(FT, g) * Δz / (convert(FT, Rᵈ) * T̄))
        T[i, j, 1] = T₀ - ΔT # lapse-rate shift; q is conserved
    end
end
