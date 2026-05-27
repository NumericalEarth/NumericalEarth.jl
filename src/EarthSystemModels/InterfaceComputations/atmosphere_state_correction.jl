#####
##### Corrections applied to the regridded atmosphere exchange state.
#####
##### A correction transforms the atmosphere state *after* it is brought onto the
##### exchange grid (`interpolate_state!`), so it applies identically whether the
##### atmosphere is prescribed (reanalysis) or a live coupled model. `nothing` is
##### a no-op (the default). A correction is constructed grid-free by the user and
##### *materialized* onto the exchange grid when the `StateExchanger` is built.
#####

"""
    ElevationCorrection(atmosphere_elevation, surface_elevation;
                        lapse_rate = 6.5e-3, gravitational_acceleration = 9.81,
                        dry_air_gas_constant = 287.052)

A moist-environmental lapse-rate correction of the near-surface atmosphere state
for the mismatch between the elevation the atmosphere data assumes
(`atmosphere_elevation`) and the desired surface elevation (`surface_elevation`).
The elevation difference `Δz = surface_elevation − atmosphere_elevation` is
materialized on the exchange grid under the hood; the user just supplies the two
elevations (anything `set!` accepts — a `Field`, function, number, or
exchange-grid array).

Applied in place to the regridded exchange state each step,

```math
T ← T − Γ\\,Δz, \\qquad p ← p\\,\\exp\\!\\left(\\frac{-g\\,Δz}{Rᵈ\\,\\bar{T}}\\right),
\\qquad \\bar{T} = T − Γ\\,Δz/2,
```

with specific humidity `q` conserved (adiabatic lifting conserves `q`). `Δz` is
naturally ≈ 0 over ocean / sea-ice (sea level), so a single correction is correct
across surfaces, and it applies to any atmosphere (prescribed or online).
"""
struct ElevationCorrection{A, S, FT, Z}
    atmosphere_elevation :: A
    surface_elevation :: S
    lapse_rate :: FT
    gravitational_acceleration :: FT
    dry_air_gas_constant :: FT
    elevation_difference :: Z # materialized Δz on the exchange grid; `nothing` until then
end

function ElevationCorrection(atmosphere_elevation, surface_elevation;
                             lapse_rate = 6.5e-3,
                             gravitational_acceleration = 9.81,
                             dry_air_gas_constant = 287.052)

    FT = promote_type(typeof(lapse_rate),
                      typeof(gravitational_acceleration),
                      typeof(dry_air_gas_constant))

    return ElevationCorrection(atmosphere_elevation,
                               surface_elevation,
                               convert(FT, lapse_rate),
                               convert(FT, gravitational_acceleration),
                               convert(FT, dry_air_gas_constant),
                               nothing)
end

#####
##### Materialization onto the exchange grid (called by `StateExchanger`).
#####

@inline materialize_atmosphere_state_correction(::Nothing, grid) = nothing

# Fill an exchange-grid field from an elevation spec (`Field`, function, number,
# or a horizontal exchange-grid array).
@inline materialize_elevation!(field, elevation) = Oceananigans.set!(field, elevation)
@inline materialize_elevation!(field, elevation::AbstractArray) =
    (Oceananigans.interior(field, :, :, 1) .= elevation; field)

function materialize_atmosphere_state_correction(c::ElevationCorrection, grid)
    zᵃ = Field{Center, Center, Nothing}(grid)
    zˢ = Field{Center, Center, Nothing}(grid)
    materialize_elevation!(zᵃ, c.atmosphere_elevation)
    materialize_elevation!(zˢ, c.surface_elevation)

    Δz = Field{Center, Center, Nothing}(grid)
    Oceananigans.interior(Δz) .= Oceananigans.interior(zˢ) .- Oceananigans.interior(zᵃ)

    return ElevationCorrection(c.atmosphere_elevation,
                               c.surface_elevation,
                               c.lapse_rate,
                               c.gravitational_acceleration,
                               c.dry_air_gas_constant,
                               Δz)
end

#####
##### Apply the correction to the atmosphere exchange state.
#####

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

@kernel function _correct_atmosphere_elevation!(T, p, Δz, Γ, g, Rᵈ)
    i, j = @index(Global, NTuple)
    FT = eltype(T)
    @inbounds begin
        δz = convert(FT, Δz[i, j, 1])
        ΔT = convert(FT, Γ) * δz
        T₀ = T[i, j, 1]
        T̄  = T₀ - ΔT / 2 # layer-mean temperature for the hydrostatic integral
        p[i, j, 1] = p[i, j, 1] * exp(- convert(FT, g) * δz / (convert(FT, Rᵈ) * T̄))
        T[i, j, 1] = T₀ - ΔT # lapse-rate shift; q is conserved
    end
end
