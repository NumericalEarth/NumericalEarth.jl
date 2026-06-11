#####
##### Corrections applied to the regridded atmosphere exchange state.
#####
##### A correction transforms the atmosphere state *after* it is brought onto the
##### exchange grid (`interpolate_state!`), so it applies identically whether the
##### atmosphere is prescribed (reanalysis) or a live coupled model. `nothing` is
##### a no-op (the default). A correction is constructed grid-free by the user and
##### *materialized* onto the exchange grid when its `ComponentExchanger` is built.
#####

"""
    ElevationCorrection(surface_elevation, atmosphere_elevation; lapse_rate = 6.5e-3)

A moist-environmental lapse-rate correction of the near-surface atmosphere state
for the mismatch between the desired surface elevation (`surface_elevation`, the
first argument) and the elevation the atmosphere data assumes
(`atmosphere_elevation`). The elevation difference
`Δz = surface_elevation − atmosphere_elevation` is materialized on the exchange
grid under the hood; the user just supplies the two elevations (anything `set!`
accepts — a `Field`, function, number, or exchange-grid array).

The gravitational acceleration and dry-air gas constant used in the hydrostatic
pressure adjustment are *not* supplied here — they are pulled from the
atmosphere's thermodynamics via [`thermodynamic_constants`](@ref) when the
atmosphere `ComponentExchanger` materializes the correction, so they aren't
duplicated.

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

function ElevationCorrection(surface_elevation, atmosphere_elevation; lapse_rate = 6.5e-3)
    FT = Oceananigans.defaults.FloatType

    # g and Rᵈ are placeholders here; they're filled from the atmosphere's
    # thermodynamics when the correction is materialized on the exchange grid.
    return ElevationCorrection(atmosphere_elevation,
                               surface_elevation,
                               convert(FT, lapse_rate),
                               zero(FT),
                               zero(FT),
                               nothing)
end

"""
    thermodynamic_constants(atmosphere)

Return the physical constants the atmosphere-state corrections need — the
gravitational acceleration and the dry-air gas constant `Rᵈ` (the latter from the
atmosphere's own thermodynamics) — so corrections like [`ElevationCorrection`](@ref)
don't hard-code or duplicate them.
"""
function thermodynamic_constants(atmosphere)
    ℂ  = thermodynamics_parameters(atmosphere)
    Rᵈ = AtmosphericThermodynamics.Parameters.R_d(ℂ)
    return (; gravitational_acceleration = default_gravitational_acceleration,
              dry_air_gas_constant = Rᵈ)
end

#####
##### Materialization onto the exchange grid (called by per-component
##### `ComponentExchanger` constructors).
#####

@inline materialize_correction(::Nothing, grid, component) = nothing

# Fill an exchange-grid field from an elevation spec (`Field`, function, number,
# or a horizontal exchange-grid array).
@inline materialize_elevation!(field, elevation) = Oceananigans.set!(field, elevation)
@inline materialize_elevation!(field, elevation::AbstractArray) =
    (Oceananigans.interior(field, :, :, 1) .= elevation; field)

function materialize_correction(c::ElevationCorrection, grid, atmosphere)
    zᵃ = Field{Center, Center, Nothing}(grid)
    zˢ = Field{Center, Center, Nothing}(grid)
    materialize_elevation!(zᵃ, c.atmosphere_elevation)
    materialize_elevation!(zˢ, c.surface_elevation)

    Δz = Field{Center, Center, Nothing}(grid)
    Oceananigans.interior(Δz) .= Oceananigans.interior(zˢ) .- Oceananigans.interior(zᵃ)

    FT = eltype(grid)
    constants = thermodynamic_constants(atmosphere)

    return ElevationCorrection(c.atmosphere_elevation,
                               c.surface_elevation,
                               c.lapse_rate,
                               convert(FT, constants.gravitational_acceleration),
                               convert(FT, constants.dry_air_gas_constant),
                               Δz)
end

#####
##### Apply the per-component correction to its exchange state.
#####

# Generic dispatcher: read the correction from the component's exchanger and
# apply it. No-op when the component is absent or carries no correction.
@inline correct_state!(::Nothing, grid) = nothing
@inline correct_state!(exchanger::ComponentExchanger, grid) =
    correct_state!(exchanger.correction, exchanger, grid)

# Per-correction-type kernels.
@inline correct_state!(::Nothing, exchanger, grid) = nothing

function correct_state!(correction::ElevationCorrection, exchanger, grid)
    arch  = architecture(grid)
    state = exchanger.state
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
