#####
##### Seeding a non-equilibrium child's condensate prognostics from the parent's hydrometeors
#####
#
# A parent `PrescribedAtmosphere` supplies four hydrometeor species `(qᶜˡ, qʳ, qᶜⁱ, qˢ)` (cloud liquid,
# rain, cloud ice, snow + graupel). An equilibrium child (`SaturationAdjustment`, moisture slot `:ρqᵉ`)
# folds the cloud species into `qᵉ` and needs nothing more. A non-equilibrium child (moisture slot `:ρqᵛ`)
# carries its condensate as separate prognostics, which the state exchanger fills from the parent through
# a "condensate seeding": an isbits parameter object built host-side from the child's microphysics, plus
#
#   seeded_prognostic_names(seeding)                         -> names of the child prognostics it fills
#   parent_condensate_prognostics(seeding, q, ρ)             -> NamedTuple of those densities at one point
#   seeded_condensate(seeding, q)                            -> (qˡ, qⁱ) the child holds as condensate
#
# where `q = (; qᶜˡ, qʳ, qᶜⁱ, qˢ)` holds point values. `seeding === nothing` (every equilibrium scheme, and
# any scheme without a method) seeds nothing, which reproduces the vapor-only exchange exactly.

using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics, AerosolActivation

"""
$(TYPEDSIGNATURES)

Return the condensate seeding for a child with `microphysics`, or `nothing` when the child's prognostics
do not carry condensate the parent can seed. Equilibrium schemes fold cloud condensate into their moisture
prognostic, so they return `nothing`, as does any scheme without a method.
"""
condensate_seeding(microphysics; kw...) = nothing

"""
    P3CondensateSeeding{FT, CN}

Parameters mapping a parent's `(qᶜˡ, qʳ, qᶜⁱ, qˢ)` onto the prognostics of a P3 child
(`PredictedParticlePropertiesMicrophysics`). See [`P3CondensateSeeding(p3)`](@ref) for the
mapping and the defaults.
"""
struct P3CondensateSeeding{FT, CN}
    rain_intercept        :: FT  # N₀ʳ [m⁻⁴]
    snow_intercept        :: FT  # N₀ˢ [m⁻⁴]
    snow_density          :: FT  # ρˢ [kg m⁻³]
    cloud_ice_mean_mass   :: FT  # m̄ᶜⁱ [kg]
    minimum_ice_mean_mass :: FT  # [kg]
    maximum_ice_mean_mass :: FT  # [kg]
    minimum_rain_slope    :: FT  # P3's λʳ bounds [m⁻¹]
    maximum_rain_slope    :: FT
    liquid_water_density  :: FT  # ρʷ [kg m⁻³]
    minimum_mass_fraction :: FT  # species below this are seeded as absent [kg kg⁻¹]
    cloud_droplet_number  :: CN  # Nᶜ [m⁻³] when P3 carries ρnᶜˡ, else `nothing`
end

"""
$(TYPEDSIGNATURES)

Build the parameters that seed a P3 child from a parent's `(qᶜˡ, qʳ, qᶜⁱ, qˢ)`:

- `ρqᶜˡ = ρ qᶜˡ` and `ρqʳ = ρ qʳ`.
- `ρqⁱ = ρ (qᶜⁱ + qˢ)`: P3 carries one ice category, so cloud ice and snow (which includes graupel for
  most parents) merge.
- Rime mass `ρqᶠ`, rime volume `ρbᶠ` and liquid on ice `ρqʷⁱ` are not seeded (they start at zero), so
  the seeded ice is unrimed and dry.
- `ρnʳ` assumes an exponential drop spectrum with the fixed intercept `rain_intercept` (N₀ʳ, default
  8×10⁶ m⁻⁴, [Marshall and Palmer (1948)](@cite MarshallPalmer1948)). The slope
  λʳ = (π ρʷ N₀ʳ / ρqʳ)^¼ is clamped to P3's `minimum_rain_slope`/`maximum_rain_slope`, and
  `ρnʳ = ρqʳ λʳ³ / (π ρʷ)` is P3's own number–slope relation, so P3 does not re-clamp it at t₀.
- `ρnⁱ` sums a cloud-ice number `ρ qᶜⁱ / m̄ᶜⁱ`, with the mean crystal mass `cloud_ice_mean_mass`
  (m̄ᶜⁱ, default 10⁻¹¹ kg, a ≈ 30 μm ice sphere), and a snow number from an exponential spectrum with
  intercept `snow_intercept` (N₀ˢ, default 3×10⁶ m⁻⁴) and bulk density `snow_density` (ρˢ, default
  100 kg m⁻³), the snow parameters of [Lin et al. (1983)](@cite LinFarleyOrville1983). The mean ice mass
  `ρqⁱ / ρnⁱ` is then clamped to `[minimum_ice_mean_mass, maximum_ice_mean_mass]`. The default bounds
  are the masses at P3's minimum and maximum mean diameters: a pure-ice sphere at the minimum, and the
  [Brown and Francis (1995)](@cite BrownFrancis1995) mass–size law `m = 0.0121 D^1.9` (SI) that P3 uses
  for large unrimed ice at the maximum.
- `ρnᶜˡ` (only when P3 carries it, that is with `AerosolActivation`) is P3's prescribed
  `cloud.number_concentration` wherever cloud liquid is seeded.

Each species with a mass fraction at or below `minimum_mass_fraction` (default P3's
`minimum_mass_mixing_ratio`) is seeded as absent, mass and number, so that no number is diagnosed from
a vanishing mass (in Float32 that can produce NaNs in P3).
"""
function P3CondensateSeeding(p3::PredictedParticlePropertiesMicrophysics;
                             rain_intercept = 8e6,
                             snow_intercept = 3e6,
                             snow_density = 100,
                             cloud_ice_mean_mass = 1e-11,
                             minimum_ice_mean_mass = default_minimum_ice_mean_mass(p3),
                             maximum_ice_mean_mass = default_maximum_ice_mean_mass(p3),
                             minimum_mass_fraction = p3.minimum_mass_mixing_ratio)
    FT = typeof(p3.minimum_mass_mixing_ratio)
    parameters = p3.process_rates
    Nᶜ = p3.aerosol isa AerosolActivation ? convert(FT, p3.cloud.number_concentration) : nothing
    return P3CondensateSeeding(convert(FT, rain_intercept),
                               convert(FT, snow_intercept),
                               convert(FT, snow_density),
                               convert(FT, cloud_ice_mean_mass),
                               convert(FT, minimum_ice_mean_mass),
                               convert(FT, maximum_ice_mean_mass),
                               convert(FT, parameters.minimum_rain_slope),
                               convert(FT, parameters.maximum_rain_slope),
                               convert(FT, parameters.liquid_water_density),
                               convert(FT, minimum_mass_fraction),
                               Nᶜ)
end

# A pure-ice sphere at P3's minimum mean diameter.
function default_minimum_ice_mean_mass(p3)
    D = p3.ice.bulk.minimum_mean_diameter
    ρⁱ = p3.process_rates.pure_ice_density
    return π / 6 * ρⁱ * D^3
end

# Brown and Francis (1995) unrimed-aggregate mass at P3's maximum mean diameter.
function default_maximum_ice_mean_mass(p3)
    D = p3.ice.bulk.maximum_mean_diameter
    return 0.0121 * D^1.9
end

condensate_seeding(p3::PredictedParticlePropertiesMicrophysics; kw...) = P3CondensateSeeding(p3; kw...)

# Re-express the parameters in the parent grid's float type, so the exchanger kernels never promote.
seeding_with_eltype(FT, ::Nothing) = nothing

function seeding_with_eltype(FT, s::P3CondensateSeeding)
    Nᶜ = isnothing(s.cloud_droplet_number) ? nothing : convert(FT, s.cloud_droplet_number)
    return P3CondensateSeeding(convert(FT, s.rain_intercept), convert(FT, s.snow_intercept),
                               convert(FT, s.snow_density), convert(FT, s.cloud_ice_mean_mass),
                               convert(FT, s.minimum_ice_mean_mass), convert(FT, s.maximum_ice_mean_mass),
                               convert(FT, s.minimum_rain_slope), convert(FT, s.maximum_rain_slope),
                               convert(FT, s.liquid_water_density), convert(FT, s.minimum_mass_fraction), Nᶜ)
end

"""
$(TYPEDSIGNATURES)

Names of the child prognostics that `seeding` fills from the parent.
"""
seeded_prognostic_names(::Nothing) = ()
seeded_prognostic_names(::P3CondensateSeeding{<:Any, Nothing}) = (:ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ)
seeded_prognostic_names(::P3CondensateSeeding) = (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ)

# The names of the mass densities among `seeded_prognostic_names`: what loads the child's total density.
seeded_mass_names(::Nothing) = ()
seeded_mass_names(::P3CondensateSeeding) = (:ρqᶜˡ, :ρqʳ, :ρqⁱ)

# A species at or below the floor is seeded as absent (this also drops negative reanalysis values).
@inline floored(q, s) = ifelse(q > s.minimum_mass_fraction, q, zero(q))

"""
$(TYPEDSIGNATURES)

Return the liquid and ice mass fractions `(qˡ, qⁱ)` that `seeding` hands the child as condensate, given
the parent's point values `q = (; qᶜˡ, qʳ, qᶜⁱ, qˢ)`. The state exchanger's θˡⁱ removes latent heat for
exactly this condensate.
"""
@inline seeded_condensate(::Nothing, q) = (zero(q.qᶜˡ), zero(q.qᶜˡ))
@inline seeded_condensate(s::P3CondensateSeeding, q) = (floored(q.qᶜˡ, s) + floored(q.qʳ, s),
                                                         floored(q.qᶜⁱ, s) + floored(q.qˢ, s))

# Number density of an exponential spectrum with intercept N₀ holding the mass density ρq of particles of
# bulk density ρᵖ (spheres): ρq = π ρᵖ N₀ / λ⁴ and n = N₀ / λ = ρq λ³ / (π ρᵖ). `max(ρq, floatmin)` keeps
# the slope finite-or-Inf (never NaN) for ρq = 0, where the number is zero anyway.
@inline exponential_slope(ρq, N₀, ρᵖ) = sqrt(sqrt(convert(typeof(ρq), π) * ρᵖ * N₀ / max(ρq, floatmin(ρq))))
@inline exponential_number(ρq, λ, ρᵖ) = ρq * λ^3 / (convert(typeof(ρq), π) * ρᵖ)

"""
$(TYPEDSIGNATURES)

Return the child prognostic densities that `seeding` derives from the parent's point values
`q = (; qᶜˡ, qʳ, qᶜⁱ, qˢ)` at total density `ρ`, as a `NamedTuple` keyed by
[`seeded_prognostic_names`](@ref). `seeding === nothing` returns an empty `NamedTuple`.
"""
@inline parent_condensate_prognostics(::Nothing, q, ρ) = NamedTuple()

@inline function parent_condensate_prognostics(s::P3CondensateSeeding, q, ρ)
    qᶜˡ = floored(q.qᶜˡ, s)
    qʳ  = floored(q.qʳ, s)
    qᶜⁱ = floored(q.qᶜⁱ, s)
    qˢ  = floored(q.qˢ, s)

    ρqᶜˡ = ρ * qᶜˡ
    ρqʳ  = ρ * qʳ
    ρqᶜⁱ = ρ * qᶜⁱ
    ρqˢ  = ρ * qˢ
    ρqⁱ  = ρqᶜⁱ + ρqˢ

    # Rain: Marshall–Palmer slope, clamped to P3's λʳ bounds, through P3's number–slope relation.
    λʳ  = clamp(exponential_slope(ρqʳ, s.rain_intercept, s.liquid_water_density),
                s.minimum_rain_slope, s.maximum_rain_slope)
    ρnʳ = exponential_number(ρqʳ, λʳ, s.liquid_water_density)

    # Ice: cloud ice at a fixed mean mass plus exponential snow, mean mass clamped to P3's size range.
    λˢ   = exponential_slope(ρqˢ, s.snow_intercept, s.snow_density)
    ρnˢ  = ifelse(ρqˢ > 0, exponential_number(ρqˢ, λˢ, s.snow_density), zero(ρqˢ))
    ρnⁱ₀ = ρqᶜⁱ / s.cloud_ice_mean_mass + ρnˢ
    ρnⁱ  = clamp(ρnⁱ₀, ρqⁱ / s.maximum_ice_mean_mass, ρqⁱ / s.minimum_ice_mean_mass)

    return seeded_cloud_number(s.cloud_droplet_number, ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ)
end

@inline seeded_cloud_number(::Nothing, ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ) = (; ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ)

@inline function seeded_cloud_number(Nᶜ, ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ)
    ρnᶜˡ = ifelse(ρqᶜˡ > 0, Nᶜ, zero(Nᶜ))
    return (; ρqᶜˡ, ρnᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ)
end
