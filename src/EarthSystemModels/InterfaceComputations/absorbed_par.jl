#####
##### Absorbed PAR — prescribed (offline default) or live from radiation.
##### Consumed as a per-unit-leaf-area quantity: the canopy up-scaling happens
##### downstream via `gᶜ = LAI · gₛ`, so `InteractiveAbsorbedPAR` divides the
##### Beer–Lambert canopy-absorbed flux by `LAI` to match the per-leaf convention
##### that `net_assimilation` and the Jarvis light factor expect. ClimaLand Eqs D9, D11.
#####

"""
    beer_lambert_absorbed_fraction(leaf_albedo, canopy_transmittance)

Fraction of incident shortwave a bulk canopy absorbs, `fᵃᵇˢ = (1 − α)(1 − ftrans)`
(ClimaLand Eq D11). `canopy_transmittance` is the Beer–Lambert `e^{−K·LAI·Ω}` of whatever
owns the canopy geometry, so a band absorbs on the same structure the shortwave split uses.
"""
@inline beer_lambert_absorbed_fraction(leaf_albedo, canopy_transmittance) =
    (1 - leaf_albedo) * (1 - canopy_transmittance)

abstract type AbstractAbsorbedPAR end

"""
    PrescribedAbsorbedPAR(value)

Fixed per-leaf absorbed PAR (mol photon m⁻² s⁻¹) — the offline default. Ignores
the radiation state and reproduces the original constant-`absorbed_par` behavior.
"""
struct PrescribedAbsorbedPAR{FT} <: AbstractAbsorbedPAR
    value :: FT
end

Base.summary(p::PrescribedAbsorbedPAR) = string("PrescribedAbsorbedPAR(", prettysummary(p.value), ")")
Base.show(io::IO, p::PrescribedAbsorbedPAR) = print(io, summary(p))

"""
    InteractiveAbsorbedPAR(FT = Float64; par_fraction, photon_per_joule,
                           leaf_albedo_par, extinction, clumping, lai_min)

Per-leaf absorbed PAR recomputed each step from the downwelling shortwave `ℐꜜˢʷ`
(W m⁻²) in the radiation state,

    APAR = fᵃᵇˢ · (fᵖᵃʳ · ℐꜜˢʷ · Qᴶ) / max(LAI, LAIᵐⁱⁿ) ,

where `fᵖᵃʳ` (`par_fraction`) is the PAR energy fraction of shortwave, `Qᴶ`
(`photon_per_joule`) converts PAR energy to a photon flux, and `fᵃᵇˢ` is the
absorbed fraction ([`beer_lambert_absorbed_fraction`](@ref)) on the canopy
transmittance the caller supplies — a [`CanopyAirSpace`](@ref) passes the one its
own shortwave split uses, so the two cannot diverge. Dividing by `LAI` returns a
per-leaf value (see the convention note above). With
the shortwave following the sun, the canopy conductance then follows the diurnal
light cycle — the dominant daytime driver of transpiration.

Fields:
- `par_fraction`     : PAR/shortwave by energy (≈ 0.45).
- `photon_per_joule` : mol photons per J in the PAR band (≈ 4.57e-6).
- `leaf_albedo_par`  : leaf albedo in the PAR band.
- `extinction`       : canopy extinction coefficient `K`, used only when no canopy supplies a
                       transmittance. A [`CanopyAirSpace`](@ref) always does.
- `clumping`         : foliage clumping index `Ω`, used as `extinction` is.
- `lai_min`          : floor on `LAI` in the per-leaf division.
"""
struct InteractiveAbsorbedPAR{FT} <: AbstractAbsorbedPAR
    par_fraction     :: FT
    photon_per_joule :: FT
    leaf_albedo_par  :: FT
    extinction       :: FT
    clumping         :: FT
    lai_min          :: FT
end

InteractiveAbsorbedPAR(FT=Oceananigans.defaults.FloatType;
                       par_fraction     = 0.45,
                       photon_per_joule = 4.57e-6,
                       leaf_albedo_par  = 0.1,
                       extinction       = 0.5,
                       clumping         = 1,
                       lai_min          = 0.1) =
    InteractiveAbsorbedPAR{FT}(par_fraction, photon_per_joule, leaf_albedo_par,
                               extinction, clumping, lai_min)

Base.summary(::InteractiveAbsorbedPAR{FT}) where FT = "InteractiveAbsorbedPAR{$FT}"
Base.show(io::IO, p::InteractiveAbsorbedPAR) = print(io, summary(p),
    "(par_fraction=", prettysummary(p.par_fraction), ")")

# Wrap a bare number as the prescribed spec (backward-compatible constructor arg).
@inline absorbed_par_spec(x::AbstractAbsorbedPAR, FT) = x
@inline absorbed_par_spec(x::Number, FT) = PrescribedAbsorbedPAR(convert(FT, x))

# A canopy hands down the transmittance its own shortwave split used, so PAR absorbs on that
# structure rather than rebuilding it. `nothing` means no canopy supplied one — the closure's own
# `extinction` and `clumping` are the fallback, and are read in no other case.
@inline canopy_transmittance(p::InteractiveAbsorbedPAR, leaf_area_index, ::Nothing) =
    exp(-p.extinction * leaf_area_index * p.clumping)
@inline canopy_transmittance(p::InteractiveAbsorbedPAR, leaf_area_index, ftrans) = ftrans

@inline absorbed_par_value(p::PrescribedAbsorbedPAR, radiation, leaf_area_index, ftrans) = p.value

@inline function absorbed_par_value(p::InteractiveAbsorbedPAR, radiation, leaf_area_index, ftrans)
    SW   = radiation.ℐꜜˢʷ                                      # downwelling shortwave (W m⁻²)
    parQ = p.par_fraction * SW * p.photon_per_joule            # canopy-incident PAR photon flux
    fᵃᵇˢ = beer_lambert_absorbed_fraction(p.leaf_albedo_par,
                                          canopy_transmittance(p, leaf_area_index, ftrans))
    return fᵃᵇˢ * parQ / max(leaf_area_index, p.lai_min)       # per-leaf
end
