#####
##### Absorbed PAR per unit leaf area: a prescribed `Number`, or `InteractiveAbsorbedPAR` from shortwave.
#####

"""
    beer_lambert_absorbed_fraction(leaf_albedo, canopy_transmittance)

Fraction of incident shortwave a bulk canopy absorbs, `fᵃᵇˢ = (1 − α)(1 − transmittance)`
(ClimaLand Eq D11).
"""
@inline beer_lambert_absorbed_fraction(leaf_albedo, canopy_transmittance) =
    (1 - leaf_albedo) * (1 - canopy_transmittance)

"""
    InteractiveAbsorbedPAR(FT = Float64; par_fraction, photon_per_joule,
                           leaf_albedo_par, extinction, clumping)

Per-leaf absorbed PAR recomputed each step from the downwelling shortwave `ℐꜜˢʷ`
(W m⁻²) in the radiation state,

    APAR = fᵃᵇˢ · (fᵖᵃʳ · ℐꜜˢʷ · Qᴶ) / LAI ,

where `fᵖᵃʳ` (`par_fraction`) is the PAR energy fraction of shortwave, `Qᴶ`
(`photon_per_joule`) converts PAR energy to a photon flux, and `fᵃᵇˢ` is the
absorbed fraction on the canopy transmittance the caller supplies. Dividing by
`LAI` returns the per-leaf value the leaf conductance models expect.

Fields:
- `par_fraction`     : PAR/shortwave by energy (≈ 0.45).
- `photon_per_joule` : mol photons per J in the PAR band (≈ 4.57e-6).
- `leaf_albedo_par`  : leaf albedo in the PAR band.
- `extinction`       : canopy extinction coefficient `K`, used only when no canopy supplies a
                       transmittance.
- `clumping`         : foliage clumping index `Ω`, used as `extinction` is.
"""
struct InteractiveAbsorbedPAR{FT}
    par_fraction     :: FT
    photon_per_joule :: FT
    leaf_albedo_par  :: FT
    extinction       :: FT
    clumping         :: FT
end

InteractiveAbsorbedPAR(FT=Oceananigans.defaults.FloatType;
                       par_fraction     = 0.45,
                       photon_per_joule = 4.57e-6,
                       leaf_albedo_par  = 0.1,
                       extinction       = 0.5,
                       clumping         = 1) =
    InteractiveAbsorbedPAR{FT}(par_fraction, photon_per_joule, leaf_albedo_par,
                               extinction, clumping)

Base.summary(::InteractiveAbsorbedPAR{FT}) where FT = "InteractiveAbsorbedPAR{$FT}"
Base.show(io::IO, p::InteractiveAbsorbedPAR) = print(io, summary(p),
    "(par_fraction=", prettysummary(p.par_fraction), ")")

# Beer–Lambert transmittance through a canopy of leaf area index `LAI`, extinction
# coefficient `K` and clumping `Ω`.
@inline canopy_transmittance(K, Ω, leaf_area_index) = exp(-K * Ω * leaf_area_index)

# `nothing` means no canopy supplied a transmittance, so the closure falls back on its
# own `extinction` and `clumping`.
@inline canopy_transmittance(p::InteractiveAbsorbedPAR, leaf_area_index, ::Nothing) =
    canopy_transmittance(p.extinction, p.clumping, leaf_area_index)
@inline canopy_transmittance(p::InteractiveAbsorbedPAR, leaf_area_index, transmittance) = transmittance

@inline absorbed_par_value(p::Number, radiation, leaf_area_index, transmittance) = p

@inline function absorbed_par_value(p::InteractiveAbsorbedPAR, radiation, leaf_area_index, transmittance)
    SW   = radiation.ℐꜜˢʷ                                      # downwelling shortwave (W m⁻²)
    parQ = p.par_fraction * SW * p.photon_per_joule            # canopy-incident PAR photon flux
    fᵃᵇˢ = beer_lambert_absorbed_fraction(p.leaf_albedo_par,
                                          canopy_transmittance(p, leaf_area_index, transmittance))
    return fᵃᵇˢ * parQ / max(leaf_area_index, sqrt(eps(leaf_area_index)))   # per-leaf; 0/0 → 0
end
