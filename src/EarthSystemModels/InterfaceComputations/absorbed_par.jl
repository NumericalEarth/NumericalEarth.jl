#####
##### Absorbed PAR per unit leaf area: a prescribed `Number`, or `InteractiveAbsorbedPAR` from shortwave.
#####

"""
    InteractiveAbsorbedPAR(FT = Float64; par_fraction, photon_per_joule,
                           leaf_albedo_par, extinction, clumping)

Per-leaf absorbed PAR recomputed each step from the downwelling shortwave `ℐꜜˢʷ`
(W m⁻², positive upward) in the radiation state,

    APAR = - fᵃᵇˢ · (fᵖᵃʳ · ℐꜜˢʷ · Qᴶ) / LAI ,

where `fᵖᵃʳ` (`par_fraction`) is the PAR energy fraction of shortwave, `Qᴶ`
(`photon_per_joule`) converts PAR energy to a photon flux, and
`fᵃᵇˢ = (1 − α)(1 − e^{−K Ω L})` is the fraction of the incident PAR the canopy
absorbs (ClimaLand Eq D11), with the PAR-band leaf albedo `α` and the Beer–Lambert
transmittance the caller supplies. Dividing by `LAI` returns the per-leaf value the
leaf conductance models expect.

Fields:
- `par_fraction`     : PAR/shortwave by energy (≈ 0.45).
- `photon_per_joule` : mol photons per J in the PAR band (≈ 4.57e-6).
- `leaf_albedo_par`  : leaf albedo in the PAR band.
- `extinction`       : Beer–Lambert extinction coefficient `K`, used only when no canopy
                       supplies a transmittance.
- `clumping`         : foliage clumping index `Ω ∈ (0, 1]` ([Chen and Black (1992)](@cite chen1992)),
                       the ratio of the effective to the true leaf area index in the
                       transmittance `e^{−K Ω L}`: `Ω = 1` is randomly distributed foliage,
                       `Ω < 1` foliage gathered into crowns and shoots, which leaves gaps and
                       transmits more light than a random canopy of the same `L`. Used as
                       `extinction` is.
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

# `nothing` means no canopy supplied a transmittance, so the absorbed-PAR model falls
# back on its own `extinction` and `clumping`.
@inline canopy_transmittance(p::InteractiveAbsorbedPAR, leaf_area_index, ::Nothing) =
    canopy_transmittance(p.extinction, p.clumping, leaf_area_index)
@inline canopy_transmittance(p::InteractiveAbsorbedPAR, leaf_area_index, transmittance) = transmittance

@inline absorbed_par_value(p::Number, radiation, leaf_area_index, transmittance) = p

@inline function absorbed_par_value(p::InteractiveAbsorbedPAR, radiation, leaf_area_index, transmittance)
    SW   = - radiation.ℐꜜˢʷ                                    # downwelling shortwave (W m⁻²)
    parQ = p.par_fraction * SW * p.photon_per_joule            # canopy-incident PAR photon flux
    fᵃᵇˢ = (1 - p.leaf_albedo_par) * (1 - canopy_transmittance(p, leaf_area_index, transmittance))
    return fᵃᵇˢ * parQ / max(leaf_area_index, sqrt(eps(leaf_area_index)))   # per-leaf; 0/0 → 0
end
