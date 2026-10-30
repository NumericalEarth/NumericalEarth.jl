#####
##### Absorbed PAR per unit leaf area: a prescribed `Number`, or `AbsorbedPhotosyntheticallyActiveRadiation` from shortwave.
#####

"""
$(TYPEDSIGNATURES)

Per-leaf absorbed photosynthetically active radiation (PAR), recomputed each step
from the downwelling shortwave `ℐꜜˢʷ`
(W m⁻², positive upward) in the radiation state,

    APAR = - fₐ · (fᵖᵃʳ · ℐꜜˢʷ · Qᴶ) / λˡᵉᵃᶠ ,

where `fᵖᵃʳ` (`par_fraction`) is the PAR energy fraction of shortwave, `Qᴶ`
(`photon_per_joule`) converts PAR energy to a photon flux, and
`fₐ = (1 − α)(1 − e^{−K Ω λˡᵉᵃᶠ})` is the fraction of the incident PAR the canopy
absorbs (ClimaLand Eq D11), with the PAR-band leaf albedo `α` and the Beer–Lambert
transmittance the caller supplies. Dividing by `λˡᵉᵃᶠ` returns the per-leaf value the
leaf conductance models expect.

Fields:
- `par_fraction`     : PAR/shortwave by energy (≈ 0.45).
- `photon_per_joule` : mol photons per J in the PAR band (≈ 4.57e-6).
- `leaf_par_albedo`  : leaf albedo in the PAR band.
- `extinction`, `clumping` : Beer–Lambert extinction `K` and foliage clumping index `Ω ∈ (0, 1]`
                       ([Chen and Black (1992)](@cite chen1992)) of the transmittance `e^{−K Ω λˡᵉᵃᶠ}`,
                       used only when no canopy supplies a transmittance.
"""
struct AbsorbedPhotosyntheticallyActiveRadiation{FT}
    par_fraction     :: FT
    photon_per_joule :: FT
    leaf_par_albedo  :: FT
    extinction       :: FT
    clumping         :: FT
end

AbsorbedPhotosyntheticallyActiveRadiation(FT=Oceananigans.defaults.FloatType;
                                        par_fraction     = 0.45,
                                        photon_per_joule = 4.57e-6,
                                        leaf_par_albedo  = 0.1,
                                        extinction       = 0.5,
                                        clumping         = 1) =
    AbsorbedPhotosyntheticallyActiveRadiation{FT}(par_fraction, photon_per_joule, leaf_par_albedo,
                                                extinction, clumping)

Base.summary(::AbsorbedPhotosyntheticallyActiveRadiation{FT}) where FT = "AbsorbedPhotosyntheticallyActiveRadiation{$FT}"
Base.show(io::IO, p::AbsorbedPhotosyntheticallyActiveRadiation) = print(io, summary(p),
    "(par_fraction=", prettysummary(p.par_fraction), ")")

# Beer–Lambert transmittance through a canopy of leaf area index `λˡᵉᵃᶠ`, extinction
# coefficient `K` and clumping `Ω`.
@inline canopy_transmittance(K, Ω, leaf_area_index) = exp(-K * Ω * leaf_area_index)

# `nothing` means no canopy supplied a transmittance, so the absorbed-PAR model falls
# back on its own `extinction` and `clumping`.
@inline canopy_transmittance(p::AbsorbedPhotosyntheticallyActiveRadiation, leaf_area_index, ::Nothing) =
    canopy_transmittance(p.extinction, p.clumping, leaf_area_index)
@inline canopy_transmittance(p::AbsorbedPhotosyntheticallyActiveRadiation, leaf_area_index, transmittance) = transmittance

@inline absorbed_par_value(p::Number, radiation, leaf_area_index, transmittance) = p

@inline function absorbed_par_value(p::AbsorbedPhotosyntheticallyActiveRadiation, radiation, leaf_area_index, transmittance)
    downwelling_shortwave = - radiation.ℐꜜˢʷ
    incident_photon_flux  = p.par_fraction * downwelling_shortwave * p.photon_per_joule
    fₐ = (1 - p.leaf_par_albedo) * (1 - canopy_transmittance(p, leaf_area_index, transmittance))
    return fₐ * incident_photon_flux / max(leaf_area_index, sqrt(eps(leaf_area_index)))   # per-leaf; 0/0 → 0
end
