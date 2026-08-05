using KernelAbstractions: @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Grids: Center, Face, inactive_node, φnode
using Oceananigans.Operators: Δzᶜᶜᶠ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using Oceananigans.TurbulenceClosures: FluxTapering, ϵSxᶠᶜᶠ, ϵSyᶜᶠᶠ
using Oceananigans.Utils: KernelParameters, launch!

# NEMO's `ldf_eiv` / `ldf_tra` with `nn_aei_ijk_t = nn_aht_ijk_t = 21` (Treguier et al. 1997), the
# setting CMCC's ORCA1 runs for OMIP. The coefficient is the internal Rossby radius squared times the
# growth rate of baroclinic instability, held depth-uniform. GM tapers linearly to zero equatorward of
# 20°; Redi reuses the same field but rises to its reference value there and carries a floor of
# `0.2 aht0`. Transcribed from NEMO `src/OCE/LDF/ldftra.F90`.
#
# The `Ro²` scaling is what a constant coefficient cannot reproduce: the internal deformation radius
# falls from ~34 km in the subtropics to ~10 km in the subpolar North Atlantic, so the coefficient
# drops by nearly an order of magnitude across the gyre boundary.

"""
    struct NEMOEddyCoefficients{F, FT}

Depth-uniform GM and Redi coefficients following NEMO's Treguier et al. (1997) option. Both fields sit
at the `IsopycnalSkewSymmetricDiffusivity` coefficient location so the closure reads them without
interpolation, and both are recomputed in place from the model state, so stepping allocates nothing.
"""
struct NEMOEddyCoefficients{F, L, FT}
    skew_coefficient      :: F    # aeiu
    symmetric_coefficient :: F    # ahtu
    slope_limiter         :: L
    parameters            :: NTuple{4, FT}   # aei0, aht0, minimum and maximum Rossby radius
end

function NEMOEddyCoefficients(grid;
                              maximum_skew_coefficient = 1000,        # aei0 = ½ rn_Ue rn_Le
                              reference_symmetric_coefficient = 1000, # aht0 = ½ rn_Ud rn_Ld
                              minimum_rossby_radius = 2e3,
                              maximum_rossby_radius = 40e3,
                              slope_limiter = FluxTapering(1e-2))

    FT = eltype(grid)
    parameters = (convert(FT, maximum_skew_coefficient),
                  convert(FT, reference_symmetric_coefficient),
                  convert(FT, minimum_rossby_radius),
                  convert(FT, maximum_rossby_radius))

    return NEMOEddyCoefficients(Field{Center, Center, Face}(grid),
                                Field{Center, Center, Face}(grid),
                                slope_limiter, parameters)
end

@inline finite_or_zero(x, grid) = ifelse(isfinite(x), x, zero(grid))

# The slope stencil reaches into the halo, and on the tripolar fold row the folded metrics can return a
# non-finite slope. One such level would otherwise poison the whole column sum below.
@inline function squared_isopycnal_slope(i, j, k, grid, limiter, buoyancy, tracers)
    Sx = ℑxᶜᵃᵃ(i, j, k, grid, ϵSxᶠᶜᶠ, limiter, buoyancy, tracers)
    Sy = ℑyᵃᶜᵃ(i, j, k, grid, ϵSyᶜᶠᶠ, limiter, buoyancy, tracers)
    return finite_or_zero(Sx^2, grid) + finite_or_zero(Sy^2, grid)
end

@kernel function _compute_nemo_eddy_coefficients!(aei, aht, grid, limiter, buoyancy, tracers, Ω, parameters)
    i, j = @index(Global, NTuple)
    aei0, aht0, Romin, Romax = parameters
    Nz = size(grid, 3)

    # Column integrals: ∫N dz for the Rossby radius, and ∫N²S² dz / ∫dz for the inverse eddy timescale.
    zn = zero(grid)
    zah = zero(grid)
    zhw = zero(grid)

    for k in 1:Nz
        inactive = inactive_node(i, j, k, grid, Center(), Center(), Face())
        Δz = ifelse(inactive, zero(grid), Δzᶜᶜᶠ(i, j, k, grid))
        N² = max(finite_or_zero(∂z_b(i, j, k, grid, buoyancy, tracers), grid), zero(grid))
        N² = ifelse(inactive, zero(grid), N²)
        zn  += sqrt(N²) * Δz
        zah += N² * squared_isopycnal_slope(i, j, k, grid, limiter, buoyancy, tracers) * Δz
        zhw += Δz
    end

    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    f = 2 * Ω * sind(φ)
    f20 = 2 * Ω * sind(convert(eltype(grid), 20))

    Ro = clamp(convert(eltype(grid), 2//5) * zn / max(abs(f), convert(eltype(grid), 1e-10)), Romin, Romax)
    growth_rate = sqrt(zah / max(zhw, one(grid)))

    # Tropical factor: GM is scaled by it (→ 0 at the equator), Redi by its complement (→ aht0).
    tropical = min(one(grid), abs(f) / f20)

    aeiw = min(tropical * Ro^2 * growth_rate, aei0)

    # NEMO's `MAX(zaht_min, aht) + zaht` has no final bound, so between the equator and 20° — where the
    # GM taper is only half applied but `aei` can still sit at its cap — it overshoots to as much as
    # `aei0 + 0.8 aht0`. Bounded here to the maximum NEMO's own namelist print documents.
    ahtmin = convert(eltype(grid), 1//5) * aht0
    ahtw = max(ahtmin, aeiw) + (one(grid) - tropical) * (aht0 - ahtmin)
    ahtw = min(ahtw, max(aei0, aht0))

    dry = zhw == zero(grid)
    aeiw = ifelse(dry, zero(grid), aeiw)
    ahtw = ifelse(dry, zero(grid), ahtw)

    for k in 1:Nz+1
        @inbounds aei[i, j, k] = aeiw
        @inbounds aht[i, j, k] = ahtw
    end
end


# With distinct, nonzero κ_skew ≠ κ_symmetric fields, the isopycnal fluxes evaluated from the
# two sides of the tripolar fold differ (the closure's (κ_symmetric − κ_skew) cross-term meets
# fold halos that are not exact images) and the closure leaks tracer at a steady rate —
# ~2e-6 of the total salt per year in an idealized closed-basin test, the rate observed in
# production. Homogenizing the band is NOT sufficient; a shared field or zero coefficients
# there are. Zeroing the last `rows` interior rows removes every closure flux touching the
# fold, so no asymmetry can survive, at no physical cost: the band is the central-Arctic rim,
# where the Treguier coefficient is near zero anyway (deformation radius at its clamp).
# Harmless on grids without a fold.
function zero_fold_rows!(coefficient_field, grid; rows = 5)
    Nz = size(grid, 3)
    launch!(architecture(grid), grid, KernelParameters((Nz + 1,), (0,)), _zero_fold_rows!, coefficient_field, grid, rows)
    return nothing
end

@kernel function _zero_fold_rows!(κ, grid, rows)
    k = @index(Global)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    for j in Ny-rows+1:Ny, i in 1:Nx
        @inbounds κ[i, j, k] = 0
    end
end

"""
    compute_nemo_eddy_coefficients!(coefficients, ocean_model)

Refresh both coefficient fields from the current buoyancy field.
"""
function compute_nemo_eddy_coefficients!(coefficients::NEMOEddyCoefficients, ocean_model)
    grid = ocean_model.grid

    # `Oceananigans.defaults` is a mutable global; reading it inside the kernel makes the device
    # dereference host memory, which faults on GPU. Resolve it here and pass the value in.
    Ω = convert(eltype(grid), Oceananigans.defaults.planet_rotation_rate)

    launch!(architecture(grid), grid, :xy, _compute_nemo_eddy_coefficients!,
            coefficients.skew_coefficient, coefficients.symmetric_coefficient, grid,
            coefficients.slope_limiter, ocean_model.buoyancy, fields(ocean_model),
            Ω, coefficients.parameters)

    zero_fold_rows!(coefficients.skew_coefficient, grid)
    zero_fold_rows!(coefficients.symmetric_coefficient, grid)

    # Without this the halos stay zero, so the two cells sharing a face across the periodic seam or the
    # tripolar fold evaluate the isopycnal flux with different κ. The face flux is then not antisymmetric
    # between them, the divergence stops telescoping, and the closure leaks salt at a steady rate.
    # A scalar κ has no halo to get wrong, which is why only the field-valued coefficient needs this.
    fill_halo_regions!(coefficients.skew_coefficient)
    fill_halo_regions!(coefficients.symmetric_coefficient)

    return nothing
end

struct RefreshNEMOEddyCoefficients{N}
    coefficients :: N
end

(r::RefreshNEMOEddyCoefficients)(sim) = compute_nemo_eddy_coefficients!(r.coefficients, sim.model.ocean.model)

# `:nemo` selects the Treguier coefficient for either diffusivity; anything else passes through to
# `IsopycnalSkewSymmetricDiffusivity` unchanged.
uses_nemo_eddy_coefficients(κ_skew, κ_symmetric) = (κ_skew === :nemo) | (κ_symmetric === :nemo)

resolve_nemo_coefficient(κ, ::Nothing, field_name) = κ
resolve_nemo_coefficient(κ, coefficients, field_name) =
    κ === :nemo ? getproperty(coefficients, field_name) : κ
