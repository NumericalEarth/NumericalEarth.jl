using Adapt: Adapt
using KernelAbstractions: @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: ∂xᵣ_b, ∂yᵣ_b, ∂z_b, buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: Center, Face, inactive_node, znode, static_column_depthᶜᶜᵃ
using Oceananigans.Operators: ∂x_zᶠᶜᶜ, ∂y_zᶠᶜᶜ, ∂x_zᶜᶠᶜ, ∂y_zᶜᶠᶜ, ∂x_zᶜᶜᶠ, ∂y_zᶜᶜᶠ,
                              ℑxyᶠᶜᵃ, ℑxzᶠᵃᶜ, ℑxyᶜᶠᵃ, ℑyzᵃᶠᶜ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶠ
using Oceananigans.TurbulenceClosures: TurbulenceClosures, IsopycnalSkewSymmetricDiffusivity, calc_tapering
using Oceananigans.Utils: launch!

# Two-sided boundary tapering of the isopycnal slopes.  `MixedLayerTapering` ramps the eddy fluxes to
# zero from the mixed-layer base to the surface, following Danabasoglu, Ferrari & McWilliams (2008) and
# NEMO's `ldfslp`.  The same argument applies at the bottom: the neutral direction must rotate to
# follow the boundary there too, and a rotated operator evaluated with the interior slope mixes a
# bottom-trapped gravity current straight into the ambient.  `BoundaryLayerTapering` carries both ramps,
# and either may be switched off — the bottom ramp by `bottom_layer_depth = 0`, the surface ramp by a
# `nothing` mixed-layer field.
#
# The vertical tracer diffusivity the isopycnal closure applies is `κ_symmetric · ϵ · (Sx² + Sy²)`,
# which `FluxTapering(Sₘ)` caps at `κ_symmetric · Sₘ²` — 0.08 m² s⁻¹ at the OMIP `orca` settings of
# 800 m² s⁻¹ and Sₘ = 10⁻². Campaign 32 measures a median of 1.6–4.5 × 10⁻⁴ m² s⁻¹ on the East
# Greenland and Labrador slopes, 20–45× the Henyey background and the order needed to erode a 200 m
# dense layer in five years, which is the timescale on which the model loses the Denmark Strait
# overflow product from the Irminger Sea.

"""
    struct BoundaryLayerTapering{FT, F}

Slope limiter for `IsopycnalSkewSymmetricDiffusivity` combining the Gerdes et al. (1991) `max_slope`
clip with a linear ramp of the tapering factor to zero at each boundary: from the local mixed-layer
base to the surface, and from `bottom_layer_depth` metres above the sea floor to the sea floor.
"""
struct BoundaryLayerTapering{FT, F}
    max_slope :: FT                    # duck-types FluxTapering for `calc_tapering`
    mixed_layer_depth :: F             # (Center, Center, Nothing), metres positive down, or `nothing`
    bottom_layer_depth :: FT           # metres; 0 switches the bottom ramp off
end

"""
    BoundaryLayerTapering(grid; max_slope = 1e-2, mixed_layer = true, bottom_layer_depth = 0)

Build the limiter. `mixed_layer = false` allocates no mixed-layer field and disables the surface ramp,
which is what isolates the bottom ramp as a single lever.
"""
function BoundaryLayerTapering(grid; max_slope = 1e-2, mixed_layer = true, bottom_layer_depth = 0)
    FT = eltype(grid)
    h = mixed_layer ? Field{Center, Center, Nothing}(grid) : nothing
    return BoundaryLayerTapering(convert(FT, max_slope), h, convert(FT, bottom_layer_depth))
end

Adapt.adapt_structure(to, limiter::BoundaryLayerTapering) =
    BoundaryLayerTapering(limiter.max_slope,
                          Adapt.adapt(to, limiter.mixed_layer_depth),
                          limiter.bottom_layer_depth)

const NoSurfaceRamp = BoundaryLayerTapering{<:Any, Nothing}

# 0 at the surface, 1 at and below the mixed-layer base
@inline surface_ramp(i, j, k, grid, ::NoSurfaceRamp) = one(grid)

@inline function surface_ramp(i, j, k, grid, limiter::BoundaryLayerTapering)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    h = @inbounds limiter.mixed_layer_depth[i, j, 1]
    return clamp(-z / max(h, one(grid)), zero(grid), one(grid))
end

# 0 at the sea floor, 1 at and above `bottom_layer_depth` metres above it
@inline function bottom_ramp(i, j, k, grid, limiter::BoundaryLayerTapering)
    δ = limiter.bottom_layer_depth
    z = znode(i, j, k, grid, Center(), Center(), Face())
    hab = z + static_column_depthᶜᶜᵃ(i, j, grid)
    ramp = clamp(hab / max(δ, one(grid)), zero(grid), one(grid))
    return ifelse(δ > 0, ramp, one(grid))
end

@inline boundary_ramp(i, j, k, grid, limiter) =
    surface_ramp(i, j, k, grid, limiter) * bottom_ramp(i, j, k, grid, limiter)

const BoundaryTaperedISSD =
    IsopycnalSkewSymmetricDiffusivity{<:Any, <:Any, <:Any, <:Any, <:Any, <:BoundaryLayerTapering}

# The three position-aware tapering factors of the diffusive fluxes (the implicit R₃₃ precomputation
# reuses the ᶜᶜᶠ one). Bodies mirror the untapered originals in
# `isopycnal_skew_symmetric_diffusivity.jl`.
@inline function TurbulenceClosures.tapering_factorᶠᶜᶜ(i, j, k, grid, closure::BoundaryTaperedISSD, tracers, buoyancy)
    by   = ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᵣ_b, buoyancy, tracers)
    bz   = ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b,  buoyancy, tracers)
    bx   =  ∂xᵣ_b(i, j, k, grid, buoyancy, tracers)
    ∂x_z = ∂x_zᶠᶜᶜ(i, j, k, grid)
    ∂y_z = ∂y_zᶠᶜᶜ(i, j, k, grid)
    ϵ = calc_tapering(bx, by, bz, ∂x_z, ∂y_z, grid, closure.isopycnal_tensor, closure.slope_limiter)
    return ϵ * boundary_ramp(i, j, k, grid, closure.slope_limiter)
end

@inline function TurbulenceClosures.tapering_factorᶜᶠᶜ(i, j, k, grid, closure::BoundaryTaperedISSD, tracers, buoyancy)
    bx   = ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᵣ_b, buoyancy, tracers)
    bz   = ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b,  buoyancy, tracers)
    by   =  ∂yᵣ_b(i, j, k, grid, buoyancy, tracers)
    ∂x_z = ∂x_zᶜᶠᶜ(i, j, k, grid)
    ∂y_z = ∂y_zᶜᶠᶜ(i, j, k, grid)
    ϵ = calc_tapering(bx, by, bz, ∂x_z, ∂y_z, grid, closure.isopycnal_tensor, closure.slope_limiter)
    return ϵ * boundary_ramp(i, j, k, grid, closure.slope_limiter)
end

@inline function TurbulenceClosures.tapering_factorᶜᶜᶠ(i, j, k, grid, closure::BoundaryTaperedISSD, tracers, buoyancy)
    bx   = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᵣ_b, buoyancy, tracers)
    by   = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᵣ_b, buoyancy, tracers)
    bz   =  ∂z_b(i, j, k, grid, buoyancy, tracers)
    ∂x_z = ∂x_zᶜᶜᶠ(i, j, k, grid)
    ∂y_z = ∂y_zᶜᶜᶠ(i, j, k, grid)
    ϵ = calc_tapering(bx, by, bz, ∂x_z, ∂y_z, grid, closure.isopycnal_tensor, closure.slope_limiter)
    return ϵ * boundary_ramp(i, j, k, grid, closure.slope_limiter)
end

# ⚠⚠ SIGNATURE-SENSITIVE (2026-09-08, campaign 32). `ϵSxᶠᶜᶠ` / `ϵSyᶜᶠᶠ` take a SLOPE LIMITER in
# Oceananigans zA2CT (what the two `orca_triad*` jobs launched at 11:53 are running) and take a CLOSURE
# in Qd5sf, which the Manifest re-resolve of 17:32 pulled in — reading `.isopycnal_tensor` and
# `.slope_limiter` instead. Under Qd5sf the two methods below no longer dispatch, so the ramp is silently
# dropped in the ADVECTIVE and BOUNDARY-VALUE formulations. The DIFFUSIVE formulation — the default, and
# the one ML_TAPER and BTAPER actually act through — goes via `tapering_factorᶠᶜᶜ/ᶜᶠᶜ/ᶜᶜᶠ` above and is
# unaffected under both. ⚠ The same signature change is what killed `orca_gmbvp` at 17:49; see
# `2026-09-08-campaign32.md` C32-39 and `temp/c32/PROPOSED_boundary_value_transport.jl`.
# Advective-formulation slope functions: the same ramps on top of the Gerdes magnitude clip.
@inline function TurbulenceClosures.ϵSxᶠᶜᶠ(i, j, k, grid, limiter::BoundaryLayerTapering, b, C)
    Sx = TurbulenceClosures.Sxᶠᶜᶠ(i, j, k, grid, b, C)
    ϵ  = TurbulenceClosures.tapering_factor(Sx, zero(grid), limiter)
    return ϵ * boundary_ramp(i, j, k, grid, limiter) * Sx
end

@inline function TurbulenceClosures.ϵSyᶜᶠᶠ(i, j, k, grid, limiter::BoundaryLayerTapering, b, C)
    Sy = TurbulenceClosures.Syᶜᶠᶠ(i, j, k, grid, b, C)
    ϵ  = TurbulenceClosures.tapering_factor(zero(grid), Sy, limiter)
    return ϵ * boundary_ramp(i, j, k, grid, limiter) * Sy
end

# Position-free magnitude clip, duck-typing FluxTapering
@inline TurbulenceClosures.tapering_factor(Sx, Sy, limiter::BoundaryLayerTapering) =
    min(one(Sx), limiter.max_slope^2 / (Sx^2 + Sy^2 + convert(typeof(Sx), 1e-40)))

"""
    compute_tapering_mixed_layer_depth!(limiter, ocean_model; Δb = 2.87e-4)

Refresh the limiter's mixed-layer-depth field. A no-op when the surface ramp is switched off.
"""
compute_tapering_mixed_layer_depth!(::NoSurfaceRamp, ocean_model; Δb = 2.87e-4) = nothing

function compute_tapering_mixed_layer_depth!(limiter::BoundaryLayerTapering, ocean_model; Δb = 2.87e-4)
    grid = ocean_model.grid
    launch!(architecture(grid), grid, :xy, _compute_tapering_mixed_layer_depth!,
            limiter.mixed_layer_depth, grid, ocean_model.buoyancy.formulation,
            ocean_model.tracers, convert(eltype(grid), Δb))
    homogenize_fold_band!(limiter.mixed_layer_depth, grid)
    fill_halo_regions!(limiter.mixed_layer_depth)
    return nothing
end

struct RefreshBoundaryLayerTapering{L}
    limiter :: L
end

(r::RefreshBoundaryLayerTapering)(sim) =
    compute_tapering_mixed_layer_depth!(r.limiter, sim.model.ocean.model)
