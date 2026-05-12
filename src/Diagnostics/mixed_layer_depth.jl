using Oceananigans.Grids: static_column_depthᶜᶜᵃ

# de Boyer Montégut et al. (2004) DR003 convention:
# - threshold: Δσ = 0.03 kg m⁻³ at ρ₀ = 1025 kg m⁻³ ⇒ Δb ≈ 2.87 × 10⁻⁴ m s⁻².
# - reference_depth: 10 m, with buoyancy obtained by vertical interpolation
const DR003_BUOYANCY_CRITERION = 2.87e-4
const DR003_REFERENCE_DEPTH    = 10.0

mutable struct MixedLayerDepthOperand{FT, B}
    buoyancy_perturbation :: B
    difference_criterion :: FT
    reference_depth :: FT
end

Base.summary(mldo::MixedLayerDepthOperand) = "MixedLayerDepthOperand"

function MixedLayerDepthOperand(bm, grid, tracers;
                                difference_criterion = DR003_BUOYANCY_CRITERION,
                                reference_depth = DR003_REFERENCE_DEPTH)
    buoyancy_perturbation = buoyancy_operation(bm, grid, tracers)
    FT = eltype(grid)
    return MixedLayerDepthOperand(buoyancy_perturbation,
                                  convert(FT, difference_criterion),
                                  convert(FT, reference_depth))
end

const MixedLayerDepthField = Field{<:Any, <:Any, <:Any, <:MixedLayerDepthOperand}

"""
    MixedLayerDepthField(bm, grid, tracers;
                         difference_criterion = 2.87e-4,
                         reference_depth = 10.0)

Mixed-layer-depth diagnostic using the [de Boyer Montégut et al. (2004)](@cite de_boyer_montegut2004mixed)
DR003 buoyancy-difference criterion.  The default `difference_criterion = 2.87e-4` m s⁻² corresponds
to `Δσ = 0.03 kg m⁻³` at `ρ₀ = 1025 kg m⁻³`. The default `reference_depth = 10` m is the
standard dBM reference; buoyancy at that depth is obtained by linear interpolation  between adjacent
cell centers, so the diagnostic is insensitive to the vertical resolution near the surface.

References
==========

* de Boyer Montégut, C., G.Madec, A. S.Fischer, A.Lazar, and D.Iudicone (2004), Mixed layer depth over
    the global ocean: An examination of profile data and a profile-based climatology, J. Geophys. Res.,
    109, C12003, doi:10.1029/2004JC002378.
"""
function MixedLayerDepthField(bm, grid, tracers;
                              difference_criterion = DR003_BUOYANCY_CRITERION,
                              reference_depth = DR003_REFERENCE_DEPTH)
    operand = MixedLayerDepthOperand(bm, grid, tracers;
                                     difference_criterion, reference_depth)
    loc = (Center(), Center(), nothing)
    indices = (:, :, :)
    bcs = FieldBoundaryConditions(grid, loc)
    data = new_data(grid, loc, indices)
    return Field(loc, grid, data, bcs, indices, operand, FieldStatus())
end

function compute!(mld::MixedLayerDepthField, time=nothing)
    compute_mixed_layer_depth!(mld)
    #@apply_regionally compute_mixed_layer_depth!(mld)
    fill_halo_regions!(mld)
    return mld
end

function compute_mixed_layer_depth!(mld)
    grid = mld.grid
    arch = architecture(grid)

    # Pass the negative of `reference_depth` so the kernel sees the
    # target z-coordinate directly as `zʳ`.
    launch!(arch, mld.grid, :xy,
            _compute_mixed_layer_depth!,
            mld, grid,
            mld.operand.buoyancy_perturbation,
            mld.operand.difference_criterion,
            -mld.operand.reference_depth)

    return mld
end

const c = Center()
const f = Face()

@kernel function _compute_mixed_layer_depth!(mld, grid, b, Δb★, zʳ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = eltype(grid)

    # Bracket cells (k⁺ above, k⁻ below) of `zʳ`).
    zn = znodes(grid, Center())
    k⁺ = min(searchsortedfirst(zn, zʳ), Nz)
    k⁻ = max(k⁺ - 1, 1)
    z⁺ = @inbounds zn[k⁺]
    z⁻ = @inbounds zn[k⁻]

    # Reference buoyancy bN at z = zʳ
    b⁺ = @inbounds b[i, j, k⁺]
    b⁻ = @inbounds b[i, j, k⁻]
    w  = clamp((zʳ - z⁻) / max(z⁺ - z⁻, eps(FT)), zero(FT), one(FT))
    bN = b⁻ + w * (b⁺ - b⁻)

    # Descend from just below the bracket until Δb crosses Δb★.
    Δb       = zero(FT)
    mixed    = true
    nk       = 1
    k        = max(k⁻ - nk, 1)
    inactive = inactive_cell(i, j, k, grid)

    while !inactive & mixed & (nk < k⁻)
        Δb = bN - @inbounds(b[i, j, k])
        mixed    = Δb < Δb★
        nk      += 1
        k        = max(k⁻ - nk, 1)
        inactive = inactive_cell(i, j, k, grid)
    end

    # Linearly interpolate the crossing depth,
    zk = znode(i, j, k, grid, c, c, c)
    Δz = zʳ - zk
    z★ = zk - Δz/Δb * (Δb★ - Δb)
    # Special case when domain is one grid cell deep
    z★ = ifelse(Δb == 0, zʳ, z★)

    # Apply various criterion
    h = -z★
    h = max(h, zero(FT))
    H = static_column_depthᶜᶜᵃ(i, j, grid)
    h = min(h, H)

    @inbounds mld[i, j, 1] = h
end
