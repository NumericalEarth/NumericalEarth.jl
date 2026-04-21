using Oceananigans.Grids: static_column_depthᶜᶜᵃ, znode

# de Boyer Montégut et al. (2004) DR003 threshold:
#   Δσ = 0.03 kg m⁻³ at ρ₀ = 1025 kg m⁻³ ⇒ Δb ≈ 2.87 × 10⁻⁴ m s⁻².
# The reference depth is left at the surface cell here — a 10 m reference
# would require horizontal interpolation that isn't defined on
# OrthogonalSphericalShellGrid, so we stay with the surface convention.
const DR003_BUOYANCY_CRITERION = 2.87e-4

mutable struct MixedLayerDepthOperand{FT, B}
    buoyancy_perturbation :: B
    difference_criterion :: FT
end

Base.summary(mldo::MixedLayerDepthOperand) = "MixedLayerDepthOperand"

function MixedLayerDepthOperand(bm, grid, tracers;
                                difference_criterion = DR003_BUOYANCY_CRITERION)
    buoyancy_perturbation = buoyancy_operation(bm, grid, tracers)
    difference_criterion = convert(eltype(grid), difference_criterion)
    return MixedLayerDepthOperand(buoyancy_perturbation, difference_criterion)
end

const MixedLayerDepthField = Field{<:Any, <:Any, <:Any, <:MixedLayerDepthOperand}

"""
    MixedLayerDepthField(bm, grid, tracers; difference_criterion = 2.87e-4)

Mixed-layer-depth diagnostic using a buoyancy-difference criterion. The
default `difference_criterion = 2.87e-4` m s⁻² corresponds to the de Boyer
Montégut et al. (2004) DR003 threshold `Δσ = 0.03 kg m⁻³` at
`ρ₀ = 1025 kg m⁻³`. The reference buoyancy is the surface cell.
"""
function MixedLayerDepthField(bm, grid, tracers;
                              difference_criterion = DR003_BUOYANCY_CRITERION)
    operand = MixedLayerDepthOperand(bm, grid, tracers; difference_criterion)
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

    launch!(arch, grid, :xy,
            _compute_mixed_layer_depth!,
            mld, grid,
            mld.operand.buoyancy_perturbation,
            mld.operand.difference_criterion)

    return mld
end

const c = Center()
const f = Face()

@kernel function _compute_mixed_layer_depth!(mld, grid, b, Δb★)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    Δb = zero(grid)
    bN = @inbounds b[i, j, Nz]
    mixed = true
    minus_k = 1
    k = Nz - 1
    inactive = inactive_cell(i, j, k, grid)

    # Run minus_k forward to facilitate Reactantification
    while !inactive & mixed & (minus_k < Nz - 1)
        Δb = @inbounds bN - b[i, j, k]
        mixed = Δb < Δb★
        minus_k += 1
        k = Nz - minus_k
        inactive = inactive_cell(i, j, k, grid)
    end

    # Linearly interpolate
    # z★ = zk + Δz/Δb * (Δb★ - Δb)
    zN = znode(i, j, Nz, grid, c, c, c)
    zk = znode(i, j, k, grid, c, c, c)
    Δz = zN - zk
    z★ = zk - Δz / Δb * (Δb★ - Δb)

    # Special case when domain is one grid cell deep
    z★ = ifelse(Δb == 0, zN, z★)

    # Apply various criterion
    h = -z★
    h = max(h, zero(grid))
    H = static_column_depthᶜᶜᵃ(i, j, grid)
    h = min(h, H)

    @inbounds mld[i, j, 1] = h
end
