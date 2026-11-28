using Oceananigans.Grids: static_column_depthᶜᶜᵃ

# de Boyer Montégut et al. (2004) DR003 convention:
# - threshold: Δσ = 0.03 kg m⁻³ at ρ₀ = 1025 kg m⁻³ ⇒ Δb ≈ 2.87 × 10⁻⁴ m s⁻².
# - reference_depth: 10 m, with buoyancy obtained by vertical interpolation
const DR003_BUOYANCY_CRITERION = 2.87e-4
const DR003_REFERENCE_DEPTH    = 10.0

mutable struct MixedLayerDepthOperand{FT, B}
    potential_buoyancy :: B
    difference_criterion :: FT
    reference_depth :: FT
end

Base.summary(mldo::MixedLayerDepthOperand) = "MixedLayerDepthOperand"

@inline function potential_buoyancyᶜᶜᶜ(i, j, k, grid, bm, tracers)
    T, S = get_temperature_and_salinity(bm, tracers)
    θ, sᴬ = θ_and_sᴬ(i, j, k, T, S)
    equation_of_state = bm.equation_of_state
    ρ′₀ = ρ′(θ, sᴬ, zero(grid), equation_of_state)
    return - bm.gravitational_acceleration * ρ′₀ / equation_of_state.reference_density
end

"""
$(SIGNATURES)

The buoyancy that the water of each cell has at the surface. Its difference between two depths leaves out the
compressibility that makes the buoyancy perturbation of a single water mass vary with depth.
"""
potential_buoyancy(bm, grid, tracers) = buoyancy_operation(bm, grid, tracers)
potential_buoyancy(bm::BuoyancyForce, grid, tracers) = potential_buoyancy(bm.formulation, grid, tracers)
potential_buoyancy(bm::SeawaterBuoyancy{<:Any, <:BoussinesqEquationOfState}, grid, tracers) =
    KernelFunctionOperation{Center, Center, Center}(potential_buoyancyᶜᶜᶜ, grid, bm, tracers)

function MixedLayerDepthOperand(bm, grid, tracers;
                                difference_criterion = DR003_BUOYANCY_CRITERION,
                                reference_depth = DR003_REFERENCE_DEPTH)
    FT = eltype(grid)
    return MixedLayerDepthOperand(potential_buoyancy(bm, grid, tracers),
                                  convert(FT, difference_criterion),
                                  convert(FT, reference_depth))
end

const MixedLayerDepthField = Field{<:Any, <:Any, <:Any, <:MixedLayerDepthOperand}

"""
$(SIGNATURES)

Mixed-layer-depth diagnostic using the [de Boyer Montégut et al. (2004)](@cite de_boyer_montegut2004mixed)
DR003 criterion: the depth at which the potential density referenced to the surface exceeds its value at
`reference_depth` by `Δσ = 0.03 kg m⁻³`. The default `difference_criterion = 2.87e-4` m s⁻² is that
difference as a buoyancy at `ρ₀ = 1025 kg m⁻³`. The default `reference_depth = 10` m is the
standard dBM reference; buoyancy at that depth is obtained by linear interpolation between adjacent
cell centers, so the diagnostic is insensitive to the vertical resolution near the surface. The depth
of the criterion is interpolated between the first cell that meets it and the cell above.

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

function Oceananigans.Fields.compute!(mld::MixedLayerDepthField, time=nothing)
    compute_mixed_layer_depth!(mld)
    # @apply_regionally compute_mixed_layer_depth!(mld)
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
            mld.operand.potential_buoyancy,
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

    # Reference depth and mixed layer depth are measured below the free surface η
    η    = znode(i, j, Nz+1, grid, c, c, f)
    zᵣ   = η + zʳ

    # Bracket cells (k⁺ above, k⁻ below) of `zᵣ`. A descending sweep replaces
    # `searchsortedfirst`, which dispatches into non-GPU-compilable methods.
    k⁺ = Nz
    @inbounds for k in Nz:-1:1
        zₖ = znode(i, j, k, grid, c, c, c)
        k⁺ = ifelse(zₖ ≥ zᵣ, k, k⁺)
    end
    k⁻ = max(k⁺ - 1, 1)
    z⁺ = znode(i, j, k⁺, grid, c, c, c)
    z⁻ = znode(i, j, k⁻, grid, c, c, c)

    # Reference buoyancy bN at z = zʳ
    b⁺ = @inbounds b[i, j, k⁺]
    b⁻ = @inbounds b[i, j, k⁻]
    w  = clamp((zᵣ - z⁻) / max(z⁺ - z⁻, eps(FT)), zero(FT), one(FT))
    bN = b⁻ + w * (b⁺ - b⁻)

    # Descend from `k⁻` (first cell below `zʳ`) to the first wet cell where Δb reaches Δb★, and
    # interpolate between that cell and the level above it, which for `k⁻` is the reference level.
    # Without such a cell the column is mixed.
    h = η + static_column_depthᶜᶜᵃ(i, j, grid)
    zₖ₊₁ = min(zᵣ, z⁺)
    Δbₖ₊₁ = zero(FT)
    mixed = true

    @inbounds for k in k⁻:-1:1
        zₖ = znode(i, j, k, grid, c, c, c)
        Δb = bN - b[i, j, k]
        wet = !inactive_cell(i, j, k, grid)
        crossing = mixed & wet & (Δb ≥ Δb★)
        z★ = zₖ₊₁ + (zₖ - zₖ₊₁) * (Δb★ - Δbₖ₊₁) / (Δb - Δbₖ₊₁)
        h = ifelse(crossing, η - z★, h)
        mixed = mixed & wet & !crossing
        zₖ₊₁ = ifelse(mixed, zₖ, zₖ₊₁)
        Δbₖ₊₁ = ifelse(mixed, Δb, Δbₖ₊₁)
    end

    @inbounds mld[i, j, 1] = h
end
