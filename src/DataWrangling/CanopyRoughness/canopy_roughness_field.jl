#####
##### Apply the drag-partition closure over a grid: (LAI, IGBP land cover, canopy height) →
##### momentum roughness length z0 and zero-plane displacement d0. Vegetated cells run the
##### closure with the measured canopy height where valid and the class-average height
##### otherwise; non-vegetated cells take the prescribed Table-3 constants.
#####

using Oceananigans: Center, launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: AbstractGrid, φnode
using Oceananigans.OutputReaders: FieldTimeSeries
using KernelAbstractions: @kernel, @index

const VON_KARMAN_CONSTANT = 0.4
const SUBLAYER_INFLUENCE = 0.193   # ψₕ (Raupach 1995)
const CLOSURE_ITERATIONS = 20
const MAXIMUM_VALID_LAI = 10       # physical LAI ceiling; larger values are fill/artifacts

# Canopy height as a native per-cell input. A missing (`nothing`) height defers to the
# class-average height everywhere (NaN → not valid); a scalar broadcasts; anything
# indexable (a `Field`, `ConstantField`, array) is read per cell.
@inline canopy_height_at(::Nothing, i, j, FT) = convert(FT, NaN)
@inline canopy_height_at(h::Number, i, j, FT) = convert(FT, h)
@inline canopy_height_at(h, i, j, FT) = @inbounds convert(FT, h[i, j, 1])

@kernel function _compute_canopy_roughness!(z0m, d0, Λ, land_cover, canopy_height, grid, κ, ψₕ, iterations)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)

    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    @inbounds lcraw = land_cover[i, j, 1]
    @inbounds Λraw = convert(FT, Λ[i, j, 1])

    valid_cover = isfinite(lcraw) & (lcraw ≥ 0) & (lcraw ≤ 17)
    lcsafe = ifelse(valid_cover, lcraw, zero(lcraw))    # finite before round (ifelse eager)
    igbp = round(Int, lcsafe)
    valid_lai = isfinite(Λraw) & (Λraw ≥ 0) & (Λraw ≤ MAXIMUM_VALID_LAI)
    Λij = ifelse(valid_lai, Λraw, zero(FT))

    vegetated = is_vegetated(igbp)
    group = drag_partition_group(igbp, φ)
    p = canopy_drag_parameters(FT, max(group, 1))       # group=0 (non-veg) discarded below

    # Measured canopy height where finite and positive; the class-average height otherwise.
    hraw = canopy_height_at(canopy_height, i, j, FT)
    hclass = class_canopy_height(FT, igbp)
    valid_h = isfinite(hraw) & (hraw > 0)
    h = ifelse(valid_h, hraw, hclass)

    z0ᵛ, d0ᵛ = canopy_roughness(Λij, h, p, κ, ψₕ, iterations)
    z0ⁿ, d0ⁿ = nonvegetated_roughness(FT, igbp)

    # Vegetated cells with no valid LAI retrieval — and cells with no valid land cover —
    # become honest gaps (NaN); non-vegetated cells take their prescribed constants.
    gap = convert(FT, NaN)
    z0veg = ifelse(valid_lai, z0ᵛ, gap)
    d0veg = ifelse(valid_lai, d0ᵛ, gap)
    @inbounds z0m[i, j, 1] = ifelse(valid_cover, ifelse(vegetated, z0veg, z0ⁿ), gap)
    @inbounds d0[i, j, 1]  = ifelse(valid_cover, ifelse(vegetated, d0veg, d0ⁿ), gap)
end

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` (metres) in place from a leaf-area-index field `Λ`, a static IGBP
`land_cover` field, and a per-cell `canopy_height` on `grid`, applying the drag-partition
closure over vegetated cells and the prescribed constants elsewhere. `canopy_height`
is the measured height field (e.g. `ETHCanopyHeight`); it may also be a scalar or
`nothing`, in which case the class-average height is used wherever it is missing or
non-positive.
"""
function compute_canopy_roughness!(z0m, d0, Λ, land_cover, canopy_height, grid;
                                   von_karman_constant = VON_KARMAN_CONSTANT,
                                   sublayer_influence = SUBLAYER_INFLUENCE,
                                   iterations = CLOSURE_ITERATIONS)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_canopy_roughness!,
            z0m, d0, Λ, land_cover, canopy_height, grid,
            convert(eltype(grid), von_karman_constant),
            convert(eltype(grid), sublayer_influence),
            iterations)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` using the class-average canopy heights alone (no measured
height field). Equivalent to passing `nothing` as the canopy height, so the closure
is usable before a canopy-height dataset is wired in.
"""
compute_canopy_roughness!(z0m, d0, Λ, land_cover, grid::AbstractGrid; kw...) =
    compute_canopy_roughness!(z0m, d0, Λ, land_cover, nothing, grid; kw...)

"""
$(TYPEDSIGNATURES)

Build cyclic climatologies of momentum roughness length `z0m` and zero-plane displacement
`d0` from a leaf-area-index `FieldTimeSeries` `lai` (one seasonal cycle of periods), a
static IGBP `land_cover` field, and a static `canopy_height` (defaults to the class
heights). Returns `(z0m, d0)` as `FieldTimeSeries` sharing `lai`'s grid and times, so both
are consumable per cell by the surface-flux solver.
"""
function canopy_roughness_climatology(lai::FieldTimeSeries, land_cover, canopy_height = nothing;
                                      von_karman_constant = VON_KARMAN_CONSTANT,
                                      sublayer_influence = SUBLAYER_INFLUENCE,
                                      iterations = CLOSURE_ITERATIONS)
    grid  = lai.grid
    times = lai.times
    LX, LY = Center, Center

    z0m = FieldTimeSeries{LX, LY, Nothing}(grid, times)
    d0  = FieldTimeSeries{LX, LY, Nothing}(grid, times)

    for n in eachindex(times)
        compute_canopy_roughness!(z0m[n], d0[n], lai[n], land_cover, canopy_height, grid;
                                  von_karman_constant, sublayer_influence, iterations)
    end

    return z0m, d0
end
