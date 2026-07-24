#####
##### Apply the drag-partition closure over a grid: (LAI, IGBP land cover, latitude) →
##### momentum roughness length z0 and zero-plane displacement d0. Vegetated cells run the
##### closure; non-vegetated cells take the prescribed Table-3 constants.
#####

using Oceananigans: Center, launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: φnode
using Oceananigans.Fields: Field, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using KernelAbstractions: @kernel, @index

const VON_KARMAN_CONSTANT = 0.4
const SUBLAYER_INFLUENCE = 0.193   # ψₕ (Raupach 1995)
const CLOSURE_ITERATIONS = 20
const MAXIMUM_VALID_LAI = 10       # physical LAI ceiling; larger values are fill/artifacts

@kernel function _compute_canopy_roughness!(z0m, d0, Λ, land_cover, grid, κ, ψₕ, iterations)
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
    h = class_canopy_height(FT, igbp)

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

Fill `z0m` and `d0` (metres) in place from a single-period leaf-area-index field `Λ` and a
static IGBP `land_cover` field on `grid`, applying the drag-partition closure over
vegetated cells and the prescribed constants elsewhere.
"""
function compute_canopy_roughness!(z0m, d0, Λ, land_cover, grid;
                                   von_karman_constant = VON_KARMAN_CONSTANT,
                                   sublayer_influence = SUBLAYER_INFLUENCE,
                                   iterations = CLOSURE_ITERATIONS)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_canopy_roughness!,
            z0m, d0, Λ, land_cover, grid,
            convert(eltype(grid), von_karman_constant),
            convert(eltype(grid), sublayer_influence),
            iterations)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Build cyclic climatologies of momentum roughness length `z0m` and zero-plane displacement
`d0` from a leaf-area-index `FieldTimeSeries` `lai` (one seasonal cycle of periods) and a
static IGBP `land_cover` field. Returns `(z0m, d0)` as `FieldTimeSeries` sharing `lai`'s
grid and times, so both are consumable per cell by the surface-flux solver.
"""
function canopy_roughness_climatology(lai::FieldTimeSeries, land_cover;
                                      von_karman_constant = VON_KARMAN_CONSTANT,
                                      sublayer_influence = SUBLAYER_INFLUENCE,
                                      iterations = CLOSURE_ITERATIONS)
    grid  = lai.grid
    times = lai.times
    LX, LY = Center, Center

    z0m = FieldTimeSeries{LX, LY, Nothing}(grid, times)
    d0  = FieldTimeSeries{LX, LY, Nothing}(grid, times)

    for n in eachindex(times)
        compute_canopy_roughness!(z0m[n], d0[n], lai[n], land_cover, grid;
                                  von_karman_constant, sublayer_influence, iterations)
    end

    return z0m, d0
end
