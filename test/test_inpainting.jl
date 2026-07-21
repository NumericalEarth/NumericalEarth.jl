include("runtests_setup.jl")

using NumericalEarth.DataWrangling: inpaint_mask!, NearestNeighborInpainting

# Build a reduced (Center, Center, Nothing) field on a small lat/lon grid from a CPU
# value matrix (`NaN` marks a gap), plus the matching NaN mask and a Bool region-label
# field. Everything is transferred to `arch` so the kernel path is exercised on GPU too.
function build_gap_fields(arch, values, labels)
    Nx, Ny = size(values)
    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nx, Ny),
                                 longitude = (0, Nx),
                                 latitude = (0, Ny),
                                 topology = (Bounded, Bounded, Flat))

    field = Field{Center, Center, Nothing}(grid)
    mask  = Field{Center, Center, Nothing}(grid, Bool)
    regions = Field{Center, Center, Nothing}(grid, Bool)

    interior(field,   :, :, 1) .= on_architecture(arch, values)
    interior(mask,    :, :, 1) .= on_architecture(arch, isnan.(values))
    interior(regions, :, :, 1) .= on_architecture(arch, labels)

    return field, mask, regions
end

# A single class-A gap (label false) at the center, with class-A neighbors west/south
# (0.90) and class-B neighbors east/north (0.98). Surface-agnostic inpainting averages
# all four; the region partition keeps the fill on the gap's own class.
gap_values() = Float32[0.95 0.90 0.95
                       0.90 NaN  0.98
                       0.95 0.98 0.95]

gap_labels() = Bool[false false false
                    false false true
                    false true  false]

@testset "Region-gated inpainting [$(typeof(arch))]" for arch in test_architectures
    # Surface-agnostic (regions = nothing): the gap is the mean of all four valid
    # neighbors, exactly as the original algorithm.
    field, mask, _ = build_gap_fields(arch, gap_values(), gap_labels())
    inpaint_mask!(field, mask; inpainting = NearestNeighborInpainting(Inf))
    agnostic = Array(interior(field, :, :, 1))
    @test agnostic[2, 2] ≈ 0.94f0 atol = 1e-5   # (0.90 + 0.98 + 0.98 + 0.90) / 4
    @test all(isfinite, agnostic)

    # Surface-aware: only the same-class west + south neighbors donate, so the gap
    # takes the land value 0.90 rather than the cross-class mean 0.94.
    field, mask, regions = build_gap_fields(arch, gap_values(), gap_labels())
    inpaint_mask!(field, mask; inpainting = NearestNeighborInpainting(Inf), regions)
    aware = Array(interior(field, :, :, 1))
    @test aware[2, 2] ≈ 0.90f0 atol = 1e-5
    @test all(isfinite, aware)
end

@testset "Region-gated inpainting: isolated-gap fallback [$(typeof(arch))]" for arch in test_architectures
    # A lone class-A (false) gap ringed by class-B (true) cells: no same-class donor
    # is reachable, so the gated loop stalls and the ungated fallback fills it from the
    # surrounding class so the field stays finite.
    values = fill(0.98f0, 3, 3)
    values[2, 2] = NaN32
    labels = trues(3, 3)
    labels[2, 2] = false

    field, mask, regions = build_gap_fields(arch, values, labels)
    # Bounded maxiter so a regression in the stall guard can't hang CI.
    inpaint_mask!(field, mask; inpainting = NearestNeighborInpainting(50), regions)
    filled = Array(interior(field, :, :, 1))
    @test all(isfinite, filled)
    @test filled[2, 2] ≈ 0.98f0
end
