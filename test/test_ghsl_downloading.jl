include("runtests_setup.jl")

using ArchGDAL  # activates NumericalEarthArchGDALExt (the Mollweide → EPSG:4326 warp)

using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.DataWrangling.GHSL: GHSBuiltH, GHSBuiltS
using NumericalEarth.Lands: urban_roughness, MorphometricRoughness

# GHSL needs no credentials, but each tile archive is tens to hundreds of MB.
# Excluded from CI in runtests.jl; run manually.

# A dense window over the City of London, as in examples/ghsl_urban_roughness.jl, on the
# 100 m products (the example's 10 m built surface is a ~470 MB tile).
const ghsl_region = BoundingBox(longitude = (-0.11, -0.07), latitude = (51.505, 51.525))

@testset "Downloading GHSL built-up morphometry" begin
    fraction = Metadatum(:built_up_fraction; dataset = GHSBuiltS(), region = ghsl_region)
    height   = Metadatum(:building_height;   dataset = GHSBuiltH(), region = ghsl_region)

    # Start from the tiles: drop the regional NetCDFs (the tile archives stay cached) so a
    # stale file cannot stand in for the download.
    for metadatum in (fraction, height)
        rm(metadata_path(metadatum); force = true)
        download(metadatum)
        @test isfile(metadata_path(metadatum))
    end

    λᵖ = Field(fraction, CPU())
    h  = Field(height, λᵖ.grid)

    built_fraction = Array(interior(λᵖ, :, :, 1))
    building_height = Array(interior(h, :, :, 1))

    built = filter(!isnan, vec(built_fraction))
    @test !isempty(built)
    @test all(x -> 0 ≤ x ≤ 1, built)
    @test length(unique(built)) > 1   # a real window, not a constant fill
    @test maximum(built) > 0.2        # the City of London is densely built

    heights = filter(!isnan, vec(building_height))
    @test !isempty(heights)
    @test all(x -> 0 ≤ x ≤ 500, heights)
    @test maximum(heights) > 10

    # The urban closure consumes the two fields as in the example, returning physical
    # (ℓᵐ, d) wherever both inputs are valid.
    closure = MorphometricRoughness(eltype(λᵖ.grid))
    ℓᵐ, d = urban_roughness(h, λᵖ; closure)
    roughness = Array(interior(ℓᵐ, :, :, 1))
    displacement = Array(interior(d, :, :, 1))

    valid = @. !isnan(built_fraction) & !isnan(building_height)
    @test all(roughness[valid] .≥ closure.bare_soil_roughness)
    @test all(0 .≤ displacement[valid] .< building_height[valid])   # displacement stays below roof level
    @test maximum(roughness[valid]) > 0.1                           # the built core is aerodynamically rough
end
