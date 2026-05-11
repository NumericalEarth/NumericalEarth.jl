using Test
using Oceananigans
using Oceananigans.Fields: interior
using NumericalEarth
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketWithSnow,
                            PrescribedSurfaceProperties, SnowModifiedSurface,
                            usgs_land_classifications, apply_land_classifications!

lc_grid() = RectilinearGrid(CPU();
                            size = (2, 1), halo = (1, 1),
                            x = (0, 2), y = (0, 1),
                            topology = (Bounded, Bounded, Flat))

@testset "apply_land_classifications! round-trip" begin
    grid = lc_grid()
    surface = PrescribedSurfaceProperties(grid)
    land = SlabLand(grid;
                    energy = SlabEnergy(),
                    hydrology = BucketWithSnow(),
                    surface = surface)

    registry = usgs_land_classifications(Float64)
    vegtype  = reshape([7, 13], 2, 1)   # grassland, evergreen broadleaf forest
    apply_land_classifications!(land, vegtype, registry)

    αv = Array(interior(surface.albedo))[:, 1, 1]
    @test αv[1] ≈ registry[7].albedo
    @test αv[2] ≈ registry[13].albedo
    @test Array(interior(surface.vegfrac))[1, 1, 1] ≈ registry[7].vegfrac
    @test Array(interior(surface.lai))[2, 1, 1]     ≈ registry[13].lai
end

@testset "apply_land_classifications! through SnowModifiedSurface" begin
    grid = lc_grid()
    base = PrescribedSurfaceProperties(grid)
    surface = SnowModifiedSurface(base; snow_albedo = 0.85)
    land = SlabLand(grid;
                    energy = SlabEnergy(),
                    hydrology = BucketWithSnow(),
                    surface = surface)

    registry = usgs_land_classifications(Float64)
    vegtype  = reshape([7, 13], 2, 1)
    apply_land_classifications!(land, vegtype, registry)

    @test Array(interior(base.albedo))[1, 1, 1] ≈ registry[7].albedo
    @test Array(interior(base.vegfrac))[2, 1, 1] ≈ registry[13].vegfrac
end
