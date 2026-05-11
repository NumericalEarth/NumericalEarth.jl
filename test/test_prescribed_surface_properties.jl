using Test
using Oceananigans
using Oceananigans.Fields: interior, CenterField
using NumericalEarth
using NumericalEarth.Lands: SlabLand, SlabEnergy, ManabeBucket,
                            PrescribedSurfaceProperties,
                            albedo, emissivity, momentum_roughness_length

psp_grid() = RectilinearGrid(CPU();
                             size = (2, 1), halo = (1, 1),
                             x = (0, 2), y = (0, 1),
                             topology = (Bounded, Bounded, Flat))

psp_col(field) = Array(interior(field))[:, 1, 1]

@testset "PrescribedSurfaceProperties" begin
    grid = psp_grid()

    @testset "scalar defaults fill per-cell Fields" begin
        sfc = PrescribedSurfaceProperties(grid;
                                          albedo = 0.2,
                                          emissivity = 0.97,
                                          roughness_length = 0.1,
                                          vegfrac = 0.5,
                                          lai = 2.0,
                                          r_smin = 100.0,
                                          is_urban = 0.0)
        @test all(psp_col(sfc.albedo) .≈ 0.2)
        @test all(psp_col(sfc.emissivity) .≈ 0.97)
        @test all(psp_col(sfc.roughness_length) .≈ 0.1)
        @test all(psp_col(sfc.vegfrac) .≈ 0.5)
        @test all(psp_col(sfc.lai) .≈ 2.0)
        @test all(psp_col(sfc.r_smin) .≈ 100.0)
        @test all(psp_col(sfc.is_urban) .≈ 0.0)
    end

    @testset "accessors return the owned Fields" begin
        sfc = PrescribedSurfaceProperties(grid;
                                          albedo = 0.3, emissivity = 0.9,
                                          roughness_length = 0.05)
        land = SlabLand(grid; energy = SlabEnergy(),
                              hydrology = ManabeBucket(),
                              surface = sfc)
        @test albedo(land) === sfc.albedo
        @test emissivity(land) === sfc.emissivity
        @test momentum_roughness_length(land) === sfc.roughness_length
    end

    @testset "per-cell Field override" begin
        α_field = CenterField(grid); α_field[1, 1, 1] = 0.1; α_field[2, 1, 1] = 0.4
        sfc = PrescribedSurfaceProperties(grid; albedo = α_field)
        @test psp_col(sfc.albedo) == [0.1, 0.4]
    end
end
