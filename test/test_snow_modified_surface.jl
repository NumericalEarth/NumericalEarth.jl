using Test
using Oceananigans
using Oceananigans.Fields: interior, CenterField
using NumericalEarth
using NumericalEarth.Lands: SlabLand, SlabEnergy, ManabeBucket,
                            PrescribedSurfaceProperties, SnowModifiedSurface,
                            albedo, emissivity, momentum_roughness_length,
                            update_diagnostics!

sms_grid() = RectilinearGrid(CPU();
                             size = (1, 1), halo = (1, 1),
                             x = (0, 1), y = (0, 1),
                             topology = (Bounded, Bounded, Flat))

sms_cell(field) = Array(interior(field))[1, 1, 1]

@testset "SnowModifiedSurface" begin
    grid = sms_grid()
    base = PrescribedSurfaceProperties(grid;
                                       albedo = 0.2,
                                       emissivity = 0.95,
                                       roughness_length = 0.1)
    smod = SnowModifiedSurface(base;
                               snow_albedo = 0.8,
                               snow_emissivity = 0.99,
                               snow_roughness_length = 0.001)

    snow_fraction = CenterField(grid)
    parameters = nothing

    @testset "snow_fraction = 0 reduces to base" begin
        fill!(snow_fraction, 0.0)
        state = (; snow_fraction)
        update_diagnostics!(smod, state, NamedTuple(), smod, parameters, grid)
        @test sms_cell(albedo(smod, state, parameters)) ≈ 0.2
        @test sms_cell(emissivity(smod, state, parameters)) ≈ 0.95
        @test sms_cell(momentum_roughness_length(smod, state, parameters)) ≈ 0.1
    end

    @testset "snow_fraction = 1 reduces to snow end-member" begin
        fill!(snow_fraction, 1.0)
        state = (; snow_fraction)
        update_diagnostics!(smod, state, NamedTuple(), smod, parameters, grid)
        @test sms_cell(albedo(smod, state, parameters)) ≈ 0.8
        @test sms_cell(emissivity(smod, state, parameters)) ≈ 0.99
        @test sms_cell(momentum_roughness_length(smod, state, parameters)) ≈ 0.001
    end

    @testset "snow_fraction = 0.5 blends linearly" begin
        fill!(snow_fraction, 0.5)
        state = (; snow_fraction)
        update_diagnostics!(smod, state, NamedTuple(), smod, parameters, grid)
        @test sms_cell(albedo(smod, state, parameters)) ≈ 0.5
        @test sms_cell(emissivity(smod, state, parameters)) ≈ 0.97
        @test sms_cell(momentum_roughness_length(smod, state, parameters)) ≈ 0.0505
    end
end

@testset "SnowModifiedSurface requires snow_fraction in state" begin
    grid = sms_grid()
    base = PrescribedSurfaceProperties(grid; albedo = 0.2)
    smod = SnowModifiedSurface(base; snow_albedo = 0.8)

    # ManabeBucket does not declare :snow_fraction → SlabLand must reject.
    @test_throws ArgumentError SlabLand(grid;
                                         energy = SlabEnergy(),
                                         hydrology = ManabeBucket(),
                                         surface = smod)
end
