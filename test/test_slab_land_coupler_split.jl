using Test
using Oceananigans
using Oceananigans.Fields: interior, CenterField
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using NumericalEarth
using NumericalEarth.Lands: _split_vapor_flux!, _split_precip_by_temperature!

split_grid() = RectilinearGrid(CPU();
                               size = (1, 1), halo = (1, 1),
                               x = (0, 1), y = (0, 1),
                               topology = (Bounded, Bounded, Flat))

@testset "vapor flux split by snow_fraction and vegfrac" begin
    grid = split_grid()
    arch = architecture(grid)
    Jᵛ = CenterField(grid); fill!(Jᵛ, -1.0e-4)   # 0.1 g m⁻² s⁻¹ upward
    snow_fraction = CenterField(grid)
    vegfrac       = CenterField(grid)
    E   = CenterField(grid)
    Sub = CenterField(grid)
    Tt  = CenterField(grid)

    @testset "no snow, full vegetation: all to transpiration" begin
        fill!(snow_fraction, 0.0); fill!(vegfrac, 1.0)
        launch!(arch, grid, :xy, _split_vapor_flux!,
                E, Sub, Tt, Jᵛ, snow_fraction, vegfrac)
        @test interior(E)[1, 1, 1]   ≈ 0.0
        @test interior(Sub)[1, 1, 1] ≈ 0.0
        @test interior(Tt)[1, 1, 1]  ≈ 1.0e-4
    end

    @testset "full snow: all to sublimation" begin
        fill!(snow_fraction, 1.0); fill!(vegfrac, 0.5)
        launch!(arch, grid, :xy, _split_vapor_flux!,
                E, Sub, Tt, Jᵛ, snow_fraction, vegfrac)
        @test interior(Sub)[1, 1, 1] ≈ 1.0e-4
        @test interior(E)[1, 1, 1]   ≈ 0.0
        @test interior(Tt)[1, 1, 1]  ≈ 0.0
    end

    @testset "half snow / half veg: three-way split" begin
        fill!(snow_fraction, 0.5); fill!(vegfrac, 0.5)
        launch!(arch, grid, :xy, _split_vapor_flux!,
                E, Sub, Tt, Jᵛ, snow_fraction, vegfrac)
        @test interior(Sub)[1, 1, 1] ≈ 5.0e-5
        @test interior(E)[1, 1, 1]   ≈ 2.5e-5
        @test interior(Tt)[1, 1, 1]  ≈ 2.5e-5
    end
end

@testset "precipitation split by air temperature" begin
    grid = split_grid()
    arch = architecture(grid)
    water_vapor = CenterField(grid)
    T_air       = CenterField(grid)
    P_rain      = CenterField(grid)
    P_snow      = CenterField(grid)

    @testset "warm air → all rain" begin
        fill!(water_vapor, 2.0e-4)   # downward
        fill!(T_air, 280.0)
        launch!(arch, grid, :xy, _split_precip_by_temperature!,
                P_rain, P_snow, water_vapor, T_air)
        @test interior(P_rain)[1, 1, 1] ≈ 2.0e-4
        @test interior(P_snow)[1, 1, 1] ≈ 0.0
    end

    @testset "cold air → all snow" begin
        fill!(water_vapor, 2.0e-4)
        fill!(T_air, 270.0)
        launch!(arch, grid, :xy, _split_precip_by_temperature!,
                P_rain, P_snow, water_vapor, T_air)
        @test interior(P_snow)[1, 1, 1] ≈ 2.0e-4
        @test interior(P_rain)[1, 1, 1] ≈ 0.0
    end

    @testset "upward water vapor → no precipitation" begin
        fill!(water_vapor, -2.0e-4)   # evaporation, no precip
        fill!(T_air, 280.0)
        launch!(arch, grid, :xy, _split_precip_by_temperature!,
                P_rain, P_snow, water_vapor, T_air)
        @test interior(P_rain)[1, 1, 1] ≈ 0.0
        @test interior(P_snow)[1, 1, 1] ≈ 0.0
    end
end
