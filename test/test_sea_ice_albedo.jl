include("runtests_setup.jl")

using NumericalEarth: stateindex
using NumericalEarth.EarthSystemModels.InterfaceComputations: SeaIceAlbedo,
                                                              InterfaceProperties,
                                                              Radiation

using Oceananigans.Units: Time

@testset "SeaIceAlbedo" begin
    for arch in test_architectures
        A = typeof(arch)

        grid = RectilinearGrid(arch;
                               size = (4, 4, 1),
                               extent = (1, 1, 1),
                               topology = (Periodic, Periodic, Bounded))

        hi = Field{Center, Center, Nothing}(grid)
        hs = Field{Center, Center, Nothing}(grid)
        Ts = Field{Center, Center, Nothing}(grid)

        @testset "Constructor [$A]" begin
            α = SeaIceAlbedo(hi, hs, Ts)
            @test α isa SeaIceAlbedo{Float64}
            @test α.ice_albedo == 0.54
            @test α.snow_albedo == 0.83
            @test α.ocean_albedo == 0.06

            α32 = SeaIceAlbedo(hi, hs, Ts; FT = Float32)
            @test α32 isa SeaIceAlbedo{Float32}
        end

        @testset "Nothing snow thickness [$A]" begin
            α = SeaIceAlbedo(hi, nothing, Ts)
            @test α.snow_thickness === nothing
        end

        @testset "Cold thick ice, no snow [$A]" begin
            # Thick ice, well below melting: should give cold bare-ice albedo
            set!(hi, 2.0)  # 2 m (>> minimum_ice_thickness = 0.5)
            set!(hs, 0.0)
            set!(Ts, -20.0) # well below melting

            α = SeaIceAlbedo(hi, hs, Ts)
            time = Time(0.0)
            loc = (Center, Center, Center)

            @allowscalar begin
                val = stateindex(α, 1, 1, 1, grid, time, loc)
                # With Ts = -20, fT = clamp((-20 - 0 + 1)/1, 0, 1) = 0
                # αi = 0.54 - 0 = 0.54, no thin-ice effect (fh=1), no snow (fs=0)
                @test val ≈ 0.54
            end
        end

        @testset "Cold thick ice, deep snow [$A]" begin
            set!(hi, 2.0)
            set!(hs, 0.1)   # >> minimum_snow_depth = 0.02
            set!(Ts, -20.0)

            α = SeaIceAlbedo(hi, hs, Ts)
            time = Time(0.0)
            loc = (Center, Center, Center)

            @allowscalar begin
                val = stateindex(α, 1, 1, 1, grid, time, loc)
                # fs = clamp(0.1/0.02, 0, 1) = 1 → full snow albedo
                @test val ≈ 0.83
            end
        end

        @testset "Melting ice, no snow [$A]" begin
            set!(hi, 2.0)
            set!(hs, 0.0)
            set!(Ts, 0.0) # at melting point

            α = SeaIceAlbedo(hi, hs, Ts)
            time = Time(0.0)
            loc = (Center, Center, Center)

            @allowscalar begin
                val = stateindex(α, 1, 1, 1, grid, time, loc)
                # fT = clamp((0 - 0 + 1)/1, 0, 1) = 1
                # αi = 0.54 - 0.075 = 0.465
                @test val ≈ 0.54 - 0.075
            end
        end

        @testset "Thin ice transitions to ocean albedo [$A]" begin
            set!(hi, 0.0)  # no ice
            set!(hs, 0.0)
            set!(Ts, -10.0)

            α = SeaIceAlbedo(hi, hs, Ts)
            time = Time(0.0)
            loc = (Center, Center, Center)

            @allowscalar begin
                val = stateindex(α, 1, 1, 1, grid, time, loc)
                # fh = clamp(0/0.5, 0, 1) = 0 → pure ocean albedo
                @test val ≈ 0.06
            end
        end

        @testset "Partial snow cover [$A]" begin
            set!(hi, 2.0)
            set!(hs, 0.01) # half of minimum_snow_depth
            set!(Ts, -20.0)

            α = SeaIceAlbedo(hi, hs, Ts)
            time = Time(0.0)
            loc = (Center, Center, Center)

            @allowscalar begin
                val = stateindex(α, 1, 1, 1, grid, time, loc)
                # fs = clamp(0.01/0.02, 0, 1) = 0.5
                # αi = 0.54, αs = 0.83
                @test val ≈ 0.5 * 0.83 + 0.5 * 0.54
            end
        end

        @testset "Adapt.adapt_structure [$A]" begin
            α = SeaIceAlbedo(hi, hs, Ts)
            # Adapt should not error
            adapted = Adapt.adapt_structure(Array, α)
            @test adapted isa SeaIceAlbedo
            @test adapted.ice_albedo == α.ice_albedo
        end

        @testset "InterfaceProperties Adapt [$A]" begin
            σ = 5.67e-8
            α = SeaIceAlbedo(hi, hs, Ts)
            ϵ = 1.0
            radiation = (; σ, α, ϵ)
            props = InterfaceProperties(radiation, nothing, nothing, nothing)
            adapted = Adapt.adapt_structure(Array, props)
            @test adapted.radiation.α isa SeaIceAlbedo
        end
    end
end
