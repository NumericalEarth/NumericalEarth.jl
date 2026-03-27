include("runtests_setup.jl")

using NumericalEarth.DataWrangling: Column, Linear, Nearest,
                                    BoundingBox, native_grid,
                                    restrict_location, dataset_location

using NumericalEarth.DataWrangling: extract_column!

using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnodes, φnodes, topology, Flat, Bounded, Periodic

@testset "extract_column! with Nearest interpolation" begin
    for arch in test_architectures
        A = typeof(arch)
        @testset "Nearest extraction on $A" begin
            # Create a LatitudeLongitudeGrid with spatially varying data
            intermediate_grid = LatitudeLongitudeGrid(arch;
                size = (4, 4, 2),
                longitude = (0, 4),
                latitude = (0, 4),
                z = (-20, 0))

            intermediate_field = CenterField(intermediate_grid)

            # Set distinct values at each horizontal point
            for i in 1:4, j in 1:4, k in 1:2
                @allowscalar intermediate_field[i, j, k] = 10 * i + j + 0.1 * k
            end
            fill_halo_regions!(intermediate_field)

            # Column near grid point (3, 2) → lon≈2.5, lat≈1.5
            col = Column(2.5, 1.5; interpolation=Nearest())
            column_grid = RectilinearGrid(arch;
                size = 2,
                x = 2.5,
                y = 1.5,
                z = (-20, 0),
                halo = 3,
                topology = (Flat, Flat, Bounded))

            column_field = Field{Nothing, Nothing, Center}(column_grid)

            extract_column!(column_field, intermediate_field, col, Nearest())

            # Find expected nearest indices
            λnodes_arr = λnodes(intermediate_grid, Center(); with_halos=false)
            φnodes_arr = φnodes(intermediate_grid, Center(); with_halos=false)
            i★ = argmin(abs.(λnodes_arr .- 2.5))
            j★ = argmin(abs.(φnodes_arr .- 1.5))

            @allowscalar begin
                for k in 1:2
                    @test column_field[1, 1, k] == intermediate_field[i★, j★, k]
                end
            end
        end

        @testset "Nearest extraction preserves vertical profile on $A" begin
            intermediate_grid = LatitudeLongitudeGrid(arch;
                size = (3, 3, 5),
                longitude = (10, 13),
                latitude = (40, 43),
                z = (-50, 0))

            intermediate_field = CenterField(intermediate_grid)

            # Set a vertical profile: value = depth level
            for k in 1:5
                interior(intermediate_field)[:, :, k] .= Float64(k)
            end
            fill_halo_regions!(intermediate_field)

            col = Column(11.5, 41.5; interpolation=Nearest())
            column_grid = RectilinearGrid(arch;
                size = 5,
                x = 11.5,
                y = 41.5,
                z = (-50, 0),
                halo = 3,
                topology = (Flat, Flat, Bounded))

            column_field = Field{Nothing, Nothing, Center}(column_grid)
            extract_column!(column_field, intermediate_field, col, Nearest())

            @allowscalar begin
                for k in 1:5
                    @test column_field[1, 1, k] == k
                end
            end
        end
    end
end

@testset "extract_column! dispatch routes on interpolation type" begin
    for arch in test_architectures
        A = typeof(arch)
        @testset "Dispatch on $A" begin
            intermediate_grid = LatitudeLongitudeGrid(arch;
                size = (4, 4, 2),
                longitude = (0, 4),
                latitude = (0, 4),
                z = (-20, 0))

            intermediate_field = CenterField(intermediate_grid)
            interior(intermediate_field) .= 42.0
            fill_halo_regions!(intermediate_field)

            column_grid = RectilinearGrid(arch;
                size = 2,
                x = 2.0,
                y = 2.0,
                z = (-20, 0),
                halo = 3,
                topology = (Flat, Flat, Bounded))

            # Column dispatch routes to the correct method
            col_nearest = Column(2.0, 2.0; interpolation=Nearest())
            cf = Field{Nothing, Nothing, Center}(column_grid)
            extract_column!(cf, intermediate_field, col_nearest)

            @allowscalar begin
                @test cf[1, 1, 1] == 42.0
                @test cf[1, 1, 2] == 42.0
            end
        end
    end
end

@testset "Column native_grid construction" begin
    @testset "ECCO4 Column grid" begin
        col = Column(35.1, 50.1)
        md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
        grid = native_grid(md)

        @test grid isa RectilinearGrid
        @test topology(grid) == (Flat, Flat, Bounded)
        _, _, Nz, _ = size(md)
        @test size(grid) == (1, 1, Nz)
    end

    @testset "ERA5 Column grid" begin
        col = Column(200.0, 35.0)
        md = Metadatum(:temperature; dataset=ERA5Hourly(),
                       date=DateTime(2020, 1, 1), region=col)
        grid = native_grid(md)

        @test grid isa RectilinearGrid
        @test topology(grid) == (Flat, Flat, Bounded)
        # ERA5 has z = (0, 1), single level
        @test size(grid) == (1, 1, 1)
    end

    @testset "Column grid uses Float32 for ECCO" begin
        col = Column(123.4, -45.6)
        md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
        grid = native_grid(md)

        # ECCO metadata has Float32 eltype
        @test eltype(grid) == Float32
    end
end
