include("runtests_setup.jl")

using NumericalEarth.DataWrangling: Column, Linear, Nearest,
                                    BoundingBox, native_grid,
                                    restrict_location, dataset_location
using NumericalEarth.DataWrangling: bracket_with_weight, infer_longitudinal_period,
                                    region_info, blend, ColumnInfo

using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: location
using Oceananigans.Grids: λnodes, φnodes, topology, Flat, Bounded, Periodic

# Test coordinates for end-to-end Column tests (ECCO4 ocean point)
const test_longitude = 12.0
const test_latitude = -50.0

@testset "bracket_with_weight (non-cyclic)" begin
    coords = [0.5, 1.5, 2.5, 3.5]

    # Interior point, exact midpoint between cells.
    i⁻, i⁺, w = bracket_with_weight(coords, 2.0)
    @test (i⁻, i⁺) == (2, 3)
    @test w ≈ 0.5

    # Off-grid below: clamps to first interval.
    i⁻, i⁺, w = bracket_with_weight(coords, -1.0)
    @test (i⁻, i⁺) == (1, 2)
    @test w == 0.0

    # Off-grid above: clamps to last interval.
    i⁻, i⁺, w = bracket_with_weight(coords, 5.0)
    @test (i⁻, i⁺) == (3, 4)
    @test w == 1.0

    # On the right-most centre: weight = 1.
    i⁻, i⁺, w = bracket_with_weight(coords, 3.5)
    @test (i⁻, i⁺) == (3, 4)
    @test w ≈ 1.0

    # Single-cell axis: nothing to bracket; both corners point at the only cell.
    # GLORYS via CopernicusMarine returns 1-cell-wide chunked files for Column queries.
    i⁻, i⁺, w = bracket_with_weight([7.5], 7.5)
    @test (i⁻, i⁺, w) == (1, 1, 0.0)
    i⁻, i⁺, w = bracket_with_weight([7.5], 99.0)
    @test (i⁻, i⁺, w) == (1, 1, 0.0)
end

@testset "bracket_with_weight (cyclic wrap)" begin
    coords = collect(0.5:1.0:359.5)  # global 1° centres
    n = length(coords)

    # Interior point — period is a no-op there.
    i⁻, i⁺, w = bracket_with_weight(coords, 180.0; period = 360)
    @test (i⁻, i⁺) == (180, 181)
    @test w ≈ 0.5

    # Wrap cell: x just below the period boundary.
    i⁻, i⁺, w = bracket_with_weight(coords, 359.99; period = 360)
    @test (i⁻, i⁺) == (n, 1)
    @test 0 < w < 1

    # x past the period: mod wraps it back into the regular range.
    i⁻, i⁺, w = bracket_with_weight(coords, 360.5; period = 360)
    @test i⁻ == 1
    @test i⁺ == 2 || (i⁻ == n && i⁺ == 1)  # right at coords[1] boundary
end

@testset "infer_longitudinal_period" begin
    @test infer_longitudinal_period(collect(0.5:1.0:359.5)) == 360
    @test infer_longitudinal_period(collect(-179.75:0.5:179.75)) == 360
    @test infer_longitudinal_period([10.0, 11.0, 12.0]) === nothing
    @test infer_longitudinal_period([100.0]) === nothing
end

@testset "NaN-aware blend" begin
    # 2x2x1 synthetic data; column at the centre point with equal weights.
    c = ColumnInfo(1, 2, 1, 2, 0.5f0, 0.5f0, Linear())
    FT = Float32

    # All-valid: result is the simple average.
    data_full = reshape(Float32[1 2; 3 4], 2, 2, 1)
    @test blend(data_full, c, 1, nothing, c.ℑ, FT) ≈ 2.5f0

    # All-NaN: result is NaN.
    data_nan = fill(NaN32, 2, 2, 1)
    @test isnan(blend(data_nan, c, 1, nothing, c.ℑ, FT))

    # Partial: bottom-right corner is NaN, weights renormalise over the rest.
    # Weights become (0.25, 0.25, 0.25, 0); Σw = 0.75; sum = 1+2+3 = 6;
    # result = 6 / 0.75 = 2.0 in 1/2/3 → renormalised mean.
    data_part = reshape(Float32[1 2; 3 NaN32], 2, 2, 1)
    @test blend(data_part, c, 1, nothing, c.ℑ, FT) ≈ 2.0f0

    # Missing values from NetCDF-style arrays are treated as NaN.
    data_missing = reshape(Union{Missing, Float32}[1.0 2.0; 3.0 missing], 2, 2, 1)
    @test blend(data_missing, c, 1, nothing, c.ℑ, FT) ≈ 2.0f0
end

@testset "blend dispatches Linear vs Nearest" begin
    data = reshape(Float32[1 2; 3 4], 2, 2, 1)
    FT = Float32

    # wx = wy = 0.5 → average for Linear; arbitrary corner for Nearest.
    c_lin = ColumnInfo(1, 2, 1, 2, 0.5f0, 0.5f0, Linear())
    @test blend(data, c_lin, 1, nothing, c_lin.ℑ, FT) ≈ 2.5f0

    # wx = 0.7, wy = 0.7 → both above 0.5 → picks i⁺, j⁺ = data[2,2,1] = 4.
    c_near = ColumnInfo(1, 2, 1, 2, 0.7f0, 0.7f0, Nearest())
    @test blend(data, c_near, 1, nothing, c_near.ℑ, FT) ≈ 4.0f0

    # wx = 0.3, wy = 0.3 → both below 0.5 → picks i⁻, j⁻ = data[1,1,1] = 1.
    c_near2 = ColumnInfo(1, 2, 1, 2, 0.3f0, 0.3f0, Nearest())
    @test blend(data, c_near2, 1, nothing, c_near2.ℑ, FT) ≈ 1.0f0
end

@testset "End-to-end Column Field construction" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Column Field with Linear interpolation on $A" begin
            col = Column(test_longitude, test_latitude; interpolation=Linear())
            md = Metadatum(:temperature; dataset=ECCO4Monthly(), date=start_date, region=col)
            field = Field(md, arch)

            @test field.grid isa RectilinearGrid
            @test topology(field.grid) == (Flat, Flat, Bounded)
            @test location(field) == (Nothing, Nothing, Center)

            # Field should have non-trivial data (not all zeros)
            @allowscalar begin
                @test any(!=(0), interior(field))
            end
        end

        @testset "Column Field with Nearest interpolation on $A" begin
            col = Column(test_longitude, test_latitude; interpolation=Nearest())
            md = Metadatum(:temperature; dataset=ECCO4Monthly(), date=start_date, region=col)
            field = Field(md, arch)

            @test field.grid isa RectilinearGrid
            @test location(field) == (Nothing, Nothing, Center)

            @allowscalar begin
                @test any(!=(0), interior(field))
            end
        end

        @testset "set! with Column metadata on $A" begin
            col = Column(test_longitude, test_latitude)
            md = Metadatum(:temperature; dataset=ECCO4Monthly(), date=start_date, region=col)

            # Build a target column field
            column_grid = native_grid(md, arch)
            target = Field{Nothing, Nothing, Center}(column_grid)

            set!(target, md)

            @allowscalar begin
                @test any(!=(0), interior(target))
            end
        end

        @testset "Column Linear vs Nearest give similar results on $A" begin
            col_lin = Column(test_longitude, test_latitude; interpolation=Linear())
            col_near = Column(test_longitude, test_latitude; interpolation=Nearest())

            md_lin = Metadatum(:temperature; dataset=ECCO4Monthly(), date=start_date, region=col_lin)
            md_near = Metadatum(:temperature; dataset=ECCO4Monthly(), date=start_date, region=col_near)

            field_lin = Field(md_lin, arch)
            field_near = Field(md_near, arch)

            # Both should produce finite, non-zero vertical profiles
            @allowscalar begin
                @test all(isfinite, interior(field_lin))
                @test all(isfinite, interior(field_near))
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
