include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Column, Linear
using Oceananigans.Fields: interpolate as oc_interpolate
using Oceananigans.Grids: topology, Bounded

@testset "JRA55 region support" begin
    arch = CPU()

    @testset "BoundingBox slices the right window" begin
        bbox = BoundingBox(longitude=(120, 240), latitude=(-30, 30))
        atm = JRA55PrescribedAtmosphere(arch;
                                        time_indices_in_memory=2,
                                        include_rivers_and_icebergs=false,
                                        region=bbox)
        Ta = atm.tracers.T
        # Coordinates of the field grid should fall inside the requested bbox.
        λnodes_T = λnodes(Ta.grid, Center())
        φnodes_T = φnodes(Ta.grid, Center())
        @test minimum(λnodes_T) ≥ 120 - 1.5  # 1.5° JRA55 spacing tolerance
        @test maximum(λnodes_T) ≤ 240 + 1.5
        @test minimum(φnodes_T) ≥ -30 - 1.5
        @test maximum(φnodes_T) ≤  30 + 1.5
        @test any(!iszero, interior(Ta))
        @test !any(isnan, interior(Ta))
        # Sub-360° span must be Bounded in x so halos do not wrap.
        @test topology(Ta.grid)[1] == Bounded
    end

    @testset "Column extracts a single point" begin
        col = Column(150.0, 0.0)  # equator, central Pacific
        atm = JRA55PrescribedAtmosphere(arch;
                                        time_indices_in_memory=2,
                                        include_rivers_and_icebergs=false,
                                        region=col)
        Ta = atm.tracers.T
        @test size(Ta.grid, 1) == 1
        @test size(Ta.grid, 2) == 1
        @test any(!iszero, interior(Ta))
        @test !any(isnan, interior(Ta))
    end

    @testset "Column matches bbox bilinear at the column point" begin
        # The Column dispatch should produce the same value as bilinearly
        # interpolating a bbox-extracted FTS at the column's (lon, lat).
        col_atm  = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2,
                                             include_rivers_and_icebergs=false,
                                             region=Column(150.0, 0.0))
        bbox_atm = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2,
                                             include_rivers_and_icebergs=false,
                                             region=BoundingBox(longitude=(148, 152),
                                                                latitude=(-2, 2)))

        Ta_col  = col_atm.tracers.T
        Ta_bbox = bbox_atm.tracers.T

        T_col_t1 = interior(Ta_col)[1, 1, 1, 1]
        loc = (Center(), Center(), Center())
        T_bbox_t1 = oc_interpolate((150.0, 0.0, 0.0), Ta_bbox[1], loc, Ta_bbox.grid)
        @test T_col_t1 ≈ T_bbox_t1  rtol = 1e-3
    end
end
