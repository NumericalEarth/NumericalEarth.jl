include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Column, metadata_path
using Oceananigans: λnodes, φnodes, Center, interior
using Oceananigans.Fields: Field
using NCDatasets
using Dates

@testset "Cross-dataset region support (snapshot path)" begin
    arch = CPU()

    @testset "ECCO4 BoundingBox loads the right window" begin
        # ECCO4Monthly stores longitude on [-180, 180]; pick a bbox far from
        # the file's SW corner (which is the Antarctic ocean, all zeros) so
        # that positional vs. coordinate indexing differ.
        bbox = BoundingBox(longitude=(-60, 60), latitude=(-30, 30))
        md = Metadatum(:temperature; dataset=ECCO4Monthly(),
                       date=DateTime(1993, 1, 1), region=bbox)
        f = Field(md, arch; inpainting=nothing, cache_inpainted_data=false)

        # Grid coordinates must fall inside the requested bbox (with a 1°
        # tolerance for ECCO4's 0.5° spacing).
        λg = λnodes(f.grid, Center())
        φg = φnodes(f.grid, Center())
        @test minimum(λg) ≥ -60 - 1.0
        @test maximum(λg) ≤  60 + 1.0
        @test minimum(φg) ≥ -30 - 1.0
        @test maximum(φg) ≤  30 + 1.0
        @test any(!iszero, interior(f))

        # Correctness: hand-extract the reference value at (0°, 0°, surface)
        # directly from the NetCDF file and compare against the bbox-restricted
        # field at the same physical point. This catches the silent SW-corner
        # positional-indexing bug in `set_metadata_field!`.
        path = metadata_path(md)
        ds = Dataset(path)
        # ECCO4 surface is k=1 in the raw file (Z=-5 m). After
        # `retrieve_data` reverses dims=3, surface becomes k=end in
        # `interior(f)`.
        T_full = ds["THETA"][:, :, 1, 1]
        λfile  = ds["longitude"][:]
        φfile  = ds["latitude"][:]
        close(ds)
        i★ = argmin(abs.(λfile .- 0))
        j★ = argmin(abs.(φfile .- 0))
        T_ref = T_full[i★, j★]
        # Same physical point in the bbox-restricted field.
        i_grid = argmin(abs.(λg .- 0))
        j_grid = argmin(abs.(φg .- 0))
        T_field = interior(f)[i_grid, j_grid, end]
        @test T_field ≈ T_ref  rtol=1e-3
    end

    @testset "ECCO4 Column extracts a single point" begin
        col = Column(150.0, 0.0)
        md = Metadatum(:temperature; dataset=ECCO4Monthly(),
                       date=DateTime(1993, 1, 1), region=col)
        f = Field(md, arch; inpainting=nothing, cache_inpainted_data=false)
        @test size(f.grid, 1) == 1
        @test size(f.grid, 2) == 1
        @test any(!iszero, interior(f))
    end
end
