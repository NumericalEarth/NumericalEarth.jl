include("runtests_setup.jl")

using Oceananigans.Grids: φnodes

# A NOAA-formatted surface reference whose value is 400 + 10 sin φ + n at time step n,
# with uncertainty 0.1 + n.
function write_ghg_reference(path)
    sine_latitude = -1:0.05:1
    open(path, "w") do io
        println(io, "# NOAA GREENHOUSE GAS MBL SURFACE REFERENCE")
        for n in 0:1
            pairs = [string(400 + 10s + n, " ", 0.1 + n) for s in sine_latitude]
            println(io, 1979 + n / 48, " ", join(pairs, " "))
        end
    end
    return path
end

for arch in test_architectures
    A = typeof(arch)

    @testset "$A NOAA marine boundary layer reference" begin
        dir = mktempdir()
        dataset = NOAAMarineBoundaryLayer()
        dates = all_dates(dataset, :carbon_dioxide)[1:2]
        metadatum = Metadatum(:carbon_dioxide; dataset, date=dates[2], dir)
        write_ghg_reference(metadata_path(metadatum))

        φ = asind.(-1:0.05:1)

        field = Field(metadatum, arch)
        @test location(field) == (Center, Face, Nothing)
        @test φnodes(field.grid, Face()) ≈ φ
        @test Array(interior(field))[1, :, 1] ≈ Float32.(401 .+ 10 .* sind.(φ))

        uncertainty = Field(Metadatum(:carbon_dioxide_uncertainty; dataset, date=dates[2], dir), arch)
        @test all(Array(interior(uncertainty)) .≈ 1.1f0)

        grid = LatitudeLongitudeGrid(arch; size=(8, 18, 1), longitude=(0, 360), latitude=(-90, 90), z=(-10, 0))
        regridded = Field(metadatum, grid)
        values = Array(interior(regridded))
        @test all(values[:, 1, 1] .≈ 391)
        @test all(values[:, end, 1] .≈ 411)

        fts = FieldTimeSeries(Metadata(:carbon_dioxide; dataset, dates, dir), arch)
        @test Array(interior(fts[1]))[1, end, 1] ≈ 410
        @test Array(interior(fts[2]))[1, end, 1] ≈ 411
    end
end
