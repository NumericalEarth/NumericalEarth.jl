include("runtests_setup.jl")

using NumericalEarth.DataWrangling: native_grid
using Oceananigans.OrthogonalSphericalShellGrids: lcc_forward, lcc_inverse

@testset "Conformal conic box geometry" begin
    for latitude in (-40, 40), orientation in (-20, 0, 20)
        box = ConformalConicBox(; origin=(170, latitude), orientation, extent=(4e6, 3e6))
        bounds = BoundingBox(box)
        south, north = bounds.latitude
        @test south < box.standard_parallels[1] < box.standard_parallels[2] < north

        grid = ConformalConicGrid(box; size=(12, 10, 1), z=(0, 1))
        projection = grid.conformal_mapping
        center_x, center_y = lcc_forward(projection, box.origin...)
        @test projection.x₁ ≈ center_x - box.extent[1] / 2
        @test projection.y₁ ≈ center_y - box.extent[2] / 2

        # Sample the continuous boundary, independently of grid resolution.
        for fraction in range(-1/2, 1/2; length=101)
            for (x, y) in ((fraction * box.extent[1], -box.extent[2] / 2),
                           (fraction * box.extent[1],  box.extent[2] / 2),
                           (-box.extent[1] / 2, fraction * box.extent[2]),
                           ( box.extent[1] / 2, fraction * box.extent[2]))
                longitude, query_latitude = lcc_inverse(projection, center_x + x, center_y + y)
                longitude = box.origin[1] + mod(longitude - box.origin[1] + 180, 360) - 180
                @test bounds.longitude[1] - 1e-10 <= longitude <= bounds.longitude[2] + 1e-10
                @test south - 1e-10 <= query_latitude <= north + 1e-10
            end
        end

        padded = BoundingBox(box; padding=1)
        @test padded.longitude == bounds.longitude .+ (-1, 1)
        @test padded.latitude == bounds.latitude .+ (-1, 1)
        @test bounds.longitude[2] > 180
        @test bounds.longitude[2] - bounds.longitude[1] < 90
    end

    box = ConformalConicBox(origin=(-105, 40), extent=(4e6, 3e6), standard_parallels=(30, 60))
    grid = ConformalConicGrid(box; size=(8, 6, 1), z=(0, 1))
    projection = grid.conformal_mapping
    north_y = projection.y₁ + box.extent[2]
    _, north = lcc_inverse(projection, 0, north_y)
    _, corner_latitude = lcc_inverse(projection, box.extent[1] / 2, north_y)
    @test BoundingBox(box).latitude[2] ≈ north
    @test north > corner_latitude + 1
    @test box.standard_parallels == (30, 60)
    @test startswith(sprint(show, box), "ConformalConicBox(origin=(-105.0, 40.0)")

    @test_throws ArgumentError ConformalConicBox(origin=(0, 0), extent=(1e5, 1e5))
    @test_throws ArgumentError ConformalConicBox(origin=(0, 40), extent=(-1e5, 1e5))
    @test_throws ArgumentError ConformalConicBox(origin=(0, 40), extent=(1e8, 1e8))
end

for arch in test_architectures
    @testset "Conic box to grid [$(typeof(arch))]" begin
        for latitude in (-40, 40), orientation in (-20, 0, 20)
            box = ConformalConicBox(; origin=(-105, latitude), orientation, extent=(2000, 1600))
            grid = ConformalConicGrid(box, arch; size=(9, 7, 1), z=(0, 1))
            longitudes = CenterField(grid)
            latitudes = CenterField(grid)
            set!(longitudes, (λ, φ, z) -> λ)
            set!(latitudes, (λ, φ, z) -> φ)
            longitude = Array(interior(longitudes))[:, :, 1]
            latitude_nodes = Array(interior(latitudes))[:, :, 1]
            i, j = (size(grid)[1:2] .+ 1) .÷ 2
            @test longitude[i, j] ≈ box.origin[1]
            @test latitude_nodes[i, j] ≈ box.origin[2]
            east = cosd(box.origin[2]) * (longitude[i+1, j] - longitude[i-1, j])
            north = latitude_nodes[i+1, j] - latitude_nodes[i-1, j]
            @test atand(north, east) ≈ orientation atol=1e-6

            projection = grid.conformal_mapping
            projected_x = CenterField(grid)
            set!(projected_x, (λ, φ, z) -> first(lcc_forward(projection, λ, φ)))
            x = Array(interior(projected_x))[:, :, 1]
            expected = projection.x₁ .+ ((1:size(grid, 1)) .- 1/2) .* projection.Δx
            @test x ≈ repeat(expected, 1, size(grid, 2)) atol=1e-7
        end
        box = ConformalConicBox(Float32; origin=(-105, 40), extent=(2e5, 2e5))
        grid = ConformalConicGrid(box, arch; size=(4, 4, 1), z=(0, 1))
        @test eltype(grid) === Float32
    end
end

@testset "Conic box to dataset region" begin
    box = ConformalConicBox(origin=(-105, 40), orientation=15, extent=(4e5, 3e5))
    bounds = BoundingBox(box)
    date = DateTime(2005, 2, 16)
    for dataset in (ERA5HourlySingleLevel(), ECCO4Monthly())
        from_box = Metadatum(:temperature; dataset, date, region=box)
        from_bounds = Metadatum(:temperature; dataset, date, region=bounds)
        @test from_box.region == bounds
        @test from_box.filename == from_bounds.filename
        @test size(native_grid(from_box)) == size(native_grid(from_bounds))

        dates = [date, date + Hour(1)]
        metadata = Metadata(:temperature; dataset, dates, region=box)
        @test metadata.region == bounds
        @test metadata[1].region == bounds
        collection = MetadataSet(:temperature; dataset, date, region=box)
        @test collection.temperature.region == bounds
    end

    dataset = SyntheticAtmosphere()
    metadata = Metadatum(:sea_level_pressure; dataset, date=DateTime(2000, 1, 1), region=box)
    grid = ConformalConicGrid(box; size=(8, 6, 1), z=(0, 1))
    pressure = CenterField(grid)
    set!(pressure, metadata; inpainting=nothing, halo=(1, 1, 1))
    @test all(value -> isapprox(value, 101325; rtol=eps(Float32)), Array(interior(pressure)))
end
