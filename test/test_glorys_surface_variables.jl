include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, z_interfaces, NearestNeighborInpainting
using NumericalEarth.DataWrangling.GLORYS: GLORYS_surface_variables, dataset_location, GLORYSMonthly, GLORYSDaily
using CopernicusMarine
using NCDatasets

const GLORYS_TEST_DIR = mktempdir()

# z_interfaces for surface-only variables must return (-1.0, 0.0) without
# attempting to open any NetCDF file (sea ice files have no depth coordinate).
@testset "z_interfaces returns surface pair for 2D GLORYS variables" begin
    date   = DateTime(2000, 1, 1)
    region = BoundingBox(longitude = (-10, 10), latitude = (50, 60))

    for dataset in (GLORYSMonthly(), GLORYSDaily())
        for var in (:free_surface, :sea_ice_thickness, :sea_ice_concentration,
                    :sea_ice_u_velocity, :sea_ice_v_velocity)
            md = Metadatum(var; dataset, region, date, dir = GLORYS_TEST_DIR)
            @test z_interfaces(md) === (-1.0, 0.0)
        end
    end
end

@testset "is_three_dimensional is false for 2D GLORYS variables" begin
    date   = DateTime(2000, 1, 1)
    region = BoundingBox(longitude = (-10, 10), latitude = (50, 60))

    for dataset in (GLORYSMonthly(), GLORYSDaily())
        for var in (:free_surface, :sea_ice_thickness, :sea_ice_concentration,
                    :sea_ice_u_velocity, :sea_ice_v_velocity)
            md = Metadatum(var; dataset, region, date, dir = GLORYS_TEST_DIR)
            @test !is_three_dimensional(md)
        end
        for var in (:temperature, :salinity, :u_velocity, :v_velocity)
            md = Metadatum(var; dataset, region, date, dir = GLORYS_TEST_DIR)
            @test is_three_dimensional(md)
        end
    end
end

@testset "dataset_location returns surface location for 2D GLORYS variables" begin
    for var in (:free_surface, :sea_ice_thickness, :sea_ice_concentration,
                :sea_ice_u_velocity, :sea_ice_v_velocity)
        @test dataset_location(GLORYSMonthly(), var) === (Center, Center, Nothing)
    end
    for var in (:temperature, :salinity)
        @test dataset_location(GLORYSMonthly(), var) === (Center, Center, Center)
    end
end

@testset "GLORYS_surface_variables contains expected variables" begin
    @test :free_surface           ∈ GLORYS_surface_variables
    @test :sea_ice_thickness      ∈ GLORYS_surface_variables
    @test :sea_ice_concentration  ∈ GLORYS_surface_variables
    @test :sea_ice_u_velocity     ∈ GLORYS_surface_variables
    @test :sea_ice_v_velocity     ∈ GLORYS_surface_variables
    @test :temperature            ∉ GLORYS_surface_variables
    @test :salinity               ∉ GLORYS_surface_variables
end

@testset "GLORYS sea ice is zero where the file has no ice" begin
    region = BoundingBox(longitude = (-60, -58), latitude = (50, 52))
    date   = DateTime(2013, 12, 1)
    dir    = mktempdir()
    grid   = LatitudeLongitudeGrid(size = (16, 16, 1), longitude = (-59.8, -58.2),
                                   latitude = (50.2, 51.8), z = (-10, 0))

    # Ice west of 59°W, and missing values (ice-free ocean) east of it
    λ = -60.5:1/12:-57.5
    φ = 49.5:1/12:52.5
    values = [λᵢ < -59 ? 0.8f0 : missing for λᵢ in λ, φⱼ in φ]
    west = λnodes(grid, Center()) .< -59.2
    east = λnodes(grid, Center()) .> -58.9

    for (name, variable) in ((:sea_ice_concentration, "siconc"), (:sea_ice_thickness, "sithick"))
        metadatum = Metadatum(name; dataset = GLORYSDaily(), region, date, dir)
        NCDataset(metadata_path(metadatum), "c") do ds
            defVar(ds, "longitude", Float32.(λ), ("longitude",))
            defVar(ds, "latitude", Float32.(φ), ("latitude",))
            defVar(ds, variable, reshape(values, size(values)..., 1), ("longitude", "latitude", "time");
                   fillvalue = -32767f0)
        end

        metadata = Metadata(name; dataset = GLORYSDaily(), region, dates = [date], dir)
        fts = FieldTimeSeries(metadata, grid; inpainting = NearestNeighborInpainting(Inf))

        for ψ in (Field(metadatum, grid), fts[1])
            ice = interior(ψ, :, :, 1)
            @test all(ice[west, :] .≈ 0.8f0)
            @test all(ice[east, :] .== 0)
        end
    end
end
