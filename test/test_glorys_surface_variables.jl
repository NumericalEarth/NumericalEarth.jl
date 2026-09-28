include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, z_interfaces
using NumericalEarth.DataWrangling.GLORYS: GLORYS_surface_variables, dataset_location, GLORYSMonthly, GLORYSDaily

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
