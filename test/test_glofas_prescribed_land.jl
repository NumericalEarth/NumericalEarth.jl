include("runtests_setup.jl")

using CDSAPI  # activates NumericalEarthCDSAPIExt (the EWDS request path)

using NumericalEarth.DataWrangling: BoundingBox

# The lower Amazon, as in the ERA5/GloFAS example: the Amazon is the only river above 10⁵ m³ s⁻¹.
@testset "GloFASPrescribedLand routes the Amazon [$arch]" for arch in test_architectures
    region = BoundingBox(longitude = (-53, -45), latitude = (-2, 5))
    ocean_grid = LatitudeLongitudeGrid(arch; size = (16, 14, 1), longitude = region.longitude,
                                       latitude = region.latitude, z = (-100, 0))
    date = DateTime(2004, 12, 27)

    # Discharge at each river mouth of a GloFASPrescribedLand
    function mouth_discharge(; kw...)
        land = GloFASPrescribedLand(ocean_grid; start_date = date, end_date = date + Day(1), region, kw...)
        routing = land.river_routing.rivers
        discharge = Array(interior(land.freshwater_flux.rivers[1], :, :, 1))
        mouths = unique(zip(Array(routing.contribution_outlet_i), Array(routing.contribution_outlet_j)))
        return [discharge[i, j] for (i, j) in mouths]
    end

    near = mouth_discharge(maximum_coastal_distance = 6kilometers)   # about one 0.05° cell
    far = mouth_discharge()                                          # the default, 100 km

    @test maximum(far) > 1e5
    @test length(far) > length(near)
    @test sum(far) > sum(near)
end
