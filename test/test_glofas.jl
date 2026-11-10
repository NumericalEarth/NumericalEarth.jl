include("runtests_setup.jl")

using CDSAPI  # the GloFAS download methods, which skip files already on disk
using NCDatasets
using NumericalEarth.DataWrangling: BoundingBox, metadata_path
using NumericalEarth.DataWrangling.GloFAS: GloFASReanalysis, GloFASStaticMaps
using NumericalEarth.Lands: river_mouth_indices
using Oceananigans.Units: kilometers

# Synthetic GloFAS files in the layout of the real ones, with ocean west of λ = 10° and three pits on
# the equator carrying 100, 200 and 400 m³ s⁻¹: on the coast, ~55 km inland and ~225 km inland.
region = BoundingBox(longitude = (9, 13), latitude = (-1, 1))
pits = [(10.025, 100), (10.525, 200), (12.025, 400)]

# `ldd_glofas_v4_0.nc`: global at 0.05°, latitudes north to south, UInt8 with 255 over the ocean
function write_drainage_direction_map(path)
    λ = range(-179.975, step = 0.05, length = 7200)
    φ = range(89.975, step = -0.05, length = 3000)
    ldd = Union{Missing, UInt8}[λᵢ < 10 ? missing : UInt8(4) for λᵢ in λ, φⱼ in φ]
    for (λᵖ, _) in pits
        ldd[findfirst(≈(λᵖ), λ), findfirst(≈(0.025), φ)] = 5
    end

    NCDataset(path, "c") do ds
        defVar(ds, "lon", collect(λ), ("lon",))
        defVar(ds, "lat", collect(φ), ("lat",))
        defVar(ds, "ldd", ldd, ("lon", "lat"); fillvalue = UInt8(255), deflatelevel = 1)
    end
end

# A regional daily discharge file as the EWDS delivers it: latitudes north to south, NaN over the ocean
function write_discharge(path, date)
    λ = range(8.775, step = 0.05, length = 90)
    φ = range(1.225, step = -0.05, length = 50)
    discharge = Float32[λᵢ < 10 ? NaN : 1 for λᵢ in λ, φⱼ in φ, n in 1:1]
    for (λᵖ, Q) in pits
        discharge[findfirst(≈(λᵖ), λ), findfirst(≈(0.025), φ), 1] = Q
    end

    NCDataset(path, "c") do ds
        defVar(ds, "valid_time", [round(Int, Dates.datetime2unix(date + Day(1)))], ("valid_time",);
               attrib = ["units" => "seconds since 1970-01-01"])
        defVar(ds, "longitude", collect(λ), ("longitude",))
        defVar(ds, "latitude", collect(φ), ("latitude",))
        defVar(ds, "avg_dis", discharge, ("longitude", "latitude", "valid_time"); fillvalue = NaN32)
    end
end

@testset "Reading the GloFAS drainage direction map [$arch]" for arch in test_architectures
    metadatum = Metadatum(:drainage_direction; dataset = GloFASStaticMaps(), dir = mktempdir(), region)
    write_drainage_direction_map(metadata_path(metadatum))

    _, _, λ, φ = river_mouth_indices(Field(metadatum, arch))
    @test λ ≈ [10.025, 10.525]
    @test φ ≈ [0.025, 0.025]
end

@testset "A larger maximum_coastal_distance routes more rivers [$arch]" for arch in test_architectures
    dir = mktempdir()
    dates = (DateTime(2010, 1, 1), DateTime(2010, 1, 2))
    write_drainage_direction_map(metadata_path(Metadatum(:drainage_direction; dataset = GloFASStaticMaps(), dir, region)))
    for date in dates
        write_discharge(metadata_path(Metadatum(:river_discharge; dataset = GloFASReanalysis(), date, dir, region)), date)
    end

    ocean_grid = LatitudeLongitudeGrid(arch; size = (8, 4, 1), longitude = (9, 13), latitude = (-1, 1), z = (-100, 0))

    # Discharge at each river mouth of a GloFASPrescribedLand
    function mouth_discharge(maximum_coastal_distance)
        land = GloFASPrescribedLand(ocean_grid; start_date = dates[1], end_date = dates[2], dir, region,
                                    maximum_coastal_distance)
        routing = land.river_routing.rivers
        discharge = Array(interior(land.freshwater_flux.rivers[1], :, :, 1))
        mouths = unique(zip(Array(routing.contribution_outlet_i), Array(routing.contribution_outlet_j)))
        return sort([discharge[i, j] for (i, j) in mouths])
    end

    @test mouth_discharge(20kilometers) == [100]
    @test mouth_discharge(100kilometers) == [100, 200]
    @test mouth_discharge(300kilometers) == [100, 200, 400]
end
