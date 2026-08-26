include("runtests_setup.jl")

using NumericalEarth: TwoColorRadiation, ChlorophyllOptics, absorption_coefficient, equivalent_chlorophyll
using NumericalEarth.Oceans: compute_absorption_coefficient!
using NumericalEarth.DataWrangling: Metadatum, is_three_dimensional, dataset_variable_name,
                                    metadata_filename, longitude_name, latitude_name, all_dates
using NumericalEarth.DataWrangling.SeaWiFS: SeaWiFSMonthly, erddap_url
using Oceananigans.Fields: Field, interior
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: Time
using Dates: DateTime

@testset "Chlorophyll optics" begin
    optics = ChlorophyllOptics()

    # Manizza et al. (2005): clear water attenuates over 43 m, a bloom over 10 m.
    @test 1 / absorption_coefficient(optics, 0) ≈ 1 / 0.0232
    @test 1 / absorption_coefficient(optics, 1) ≈ 10.288 atol=1e-3
    @test absorption_coefficient(optics, 1) > absorption_coefficient(optics, 0.1)

    # Negative chlorophyll, which inpainting can produce at the edge of a gap, is floored not raised.
    @test absorption_coefficient(optics, -1) == absorption_coefficient(optics, 0)

    for κ in (1/23, 1/17, 1/10)
        @test absorption_coefficient(optics, equivalent_chlorophyll(optics, κ)) ≈ κ
    end

    @test eltype(ChlorophyllOptics(Float32).clear_water_attenuation) == Float32
end

@testset "TwoColorRadiation" begin
    grid = LatitudeLongitudeGrid(size=(4, 4, 4), longitude=(0, 360), latitude=(-60, 60), z=(-100, 0))
    clock = Clock(grid)

    # The default must remain the Jerlov Type I 23 m decay scale it has always been.
    radiation = TwoColorRadiation(grid)
    @test absorption_coefficient(radiation.chlorophyll_optics, radiation.chlorophyll) ≈ 1 / 23
    @test radiation.first_absorption_coefficient ≈ 1 / 0.35
    @test radiation.first_color_fraction == 0.58

    # Uniform chlorophyll resolves to a scalar coefficient, so the flux divergence reads no field.
    @test radiation.second_absorption_coefficient isa Number
    @test radiation.second_absorption_coefficient ≈ 1 / 23

    # A uniform surface flux heats every level, most strongly at the top, and the column
    # integral of the flux divergence returns the surface flux.
    fill!(parent(radiation.surface_flux), 1)
    heating = [radiation(2, 2, k, grid, clock, nothing) for k in 1:4]
    @test all(heating .> 0)
    @test issorted(heating)

    Δz = 25
    @test sum(heating) * Δz ≈ 1 atol=1e-10

    # Turbid water traps the same surface flux nearer the surface.
    turbid = TwoColorRadiation(grid; chlorophyll = 2)
    fill!(parent(turbid.surface_flux), 1)
    turbid_heating = [turbid(2, 2, k, grid, clock, nothing) for k in 1:4]
    @test turbid_heating[4] > heating[4]
    @test turbid_heating[1] < heating[1]
    @test sum(turbid_heating) * Δz ≈ 1 atol=1e-10

    # Chlorophyll that varies in space makes the optics vary with it.
    patchy = TwoColorRadiation(grid; chlorophyll = (λ, φ, z, t) -> φ > 0 ? 2.0 : 0.02)
    fill!(parent(patchy.surface_flux), 1)
    compute_absorption_coefficient!(patchy, Time(0))
    north = patchy(2, 3, 4, grid, clock, nothing)
    south = patchy(2, 2, 4, grid, clock, nothing)
    @test north > south

    chlorophyll = Field{Center, Center, Nothing}(grid)
    set!(chlorophyll, 2)
    from_field = TwoColorRadiation(grid; chlorophyll)
    fill!(parent(from_field.surface_flux), 1)
    compute_absorption_coefficient!(from_field, Time(0))
    @test from_field(2, 2, 4, grid, clock, nothing) ≈ turbid_heating[4]

    # Chlorophyll that varies in time is resolved at the time it is refreshed, not at the level
    # the flux divergence is evaluated on.
    seasonal = TwoColorRadiation(grid; chlorophyll = (λ, φ, z, t) -> t.time < 1 ? 0.02 : 2.0)
    fill!(parent(seasonal.surface_flux), 1)
    compute_absorption_coefficient!(seasonal, Time(0))
    winter = seasonal(2, 2, 4, grid, clock, nothing)
    compute_absorption_coefficient!(seasonal, Time(2))
    summer = seasonal(2, 2, 4, grid, clock, nothing)
    @test summer > winter
    @test summer ≈ turbid_heating[4]
end

@testset "SeaWiFS chlorophyll metadata" begin
    dataset = SeaWiFSMonthly()

    @test length(all_dates(dataset, :chlorophyll)) == 160
    @test first(all_dates(dataset, :chlorophyll)) == DateTime(1997, 9, 1)
    @test size(dataset, :chlorophyll) == (360, 180, 1)
    @test size(SeaWiFSMonthly(resolution=2), :chlorophyll) == (180, 90, 1)

    metadatum = Metadatum(:chlorophyll; dataset, date = DateTime(2000, 6, 1), dir = ".")

    @test !is_three_dimensional(metadatum)
    @test dataset_variable_name(metadatum) == "chlorophyll"
    @test longitude_name(metadatum) == "longitude"
    @test latitude_name(metadatum) == "latitude"
    @test metadata_filename(dataset, :chlorophyll, DateTime(2000, 6, 1), nothing) ==
          "SeaWiFS_chlorophyll_1deg_2000-06.nc"

    url = erddap_url(metadatum)
    @test occursin("erdSW2018chlamday", url)
    @test occursin("(2000-06-01)", url)
    @test occursin("(2000-06-30)", url)
    @test occursin("0:12:2159", url)
end
