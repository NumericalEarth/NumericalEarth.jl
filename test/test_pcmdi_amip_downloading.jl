include("runtests_setup.jl")

using Oceananigans.Grids: φnodes
using Oceananigans.OutputReaders: Clamp
using Oceananigans.Units: Time, days
using Statistics: mean

dir = mktempdir()

@testset "Downloading PCMDI-AMIP data" begin
    for dataset in (PCMDIAMIPBoundaryConditions(), PCMDIAMIPObservations()), name in (:sea_surface_temperature, :sea_ice_concentration)
        metadatum = Metadatum(name; dataset, dir)
        download(metadatum)
        @test isfile(metadata_path(metadatum))
    end
end

@testset "PCMDI-AMIP boundary conditions recover the observed monthly means" begin
    dates = (DateTime(1979, 5, 1), DateTime(1979, 7, 1))
    series(name, dataset) = FieldTimeSeries(Metadata(name; dataset, dates, dir); time_indexing=Clamp())

    observed_sst = series(:sea_surface_temperature, PCMDIAMIPObservations())
    observed_ice = series(:sea_ice_concentration, PCMDIAMIPObservations())
    @test all(271 .< interior(observed_sst) .< 310)
    @test all(0 .≤ interior(observed_ice) .≤ 1)

    # June 1979, averaged over the days of the series interpolated between mid-month values.
    # Away from sea ice no bound is active, so the mean is the observed one.
    sst = series(:sea_surface_temperature, PCMDIAMIPBoundaryConditions())
    june_days = (31 .+ (0.5:29.5)) .* days
    june_mean = mean(interior(sst[Time(t)], :, :, 1) for t in june_days)
    tropics = abs.(φnodes(sst.grid, Center())) .< 30
    @test maximum(abs, june_mean[:, tropics] .- interior(observed_sst[2], :, tropics, 1)) < 0.01
end
