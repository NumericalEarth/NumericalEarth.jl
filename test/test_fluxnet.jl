include("runtests_setup.jl")

using NumericalEarth.FLUXNET
using NumericalEarth.Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using NumericalEarth.Radiations: PrescribedRadiation
using Dates: DateTime, Minute, format
using Printf: @sprintf

# Write a small synthetic FLUXNET2015 half-hourly CSV covering a few diurnal cycles.
# `with_relative_humidity` toggles the FULLSET (`RH` present) vs SUBSET (`VPD_F` only)
# humidity path. A handful of `-9999` and elevated QC flags exercise gap filling and
# quality-control masking.
function write_synthetic_fluxnet(dir; site="XX-Tst", kind="FULLSET",
                                 start = DateTime(2020, 1, 1), ndays = 3,
                                 with_relative_humidity = true)
    Δ = Minute(30)
    times = start : Δ : (start + Minute(30) * (48 * ndays - 1))

    columns = ["TIMESTAMP_START", "TIMESTAMP_END",
               "TA_F", "TA_F_QC", "PA_F", "WS_F",
               "VPD_F", "P_F", "SW_IN_F", "LW_IN_F", "CO2_F_MDS",
               "H_F_MDS", "H_F_MDS_QC", "LE_F_MDS", "LE_F_MDS_QC",
               "USTAR", "NETRAD", "G_F_MDS", "TS_F_MDS_1", "SWC_F_MDS_1"]
    with_relative_humidity && insert!(columns, findfirst(==("VPD_F"), columns), "RH")

    rows = String[join(columns, ",")]
    for (i, t) in enumerate(times)
        hour = (i - 1) % 48 / 2                # local hour of day
        daylight = max(0.0, sinpi((hour - 6) / 12))
        Ta = 15 + 8 * sinpi((hour - 9) / 12)   # °C
        VPD = 2 + 8 * daylight                 # hPa
        RH = 90 - 40 * daylight                # %
        SW = 800 * daylight                    # W/m²
        H  = 220 * daylight                    # W/m²
        LE = 160 * daylight                    # W/m²
        rain = (i % 40 == 0) ? 0.6 : 0.0       # mm/30min, occasional
        # Missing air-temperature sample (short gap) and a poor-quality H flag.
        Ta_str = i == 5 ? "-9999" : @sprintf("%.3f", Ta)
        H_qc   = i in (10, 11) ? 3 : 0

        fields = [format(t, "yyyymmddHHMM"),
                  format(t + Δ, "yyyymmddHHMM"),
                  Ta_str, "0",
                  @sprintf("%.3f", 101.3),     # PA_F  kPa
                  @sprintf("%.3f", 2.5),       # WS_F  m/s
                  ]
        with_relative_humidity && push!(fields, @sprintf("%.3f", RH))
        append!(fields, [@sprintf("%.3f", VPD),
                         @sprintf("%.4f", rain),
                         @sprintf("%.3f", SW),
                         @sprintf("%.3f", 320.0),   # LW_IN_F
                         @sprintf("%.3f", 400.0),   # CO2_F_MDS
                         @sprintf("%.3f", H), string(H_qc),
                         @sprintf("%.3f", LE), "0",
                         @sprintf("%.4f", 0.35),    # USTAR
                         @sprintf("%.3f", SW - 120),# NETRAD
                         @sprintf("%.3f", 10.0),    # G_F_MDS
                         @sprintf("%.3f", 12.0),    # TS_F_MDS_1  °C
                         @sprintf("%.3f", 25.0)])   # SWC_F_MDS_1 %
        push!(rows, join(fields, ","))
    end

    filename = "FLX_$(site)_FLUXNET2015_$(kind)_HH_2020-2020_1-1.csv"
    path = joinpath(dir, filename)
    open(io -> foreach(r -> println(io, r), rows), path, "w")
    return path
end

@testset "FLUXNET metadata + FieldTimeSeries parsing" begin
    for arch in test_architectures
        dir = mktempdir()
        write_synthetic_fluxnet(dir)
        site = FLUXNETSite("XX-Tst"; dir, longitude = -120.95, latitude = 38.41)

        @test all_dates(site, :air_temperature)[1] == DateTime(2020, 1, 1)
        @test length(all_dates(site, :air_temperature)) == 48 * 3

        grid = RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))

        # Air temperature: °C → K, with the injected -9999 filled by fill_gaps!.
        md = Metadata(:air_temperature; dataset = site, dir)
        Ta = FieldTimeSeries(md, grid; time_indices_in_memory = length(md))
        NumericalEarth.DataWrangling.fill_gaps!(Ta; max_gap = 4)
        Ta_data = Array(interior(Ta))
        @test all(isfinite, Ta_data)
        @test all(240 .< Ta_data .< 320)     # sane Kelvin range

        # Pressure: kPa → Pa.
        pmd = Metadata(:surface_pressure; dataset = site, dir)
        pa = FieldTimeSeries(pmd, grid; time_indices_in_memory = length(pmd))
        @test all(Array(interior(pa)) .≈ 101300)
    end
end

@testset "FLUXNETPrescribedAtmosphere" begin
    for arch in test_architectures
        for with_rh in (true, false)
            dir = mktempdir()
            write_synthetic_fluxnet(dir; with_relative_humidity = with_rh)
            site = FLUXNETSite("XX-Tst"; dir)

            atmosphere = FLUXNETPrescribedAtmosphere(site, arch)

            @test atmosphere isa PrescribedAtmosphere
            @test haskey(atmosphere.velocities, :u)
            @test haskey(atmosphere.velocities, :v)
            @test atmosphere.tracers.T isa FieldTimeSeries
            @test atmosphere.tracers.q isa FieldTimeSeries
            @test atmosphere.freshwater_flux isa PrescribedPrecipitationFlux

            T_data = Array(interior(atmosphere.tracers.T))
            q_data = Array(interior(atmosphere.tracers.q))
            u_data = Array(interior(atmosphere.velocities.u))
            v_data = Array(interior(atmosphere.velocities.v))
            rain_data = Array(interior(atmosphere.freshwater_flux.rain))

            @test all(240 .< T_data .< 320)
            @test all(0 .< q_data .< 0.05)       # physical specific humidity
            @test all(isfinite, q_data)
            @test all(u_data .≈ 2.5)             # scalar wind speed in the eastward slot
            @test all(v_data .== 0)
            @test all(rain_data .≥ 0)
            @test maximum(rain_data) ≈ 0.6 / 1800  # mm/30min → kg/m²/s
        end
    end
end

@testset "FLUXNETPrescribedRadiation" begin
    for arch in test_architectures
        dir = mktempdir()
        write_synthetic_fluxnet(dir)
        site = FLUXNETSite("XX-Tst"; dir)

        radiation = FLUXNETPrescribedRadiation(site, arch)
        @test radiation isa PrescribedRadiation
        @test haskey(radiation.surface_properties, :land)

        sw = Array(interior(radiation.downwelling_shortwave))
        lw = Array(interior(radiation.downwelling_longwave))
        @test all(sw .≥ 0)
        @test maximum(sw) < 1500
        @test all(lw .> 0)
    end
end

@testset "fluxnet_flux_observations + quality control" begin
    for arch in test_architectures
        dir = mktempdir()
        write_synthetic_fluxnet(dir)
        site = FLUXNETSite("XX-Tst"; dir)

        observations = fluxnet_flux_observations(site, arch)
        @test keys(observations) == (:H, :LE, :ustar, :Rn, :G)
        for fts in observations
            @test fts isa FieldTimeSeries
        end

        # Without masking, all values are present.
        H_all = Array(interior(observations.H))
        @test all(isfinite, H_all)

        # With QC ≤ 1, the two H_F_MDS_QC = 3 samples (indices 10, 11) become NaN.
        masked = fluxnet_flux_observations(site, arch; quality_control = 1)
        H_masked = vec(Array(interior(masked.H)))
        @test isnan(H_masked[10]) && isnan(H_masked[11])
        @test count(isnan, H_masked) == 2
        # USTAR has no QC column, so masking leaves it untouched.
        @test all(isfinite, Array(interior(masked.ustar)))
    end
end
