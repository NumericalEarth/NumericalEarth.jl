include("runtests_setup.jl")

using Statistics: median
using NumericalEarth.Atmospheres: PrescribedAtmosphere, TwoBandDownwellingRadiation
using NumericalEarth.ECCO: ECCOPrescribedAtmosphere, ECCO4Monthly
using NumericalEarth.DataWrangling: higher_bound

@testset "ECCO Prescribed Atmosphere" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing ECCOPrescribedAtmosphere on $A..."

        dataset = ECCO4Monthly()
        start_date = DateTime(1992, 1, 1)
        end_date = DateTime(1992, 3, 1)

        atmosphere = ECCOPrescribedAtmosphere(arch;
                                              dataset,
                                              start_date,
                                              end_date,
                                              time_indices_in_memory = 2)

        @test atmosphere isa PrescribedAtmosphere

        # Test that all expected fields are present
        @test haskey(atmosphere.velocities, :u)
        @test haskey(atmosphere.velocities, :v)
        @test haskey(atmosphere.tracers, :T)
        @test haskey(atmosphere.tracers, :q)
        @test !isnothing(atmosphere.pressure)
        @test !isnothing(atmosphere.downwelling_radiation)
        @test haskey(atmosphere.freshwater_flux, :rain)

        # Test downwelling radiation components
        ℐꜜˢʷ = atmosphere.downwelling_radiation.shortwave
        ℐꜜˡʷ = atmosphere.downwelling_radiation.longwave

        @test ℐꜜˢʷ isa FieldTimeSeries
        @test ℐꜜˡʷ isa FieldTimeSeries

        # Test that downwelling radiation has the correct sign convention
        # Downwelling radiation should be positive (energy coming down to the surface)
        # This is consistent with JRA55 convention
        CUDA.@allowscalar begin
            # Get some sample values (not all zeros)
            # Use interior() to avoid checking halo regions which may
            # contain uninitialized memory on GPU
            ℐꜜˢʷ_data = interior(ℐꜜˢʷ)
            ℐꜜˡʷ_data = interior(ℐꜜˡʷ)

            # Longwave radiation should be positive (always some downwelling longwave)
            @test all(ℐꜜˡʷ_data .>= 0)

            # Typical ranges for radiation (sanity checks)
            # Shortwave: 0 to ~1400 W/m² (solar constant at TOA)
            # Longwave: ~100 to ~500 W/m² (typical atmospheric emission)
            @test maximum(ℐꜜˢʷ_data) < 1500  # W/m²
            @test maximum(ℐꜜˡʷ_data) < 600   # W/m²
            @test maximum(ℐꜜˡʷ_data) > 50    # W/m² - should have some reasonable values
        end

        # Test that higher_bound for sea_level_pressure is large enough
        # to avoid masking valid pressure data (~101325 Pa).
        pa_metadata = Metadata(:sea_level_pressure; dataset, start_date, end_date)
        other_metadata = Metadata(:temperature; dataset, start_date, end_date)
        @test higher_bound(pa_metadata, Val(:sea_level_pressure)) == 1f10
        @test higher_bound(other_metadata, Val(:temperature)) == 1f5  # default

        # Verify pressure field contains physically reasonable values
        CUDA.@allowscalar begin
            pa_data = interior(atmosphere.pressure)
            valid = pa_data[pa_data .!= 0]
            @test length(valid) > 0
            @test median(valid) > 1f5   # typical sea level pressure exceeds 1e5 Pa
            @test median(valid) < 1.1f5
        end

        # Test grid consistency
        @test atmosphere.velocities.u.grid isa LatitudeLongitudeGrid
        @test atmosphere.velocities.u.grid == atmosphere.velocities.v.grid
        @test atmosphere.velocities.u.grid == atmosphere.tracers.T.grid
    end
end
