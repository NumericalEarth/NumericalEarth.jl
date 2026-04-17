include("runtests_setup.jl")
include("download_utils.jl")

using NumericalEarth.JRA55: download_JRA55_cache
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.DataWrangling: compute_native_date_range

@testset "JRA55 and data wrangling utilities" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing reanalysis_field_time_series on $A..."

        # This should download files called "RYF.rsds.1990_1991.nc" and "RYF.tas.1990_1991.nc"
        for test_name in (:downwelling_shortwave_radiation, :temperature)
            dates = NumericalEarth.DataWrangling.all_dates(JRA55.RepeatYearJRA55(), test_name)
            end_date = dates[3]

            JRA55_fts = FieldTimeSeries(Metadata(test_name; dataset=JRA55.RepeatYearJRA55(), end_date), arch)
            test_filename = joinpath(download_JRA55_cache, "RYF.rsds.1990_1991.nc")

            @test JRA55_fts isa FieldTimeSeries
            @test JRA55_fts.grid isa LatitudeLongitudeGrid

            Nx, Ny, Nz, Nt = size(JRA55_fts)
            @test Nx == 640
            @test Ny == 320
            @test Nz == 1
            @test Nt == 3

            if test_name == :downwelling_shortwave_radiation
                CUDA.@allowscalar begin
                    @test JRA55_fts[1, 1, 1, 1]   == 430.98105f0
                    @test JRA55_fts[641, 1, 1, 1] == 430.98105f0
                end
            end

            # Test that halo regions were filled to respect boundary conditions
            CUDA.@allowscalar begin
                @test view(JRA55_fts.data, 1, :, 1, :) == view(JRA55_fts.data, Nx+1, :, 1, :)
            end

            @info "Testing Cyclical time_indices for JRA55 data on $A..."
            Nb = 4
            netcdf_JRA55_fts = FieldTimeSeries(Metadata(test_name; dataset=JRA55.RepeatYearJRA55()), arch;
                                               time_indices_in_memory=Nb)

            Nt = length(netcdf_JRA55_fts.times)
            @test Oceananigans.OutputReaders.time_indices(netcdf_JRA55_fts) == (1, 2, 3, 4)
            f₁ = view(parent(netcdf_JRA55_fts), :, :, 1, 1)
            f₁ = Array(f₁)

            netcdf_JRA55_fts.backend = Oceananigans.OutputReaders.new_backend(netcdf_JRA55_fts.backend, Nt-2, Nb)
            @test Oceananigans.OutputReaders.time_indices(netcdf_JRA55_fts) == (Nt-2, Nt-1, Nt, 1)
            set!(netcdf_JRA55_fts)

            f₁′ = view(parent(netcdf_JRA55_fts), :, :, 1, 4)
            f₁′ = Array(f₁′)
            @test f₁ == f₁′

            @info "Testing PrefetchingBackend on $A for $test_name..."
            # Build a reference (cold) FTS and a prefetching FTS over the same
            # window, then drive each through several reloads. After every
            # reload the parent data of the prefetching FTS must be byte-
            # identical to the reference. The first reload exercises the cold
            # fallback (no prior prefetch); subsequent reloads exercise the
            # hot path; the wrap from `Nt-3..Nt` back to `1..Nb` exercises the
            # cyclical prefetch logic (`mod1(start+Nm, Nt)`).
            ref_fts = FieldTimeSeries(Metadata(test_name; dataset=JRA55.RepeatYearJRA55()), arch;
                                      time_indices_in_memory=Nb)
            pf_fts  = FieldTimeSeries(Metadata(test_name; dataset=JRA55.RepeatYearJRA55()), arch;
                                      time_indices_in_memory=Nb, prefetch=true)

            @test pf_fts.backend isa NumericalEarth.DataWrangling.PrefetchingBackend
            @test parent(pf_fts.data) == parent(ref_fts.data)              # cold load alignment
            @test pf_fts.backend.next_start == Nb + 1                       # next prefetch scheduled

            # Reload sequence:
            #   * Nb+1, 2Nb+1     → straight hot-path advances
            #   * Nt-Nb+1         → places the next prefetch's window across
            #                       the end-of-times boundary, exercising the
            #                       `mod1(start+Nm, Nt)` wrap when scheduling
            #                       the prefetch
            #   * 1               → consumes that wrapped prefetch (hot path)
            #                       at the start of the cycle
            for next_start in (Nb + 1, 2Nb + 1, Nt - Nb + 1, 1)
                ref_fts.backend = Oceananigans.OutputReaders.new_backend(ref_fts.backend, next_start, Nb)
                pf_fts.backend  = Oceananigans.OutputReaders.new_backend(pf_fts.backend,  next_start, Nb)
                set!(ref_fts)
                set!(pf_fts)
                @test parent(pf_fts.data) == parent(ref_fts.data)
                @test pf_fts.backend.next_start == mod1(next_start + Nb, Nt)
            end
        end

        @info "Testing interpolate_field_time_series! on $A..."

        name  = :downwelling_shortwave_radiation
        dates = NumericalEarth.DataWrangling.all_dates(JRA55.RepeatYearJRA55(), name)
        end_date = dates[3]
        JRA55_fts = FieldTimeSeries(Metadata(name; dataset=JRA55.RepeatYearJRA55(), end_date), arch)

        # Make target grid and field
        resolution = 1 # degree, eg 1/4
        Nx = Int(360 / resolution)

        southern_limit = -79
        northern_limit = -30
        j₁ = (90 + southern_limit) / resolution
        j₂ = (90 + northern_limit) / resolution + 1
        Ny = Int(j₂ - j₁ + 1)

        target_grid = LatitudeLongitudeGrid(arch,
                                            size = (Nx, Ny, 1);
                                            longitude = (0, 360),
                                            latitude = (southern_limit, northern_limit),
                                            z = (0, 1),
                                            topology = (Periodic, Bounded, Bounded))

        times = JRA55_fts.times
        boundary_conditions = JRA55_fts.boundary_conditions
        target_fts = FieldTimeSeries{Center, Center, Nothing}(target_grid, times; boundary_conditions)
        interpolate!(target_fts, JRA55_fts)

        # Random regression test
        CUDA.@allowscalar begin
            @test Float32(target_fts[1, 1, 1, 1]) ≈ Float32(222.24313354492188)

            # Only include this if we are filling halo regions within
            # interpolate_field_time_series
            @test Float32(target_fts[Nx + 1, 1, 1, 1]) ≈ Float32(222.24313354492188)
        end

        @test target_fts.times == JRA55_fts.times

        # What else might we test?

        @info "Testing save_field_time_series! on $A..."
        filepath = "JRA55_downwelling_shortwave_radiation_test_$(string(typeof(arch))).jld2" # different filename for each arch so that the CPU and GPU tests do not crash
        NumericalEarth.DataWrangling.save_field_time_series!(target_fts, path=filepath, name="Qsw",
                                                         overwrite_existing = true)
        @test isfile(filepath)

        # Test that we can load the data back
        Qswt = FieldTimeSeries(filepath, "Qsw")
        @test on_architecture(CPU(), parent(Qswt.data)) == on_architecture(CPU(), parent(target_fts.data))
        @test Qswt.times == target_fts.times
        rm(filepath)

        #####
        ##### JRA55 prescribed atmosphere
        #####

        atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2, include_rivers_and_icebergs=false)
        @test atmosphere isa PrescribedAtmosphere
        @test isnothing(atmosphere.auxiliary_freshwater_flux)

        # Test that rivers and icebergs are included in the JRA55 data with the correct frequency
        atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2, include_rivers_and_icebergs=true)
        @test haskey(atmosphere.auxiliary_freshwater_flux, :rivers)
        @test haskey(atmosphere.auxiliary_freshwater_flux, :icebergs)

        rivers_times = atmosphere.auxiliary_freshwater_flux.rivers.times
        pressure_times = atmosphere.pressure.times
        @test rivers_times != pressure_times
        @test length(rivers_times) != length(pressure_times)
        @test rivers_times[2] - rivers_times[1] == 86400

        @info "Testing MultiYearJRA55 data on $A..."
        dataset = JRA55.MultiYearJRA55()

        # Test that when date range spans two years both netCDF files are downloaded
        # and concatenated when reading the data.
        start_date = DateTime("1959-01-01T00:00:00") - 15 * Day(1) # sometime in 1958
        end_date   = DateTime("1959-01-01T00:00:00") + 85 * Day(1) # sometime in 1959

        # Use a temporary directory so different architectures don't clash
        mktempdir("./") do dir
            # Compute expected file paths so we can fall back to artifacts if needed
            native_dates = NumericalEarth.DataWrangling.all_dates(dataset, :temperature)
            dates = compute_native_date_range(native_dates, start_date, end_date)
            metadata = Metadata(:temperature; dataset, dates, dir)
            filepaths = unique(metadata_path(metadata))

            Ta = download_dataset_with_fallback(filepaths; dataset_name="MultiYearJRA55 :temperature") do
                FieldTimeSeries(metadata, arch; time_indices_in_memory=10)
            end
            @test Second(end_date - start_date).value ≈ Ta.times[end] - Ta.times[1]

            # Test we can access all the data
            for t in eachindex(Ta.times)
                @test Ta[t] isa Field
            end
        end

        @info "Testing MultiYearJRA55 single-window crossing year boundary on $A..."

        # Force a single in-memory window to straddle the 1958 → 1959 file
        # boundary. Before the per-file `ftsn_loc` fix, the second file's
        # iteration in `set!` would clobber the outer `ftsn` and write to the
        # wrong slots; this regression test would then leave some in-memory
        # slots untouched (zero-valued).
        start_date_span = DateTime("1958-12-27T00:00:00")
        end_date_span   = DateTime("1959-01-05T00:00:00")

        mktempdir("./") do dir
            native_dates = NumericalEarth.DataWrangling.all_dates(dataset, :temperature)
            dates = compute_native_date_range(native_dates, start_date_span, end_date_span)
            metadata = Metadata(:temperature; dataset, dates, dir)
            filepaths = unique(metadata_path(metadata))

            Ta_span = download_dataset_with_fallback(filepaths;
                                                    dataset_name="MultiYearJRA55 :temperature year-boundary window") do
                # backend window of 80 holds the whole range in a single window
                FieldTimeSeries(metadata, arch; time_indices_in_memory=80)
            end

            # Every slot in the single in-memory window must carry valid
            # (non-zero) atmospheric temperature data.
            CUDA.@allowscalar begin
                for t in eachindex(Ta_span.times)
                    @test maximum(abs, interior(Ta_span[t])) > 0
                end
            end
        end
    end
end
