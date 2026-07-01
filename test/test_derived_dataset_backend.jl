include("runtests_setup.jl")

using NumericalEarth.DataWrangling: DerivedDatasetBackend
using Oceananigans.OutputReaders: Cyclical, FieldTimeSeries, time_indices

@inline product_of_sources(i, j, k, grid, a, b) = @inbounds a[i, j, k] * b[i, j, k]
@inline scaled_source(i, j, k, grid, a, scale) = @inbounds scale * a[i, j, k]

@testset "DerivedDatasetBackend" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (4, 4), latitude = (0, 1), longitude = (0, 1),
                                     topology = (Bounded, Bounded, Flat))
        times = 0.0:1.0:9.0
        Nt = length(times)

        a = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        b = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        for n in 1:Nt
            set!(a[n], (λ, φ) -> n + λ)
            set!(b[n], (λ, φ) -> 10n + φ)
        end

        window = 3
        backend = DerivedDatasetBackend(window, product_of_sources, (a, b))
        derived = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend,
                                                           time_indexing = Cyclical())
        set!(derived)

        @test length(derived.backend) == window
        @test collect(time_indices(derived)) == [1, 2, 3]

        # Slide the window forward, jump backward, and wrap past the last index:
        # every access must match the directly computed product, and the number
        # of resident slices must never exceed the window.
        for n in (1, 2, 3, 7, 8, 4, 1, Nt)
            expected = Array(interior(a[n])) .* Array(interior(b[n]))
            @test Array(interior(derived[n])) ≈ expected
            @test length(time_indices(derived)) == window
        end

        # Parameters are appended to the kernel function call.
        scale = 2.5
        scaled = FieldTimeSeries{Center, Center, Nothing}(grid, times;
                     backend = DerivedDatasetBackend(window, scaled_source, (a,), (scale,)),
                     time_indexing = Cyclical())
        set!(scaled)

        for n in (2, 6, Nt)
            @test Array(interior(scaled[n])) ≈ scale .* Array(interior(a[n]))
        end
    end
end
