using Test, Oceananigans, NumericalEarth, CUDA
using Oceananigans.Fields: interpolate, interpolate!
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: node
using Oceananigans.Units: Time
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using NumericalEarth.Grids: PressureLevelVerticalDiscretization

@inline function sample_pressure_series(i, j, k, grid, series, source_grid)
    location = (Center(), Center(), Center())
    point = node(i, j, k, grid, location...)
    return interpolate(point, Time(0.4), series, location, source_grid)
end

@inline function sample_log_pressure(i, j, k, grid, field, source_grid)
    location = (Center(), Center(), Center())
    point = node(i, j, k, grid, location...)
    return interpolate(log, point, field, location, source_grid)
end

architectures = parse(Bool, get(ENV, "GPU_TEST", "false")) ? (GPU(),) : (CPU(),)
CUDA.allowscalar(false)
for arch in architectures, FT in (Float32, Float64)
    @testset "Column-wise pressure interpolation $arch $FT" begin
        source_grid = LatitudeLongitudeGrid(arch, FT; size=(3, 3, 3),
            longitude=(0, 3), latitude=(0, 3), z=(0, 3), topology=(Bounded, Bounded, Bounded))
        geopotential = CenterField(source_grid)
        set!(geopotential, (x, y, z) -> 100 + 900z + 200x + 150y)
        vertical = PressureLevelVerticalDiscretization(geopotential; gravitational_acceleration=1)
        grid = LatitudeLongitudeGrid(arch, FT; size=size(source_grid),
            longitude=(0, 3), latitude=(0, 3), z=vertical, topology=(Bounded, Bounded, Bounded))
        source = CenterField(grid)
        set!(source, (x, y, z) -> z + 2x + 3y)
        target_grid = LatitudeLongitudeGrid(arch, FT; size=(6, 3, 2),
            longitude=(1.4, 1.6), latitude=(1.2, 1.8), z=(1600, 1800), topology=(Bounded, Bounded, Bounded))
        target, expected = CenterField(target_grid), CenterField(target_grid)
        set!(expected, (x, y, z) -> z + 2x + 3y)
        interpolate!(target, source)
        @test Array(interior(target)) ≈ Array(interior(expected)) rtol=20eps(FT)

        geometry = FieldTimeSeries{Center, Center, Center}(source_grid, [0., 1.])
        set!(geometry[1], (x, y, z) -> 100 + 900z + 200x + 150y)
        set!(geometry[2], (x, y, z) -> 200 + 900z + 200x + 150y)
        clock = Clock(time=0.0)
        moving_vertical = PressureLevelVerticalDiscretization(TimeSeriesInterpolation(geometry, source_grid; clock);
            gravitational_acceleration=1)
        moving_grid = LatitudeLongitudeGrid(arch, FT; size=size(source_grid),
            longitude=(0, 3), latitude=(0, 3), z=moving_vertical, topology=(Bounded, Bounded, Bounded))
        moving_series = FieldTimeSeries{Center, Center, Center}(moving_grid, [0., 1.])
        set!(moving_series[1], (x, y, z) -> z + 2x + 3y)
        clock.time = 1.0
        set!(moving_series[2], (x, y, z) -> z + 2x + 3y + 20)
        clock.time = 0.4
        moving_sample = Field(KernelFunctionOperation{Center, Center, Center}(
            sample_pressure_series, target_grid, moving_series, moving_grid))
        compute!(moving_sample)
        @test Array(interior(moving_sample)) ≈ Array(interior(expected)) .+ 8 rtol=20eps(FT)

        series = FieldTimeSeries{Center, Center, Center}(grid, [0., 1.])
        set!(series[1], (x, y, z) -> z + 2x + 3y)
        set!(series[2], (x, y, z) -> z + 2x + 3y + 20)
        sampled = Field(KernelFunctionOperation{Center, Center, Center}(
            sample_pressure_series, target_grid, series, grid))
        compute!(sampled)
        @test Array(interior(sampled)) ≈ Array(interior(expected)) .+ 8 rtol=20eps(FT)

        set!(source, (x, y, z) -> exp((z + 2x + 3y) / 1000))
        logarithm = Field(KernelFunctionOperation{Center, Center, Center}(
            sample_log_pressure, target_grid, source, grid))
        compute!(logarithm)
        @test Array(interior(logarithm)) ≈ Array(interior(expected)) ./ 1000 rtol=20eps(FT)

        surface = Field{Center, Center, Nothing}(source_grid)
        set!(surface, FT(1500))
        heights = reshape(FT[1400, 1500 + 2eps(FT(1500)), 2500], 1, 1, :)
        set!(geopotential, repeat(heights, size(source_grid, 1), size(source_grid, 2), 1))
        clipped = PressureLevelVerticalDiscretization(geopotential;
            gravitational_acceleration=1, surface_geopotential=surface)
        clipped_grid = LatitudeLongitudeGrid(arch, FT; size=size(source_grid),
            longitude=(0, 3), latitude=(0, 3), z=clipped, topology=(Bounded, Bounded, Bounded))
        clipped_source = CenterField(clipped_grid)
        values = reshape(FT[-1e6, -1e6, 42], 1, 1, :)
        set!(clipped_source, repeat(values, size(source_grid, 1), size(source_grid, 2), 1))
        interpolate!(target, clipped_source)
        @test Array(interior(target)) ≈ fill(FT(42), size(target)) rtol=20eps(FT)

        extended = PressureLevelVerticalDiscretization(geopotential;
            gravitational_acceleration=1, surface_geopotential=surface, clip_subsurface=false)
        extended_grid = LatitudeLongitudeGrid(arch, FT; size=size(source_grid),
            longitude=(0, 3), latitude=(0, 3), z=extended, topology=(Bounded, Bounded, Bounded))
        extended_source = CenterField(extended_grid)
        set!(extended_source, (x, y, z) -> z)
        interpolate!(target, extended_source)
        set!(expected, (x, y, z) -> z)
        @test Array(interior(target)) ≈ Array(interior(expected)) rtol=20eps(FT)
    end
end
