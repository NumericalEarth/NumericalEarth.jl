include("runtests_setup.jl")
include("download_utils.jl")

using NumericalEarth.DataWrangling: NearestNeighborInpainting, native_times
using Oceananigans.Grids: topology, λnodes, φnodes, znodes
using Oceananigans.OutputReaders: time_indices
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units

dataset = SyntheticOcean()
dates = all_dates(dataset, :temperature)[1:4]
inpainting = NearestNeighborInpainting(10)

for arch in test_architectures
    A = typeof(arch)

    @testset "$A synthetic ocean dataset" begin
        @info "Running synthetic dataset tests on $A..."

        @testset "Native-grid values" begin
            datum = Metadatum(:temperature; dataset)
            field = Field(datum, arch; inpainting = nothing)
            values = Array(interior(field))
            grid = on_architecture(CPU(), field.grid)
            λ = λnodes(grid, Center())
            φ = φnodes(grid, Center())
            z = znodes(grid, Center())
            expected = [synthetic_value(dataset, :temperature, λ[i], φ[j], z[k])
                        for i in eachindex(λ), j in eachindex(φ), k in eachindex(z)]
            @test all(isequal.(values, Float32.(expected)))
            @test any(isnan, values)
        end

        @testset "Setting a field from a dataset" begin
            test_setting_from_metadata(arch, dataset, dates[1], inpainting)
        end

        @testset "Timestepping with a dataset" begin
            test_timestepping_with_dataset(arch, dataset, dates[1], inpainting)
        end

        @testset "Field utilities" begin
            test_ocean_metadata_utilities(arch, dataset, dates, inpainting)
        end

        @testset "DatasetRestoring with LinearlyTaperedPolarMask" begin
            test_dataset_restoring(arch, dataset, dates, inpainting)
        end

        @testset "Timestepping with DatasetRestoring" begin
            test_timestepping_with_dataset_restoring(arch, dataset, dates, inpainting)
        end

        @testset "Dataset cycling boundaries" begin
            test_cycling_dataset_restoring(arch, dataset, dates, inpainting)
        end

        @testset "Inpainting algorithm" begin
            test_inpainting_algorithm(arch, dataset, dates[1], inpainting)
        end
    end

    @testset "$A synthetic bathymetry" begin
        grid = LatitudeLongitudeGrid(arch; size = (36, 18, 1), longitude = (-180, 180), latitude = (-90, 90), z = (0, 1))
        bottom_height = synthetic_bottom_height(grid; major_basins = Inf)
        values = Array(interior(bottom_height, :, :, 1))
        cpu_grid = on_architecture(CPU(), grid)
        λ = λnodes(cpu_grid, Center())
        φ = φnodes(cpu_grid, Center())
        expected = [synthetic_value(SyntheticBathymetry(), :bottom_height, λ[i], φ[j], 0) for i in eachindex(λ), j in eachindex(φ)]
        @test values == expected
    end

    @testset "$A synthetic prescribed components" begin
        atmosphere = synthetic_prescribed_atmosphere(arch)
        radiation = synthetic_prescribed_radiation(arch)
        land = synthetic_prescribed_land(arch)

        @test length(atmosphere.times) == 12
        @test all(Array(interior(atmosphere.pressure[1])) .== 101325)
        @test all(Array(interior(radiation.downwelling_longwave[1])) .== 250)
        @test all(Array(interior(land.freshwater_flux.rivers[1])) .== 1f-5)
    end
end
