include("runtests_setup.jl")

using NumericalEarth.DataWrangling: x_interfaces, y_interfaces, dataset_variable_name, metadata_path,
                                    validate_dataset_coverage, default_inpainting, is_three_dimensional,
                                    native_grid, projected_window, projected_window_centers
using NumericalEarth.DataWrangling.BedMachine: BedMachineAntarctica_url, window_to_netcdf
using Oceananigans.Grids: xnodes, ynodes, topology, Bounded, Flat
using NCDatasets: NCDataset, defDim, defVar

# A synthetic file at the north-west corner of the real 500 m lattice stands in for the 1.2 GB
# BedMachine file; it carries the real file name, so the cached download finds it.

dataset = BedMachineAntarctica()
x₁, _ = x_interfaces(dataset)
_, y₂ = y_interfaces(dataset)
Δ = 500

linear_bed(x, y) = 1e-3 * (x - x₁) + 2e-3 * (y₂ - y) - 1000

function write_synthetic_bedmachine(path; Nx = 40, Ny = 30, spacing = Δ, standard_parallel = -71.0)
    x = [x₁ + (i - 1/2) * spacing for i in 1:Nx]
    y = [y₂ - (j - 1/2) * spacing for j in 1:Ny]
    bed = Float32[linear_bed(xᵢ, yⱼ) for xᵢ in x, yⱼ in y]
    bed[6, 6] = -9999
    mask = Int8[ifelse(xᵢ < x₁ + 10Δ, 2, 3) for xᵢ in x, yⱼ in y]
    NCDataset(path, "c") do ds
        defDim(ds, "x", Nx)
        defDim(ds, "y", Ny)
        defVar(ds, "x", Int32.(round.(x)), ("x",))
        defVar(ds, "y", Int32.(round.(y)), ("y",))
        defVar(ds, "mapping", Int32, (); attrib = ["grid_mapping_name" => "polar_stereographic",
                                                  "latitude_of_projection_origin" => -90.0,
                                                  "standard_parallel" => standard_parallel,
                                                  "straight_vertical_longitude_from_pole" => 0.0])
        defVar(ds, "bed", bed, ("x", "y"); fillvalue = -9999f0)
        defVar(ds, "mask", mask, ("x", "y"))
    end
    return path
end

region = ProjectedBoundingBox(x = (x₁ + 3000, x₁ + 15000), y = (y₂ - 12000, y₂ - 3000))

@testset "BedMachineAntarctica metadata interfaces" begin
    metadatum = Metadatum(:bottom_height; dataset, region)
    @test dataset_variable_name(metadatum) == "bed"
    @test !is_three_dimensional(metadatum)
    @test isnothing(default_inpainting(metadatum))
    @test metadatum.filename == "BedMachine_Antarctica_bottom_height_x_-3330250_-3318250_y_3321250_3330250.nc"

    @test_throws ErrorException validate_dataset_coverage(nothing, Metadatum(:bottom_height; dataset))
    outside = ProjectedBoundingBox(x = (4e6, 4.1e6), y = (0, 1e5))
    @test_throws ErrorException validate_dataset_coverage(nothing, Metadatum(:bottom_height; dataset, region = outside))
end

@testset "BedMachineAntarctica window matches the native grid" begin
    # Pine Island Glacier
    pine_island = ProjectedBoundingBox(x = (-1.62e6, -1.59e6), y = (-2.8e5, -2.5e5))
    grid = native_grid(Metadatum(:ice_thickness; dataset, region = pine_island))
    window = projected_window(dataset, pine_island)
    x, y = projected_window_centers(dataset, window)

    @test topology(grid) == (Bounded, Bounded, Flat)
    @test size(grid)[1:2] == (window.Nx, window.Ny)
    @test xnodes(grid, Center()) ≈ x
    @test ynodes(grid, Center()) ≈ y
    @test first(x) ≤ -1.62e6 && last(x) ≥ -1.59e6
    @test first(y) ≤ -2.8e5 && last(y) ≥ -2.5e5
    @test all(diff(x) .≈ Δ) && all(diff(y) .≈ Δ)
end

@testset "BedMachineAntarctica rejects a source off the lattice or projection" begin
    dir = mktempdir()
    for (name, kw) in (("coarse", (; spacing = 1000)), ("projection", (; standard_parallel = -70.0)))
        source = write_synthetic_bedmachine(joinpath(dir, name * "_source.nc"); kw...)
        @test_throws ErrorException window_to_netcdf(source, joinpath(dir, name * ".nc"), :bottom_height, dataset, region)
    end
end

@testset "BedMachineAntarctica native and regridded fields on $arch" for arch in test_architectures
    dir = mktempdir()
    write_synthetic_bedmachine(joinpath(dir, basename(BedMachineAntarctica_url)))

    native = Field(Metadatum(:bottom_height; dataset, region, dir), arch)
    x = xnodes(native.grid, Center())
    y = ynodes(native.grid, Center())
    values = Array(interior(native, :, :, 1))
    expected = Float32[linear_bed(xᵢ, yⱼ) for xᵢ in x, yⱼ in y]

    # Only the fill cell, sixth column and row from the north-west corner, is NaN.
    missing_cells = findall(isnan, values)
    @test length(missing_cells) == 1
    @test x[missing_cells[1][1]] ≈ x₁ + 5.5Δ
    @test y[missing_cells[1][2]] ≈ y₂ - 5.5Δ
    @test all(isapprox.(values[isfinite.(values)], expected[isfinite.(values)]; rtol = 1e-5))

    mask = Array(interior(Field(Metadatum(:ice_mask; dataset, region, dir), arch), :, :, 1))
    @test mask == [ifelse(xᵢ < x₁ + 10Δ, 2, 3) for xᵢ in x, yⱼ in y]

    target_grid = RectilinearGrid(arch; size = (8, 6), x = (x₁ + 4000, x₁ + 14000), y = (y₂ - 11000, y₂ - 4000),
                                  topology = (Bounded, Bounded, Flat))
    regridded = Field(Metadatum(:bottom_height; dataset, region, dir), target_grid)
    xᵗ = xnodes(target_grid, Center())
    yᵗ = ynodes(target_grid, Center())
    @test all(isapprox.(Array(interior(regridded, :, :, 1)),
                        [linear_bed(xᵢ, yⱼ) for xᵢ in xᵗ, yⱼ in yᵗ]; rtol = 1e-5))
end
