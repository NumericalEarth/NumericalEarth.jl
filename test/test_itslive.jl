include("runtests_setup.jl")

using NumericalEarth.DataWrangling: x_interfaces, y_interfaces, dataset_variable_name, metadata_path,
                                    validate_dataset_coverage, default_inpainting, is_three_dimensional,
                                    native_grid, projected_window, projected_window_centers
using NumericalEarth.DataWrangling.ITSLIVE: cog_window_to_netcdf, validate_source_lattice
using Oceananigans.Grids: xnodes, ynodes, topology, Bounded, Flat

using ArchGDAL

# The real /vsicurl reads need network access and live in test_itslive_downloading.jl. Here a
# synthetic GeoTIFF placed at the north-west corner of the real 120 m lattice stands in for the mosaic.

dataset = ITSLIVEAntarctica()
x₁, _ = x_interfaces(dataset)
_, y₂ = y_interfaces(dataset)
Δ = 120.0

# Linear in x and y, so bilinear regridding reproduces it to rounding.
linear_velocity(x, y) = 1e-3 * (x - x₁) + 2e-3 * (y₂ - y)

function write_synthetic_mosaic(path; Nx = 40, Ny = 30, epsg = 3031, spacing = Δ)
    raw = Float32[linear_velocity(x₁ + (i - 1/2) * spacing, y₂ - (j - 1/2) * spacing) for i in 1:Nx, j in 1:Ny]
    raw[6, 6] = -32767
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"), width = Nx, height = Ny, nbands = 1, dtype = Float32) do ds
        ArchGDAL.setgeotransform!(ds, [x₁, spacing, 0.0, y₂, 0.0, -spacing])
        ArchGDAL.setproj!(ds, ArchGDAL.toWKT(ArchGDAL.importEPSG(epsg)))
        band = ArchGDAL.getband(ds, 1)
        ArchGDAL.setnodatavalue!(band, -32767.0)
        ArchGDAL.write!(band, raw)
    end
    return path
end

region = ProjectedBoundingBox(x = (x₁ + 600, x₁ + 3000), y = (y₂ - 3000, y₂ - 600))

@testset "ITSLIVEAntarctica metadata interfaces" begin
    metadatum = Metadatum(:velocity_x; dataset, region)
    @test dataset_variable_name(metadatum) == "vx"
    @test !is_three_dimensional(metadatum)
    @test isnothing(default_inpainting(metadatum))
    @test metadatum.filename == "ITSLIVE_Antarctica_velocity_x_x_-2.6994075e6_-2.6970075e6_y_2.3970075e6_2.3994075e6.nc"

    @test_throws ErrorException validate_dataset_coverage(nothing, Metadatum(:velocity_x; dataset))
    outside = ProjectedBoundingBox(x = (3e6, 3.1e6), y = (0, 1e5))
    @test_throws ErrorException validate_dataset_coverage(nothing, Metadatum(:velocity_x; dataset, region = outside))
end

@testset "ITSLIVEAntarctica window matches the native grid" begin
    # Pine Island Glacier
    pine_island = ProjectedBoundingBox(x = (-1.62e6, -1.59e6), y = (-2.8e5, -2.5e5))
    metadatum = Metadatum(:velocity_x; dataset, region = pine_island)
    grid = native_grid(metadatum)
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

@testset "ITSLIVEAntarctica rejects a source off the mosaic lattice" begin
    dir = mktempdir()
    for (name, kw) in (("geographic", (; epsg = 4326)), ("coarse", (; spacing = 240.0)))
        tif = write_synthetic_mosaic(joinpath(dir, name * ".tif"); kw...)
        @test_throws ErrorException cog_window_to_netcdf(tif, joinpath(dir, name * ".nc"), "vx", dataset, region)
    end
end

@testset "ITSLIVEAntarctica native and regridded fields on $arch" for arch in test_architectures
    dir = mktempdir()
    metadatum = Metadatum(:velocity_x; dataset, region, dir)
    tif = write_synthetic_mosaic(joinpath(dir, "vx.tif"))
    cog_window_to_netcdf(tif, metadata_path(metadatum), "vx", dataset, region)

    native = Field(metadatum, arch)
    x = xnodes(native.grid, Center())
    y = ynodes(native.grid, Center())
    values = Array(interior(native, :, :, 1))
    expected = Float32[linear_velocity(xᵢ, yⱼ) for xᵢ in x, yⱼ in y]

    # Only the nodata cell, sixth column and row from the north-west corner and outside the regridding stencil, is NaN.
    nodata = findall(isnan, values)
    @test length(nodata) == 1
    @test x[nodata[1][1]] ≈ x₁ + 5.5Δ
    @test y[nodata[1][2]] ≈ y₂ - 5.5Δ
    @test all(isapprox.(values[isfinite.(values)], expected[isfinite.(values)]; rtol = 1e-5))

    target_grid = RectilinearGrid(arch; size = (8, 6), x = (x₁ + 900, x₁ + 2700), y = (y₂ - 2700, y₂ - 900),
                                  topology = (Bounded, Bounded, Flat))
    regridded = Field(metadatum, target_grid)
    xᵗ = xnodes(target_grid, Center())
    yᵗ = ynodes(target_grid, Center())
    @test all(isapprox.(Array(interior(regridded, :, :, 1)),
                        [linear_velocity(xᵢ, yⱼ) for xᵢ in xᵗ, yⱼ in yᵗ]; rtol = 1e-5))
end
