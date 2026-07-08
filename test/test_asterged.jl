include("runtests_setup.jl")

using NCDatasets: NCDataset, defDim, defVar

using NumericalEarth.DataWrangling.ASTERGED
using NumericalEarth.DataWrangling.ASTERGED: decode_mean, decode_sdev,
    broadband_emissivity, broadband_emissivity_map, broadband_uncertainty_map,
    fill_water, OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS, ASTERGED_WATER_CODE
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces,
    dataset_variable_name, validate_dataset_coverage, metadata_filename,
    is_three_dimensional, default_inpainting, missing_value, native_grid,
    NearestNeighborInpainting
using NumericalEarth: stateindex

using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.Fields: location

# The real HDF5 tile read + Earthdata download live in the ArchGDAL extension and
# need network + NASA Earthdata credentials, so the pure decode/broadband/fill
# core, the dataset-interface logic, and a synthetic-NetCDF regional Field
# (CPU and GPU) are exercised here; the live download path is verified
# end-to-end separately (see future_plans/status/aster-ged_STATUS.md).

@testset "ASTER GED decode scaling" begin
    # Mean scale is 0.001; fill −9999 → NaN (not −9.999).
    @test decode_mean(950) ≈ 0.95
    @test decode_mean(982) ≈ 0.982
    @test isnan(decode_mean(-9999))
    @test decode_mean(-9999) != -9.999

    # SDev scale is 0.0001 — 10× smaller than Mean. Decoding with the Mean scale
    # would be a silent 10× error.
    @test decode_sdev(120) ≈ 0.012
    @test decode_sdev(120) != decode_mean(120)
    @test decode_mean(120) ≈ 10 * decode_sdev(120)
    @test isnan(decode_sdev(-9999))
end

@testset "ASTER GED broadband synthesis" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # The Ogawa & Schmugge (2004) weights form a convex combination (they sum
    # to exactly unity — no dropped intercept — and are all nonnegative).
    @test sum(coefficients) ≈ 1
    @test all(coefficients .>= 0)
    @test length(coefficients) == 5

    # A convex combination of equal band emissivities returns that value.
    @test broadband_emissivity(fill(0.96, 5), coefficients) ≈ 0.96

    # Broadband stays within the range of the band emissivities → in [0.7, 1.0]
    # when all bands are typical land values.
    ε = [0.92, 0.88, 0.90, 0.97, 0.98]
    εᵇᵇ = broadband_emissivity(ε, coefficients)
    @test minimum(ε) <= εᵇᵇ <= maximum(ε)
    @test 0.7 <= εᵇᵇ <= 1.0
end

@testset "ASTER GED band index is first dimension" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # Synthetic tile: (5 bands, Nx, Ny). Bands differ so a transposed reduction
    # would give a different (wrong) answer.
    Nx, Ny = 4, 3
    decoded = Array{Float64}(undef, 5, Nx, Ny)
    for b in 1:5, i in 1:Nx, j in 1:Ny
        decoded[b, i, j] = 0.90 + 0.01 * b
    end

    ε = broadband_emissivity_map(decoded, coefficients)
    @test size(ε) == (Nx, Ny)

    expected = broadband_emissivity([0.91, 0.92, 0.93, 0.94, 0.95], coefficients)
    @test all(ε .≈ Float32(expected))
    @test all(0.7 .<= ε .<= 1.0)
end

@testset "ASTER GED fill values propagate to NaN" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # A fill (−9999) in any band decodes to NaN and propagates to the broadband.
    raw = fill(Int16(950), 5, 2, 2)
    raw[3, 1, 1] = Int16(-9999)
    ε = broadband_emissivity_map(decode_mean.(raw), coefficients)
    @test isnan(ε[1, 1])
    @test all(isfinite, ε[2:end, :][:])
end

@testset "ASTER GED water filling" begin
    ε = fill(0.95f0, 3, 3)

    # LWmap with a water pixel (verified coding: land == 0, water == 1, so the
    # default ASTERGED_WATER_CODE == 1). Water gets the prescribed emissivity,
    # not NaN — a NaN would poison downstream flux kernels.
    lwmap = fill(0, 3, 3)
    lwmap[2, 2] = ASTERGED_WATER_CODE
    filled = fill_water(ε, lwmap, 0.985)
    @test filled[2, 2] == 0.985f0
    @test eltype(filled) == Float32
    @test filled[1, 1] == 0.95f0
    @test count(==(0.985f0), filled) == 1

    # Verify the keyword overrides the default coding.
    lwmap2 = fill(1, 3, 3)
    lwmap2[1, 3] = 2
    filled2 = fill_water(ε, lwmap2, 0.985; water_code = 2)
    @test filled2[1, 3] == 0.985f0
    @test count(==(0.985f0), filled2) == 1
end

@testset "ASTER GED uncertainty broadband" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # SDev raw of 120 → 0.012 per band. Broadband uncertainty ≤ max band σ.
    raw = fill(Int16(120), 5, 2, 2)
    σ = broadband_uncertainty_map(decode_sdev.(raw), coefficients)
    @test size(σ) == (2, 2)
    @test all(0 .< σ .<= 0.012 + eps())
end

@testset "ASTER GED dataset interfaces" begin
    for resolution in (:AG100, :AG1km)
        ds = ASTERGEDv3(; resolution)
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, 90)

        Nx, Ny, Nz = size(ds)
        @test Nz == 1
        @test Nx == (resolution === :AG100 ? 360_000 : 36_000)
        @test Ny == (resolution === :AG100 ? 180_000 : 18_000)

        region = BoundingBox(longitude = (110, 112), latitude = (0, 2))
        meta = Metadatum(:emissivity; dataset = ds, region)
        @test dataset_variable_name(meta) == "/Emissivity/Mean"
        @test is_three_dimensional(meta) == false
        @test default_inpainting(meta) isa NearestNeighborInpainting
        @test missing_value(meta) == -9999

        meta_unc = Metadatum(:emissivity_uncertainty; dataset = ds, region)
        @test dataset_variable_name(meta_unc) == "/Emissivity/SDev"

        filename = metadata_filename(ds, :emissivity, nothing, region)
        @test startswith(filename, "ASTERGED_")
        @test occursin(resolution === :AG100 ? "AG100" : "AG1km", filename)
        @test endswith(filename, ".nc")
    end

    @test_throws ArgumentError ASTERGEDv3(resolution = :AG5km)
end

@testset "ASTER GED region-keyed filenames are distinct" begin
    ds = ASTERGEDv3()
    region_a = BoundingBox(longitude = (110, 112), latitude = (0, 2))
    region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))
    name_a = metadata_filename(ds, :emissivity, nothing, region_a)
    name_b = metadata_filename(ds, :emissivity, nothing, region_b)
    @test name_a != name_b
end

@testset "ASTER GED requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (10, 10),
                                 longitude = (110, 112),
                                 latitude = (0, 2),
                                 topology = (Bounded, Bounded, Flat))

    meta_global = Metadatum(:emissivity; dataset = ASTERGEDv3())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)
    @test_throws ErrorException download(meta_global)

    region = BoundingBox(longitude = (110, 112), latitude = (0, 2))
    meta_region = Metadatum(:emissivity; dataset = ASTERGEDv3(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end

# Write a regional NetCDF of raw digital numbers with the exact layout the
# ArchGDAL download step produces, so `Field(metadatum, arch)` runs the full
# retrieve/decode/fill/inpaint pipeline without credentials or network.
function write_synthetic_asterged_netcdf(path, λc, φc, mean_dn, sdev_dn, lwmap)
    NCDataset(path, "c") do ds
        defDim(ds, "band", 5)
        defDim(ds, "lon", length(λc))
        defDim(ds, "lat", length(φc))
        defVar(ds, "lon", Float64, ("lon",))[:] = λc
        defVar(ds, "lat", Float64, ("lat",))[:] = φc
        defVar(ds, "emissivity_mean", Int16, ("band", "lon", "lat"))[:, :, :] = mean_dn
        defVar(ds, "emissivity_sdev", Int16, ("band", "lon", "lat"))[:, :, :] = sdev_dn
        defVar(ds, "land_water_map", Int16, ("lon", "lat"))[:, :] = lwmap
    end
    return path
end

@testset "ASTER GED synthetic regional Field [$(typeof(arch))]" for arch in test_architectures
    mktempdir() do dir
        dataset = ASTERGEDv3()
        region  = BoundingBox(longitude = (10.0, 10.05), latitude = (45.0, 45.05))
        metadatum = Metadatum(:emissivity; dataset, region, dir)

        # File coordinates = native-grid cell centers, as the download step writes.
        grid_native = native_grid(metadatum, CPU())
        λc = λnodes(grid_native, Center())
        φc = φnodes(grid_native, Center())
        Nx, Ny = length(λc), length(φc)

        mean_dn = Array{Int16}(undef, 5, Nx, Ny)
        for b in 1:5
            mean_dn[b, :, :] .= Int16(930 + 10b)   # bands 0.94, 0.95, …, 0.98
        end
        mean_dn[:, 2, 2] .= Int16(-9999)           # retrieval gap → inpainted
        sdev_dn = fill(Int16(120), 5, Nx, Ny)
        lwmap = zeros(Int16, Nx, Ny)
        lwmap[1, 1] = ASTERGED_WATER_CODE          # water → water_emissivity

        write_synthetic_asterged_netcdf(metadata_path(metadatum), λc, φc, mean_dn, sdev_dn, lwmap)

        field = Field(metadatum, arch)
        @test location(field) == (Center, Center, Nothing)

        values = Array(interior(field, :, :, 1))
        expected = broadband_emissivity(0.001 .* (940:10:980), dataset.broadband_coefficients)

        @test all(isfinite, values)                # gap was inpainted, water filled
        @test values[1, 1] ≈ dataset.water_emissivity atol = 1e-6
        @test values[3, 3] ≈ expected atol = 1e-4
        @test values[2, 2] ≈ expected atol = 1e-4  # inpainted from uniform neighbors
        @test all(0.7 .<= values .<= 1.0)

        # The reduced (Center, Center, Nothing) field slots directly into
        # `SurfaceRadiationProperties` and is safely `stateindex`-able at any k,
        # as the air-land interface flux kernels do at k = Nz.
        properties = SurfaceRadiationProperties(albedo = 0.3, emissivity = field)
        if arch isa CPU
            ϵ = stateindex(properties.emissivity, 3, 3, 7, field.grid, nothing, (Center, Center, Center))
            @test ϵ ≈ expected atol = 1e-4
        end
    end
end
