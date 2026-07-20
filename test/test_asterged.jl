include("runtests_setup.jl")

using NCDatasets: NCDataset, defDim, defVar

using NumericalEarth.DataWrangling.ASTERGED
using NumericalEarth.DataWrangling.ASTERGED: asterged_decode_emissivity, asterged_decode_uncertainty,
    broadband_emissivity, broadband_uncertainty, broadband_map, place_tile!,
    OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces,
    dataset_variable_name, validate_dataset_coverage, metadata_filename,
    is_three_dimensional, default_inpainting, native_grid, default_horizontal_padding,
    NearestNeighborInpainting, supported_datasets
using NumericalEarth: stateindex

using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.Fields: location

# The real HDF5 tile read + Earthdata download live in the ArchGDAL extension and
# need network + NASA Earthdata credentials. Everything else — the decode/broadband
# core, the analytic tile placement (`place_tile!`), the dataset-interface logic,
# and a synthetic-NetCDF regional Field (CPU and GPU) — is credential-free and
# exercised here.

@testset "ASTER GED decode scaling" begin
    # Mean scale is 0.001; fill −9999 → NaN (not −9.999).
    @test asterged_decode_emissivity(950) ≈ 0.95
    @test asterged_decode_emissivity(982) ≈ 0.982
    @test isnan(asterged_decode_emissivity(-9999))
    @test asterged_decode_emissivity(-9999) != -9.999

    # SDev scale is 0.0001 — 10× smaller than Mean. Decoding with the Mean scale
    # would be a silent 10× error.
    @test asterged_decode_uncertainty(120) ≈ 0.012
    @test asterged_decode_uncertainty(120) != asterged_decode_emissivity(120)
    @test asterged_decode_emissivity(120) ≈ 10 * asterged_decode_uncertainty(120)
    @test isnan(asterged_decode_uncertainty(-9999))
end

@testset "ASTER GED broadband synthesis" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # The write-time collapse assumes normalized weights; guard the const against drift.
    @test sum(coefficients) ≈ 1
    @test length(coefficients) == 5

    # Equal bands → that value; a non-uniform vector → the plain dot product.
    @test broadband_emissivity(fill(0.96, 5), coefficients) ≈ 0.96
    ε = [0.92, 0.88, 0.90, 0.97, 0.98]
    @test broadband_emissivity(ε, coefficients) ≈ sum(coefficients .* ε)
end

@testset "ASTER GED broadband uncertainty (correlated upper bound)" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS
    σ = fill(0.012, 5)

    # Fully-correlated band errors: σ = Σ cᵢ σᵢ (= 0.012 here, since Σ cᵢ = 1), and
    # strictly larger than the independence-assuming RSS which biases σ low.
    @test broadband_uncertainty(σ, coefficients) ≈ 0.012
    @test broadband_uncertainty(σ, coefficients) > sqrt(sum(coefficients .^ 2 .* σ .^ 2))
end

@testset "ASTER GED broadband_map: band index is first dimension" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # Synthetic tile: (5 bands, Nx, Ny). Bands differ so a transposed reduction
    # would give a different (wrong) answer.
    Nx, Ny = 4, 3
    decoded = Array{Float32}(undef, 5, Nx, Ny)
    for b in 1:5, i in 1:Nx, j in 1:Ny
        decoded[b, i, j] = 0.90 + 0.01 * b
    end

    ε = broadband_map(decoded, coefficients)
    @test size(ε) == (Nx, Ny)
    @test eltype(ε) == Float32
    @test all(ε .≈ Float32(broadband_emissivity([0.91, 0.92, 0.93, 0.94, 0.95], coefficients)))
end

@testset "ASTER GED fill values propagate to NaN" begin
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # A fill (−9999) in any band decodes to NaN and propagates to the broadband.
    raw = fill(Int16(950), 5, 2, 2)
    raw[3, 1, 1] = Int16(-9999)
    ε = broadband_map(asterged_decode_emissivity.(raw), coefficients)
    @test isnan(ε[1, 1])
    @test all(isfinite, ε[2:end, :][:])
end

@testset "ASTER GED place_tile! analytic placement" begin
    # Native regional axes: 5×4 uniform cells, Δ = 0.01.
    longitude = collect(10.005:0.01:10.045)   # 5 cells
    latitude  = collect(45.005:0.01:45.035)   # 4 cells
    field = fill(NaN32, length(longitude), length(latitude))

    # A tile over the middle, latitude DESCENDING (as ASTER GED tiles store it),
    # with distinct values so a wrong index or orientation is caught.
    tile_longitude = [10.015, 10.025, 10.035]
    tile_latitude  = [45.035, 45.025, 45.015, 45.005]           # north → south
    tile_values    = Float32[10il + jl for il in 1:3, jl in 1:4]  # (3, 4)

    place_tile!(field, tile_values, tile_longitude, tile_latitude, longitude, latitude)

    # Tile starts at 10.015 → native column 2, so columns 1 and 5 stay untouched.
    @test all(isnan, field[1, :])
    @test all(isnan, field[5, :])
    # (il, jl) = (1, 1): lon 10.015 → col 2, lat 45.035 → row 4.
    @test field[2, 4] == tile_values[1, 1]
    # (il, jl) = (3, 4): lon 10.035 → col 4, lat 45.005 → row 1.
    @test field[4, 1] == tile_values[3, 4]
    # Descending tile latitude mapped by value, not order: (1, 4) → row 1.
    @test field[2, 1] == tile_values[1, 4]
end

@testset "ASTER GED place_tile! mosaics adjacent tiles; gaps stay NaN" begin
    longitude = collect(0.005:0.01:0.055)   # 6 cells
    latitude  = collect(0.005:0.01:0.025)   # 3 cells
    field = fill(NaN32, length(longitude), length(latitude))

    tileA_longitude = [0.005, 0.015, 0.025]   # native columns 1–3
    tileB_longitude = [0.035, 0.045, 0.055]   # native columns 4–6
    tile_latitude   = [0.005, 0.015, 0.025]
    valuesA = fill(0.90f0, 3, 3)
    valuesB = fill(0.95f0, 3, 3)
    valuesB[2, 2] = NaN32                     # a retrieval gap inside tile B

    place_tile!(field, valuesA, tileA_longitude, tile_latitude, longitude, latitude)
    place_tile!(field, valuesB, tileB_longitude, tile_latitude, longitude, latitude)

    @test all(field[1:3, :] .== 0.90f0)
    @test field[4, 1] == 0.95f0
    @test isnan(field[5, 2])          # gap left NaN (for the downstream inpainting)
    @test count(isnan, field) == 1
end

@testset "ASTER GED dataset interfaces" begin
    for resolution in (:high_100m, :low_1km)
        ds = ASTERGEDv3(; resolution)
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, 90)
        @test default_horizontal_padding(ds) > 0

        Nx, Ny, Nz = size(ds)
        @test Nz == 1
        @test Nx == (resolution === :high_100m ? 360_000 : 36_000)
        @test Ny == (resolution === :high_100m ? 180_000 : 18_000)

        region = BoundingBox(longitude = (110, 112), latitude = (0, 2))
        meta = Metadatum(:emissivity; dataset = ds, region)
        @test dataset_variable_name(meta) == "emissivity"
        @test is_three_dimensional(meta) == false
        @test default_inpainting(meta) isa NearestNeighborInpainting

        meta_unc = Metadatum(:emissivity_uncertainty; dataset = ds, region)
        @test dataset_variable_name(meta_unc) == "emissivity_uncertainty"

        filename = metadata_filename(ds, :emissivity, nothing, region)
        @test startswith(filename, "ASTERGED_")
        @test occursin(string(resolution), filename)
        @test endswith(filename, ".nc")
    end

    @test ASTERGEDv3().resolution == :low_1km          # coarse product is the default
    @test_throws ArgumentError ASTERGEDv3(resolution = :medium_500m)

    # Non-parametric concrete struct → discoverable by `supported_datasets()`.
    @test ASTERGEDv3 in supported_datasets()
end

@testset "ASTER GED filenames: region-keyed, variable-independent" begin
    ds = ASTERGEDv3()
    region_a = BoundingBox(longitude = (110, 112), latitude = (0, 2))
    region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))
    @test metadata_filename(ds, :emissivity, nothing, region_a) !=
          metadata_filename(ds, :emissivity, nothing, region_b)

    # One regional file serves both variables (the tile download produces both).
    @test metadata_filename(ds, :emissivity, nothing, region_a) ==
          metadata_filename(ds, :emissivity_uncertainty, nothing, region_a)
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

# Write a regional NetCDF of decoded broadband floats with the exact layout the
# ArchGDAL download step produces, so `Field(metadatum, arch)` runs the full
# retrieve/inpaint pipeline without credentials or network.
function write_synthetic_asterged_netcdf(path, λc, φc, emissivity, uncertainty)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", length(λc))
        defDim(ds, "lat", length(φc))
        defVar(ds, "lon", Float64, ("lon",))[:] = λc
        defVar(ds, "lat", Float64, ("lat",))[:] = φc
        defVar(ds, "emissivity", Float32, ("lon", "lat"))[:, :] = emissivity
        defVar(ds, "emissivity_uncertainty", Float32, ("lon", "lat"))[:, :] = uncertainty
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

        emissivity = fill(0.95f0, Nx, Ny)
        emissivity[2, 2] = NaN32                # retrieval gap → inpainted
        uncertainty = fill(0.01f0, Nx, Ny)

        write_synthetic_asterged_netcdf(metadata_path(metadatum), λc, φc, emissivity, uncertainty)

        field = Field(metadatum, arch)
        @test location(field) == (Center, Center, Nothing)

        values = Array(interior(field, :, :, 1))
        @test all(isfinite, values)             # gap was inpainted
        @test values[3, 3] ≈ 0.95 atol = 1e-4
        @test values[2, 2] ≈ 0.95 atol = 1e-4   # inpainted from uniform neighbors

        # The reduced (Center, Center, Nothing) field slots directly into
        # `SurfaceRadiationProperties` and is safely `stateindex`-able at any k,
        # as the air-land interface flux kernels do at k = Nz.
        properties = SurfaceRadiationProperties(albedo = 0.3, emissivity = field)
        if arch isa CPU
            ϵ = stateindex(properties.emissivity, 3, 3, 7, field.grid, nothing, (Center, Center, Center))
            @test ϵ ≈ 0.95 atol = 1e-4
        end
    end
end
