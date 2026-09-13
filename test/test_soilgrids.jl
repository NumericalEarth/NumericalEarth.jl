include("runtests_setup.jl")

using NumericalEarth.DataWrangling.SoilGrids
using NumericalEarth.DataWrangling.SoilGrids: Statistic, Mean, Q5, Q50, Q95,
                                              SoilGrids2_dataset_variable_names,
                                              SoilGrids2_z_interfaces, SoilGrids2_depth_ranges,
                                              soilgrids_spacing_meters,
                                              soilgrids_statistic_url_name,
                                              soilgrids_vsicurl_source,
                                              soilgrids_raster_geometry,
                                              soilgrids_variable_to_netcdf
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, native_grid,
                                    longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, available_variables,
                                    is_three_dimensional, default_inpainting, missing_value,
                                    windowed_retrieval, conversion_units, convert_units,
                                    GramPerKilogram, CentigramPerCubicCentimeter,
                                    HectogramPerCubicMeter, DecigramPerKilogram

using Oceananigans.Fields: location
using Oceananigans.Grids: x_domain, y_domain, λnodes, φnodes

#####
##### Pure logic: no network, no ArchGDAL needed.
#####

@testset "SoilGrids2 constructor and defaults" begin
    ds = SoilGrids2()
    @test ds.statistic == Mean
    @test ds.resolution == Grid1000m
    @test summary(ds) == "SoilGrids2(statistic = Mean, resolution = Grid1000m)"

    ds2 = SoilGrids2(statistic = Q95, resolution = Grid250m)
    @test ds2.statistic == Q95
    @test ds2.resolution == Grid250m
end

@testset "SoilGrids2 Statistic → ISRIC URL fragment" begin
    @test soilgrids_statistic_url_name(Mean) == "mean"
    @test soilgrids_statistic_url_name(Q5)   == "Q0.05"
    @test soilgrids_statistic_url_name(Q50)  == "Q0.5"
    @test soilgrids_statistic_url_name(Q95)  == "Q0.95"
end

@testset "SoilGrids2 vsicurl source URL" begin
    url = soilgrids_vsicurl_source("clay", "0-5cm", "mean")
    @test startswith(url, "/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=")
    @test endswith(url, "https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt")

    # Every combination of variable, depth, and statistic is representable.
    for name in values(SoilGrids2_dataset_variable_names), depth in SoilGrids2_depth_ranges,
        stat in (Mean, Q5, Q50, Q95)
        url = soilgrids_vsicurl_source(name, depth, soilgrids_statistic_url_name(stat))
        @test occursin("$(name)_$(depth)_$(soilgrids_statistic_url_name(stat)).vrt", url)
    end
end

@testset "SoilGrids2 native resolution" begin
    @test soilgrids_spacing_meters(Clenshaw10km) == 10_000
    @test soilgrids_spacing_meters(Grid1000m) == 1000
    @test soilgrids_spacing_meters(Grid250m) == 250
end

@testset "SoilGrids2 dataset interface across resolutions" begin
    # Clenshaw10km: unchanged legacy layout.
    legacy = SoilGrids2(resolution = Clenshaw10km)
    @test longitude_interfaces(legacy) == (0, 360)
    @test latitude_interfaces(legacy)  == (-90, 90)
    @test z_interfaces(legacy) == [-200, -100, -60, -30, -15, -5, 0]
    Nx, Ny, Nz = size(legacy, :clay_fraction)
    @test (Nx, Ny, Nz) == (3956, 1979, 6)

    # Grid1000m / Grid250m: real meters, standard (-180,180) convention.
    for resolution in (Grid1000m, Grid250m)
        dataset = SoilGrids2(; resolution)
        @test longitude_interfaces(dataset) == (-180, 180)
        @test latitude_interfaces(dataset)  == (-90, 90)
        @test z_interfaces(dataset) == SoilGrids2_z_interfaces
        @test z_interfaces(dataset)[1] == -2.0   # bottom face at 2 m, the true 100-200 cm depth
        Nx, Ny, Nz = size(dataset, :clay_fraction)
        @test Nz == 6
        @test Nx > Ny > 0
    end

    # 250 m is 4× finer than 1000 m.
    Nx250, _, _ = size(SoilGrids2(resolution = Grid250m), :clay_fraction)
    Nx1000, _, _ = size(SoilGrids2(resolution = Grid1000m), :clay_fraction)
    @test Nx250 ≈ 4 * Nx1000 rtol = 1e-2

    @test windowed_retrieval(SoilGrids2(resolution = Grid250m))
    @test !windowed_retrieval(SoilGrids2(resolution = Grid1000m))
    @test !windowed_retrieval(SoilGrids2(resolution = Clenshaw10km))
end

@testset "SoilGrids2 metadatum interface" begin
    region = BoundingBox(longitude = (-0.35, -0.25), latitude = (51.20, 51.27))

    for resolution in (Grid250m, Grid1000m, Clenshaw10km)
        dataset = SoilGrids2(; resolution)
        md = Metadatum(:clay_fraction; dataset, region)
        @test dataset_variable_name(md) == "clay"
        @test is_three_dimensional(md)
        @test default_inpainting(md) === nothing
        @test location(md) == (Center, Center, Center)
    end

    @test Set(keys(available_variables(SoilGrids2()))) ==
          Set((:sand_fraction, :silt_fraction, :clay_fraction, :coarse_fraction,
               :bulk_density, :organic_carbon_density, :soil_organic_carbon))

    # The legacy path carries its own raw integer sentinel; the ISRIC-direct pipeline masks
    # nodata to NaN itself during the reprojection, so no sentinel remains.
    md_legacy = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Clenshaw10km))
    @test missing_value(md_legacy) == -32768
    md_isric = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Grid1000m))
    @test missing_value(md_isric) === missing
end

@testset "SoilGrids2 unit conversions" begin
    # Texture / coarse fraction: g/kg → kg/kg.
    @test conversion_units(Metadatum(:clay_fraction; dataset = SoilGrids2())) isa GramPerKilogram
    @test conversion_units(Metadatum(:sand_fraction; dataset = SoilGrids2())) isa GramPerKilogram
    @test conversion_units(Metadatum(:coarse_fraction; dataset = SoilGrids2())) isa GramPerKilogram
    @test convert_units(224.0f0, GramPerKilogram()) ≈ 0.224f0

    # Bulk density: cg/cm³ → kg/dm³.
    @test conversion_units(Metadatum(:bulk_density; dataset = SoilGrids2())) isa CentigramPerCubicCentimeter

    # Organic carbon density: hg/m³ → kg/m³.
    @test conversion_units(Metadatum(:organic_carbon_density; dataset = SoilGrids2())) isa HectogramPerCubicMeter

    # Soil organic carbon: dg/kg → g/kg.
    @test conversion_units(Metadatum(:soil_organic_carbon; dataset = SoilGrids2())) isa DecigramPerKilogram
end

@testset "SoilGrids2 filename disambiguation" begin
    region_a = BoundingBox(longitude = (-0.35, -0.25), latitude = (51.20, 51.27))
    region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))

    # Clenshaw10km always shares the same filename, regardless of variable/statistic/region.
    legacy = SoilGrids2(resolution = Clenshaw10km)
    @test metadata_filename(legacy, :clay_fraction, nothing, nothing) ==
          metadata_filename(legacy, :sand_fraction, nothing, region_a)

    # Grid1000m is global by default: same filename with or without an (unused) region,
    # but distinct per variable and statistic.
    kilo = SoilGrids2(resolution = Grid1000m)
    @test metadata_filename(kilo, :clay_fraction, nothing, nothing) ==
          "SoilGrids2_clay_fraction_Mean_1000m_global.nc"
    @test metadata_filename(kilo, :clay_fraction, nothing, nothing) !=
          metadata_filename(kilo, :sand_fraction, nothing, nothing)
    @test metadata_filename(kilo, :clay_fraction, nothing, nothing) !=
          metadata_filename(SoilGrids2(resolution = Grid1000m, statistic = Q5), :clay_fraction, nothing, nothing)

    # Grid250m filenames are additionally keyed by region.
    fine = SoilGrids2(resolution = Grid250m)
    @test metadata_filename(fine, :clay_fraction, nothing, region_a) !=
          metadata_filename(fine, :clay_fraction, nothing, region_b)
    @test metadata_filename(fine, :clay_fraction, nothing, region_a) !=
          metadata_filename(kilo, :clay_fraction, nothing, nothing)
end

@testset "SoilGrids2 requires a bounded region at 250 m only" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 6),
                                 longitude = (-0.35, -0.25), latitude = (51.20, 51.27),
                                 z = SoilGrids2_z_interfaces)
    region = BoundingBox(longitude = (-0.35, -0.25), latitude = (51.20, 51.27))

    meta_global_fine = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Grid250m))
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global_fine)
    @test_throws ErrorException download(meta_global_fine)

    meta_region_fine = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Grid250m), region)
    @test validate_dataset_coverage(grid, meta_region_fine) === nothing

    # 1000 m and the legacy 10 km path stay global; no region is required.
    for resolution in (Grid1000m, Clenshaw10km)
        meta_global = Metadatum(:clay_fraction; dataset = SoilGrids2(; resolution))
        @test validate_dataset_coverage(grid, meta_global) === nothing
    end
end

@testset "SoilGrids2 raster geometry matches native_grid" begin
    region = BoundingBox(longitude = (-0.35, -0.25), latitude = (51.20, 51.27))

    # Bounded region (Grid250m's required case).
    md_region = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Grid250m), region)
    grid_region = native_grid(md_region)
    raster_region = soilgrids_raster_geometry(md_region)
    @test (raster_region.Nx, raster_region.Ny) == size(grid_region)[1:2]
    @test (raster_region.west, raster_region.east) == x_domain(grid_region)
    @test (raster_region.south, raster_region.north) == y_domain(grid_region)
    @test raster_region.longitude == collect(λnodes(grid_region, Center()))
    @test raster_region.latitude  == collect(φnodes(grid_region, Center()))

    # Global (Grid1000m's default case) — same function, no region needed.
    md_global = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Grid1000m))
    grid_global = native_grid(md_global)
    raster_global = soilgrids_raster_geometry(md_global)
    @test (raster_global.Nx, raster_global.Ny) == size(grid_global)[1:2]
    @test raster_global.west  ≈ -180.0 atol = 1e-2
    @test raster_global.east  ≈  180.0 atol = 1e-2
    @test raster_global.south ≈  -90.0 atol = 1e-2
    @test raster_global.north ≈   90.0 atol = 1e-2
end

@testset "SoilGrids2 ISRIC read is extension-gated" begin
    meta = Metadatum(:clay_fraction; dataset = SoilGrids2(resolution = Grid1000m))
    if isnothing(Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt))
        @test_throws ErrorException soilgrids_variable_to_netcdf(meta, tempname() * ".nc")
    end
end

#####
##### Clenshaw10km: real network (small pre-baked global file, unchanged from before this
##### rewrite). Excluded from CI via `remote_data_tests` in runtests.jl.
#####

@testset "SoilGrids2 (10 km legacy) metadata download" begin
    legacy = SoilGrids2(resolution = Clenshaw10km)
    sg_sand = Metadatum(:sand_fraction, dataset = legacy)
    download(sg_sand)
    @test isfile(metadata_path(sg_sand))
    for var in keys(SoilGrids2_dataset_variable_names)
        sgmd = Metadatum(var, dataset = legacy)
        field = Field(sgmd)
        @test any(isfinite.(field))
    end
    for var in (:sand_fraction, :silt_fraction, :clay_fraction)
        sgmd = Metadatum(var, dataset = legacy)
        field = Field(sgmd)
        # Check that sand/silt/clay fractions are between 0 and 1
        min, max = extrema(filter(isfinite, field.data))
        @test 0 <= min <= 1
    end
end
