include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Metadatum, Metadata, native_grid,
    is_three_dimensional, default_inpainting, default_horizontal_padding,
    dataset_variable_name, metadata_filename,
    longitude_name, latitude_name, all_dates, native_times,
    longitude_interfaces, latitude_interfaces,
    retrieve_data, read_file_coords, region_info, dekadal_dates
using NumericalEarth.DataWrangling.CopernicusLandVegetation: vegetation_request_area,
    vegetation_cds_request_variables, retrieval_flag_masks
using Oceananigans.Grids: λnodes, φnodes, topology, Bounded
using NCDatasets: NCDataset, defDim, defVar
using Dates: DateTime, Day, day, month, daysinmonth

# The product's native cell size, 1/336°.
const Δᶜᵍˡˢ = 1/336

# Write a file with the layout of a delivered C3S 300 m file: packed `UInt16` LAI with CF
# scaling, a `UInt32` `retrieval_flag` that also carries a `scale_factor`, coordinates
# labelling each cell's west/south edge, and latitude stored north→south.
function write_synthetic_vegetation_file(path, west_corner, south_corner, Nx, Ny;
                                        fill_at = (), flag_at = Dict())
    scale = 0.00015260186f0
    λ = [west_corner + (i - 1) * Δᶜᵍˡˢ for i in 1:Nx]
    φ = [south_corner + (j - 1) * Δᶜᵍˡˢ for j in 1:Ny]

    packed = Array{UInt16}(undef, Nx, Ny)
    flags = zeros(UInt32, Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        # A ramp unique per cell, so a shifted read cannot pass by coincidence.
        packed[i, j] = UInt16(clamp((i - 1) * Ny + (j - 1), 0, 65534))
    end
    for (i, j) in fill_at
        packed[i, j] = 0xffff
    end
    for ((i, j), flag) in flag_at
        flags[i, j] = flag
    end

    NCDataset(path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defVar(ds, "lon", λ, ("lon",))
        defVar(ds, "lat", reverse(φ), ("lat",))   # stored north→south

        Λ = defVar(ds, "LAI", UInt16, ("lon", "lat");
                   attrib = ["scale_factor" => scale, "add_offset" => 0f0],
                   fillvalue = 0xffff)
        Λ.var[:, :] = reverse(packed, dims = 2)

        flag = defVar(ds, "retrieval_flag", UInt32, ("lon", "lat");
                      attrib = ["scale_factor" => 1f0, "add_offset" => 0f0])
        flag.var[:, :] = reverse(flags, dims = 2)
    end

    return scale, packed, flags
end

@testset "Copernicus vegetation retrieval flags" begin
    @test retrieval_flag_mask() == 0x00000000
    @test retrieval_flag_mask(:obs_unusable) == 0x00000080
    @test retrieval_flag_mask(:obs_unusable, :tip_untrusted) == 0x000000c0
    @test retrieval_flag_mask(:obs_unusable, :obs_unusable) == 0x00000080   # idempotent

    # The recommended screen is exactly the three "no usable retrieval" bits.
    @test unusable_retrieval_flags() ==
          retrieval_flag_mask(:obs_is_fillvalue, :tip_untrusted, :obs_unusable)
    @test unusable_retrieval_flags() == 0x000000c1

    # High-uncertainty bits are deliberately *not* screened by default.
    @test unusable_retrieval_flags() & retrieval_flag_masks.obs_nosnow_hiunc == 0
    @test unusable_retrieval_flags() & retrieval_flag_masks.obs_snow_hiunc == 0

    # Nor are the descriptive bits. `obs_inconsistent` and `obs_nosnow_only` are set on
    # ~95% and ~100% of a mid-latitude July scene, so screening on either — which their
    # names invite — throws away nearly the whole field.
    @test unusable_retrieval_flags() & retrieval_flag_masks.obs_inconsistent == 0
    @test unusable_retrieval_flags() & retrieval_flag_masks.obs_nosnow_only == 0

    # Every mask is a distinct single bit of the product's bitfield.
    masks = collect(values(retrieval_flag_masks))
    @test all(mask -> count_ones(mask) == 1, masks)
    @test length(unique(masks)) == length(masks)

    @test_throws ArgumentError retrieval_flag_mask(:not_a_flag)
end

@testset "Copernicus vegetation dataset interface" begin
    dataset = CopernicusVegetation()
    @test dataset.screened_flags == 0x00000000
    @test CopernicusVegetation(screened_flags = unusable_retrieval_flags()).screened_flags == 0x000000c1

    # Global 1/336° grid, 360° of longitude and the 80°N–60°S band.
    @test size(dataset, :leaf_area_index) == (120960, 47040, 1)

    date = DateTime(2021, 7, 20)
    metadatum = Metadatum(:leaf_area_index; dataset, date)
    @test !is_three_dimensional(metadatum)
    @test isnothing(default_inpainting(metadatum))
    @test dataset_variable_name(metadatum) == "LAI"
    @test longitude_name(metadatum) == "lon"
    @test latitude_name(metadatum) == "lat"
    @test location(metadatum) == (Center, Center, Nothing)
    @test vegetation_cds_request_variables[:leaf_area_index] == "lai"

    # The `GeoTransform` origin is a cell corner, so the interfaces are whole degrees and
    # the cell size comes out exactly 1/336°.
    Nx, Ny, _ = size(dataset, :leaf_area_index)
    λ₁, λ₂ = longitude_interfaces(metadatum)
    φ₁, φ₂ = latitude_interfaces(metadatum)
    @test (λ₁, λ₂) == (-180, 180)
    @test (φ₁, φ₂) == (-60, 80)
    @test (λ₂ - λ₁) / Nx ≈ Δᶜᵍˡˢ
    @test (φ₂ - φ₁) / Ny ≈ Δᶜᵍˡˢ
    @test default_horizontal_padding(dataset) ≈ 4Δᶜᵍˡˢ

    @test CopernicusVegetation in supported_datasets()
end

@testset "Copernicus vegetation dates" begin
    dataset = CopernicusVegetation()
    dates = all_dates(dataset, :leaf_area_index)

    # Sentinel-3 300 m coverage: the record opens mid-2018, the consolidated CDR ends 2024.
    @test first(dates) == DateTime(2018, 7, 10)
    @test last(dates) == DateTime(2024, 12, 31)
    @test issorted(dates)

    # Three composites a month, on day 10, 20, and the last day of the month.
    @test all(d -> day(d) in (10, 20, daysinmonth(d)), dates)
    @test DateTime(2020, 2, 29) in dates   # leap February
    @test DateTime(2021, 2, 28) in dates
    @test DateTime(2021, 9, 30) in dates
    @test count(d -> month(d) == 7 && Dates.year(d) == 2021, dates) == 3

    # The 2018 record starts in July, so that year is half a year short.
    @test count(d -> Dates.year(d) == 2018, dates) == 18
    @test count(d -> Dates.year(d) == 2021, dates) == 36

    @test dates == dekadal_dates(DateTime(2018, 7, 10), DateTime(2024, 12, 31))

    # A (start_date, end_date) window expands to the product's own cadence, bracketing
    # the requested window so it stays interpolatable at both ends.
    metadata = Metadata(:leaf_area_index; dataset,
                        dates = (DateTime(2021, 6, 15), DateTime(2021, 8, 5)))
    @test metadata.dates == [DateTime(2021, 6, 10), DateTime(2021, 6, 20),
                             DateTime(2021, 6, 30), DateTime(2021, 7, 10),
                             DateTime(2021, 7, 20), DateTime(2021, 7, 31),
                             DateTime(2021, 8, 10)]
    @test length(native_times(metadata)) == length(metadata.dates)
end

@testset "Copernicus vegetation filenames" begin
    dataset = CopernicusVegetation()
    date = DateTime(2021, 7, 20)
    region = BoundingBox(longitude = (-92, -91), latitude = (37, 38))
    other_region = BoundingBox(longitude = (10, 11), latitude = (37, 38))

    # Each region is a separate server-side subset, so the filename is keyed by region as
    # well as date — unlike the global-download datasets.
    @test metadata_filename(dataset, :leaf_area_index, date, region) !=
          metadata_filename(dataset, :leaf_area_index, date, other_region)
    @test metadata_filename(dataset, :leaf_area_index, date, region) !=
          metadata_filename(dataset, :leaf_area_index, date + Day(10), region)
    @test metadata_filename(dataset, :leaf_area_index, date, nothing) ==
          metadata_filename(dataset, :leaf_area_index, date, BoundingBox())
    @test occursin("global", metadata_filename(dataset, :leaf_area_index, date, nothing))

    # Filenames must be usable on every filesystem: no minus signs, one extension.
    filename = metadata_filename(dataset, :leaf_area_index, date, region)
    @test !occursin("-9", filename)
    @test endswith(filename, ".nc")
    @test occursin("2021-07-20", filename)
end

@testset "Copernicus vegetation request area" begin
    @test isnothing(vegetation_request_area(nothing))
    @test isnothing(vegetation_request_area(BoundingBox()))

    region = BoundingBox(longitude = (-92, -91), latitude = (37, 38))
    north, west, south, east = vegetation_request_area(region)

    # Six native cells of margin on each side, in CDS's [N, W, S, E] order.
    @test north ≈ 38 + 6Δᶜᵍˡˢ
    @test south ≈ 37 - 6Δᶜᵍˡˢ
    @test west ≈ -92 - 6Δᶜᵍˡˢ
    @test east ≈ -91 + 6Δᶜᵍˡˢ

    # The margin never runs past the product's latitude coverage.
    polar = vegetation_request_area(BoundingBox(longitude = (10, 11), latitude = (79.99, 80)))
    @test polar[1] == 80
    austral = vegetation_request_area(BoundingBox(longitude = (10, 11), latitude = (-60, -59.99)))
    @test austral[3] == -60

    # A [0, 360] bounding box is mapped into the product's [-180, 180] convention.
    mapped = vegetation_request_area(BoundingBox(longitude = (268, 269), latitude = (37, 38)))
    @test mapped[2] ≈ -92 - 6Δᶜᵍˡˢ
    @test mapped[4] ≈ -91 + 6Δᶜᵍˡˢ

    # Windows straddling the ±180 seam would silently pull the whole globe at 300 m.
    @test_throws ArgumentError vegetation_request_area(BoundingBox(longitude = (179, 181),
                                                                   latitude = (37, 38)))
end

@testset "Copernicus vegetation read path" begin
    mktempdir() do dir
        dataset = CopernicusVegetation()
        date = DateTime(2021, 7, 20)
        region = BoundingBox(longitude = (-92, -91), latitude = (37, 38))
        metadatum = Metadatum(:leaf_area_index; dataset, region, date, dir)
        grid = native_grid(metadatum)
        Nx, Ny = size(grid, 1), size(grid, 2)

        # A file covering the native window with the download's margin on every side.
        margin = 6
        gλ = Array(λnodes(grid, Center(), Center(), Center()))
        gφ = Array(φnodes(grid, Center(), Center(), Center()))
        west_corner = gλ[1] - Δᶜᵍˡˢ/2 - margin * Δᶜᵍˡˢ
        south_corner = gφ[1] - Δᶜᵍˡˢ/2 - margin * Δᶜᵍˡˢ

        path = joinpath(dir, metadatum.filename)
        scale, packed, flags = write_synthetic_vegetation_file(path, west_corner, south_corner,
                                                              Nx + 2margin, Ny + 2margin;
                                                              fill_at = ((margin + 2, margin + 3),),
                                                              flag_at = Dict((margin + 4, margin + 5) => 0x00000080,
                                                                             (margin + 6, margin + 7) => 0x00000200))

        # The window handed to `set_region_data!` is exactly the native grid, which pins
        # the region offset to zero rather than leaving it to a float comparison.
        Λ = retrieve_data(metadatum)
        λc, φc = read_file_coords(metadatum)
        @test size(Λ) == (Nx, Ny)
        @test length(λc) == Nx
        @test length(φc) == Ny
        @test region_info(region, Field{Center, Center, Nothing}(grid), λc, φc) ==
              NumericalEarth.DataWrangling.BoundingBoxOffset(0, 0)

        # Coordinates come back as cell centers (the file labels west/south edges) and
        # ascending in latitude (the file stores it north→south).
        # The grid stores nodes in Float32, so allow that rounding but stay far below the
        # half cell a shifted read would move the coordinates by.
        node_atol = Δᶜᵍˡˢ / 20
        @test issorted(λc)
        @test issorted(φc)
        @test λc[1] ≈ gλ[1] atol = node_atol
        @test φc[1] ≈ gφ[1] atol = node_atol
        @test λc[end] ≈ gλ[end] atol = node_atol
        @test φc[end] ≈ gφ[end] atol = node_atol

        # Values line up cell by cell with the window of the synthetic ramp, so a
        # half-pixel shift or a missing latitude flip cannot pass.
        expected = Float32.(packed[margin+1:margin+Nx, margin+1:margin+Ny]) .* scale
        fill_cell = (2, 3)   # written at (margin+2, margin+3), so window cell (2, 3)
        for (i, j) in ((1, 1), (2, 5), (Nx, Ny), (Nx - 3, 7))
            @test Λ[i, j] ≈ expected[i, j]
        end

        # `_FillValue` decodes to NaN, and nothing else does.
        @test isnan(Λ[fill_cell...])
        @test count(isnan, Λ) == 1

        # Flags are ignored by default.
        @test !isnan(Λ[4, 5])
        @test !isnan(Λ[6, 7])

        # Screening reads the bitfield exactly: `retrieval_flag` carries a `scale_factor`,
        # so a CF-decoded read would return a lossy Float32 instead of the raw bits.
        screened = Metadatum(:leaf_area_index;
                             dataset = CopernicusVegetation(screened_flags = unusable_retrieval_flags()),
                             region, date, dir)
        Λscreened = retrieve_data(screened)
        @test isnan(Λscreened[4, 5])         # obs_unusable is in the screen
        @test !isnan(Λscreened[6, 7])        # obs_nosnow_hiunc is not
        @test isnan(Λscreened[fill_cell...])
        @test count(isnan, Λscreened) == 2

        # Screening only ever removes data.
        @test all(i -> isnan(Λscreened[i]) || Λscreened[i] == Λ[i], eachindex(Λ))

        # A hand-picked mask screens exactly what it names.
        high_uncertainty = Metadatum(:leaf_area_index;
                                     dataset = CopernicusVegetation(screened_flags = retrieval_flag_mask(:obs_nosnow_hiunc)),
                                     region, date, dir)
        Λhigh = retrieve_data(high_uncertainty)
        @test isnan(Λhigh[6, 7])
        @test !isnan(Λhigh[4, 5])

        # A field built from the file is the same window, on the native grid.
        field = Field{Center, Center, Nothing}(grid)
        NumericalEarth.DataWrangling.set_metadata_field!(field, Λ, metadatum)
        values = Array(interior(field, :, :, 1))
        @test size(values) == (Nx, Ny)
        @test values[1, 1] ≈ expected[1, 1]
        @test values[Nx, Ny] ≈ expected[Nx, Ny]
        @test isnan(values[fill_cell...])

        # A sub-360° window must be Bounded in x so halos do not wrap.
        @test topology(grid)[1] == Bounded
    end
end

@testset "Copernicus vegetation read path rejects a file that misses the region" begin
    mktempdir() do dir
        dataset = CopernicusVegetation()
        date = DateTime(2021, 7, 20)
        region = BoundingBox(longitude = (-92, -91), latitude = (37, 38))
        metadatum = Metadatum(:leaf_area_index; dataset, region, date, dir)
        grid = native_grid(metadatum)
        gλ = Array(λnodes(grid, Center(), Center(), Center()))
        gφ = Array(φnodes(grid, Center(), Center(), Center()))

        # A file that starts inside the region: too small to fill the native grid. Reading
        # it must fail loudly rather than clamp the edges and silently replicate cells.
        path = joinpath(dir, metadatum.filename)
        write_synthetic_vegetation_file(path,
                                       gλ[1] - Δᶜᵍˡˢ/2 + 4Δᶜᵍˡˢ,
                                       gφ[1] - Δᶜᵍˡˢ/2 + 4Δᶜᵍˡˢ,
                                       size(grid, 1), size(grid, 2))

        @test_throws ErrorException retrieve_data(metadatum)
        @test_throws ErrorException read_file_coords(metadatum)
    end
end
