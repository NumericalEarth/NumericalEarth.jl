include("runtests_setup.jl")

using NumericalEarth.DataWrangling: FieldRegridding, field_cache_filename, field_cache_path,
                                    load_field_cache, save_field_cache

@testset "Field cache keys" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (36, 18, 4),
                                 longitude = (0, 90),
                                 latitude = (-45, 45),
                                 z = (-1000, 0))

    metadatum = Metadatum(:temperature; dataset = EN4Monthly(), date = start_date)

    config1 = FieldRegridding(grid, metadatum, (;))
    config2 = FieldRegridding(grid, metadatum, (;))
    @test config1 == config2
    @test hash(config1) == hash(config2)
    @test field_cache_filename(config1) == field_cache_filename(config2)

    # Different grid geometry, read keywords, or date must key differently.
    other_grid = LatitudeLongitudeGrid(CPU();
                                       size = (18, 18, 4),
                                       longitude = (0, 90),
                                       latitude = (-45, 45),
                                       z = (-1000, 0))
    @test field_cache_filename(FieldRegridding(other_grid, metadatum, (;))) !=
          field_cache_filename(config1)
    @test field_cache_filename(FieldRegridding(grid, metadatum, (; inpainting = 7))) !=
          field_cache_filename(config1)

    other_datum = Metadatum(:salinity; dataset = EN4Monthly(), date = start_date)
    @test field_cache_filename(FieldRegridding(grid, other_datum, (;))) !=
          field_cache_filename(config1)

    # A mismatched stored config is rejected rather than served.
    save_field_cache(config1, zeros(4, 4, 4))
    stale = FieldRegridding(grid, other_datum, (;))
    @test isnothing(load_field_cache(stale))
    @test load_field_cache(config1) == zeros(4, 4, 4)
    rm(field_cache_path(config1); force = true)
end

@testset "Field cache round-trip" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (40, 30, 5),
                                 longitude = (0, 60),
                                 latitude = (-30, 30),
                                 z = (-800, 0))

    metadatum = Metadatum(:temperature; dataset = EN4Monthly(), date = start_date)

    # Key the cache only after the dataset file exists — the key stamps it
    download(metadatum)
    config = FieldRegridding(grid, metadatum, (;))
    rm(field_cache_path(config); force = true)

    first_read  = Field(metadatum, grid; cache = true)
    @test isfile(field_cache_path(config))

    second_read = Field(metadatum, grid; cache = true)
    @test all(isequal.(parent(first_read), parent(second_read)))

    uncached = Field(metadatum, grid)
    @test all(isequal.(parent(first_read), parent(uncached)))

    # overwrite_cache = true skips the lookup and refreshes the entry
    save_field_cache(config, zeros(size(grid)))
    poisoned = Field(metadatum, grid; cache = true)
    @test all(iszero, interior(poisoned))

    refreshed = Field(metadatum, grid; cache = true, overwrite_cache = true)
    @test all(isequal.(parent(refreshed), parent(first_read)))

    reread = Field(metadatum, grid; cache = true)
    @test all(isequal.(parent(reread), parent(first_read)))

    rm(field_cache_path(config); force = true)
end
