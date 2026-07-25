include("runtests_setup.jl")

using CDSAPI  # loads NumericalEarthCDSAPIExt (the `satellite-lai-fapar` download path)
using NumericalEarth.DataWrangling: BoundingBox, Metadata, Metadatum, native_grid,
    retrieve_data, read_file_coords, region_info, BoundingBoxOffset
using Oceananigans.Grids: λnodes, φnodes, topology, Bounded
using Downloads: Downloads
using Dates: DateTime

# Network-gated: hits the Climate Data Store, which queues requests. Excluded from the
# default suite in runtests.jl, mirroring the other *_downloading tests.
@testset "Copernicus vegetation download and regional read" begin
    dataset = CopernicusVegetation()
    region = BoundingBox(longitude = (-91.6, -91.2), latitude = (37.2, 37.6))
    date = DateTime(2021, 7, 20)

    metadatum = Metadatum(:leaf_area_index; dataset, region, date)
    grid = native_grid(metadatum)
    Λ = Field(metadatum)
    values = Array(interior(Λ, :, :, 1))

    # The delivered file is subset server-side and read back as exactly the native-grid
    # window, so the region offset is zero and no cell is filled by an edge clamp.
    @test size(values) == (size(grid, 1), size(grid, 2))
    λc, φc = read_file_coords(metadatum)
    @test length(λc) == size(grid, 1)
    @test length(φc) == size(grid, 2)
    @test region_info(region, Λ, λc, φc) == BoundingBoxOffset(0, 0)

    valid = filter(!isnan, vec(values))
    @test !isempty(valid)
    @test all(Λ -> 0 ≤ Λ ≤ 10, valid)      # the product's packed range decodes to [0, 10]
    @test length(unique(valid)) > 1          # a real window, not a constant fill

    # Every cell holds the file pixel it is geographically nearest to — the check that
    # catches a half-pixel coordinate convention error or a missing latitude flip.
    data = retrieve_data(metadatum)
    gλ = Array(λnodes(grid, Center(), Center(), Center()))
    gφ = Array(φnodes(grid, Center(), Center(), Center()))
    mismatches = 0
    for i in axes(values, 1), j in axes(values, 2)
        nearest = data[argmin(abs.(λc .- gλ[i])), argmin(abs.(φc .- gφ[j]))]
        held = values[i, j]
        (held === nearest || (isnan(held) && isnan(nearest))) || (mismatches += 1)
    end
    @test mismatches == 0

    # Sub-360° window must be Bounded in x so halos do not wrap.
    @test topology(grid)[1] == Bounded

    # Screening the unusable-retrieval bits can only remove pixels, never add them.
    screened = Metadatum(:leaf_area_index;
                         dataset = CopernicusVegetation(screened_flags = unusable_retrieval_flags()),
                         region, date)
    screened_values = Array(interior(Field(screened), :, :, 1))
    @test count(isnan, screened_values) ≥ count(isnan, values)
    kept = .!isnan.(screened_values)
    @test all(screened_values[kept] .== values[kept])
end

@testset "Copernicus vegetation agrees across two independent deliveries" begin
    dataset = CopernicusVegetation()
    date = DateTime(2021, 7, 20)

    # Two separate requests whose areas snap onto the pixel lattice differently, the second
    # box wholly inside the first. Every cell of the overlap must carry the same value: an
    # off-by-one in the coordinate convention, the latitude flip, or the read window would
    # break the agreement even though each read looks self-consistent on its own.
    outer = Metadatum(:leaf_area_index; dataset, date,
                      region = BoundingBox(longitude = (-91.6, -91.2), latitude = (37.2, 37.6)))
    inner = Metadatum(:leaf_area_index; dataset, date,
                      region = BoundingBox(longitude = (-91.4731, -91.2077),
                                           latitude = (37.3313, 37.5519)))

    Downloads.download(outer)
    Downloads.download(inner)

    Λouter, Λinner = retrieve_data(outer), retrieve_data(inner)
    λouter, φouter = read_file_coords(outer)
    λinner, φinner = read_file_coords(inner)

    Δ = 1/336
    matched = 0
    agreed = 0
    for i in eachindex(λinner), j in eachindex(φinner)
        io = findfirst(λ -> abs(λ - λinner[i]) < Δ/8, λouter)
        jo = findfirst(φ -> abs(φ - φinner[j]) < Δ/8, φouter)
        (isnothing(io) || isnothing(jo)) && continue
        matched += 1
        a, b = Λinner[i, j], Λouter[io, jo]
        (a === b || (isnan(a) && isnan(b))) && (agreed += 1)
    end

    @test matched == length(Λinner)   # the inner box lies entirely inside the outer one
    @test agreed == matched
end

@testset "Copernicus vegetation seasonal FieldTimeSeries" begin
    region = BoundingBox(longitude = (-91.6, -91.2), latitude = (37.2, 37.6))
    grid = LatitudeLongitudeGrid(CPU(), Float32; size = (16, 16),
                                 longitude = (-91.6, -91.2), latitude = (37.2, 37.6),
                                 topology = (Bounded, Bounded, Flat))

    # One dekad per season, so green-up and senescence are both in the series.
    metadata = Metadata(:leaf_area_index; dataset = CopernicusVegetation(), region,
                        dates = [DateTime(2021, 1, 20), DateTime(2021, 4, 20),
                                 DateTime(2021, 7, 20), DateTime(2021, 10, 20)])

    Λ = FieldTimeSeries(metadata, grid; time_indices_in_memory = 2)
    @test length(Λ.times) == 4

    seasonal_mean = map(1:4) do n
        values = filter(!isnan, vec(Array(interior(Λ[n], :, :, 1))))
        sum(values) / length(values)
    end

    @test all(Λ -> 0 ≤ Λ ≤ 10, seasonal_mean)
    # Deciduous forest and cropland: summer LAI clearly exceeds winter LAI.
    @test seasonal_mean[3] > seasonal_mean[1]
end
