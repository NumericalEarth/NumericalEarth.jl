include("runtests_setup.jl")

using CDSAPI
using Dates
using NCDatasets

using NumericalEarth.DataWrangling: metadata_path, BoundingBox, Column
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel

date = DateTime(2005, 2, 16, 12)
global_longitudes = collect(0:0.25:359.75)
global_latitudes = collect(90:-0.25:-90)

# Distinct at every point, and periodic in longitude so a relabeled longitude carries the same value
surface_value(λ, φ) = Float32(mod(λ, 360) + 1000φ)

function write_era5_file(path, λ, φ)
    NCDataset(path, "c") do ds
        defDim(ds, "longitude", length(λ))
        defDim(ds, "latitude", length(φ))
        defDim(ds, "valid_time", 1)
        defVar(ds, "longitude", λ, ("longitude",))
        defVar(ds, "latitude", φ, ("latitude",))
        time = defVar(ds, "valid_time", Float64, ("valid_time",), attrib = Dict("units" => "seconds since 1970-01-01"))
        time[:] = [date]
        defVar(ds, "t2m", [surface_value(λ, φ) for λ in λ, φ in φ, _ in 1:1], ("longitude", "latitude", "valid_time"))
    end
    return path
end

# Stands in for the CDS: answers with the points of the global grid inside the request's area,
# labeled in the area's longitude convention, and counts the requests.
retrievals = Ref(0)

function fake_retrieve(product, request, path)
    retrievals[] += 1
    haskey(request, "area") || return write_era5_file(path, global_longitudes, global_latitudes)
    north, west, south, east = request["area"]
    λ = sort([west + mod(λ - west, 360) for λ in global_longitudes if mod(λ - west, 360) ≤ east - west])
    φ = filter(φ -> south ≤ φ ≤ north, global_latitudes)
    return write_era5_file(path, λ, φ)
end

temperature(region, dir) = Metadatum(:temperature; dataset = ERA5HourlySingleLevel(), date, region, dir)

# Download `metadatum` through the fake CDS and return its Field and the number of retrievals it took
function field_and_retrievals(metadatum)
    retrievals[] = 0
    download(metadatum; retrieve = fake_retrieve)
    return interior(Field(metadatum)), retrievals[]
end

function with_global_file(f)
    mktempdir() do dir
        write_era5_file(joinpath(dir, temperature(nothing, dir).filename), global_longitudes, global_latitudes)
        f(dir)
    end
end

@testset "Regional requests are served from a cached global file" begin
    regions = (BoundingBox(longitude = (10, 20), latitude = (40, 50)),
               BoundingBox(longitude = (-10, 10), latitude = (-5, 5)),    # across the file's 0° seam
               BoundingBox(longitude = (350, 370), latitude = (60, 70)),  # east of 360°
               Column(15.1, 45.1))

    with_global_file() do global_dir
        for region in regions
            served, served_retrievals = field_and_retrievals(temperature(region, global_dir))
            downloaded, downloaded_retrievals = mktempdir(dir -> field_and_retrievals(temperature(region, dir)))
            @test served_retrievals == 0
            @test downloaded_retrievals == 1
            @test served == downloaded
        end

        # A Column is interpolated from the global file's 2 × 2 stencil
        column, _ = field_and_retrievals(temperature(Column(15.1, 45.1), global_dir))
        @test only(column) ≈ surface_value(15.1, 45.1)

        # Multi-date downloads skip dates the global file serves
        metadata = Metadata(:temperature; dataset = ERA5HourlySingleLevel(), dates = [date],
                            region = BoundingBox(longitude = (10, 20), latitude = (40, 50)), dir = global_dir)
        retrievals[] = 0
        @test download(metadata; retrieve = fake_retrieve) == [joinpath(global_dir, temperature(nothing, global_dir).filename)]
        @test retrievals[] == 0
    end
end

@testset "A larger regional file serves the requests it covers" begin
    mktempdir() do dir
        large = temperature(BoundingBox(longitude = (-20, 20), latitude = (30, 60)), dir)
        download(large; retrieve = fake_retrieve)
        large_path = metadata_path(large)

        for region in (BoundingBox(longitude = (0, 10), latitude = (40, 50)),
                       BoundingBox(longitude = (345, 355), latitude = (40, 50)),  # other longitude convention
                       Column(-5.1, 45.1))
            served, served_retrievals = field_and_retrievals(temperature(region, dir))
            downloaded, _ = mktempdir(other_dir -> field_and_retrievals(temperature(region, other_dir)))
            @test metadata_path(temperature(region, dir)) == large_path
            @test served_retrievals == 0
            @test served == downloaded
        end

        for region in (BoundingBox(longitude = (10, 30), latitude = (40, 50)),
                       BoundingBox(longitude = (0, 10), latitude = (50, 70)))
            uncovered = temperature(region, dir)
            @test metadata_path(uncovered) == joinpath(dir, uncovered.filename)
            @test last(field_and_retrievals(uncovered)) == 1
        end
    end
end
