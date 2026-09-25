#####
##### The area a download fetches for a region
#####

build_era5_area(::Nothing) = nothing

# Columns and unbounded regions: the area is a pure function of the region.
era5_request_area(region, dataset, name) = build_era5_area(region)

# Bounding box: the native grid is built by center-bracketing `restrict`, which
# can reach one cell past a boundary-aligned edge. Fetch two native cells of
# margin (in the bbox's own longitude convention) so the downloaded file always
# covers the grid the data is interpolated onto — otherwise downscaling leaves
# NaNs at the domain edges. Over-fetching is harmless: `restrict` selects the
# exact cells from the larger file.
function era5_request_area(bbox::BoundingBox, dataset, name)
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    Nx, Ny, _ = size(dataset, name)
    Δλ = 360 / Nx
    Δφ = 180 / Ny
    lon = bbox.longitude
    lat = bbox.latitude
    padded = BoundingBox(longitude = (lon[1] - 2Δλ, lon[2] + 2Δλ),
                         latitude  = (max(lat[1] - 2Δφ, -90), min(lat[2] + 2Δφ, 90)))
    return build_era5_area(padded)
end

function build_era5_area(bbox::BoundingBox)
    lon = bbox.longitude
    lat = bbox.latitude

    if isnothing(lon) || isnothing(lat)
        return nothing
    end

    west  = lon[1]
    east  = lon[2]
    south = lat[1]
    north = lat[2]

    return [north, west, south, east]
end

# Column with Nearest interpolation: tight box; CDS returns the nearest cell.
function build_era5_area(col::Column{<:Any, <:Any, <:Any, <:Nearest})
    lon, lat = col.longitude, col.latitude
    ε = 1e-3
    return [lat + ε, lon - ε, lat - ε, lon + ε]  # [N, W, S, E]
end

# Column with Linear interpolation: pad by slightly more than ERA5's native
# 0.25° spacing so the file contains the 2x2 stencil bilinear interp needs.
function build_era5_area(col::Column{<:Any, <:Any, <:Any, <:DataWrangling.Linear})
    lon, lat = col.longitude, col.latitude
    ε = 0.3
    return [lat + ε, lon - ε, lat - ε, lon + ε]
end

#####
##### Serving a request from a cached file that covers it
#####

# Products stored one snapshot per file, which any file of the same variable and date can serve
const ERA5SnapshotMetadatum = Metadatum{<:Union{ERA5HourlySingleLevel, ERA5MonthlySingleLevel, ERA5PressureLevelsDataset}}

# Degrees within which a file coordinate matches a requested bound
const coordinate_tolerance = 1e-4

# Sorted file names in `dir`, listed again when `dir` changes
const directory_listings = Dict{String, Tuple{Float64, Float64, Vector{String}}}()

function cached_filenames(dir)
    isdir(dir) || return String[]
    modified = mtime(dir)
    listing = get(directory_listings, dir, nothing)

    # Timestamps are coarse: a listing taken within a second of a change may miss another change in the same tick
    if isnothing(listing) || listing[1] != modified || listing[2] - modified < 1
        listing = directory_listings[dir] = (modified, time(), readdir(dir))
    end

    return listing[3]
end

struct FileExtent
    west :: Float64
    east :: Float64
    south :: Float64
    north :: Float64
    periodic :: Bool
end

const file_extents = Dict{Tuple{String, Float64}, FileExtent}()

function file_extent(path)
    return get!(file_extents, (path, mtime(path))) do
        λ, φ = NCDatasets.Dataset(ds -> (ds["longitude"][:], ds["latitude"][:]), path)
        FileExtent(first(λ), last(λ), minimum(φ), maximum(φ), !isnothing(infer_longitudinal_period(λ)))
    end
end

function covers(extent, area)
    north, west, south, east = area
    ε = coordinate_tolerance
    covers_latitude = extent.south ≤ south + ε && extent.north ≥ north - ε
    west = extent.west + mod(west - extent.west + ε, 360) - ε
    covers_longitude = extent.periodic || west + (area[4] - area[2]) ≤ extent.east + ε
    return covers_latitude && covers_longitude
end

# A region's four bounds, as regional filenames end
const region_suffix_pattern = r"^(_(-?\d+\.\d|nothing)){4}\.nc$"

"""
    covering_cached_file(metadatum::ERA5SnapshotMetadatum, dir)

The first file in `dir` holding `metadatum`'s variable and date — the global file, then
regional ones — whose longitudes and latitudes contain the area a download for `metadatum`
would fetch, or `nothing`.
"""
function DataWrangling.covering_cached_file(metadatum::ERA5SnapshotMetadatum, dir)
    area = era5_request_area(metadatum.region, metadatum.dataset, metadatum.name)
    isnothing(area) && return nothing

    global_filename = DataWrangling.metadata_filename(metadatum.dataset, metadatum.name, metadatum.dates, nothing)
    stem = chopsuffix(global_filename, ".nc")
    filenames = cached_filenames(dir)

    for filename in view(filenames, searchsortedfirst(filenames, stem):length(filenames))
        startswith(filename, stem) || break
        suffix = chopprefix(filename, stem)
        suffix == ".nc" || occursin(region_suffix_pattern, suffix) || continue
        path = joinpath(dir, filename)
        isfile(path) || continue
        covers(file_extent(path), area) && return path
    end

    return nothing
end

"""
    cached_window(metadatum, path)

Longitude indices, latitude indices, longitudes, and latitudes of the part of the file at `path`
that a download for `metadatum` fetches: all of `metadatum`'s own file, or the points of a
covering file inside the requested area, with longitudes relabeled into the area's convention.
"""
function cached_window(metadatum, path)
    λ, φ = NCDatasets.Dataset(ds -> (ds["longitude"][:], ds["latitude"][:]), path)
    basename(path) == metadatum.filename && return eachindex(λ), eachindex(φ), λ, φ

    north, west, south, east = era5_request_area(metadatum.region, metadatum.dataset, metadatum.name)
    ε = coordinate_tolerance
    λ = @. λ + 360 * ceil((west - ε - λ) / 360)
    i = filter(i -> λ[i] ≤ east + ε, sortperm(λ))
    j = findall(φ -> south - ε ≤ φ ≤ north + ε, φ)

    # A Nearest column's area holds no point: take the one closest to it
    isempty(i) && (i = [argmin(@. abs(mod(λ - (west + east) / 2 + 180, 360) - 180))])
    isempty(j) && (j = [argmin(@. abs(φ - (south + north) / 2))])

    return i, first(j):last(j), λ[i], φ[first(j):last(j)]
end

function DataWrangling.read_file_coords(metadatum::ERA5SnapshotMetadatum)
    _, _, λ, φ = cached_window(metadatum, metadata_path(metadatum))
    return λ, reverse(φ)
end
