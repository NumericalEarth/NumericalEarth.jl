module NumericalEarthZarrExt

using Zarr: Zarr, zopen
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans.Grids: x_domain, y_domain
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: native_grid

const CopernicusDEM = NumericalEarth.DataWrangling.CopernicusDEM

#####
##### bitround filter
#####

# The GLO-90 store applies the numcodecs `bitround` filter (lossy mantissa
# rounding done at write time). Decoding is a passthrough: the stored values are
# already the rounded floats, so no inverse is needed. Zarr.jl has no built-in
# bitround filter, so we register a minimal one in its global `filterdict`.
struct BitRoundFilter{T, Tenc} <: Zarr.Filter{T, Tenc}
    keepbits::Int32
end

BitRoundFilter(; keepbits = 14, T = Float32, Tenc = T) = BitRoundFilter{T, Tenc}(Int32(keepbits))

Zarr.zencode(data::AbstractArray, ::BitRoundFilter) = data
Zarr.zdecode(data::AbstractArray, ::BitRoundFilter) = data
Zarr.JSON.lower(filter::BitRoundFilter) = Dict("id" => "bitround", "keepbits" => filter.keepbits)
Zarr.getfilter(::Type{<:BitRoundFilter}, d) = BitRoundFilter(; keepbits = d["keepbits"])
Zarr.filterdict["bitround"] = BitRoundFilter

#####
##### Copernicus DEM Zarr → regional NetCDF
#####

# The Earth Data Hub Copernicus DEM stores name their coordinates "lon"/"lat" and
# store `dsm` as (lon, lat); both coordinates are ascending. Ascending vs descending
# is detected and handled regardless, so only the coordinate names and dimension
# order are assumed here.
function CopernicusDEM.zarr_to_netcdf(metadatum::CopernicusDEM.CopernicusDEMMetadatum, nc_path)
    token = get(ENV, "DESTINE_ACCESS_TOKEN", nothing)
    isnothing(token) && error(
        "Set the DESTINE_ACCESS_TOKEN environment variable to read Copernicus DEM. " *
        "Register at https://platform.destine.eu/ and create a token at " *
        "https://earthdatahub.destine.eu/account-settings#my-personal-access-tokens.")

    url = string("https://edh:", token, "@", CopernicusDEM.zarr_host_path(metadatum.dataset))
    store = zopen(url; consolidated = true)

    grid = native_grid(metadatum)
    λ₁, λ₂ = x_domain(grid)
    φ₁, φ₂ = y_domain(grid)
    Nx, Ny, _ = size(grid)

    Δλ = (λ₂ - λ₁) / Nx
    Δφ = (φ₂ - φ₁) / Ny
    first_longitude = λ₁ + Δλ / 2
    first_latitude  = φ₁ + Δφ / 2

    longitude = store["lon"][:]
    latitude  = store["lat"][:]

    longitude_range, longitude_ascending = ascending_window(longitude, first_longitude, Nx)
    latitude_range,  latitude_ascending  = ascending_window(latitude,  first_latitude,  Ny)

    variable_name = CopernicusDEM.dataset_zarr_variable_name
    elevation = Float32.(store[variable_name][longitude_range, latitude_range])  # (lon, lat)

    # The native LatitudeLongitudeGrid is ascending in both lon and lat.
    longitude_ascending || (elevation = reverse(elevation, dims = 1))
    latitude_ascending  || (elevation = reverse(elevation, dims = 2))

    window_longitude = longitude_ascending ? longitude[longitude_range] : reverse(longitude[longitude_range])
    window_latitude  = latitude_ascending  ? latitude[latitude_range]   : reverse(latitude[latitude_range])

    NCDataset(nc_path, "c") do dataset
        defDim(dataset, "lon", length(window_longitude))
        defDim(dataset, "lat", length(window_latitude))

        longitude_variable = defVar(dataset, "lon", Float64, ("lon",);
                                    attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        latitude_variable  = defVar(dataset, "lat", Float64, ("lat",);
                                    attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        elevation_variable = defVar(dataset, "z", Float32, ("lon", "lat");
                                    attrib = ["units" => "m", "long_name" => "surface elevation"])

        longitude_variable[:] = window_longitude
        latitude_variable[:]  = window_latitude
        elevation_variable[:, :] = elevation
    end

    return nothing
end

# A contiguous block of `count` storage indices into `coordinate` whose values
# bracket the window starting near `target_first`, returned in storage order
# together with whether `coordinate` is ascending. Assumes the store resolution
# matches the native grid, so a contiguous block of length `count` is exact.
function ascending_window(coordinate, target_first, count)
    n = length(coordinate)
    ascending = coordinate[1] < coordinate[end]
    ascending_coordinate = ascending ? coordinate : reverse(coordinate)

    start = searchsortednearest(ascending_coordinate, target_first)
    start = clamp(start, 1, n - count + 1)
    ascending_range = start:(start + count - 1)

    storage_range = ascending ? ascending_range :
                                (n - ascending_range.stop + 1):(n - ascending_range.start + 1)

    return storage_range, ascending
end

function searchsortednearest(sorted, value)
    i = searchsortedfirst(sorted, value)
    i == 1 && return 1
    i > length(sorted) && return length(sorted)
    return abs(sorted[i] - value) < abs(sorted[i-1] - value) ? i : i - 1
end

end # module NumericalEarthZarrExt
