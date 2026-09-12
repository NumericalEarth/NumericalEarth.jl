module TPXO

export TPXO10Atlas

using DocStringExtensions: TYPEDSIGNATURES
using NCDatasets: Dataset
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: Bounded, Center, Face, Flat, LatitudeLongitudeGrid

using NumericalEarth.Oceans: Oceans
using ..DataWrangling: DataWrangling, AbstractStaticDataset, BoundingBox, inpaint_mask!

download_TPXO_cache::String = ""
function __init__()
    global download_TPXO_cache = DataWrangling.download_cache("TPXO10")
end

"""
    TPXO10Atlas()

The TPXO10-atlas-v2 solution for the barotropic tide, on a 1/30° grid
[Egbert and Erofeeva (2002)](@cite egbert2002efficient).

TPXO is licensed, so its files cannot be downloaded: request them at https://www.tpxo.net, then
unpack `grid_tpxo10atlas_v2.nc` together with the elevation and transport files of the constituents
to be forced into `dir`.
"""
struct TPXO10Atlas <: AbstractStaticDataset end

DataWrangling.default_download_directory(::TPXO10Atlas) = download_TPXO_cache

constituent_path(::TPXO10Atlas, dir, variable, constituent) =
    joinpath(dir, string(variable, "_", lowercase(string(constituent)), "_tpxo10_atlas_30_v2.nc"))

atlas_grid_path(::TPXO10Atlas, dir) = joinpath(dir, "grid_tpxo10atlas_v2.nc")

# The atlas grid is an Arakawa C-grid: `lon_u` holds the western faces of the elevation cells and
# `lat_v` their southern faces, so it maps onto a `LatitudeLongitudeGrid` cell for cell.
function atlas_window(dataset, dir, region)
    path = atlas_grid_path(dataset, dir)
    isfile(path) ||
        error("$path not found; TPXO10 is licensed and must be downloaded by hand from https://www.tpxo.net")

    λᶠ, φᶠ = Dataset(path) do atlas
        (atlas["lon_u"][:], atlas["lat_v"][:])
    end

    west, east = mod.(region.longitude, 360)
    south, north = region.latitude

    columns = searchsortedlast(λᶠ, west):(searchsortedfirst(λᶠ, east) - 1)
    rows = searchsortedlast(φᶠ, south):(searchsortedfirst(φᶠ, north) - 1)

    grid = LatitudeLongitudeGrid(size = (length(columns), length(rows)),
                                 longitude = (λᶠ[first(columns)], λᶠ[last(columns) + 1]),
                                 latitude = (φᶠ[first(rows)], φᶠ[last(rows) + 1]),
                                 topology = (Bounded, Bounded, Flat))

    return (; grid, rows, columns)
end

# The atlas stores its arrays as [latitude, longitude], and land as exactly zero in both parts.
function atlas_field(window, path, names, (LX, LY), conversion)
    columns = LX === Face ? (first(window.columns):(last(window.columns) + 1)) : window.columns
    rows = LY === Face ? (first(window.rows):(last(window.rows) + 1)) : window.rows

    real_data, imaginary_data = Dataset(path) do atlas
        map(name -> permutedims(atlas[name][rows, columns]), names)
    end

    mask = Field{LX, LY, Nothing}(window.grid, Bool)
    interior(mask, :, :, 1) .= @. (real_data == 0) & (imaginary_data == 0)

    real_part, imaginary_part = map((real_data, imaginary_data)) do data
        field = Field{LX, LY, Nothing}(window.grid)
        interior(field, :, :, 1) .= conversion .* data
        inpaint_mask!(field, mask)
        field
    end

    constants = Field{LX, LY, Nothing}(window.grid, ComplexF64)
    interior(constants, :, :, 1) .= complex.(interior(real_part, :, :, 1), interior(imaginary_part, :, :, 1))

    return constants
end

"""
$(TYPEDSIGNATURES)

Read the harmonic constants of `constituent` over a window of the atlas covering `grid`. Elevations
are stored in millimeters and transports in centimeters squared per second.
"""
function Oceans.tidal_atlas_constants(dataset::TPXO10Atlas, grid, constituent;
                                      dir = DataWrangling.default_download_directory(dataset))

    window = atlas_window(dataset, dir, BoundingBox(grid; padding = 1))
    elevation_path = constituent_path(dataset, dir, "h", constituent)
    transport_path = constituent_path(dataset, dir, "u", constituent)

    sea_surface_height = atlas_field(window, elevation_path, ("hRe", "hIm"), (Center, Center), 1e-3)
    eastward_transport = atlas_field(window, transport_path, ("uRe", "uIm"), (Face, Center), 1e-4)
    northward_transport = atlas_field(window, transport_path, ("vRe", "vIm"), (Center, Face), 1e-4)

    return (; sea_surface_height, eastward_transport, northward_transport)
end

end # module TPXO
