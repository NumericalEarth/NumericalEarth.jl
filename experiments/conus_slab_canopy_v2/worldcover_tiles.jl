# ESA WorldCover is read one 3° × 3° tile at a time: a single continental window would
# materialize the 10 m raster (~340 GB). Each tile is counted onto a 0.01° lattice
# (`aggregation_factor = 120`, 1200 pixels per side per aggregated cell) and cached as its
# own NetCDF, so the ingestion is resumable and parallelizable by tile.

worldcover_dataset() = ESAWorldCover(aggregation_factor = 120)
worldcover_lattice_step = 120 / 12000

# SW corners of the 3° tiles covering the CONUS ingestion window.
function worldcover_tile_corners(longitude, latitude)
    λ₁, λ₂ = longitude
    φ₁, φ₂ = latitude
    return [(Float64(λ), Float64(φ)) for φ in (3 * fld(φ₁, 3)):3:(3 * fld(φ₂, 3))
                                      for λ in (3 * fld(λ₁, 3)):3:(3 * fld(λ₂, 3))]
end

worldcover_tile_region((λ₀, φ₀)) = BoundingBox(longitude = (λ₀, λ₀ + 3.0), latitude = (φ₀, φ₀ + 3.0))

worldcover_tile_metadatum(name, corner) =
    Metadatum(name; dataset = worldcover_dataset(), region = worldcover_tile_region(corner))
