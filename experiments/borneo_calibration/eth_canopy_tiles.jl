# ETH canopy height is read one 3° × 3° tile at a time onto a 0.01° lattice (≈ 1.1 km, some
# 12,000 native pixels averaged per cell) and cached per tile.

eth_canopy_lattice_step = 0.01

eth_canopy_tile_grid((λ₀, φ₀)) =
    LatitudeLongitudeGrid(CPU(); size = (300, 300), longitude = (λ₀, λ₀ + 3.0), latitude = (φ₀, φ₀ + 3.0),
                          topology = (Bounded, Bounded, Flat))

eth_canopy_tile_path((λ₀, φ₀)) =
    joinpath(NumericalEarth.DataWrangling.default_download_directory(ETHSentinel2CanopyHeight()),
             "canopy_height_0.01deg_lon_$(λ₀)_lat_$(φ₀).jld2")
