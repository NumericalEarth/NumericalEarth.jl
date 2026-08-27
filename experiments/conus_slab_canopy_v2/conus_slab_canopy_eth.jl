# The CONUS canopy-land hindcast with the canopy height measured by the ETH Sentinel-2 10 m
# product (area-averaged per cell) in place of the IGBP class heights. Where the product sees
# no trees — crops, grass, shrubs — the class height stands in as the floor, so herbaceous
# cover keeps its roughness. Everything else is `conus_slab_canopy_v2.jl`; the ETH tiles are
# fetched beforehand by `ingest_eth_canopy.jl`.

ENV["CANOPY_HEIGHT"] = "eth"
get!(ENV, "LANDCOVER", "worldcover")
get!(ENV, "TAG", "conus12km_eth")

include(joinpath(@__DIR__, "conus_slab_canopy_v2.jl"))
