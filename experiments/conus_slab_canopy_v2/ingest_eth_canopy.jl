# Area-average the ETH Sentinel-2 10 m canopy height onto a 0.01° lattice, one 3° tile per
# call, one worker of `NWORKERS` handling every `NWORKERS`-th tile:
#   WORKER=0 NWORKERS=24 julia --project=<docs> ingest_eth_canopy.jl
# Each tile lands in its own JLD2 under the dataset's cache directory, so the ingestion is
# resumable; tiles that fail (open ocean has no published tile) are logged and skipped.

using NumericalEarth
using Oceananigans
using ArchGDAL
using JLD2

include(joinpath(@__DIR__, "worldcover_tiles.jl"))
include(joinpath(@__DIR__, "eth_canopy_tiles.jl"))

worker   = parse(Int, get(ENV, "WORKER", "0"))
nworkers = parse(Int, get(ENV, "NWORKERS", "1"))

corners = worldcover_tile_corners((-134.2, -60.8), (20.8, 53.2))
mine = corners[(worker + 1):nworkers:end]
@info "worker $worker / $nworkers: $(length(mine)) of $(length(corners)) tiles"

for (n, corner) in enumerate(mine)
    path = eth_canopy_tile_path(corner)
    if isfile(path)
        @info "[$n/$(length(mine))] cached $(basename(path))"
        continue
    end
    elapsed = @elapsed try
        grid = eth_canopy_tile_grid(corner)
        height = canopy_height_field(grid, ETHSentinel2CanopyHeight())
        mkpath(dirname(path))
        jldsave(path; height = Array(interior(height, :, :, 1)), corner)
    catch err
        @warn "[$n/$(length(mine))] tile $corner skipped" exception = (err, catch_backtrace())
    end
    @info "[$n/$(length(mine))] tile $corner done in $(round(elapsed; digits = 1)) s"
    GC.gc()
end

@info "worker $worker finished"
