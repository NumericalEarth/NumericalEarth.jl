# Download and aggregate the ESA WorldCover tiles covering the CONUS window, one worker of
# `NWORKERS` handling every `NWORKERS`-th tile:
#   WORKER=0 NWORKERS=24 julia --project=<docs> ingest_worldcover.jl
# Tiles that fail (ocean cells with no published tile, transient S3 errors) are logged and
# skipped; rerunning fills in whatever is missing.

using NumericalEarth
using Oceananigans
using ArchGDAL

include(joinpath(@__DIR__, "worldcover_tiles.jl"))

worker   = parse(Int, get(ENV, "WORKER", "0"))
nworkers = parse(Int, get(ENV, "NWORKERS", "1"))

corners = worldcover_tile_corners((-134.2, -60.8), (20.8, 53.2))
mine = corners[(worker + 1):nworkers:end]
@info "worker $worker / $nworkers: $(length(mine)) of $(length(corners)) tiles"

for (n, corner) in enumerate(mine)
    metadatum = worldcover_tile_metadatum(:vegetation_fraction, corner)
    path = NumericalEarth.DataWrangling.metadata_path(metadatum)
    if isfile(path)
        @info "[$n/$(length(mine))] cached $(basename(path))"
        continue
    end
    elapsed = @elapsed try
        Field(metadatum, CPU())
    catch err
        @warn "[$n/$(length(mine))] tile $corner skipped" exception = (err, catch_backtrace())
    end
    @info "[$n/$(length(mine))] tile $corner done in $(round(elapsed; digits = 1)) s"
    GC.gc()
end

@info "worker $worker finished"
