include("../omip_defaults.jl")

using Oceananigans.Operators: Δzᶜᶜᶜ
using WorldOceanAtlasTools

arch = GPU()
Nz   = 100
z_faces = ExponentialDiscretization(Nz, -5500, 0; scale=1600, mutable=true)

grid = ORCAGrid(arch; Nz,
                z = z_faces,
                halo = (7, 7, 7))

Nx, Ny, Nz = size(grid)

const zˢ  = CUDA.@allowscalar znode(Nz, grid, Face()) - 1
const Δzˢ = CUDA.@allowscalar Δzᶜᶜᶜ(1, 1, Nz, grid)

omip = omip_simulation(grid; forcing_dir="forcing_data", restoring_dir="climatology", filename = "orca")

run!(omip)

omip.Δt = 30minutes
omip.stop_time = 300 * 365days

run!(omip)
