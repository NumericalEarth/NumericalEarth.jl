include("../omip_defaults.jl")

using Oceananigans.Operators: Δzᶜᶜᶜ
using WorldOceanAtlasTools

restart_iteration = "120000"

arch = GPU()

Nx = 720
Ny = 360
Nz = 100

z_faces = ExponentialDiscretization(Nz, -5500, 0; scale=1600, mutable=true)

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid; minimum_depth=20, major_basins=1, interpolation_passes=25)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

const zˢ  = CUDA.@allowscalar znode(Nz, grid, Face()) - 1
const Δzˢ = CUDA.@allowscalar Δzᶜᶜᶜ(1, 1, Nz, grid)

omip = omip_simulation(grid; forcing_dir="forcing_data", restart_iteration, restoring_dir="climatology",filename = "halfdegree")

run!(omip)

omip.Δt = 30minutes
omip.stop_time = 300 * 365days

run!(omip)
