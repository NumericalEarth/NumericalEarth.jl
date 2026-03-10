include("../omip_defaults.jl")

arch = GPU()

z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800, mutable=true)

const z_surf = z_faces.cᵃᵃᶠ(Nz)

grid = ORCAGrid(arch; Nz,
                z = z_faces,
                halo = (7, 7, 7))

Nx, Ny, Nz = size(grid)

omip = omip_simulation(grid; forcing_dir, restoring_dir, filename = "orca")

run!(omip)

omip.Δt = 30minutes
omip.stop_time = 300 * 365days

run!(omip)