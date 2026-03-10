include("../omip_defaults.jl")

arch = GPU()

z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800, mutable=true)

const z_surf = z_faces.cᵃᵃᶠ(Nz)

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid; minimum_depth=20, major_basins=1, interpolation_passes=25)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

omip = omip_simulation(grid; forcing_dir, restoring_dir, filename = "halfdegree")

run!(omip)

omip.Δt = 30minutes
omip.stop_time = 300 * 365days

run!(omip)