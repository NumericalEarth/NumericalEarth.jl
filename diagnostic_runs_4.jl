include("diagnostic_runs.jl")

# Default configuration
grid = default_one_degree_grid()
ocean_kwargs = default_ocean_configuration(grid)

static_grid = default_one_degree_grid(; mutable=false)
static_ocean_kwargs = default_ocean_configuration(grid)

# Run #6 - ORCA grid
Nz = 75
depth = 6000meters 

static_orca_grid = ORCAGrid(GPU(); halo = (7, 7, 7), Nz, z = ExponentialDiscretization(Nz, -depth, 0; mutable=false, scale=1800))
orca_ocean_kwargs = default_ocean_configuration(static_orca_grid)
run_one_degree_diagnostic("static_orca_grid", GPU(); grid = static_orca_grid, ocean_kwargs = orca_ocean_kwargs)

# Run #7 - static ORCA grid
orca_grid = ORCAGrid(GPU(); halo = (7, 7, 7), Nz, z = ExponentialDiscretization(Nz, -depth, 0; mutable=true, scale=1800))
orca_ocean_kwargs = default_ocean_configuration(_orca_grid)
run_one_degree_diagnostic("orca_grid", GPU(); grid = static_orca_grid, ocean_kwargs = static_orca_ocean_kwargs)

