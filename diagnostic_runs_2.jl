include("diagnostic_runs.jl")

grid = default_one_degree_grid()
ocean_kwargs = default_ocean_configuration(grid)

sim = run_one_degree_diagnostic("new_baseline_", GPU(); grid, ocean_kwargs)

# Default configuration
grid = default_one_degree_grid()
ocean_kwargs = default_ocean_configuration(grid)

# Run #2 — only change closure
coriolis = HydrostaticSphericalCoriolis(; scheme = Oceananigans.Coriolis.ActiveWeightedEnergyConserving())

ocean_kwargs = merge(ocean_kwargs, (; coriolis))

sim = run_one_degree_diagnostic("with_coriolis_awen_", GPU(); grid, ocean_kwargs)
