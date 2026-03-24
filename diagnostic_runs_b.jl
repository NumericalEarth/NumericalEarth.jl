include("diagnostic_runs.jl")

# Default configuration
grid = default_one_degree_grid()
ocean_kwargs = default_ocean_configuration(grid)
ocean_kwargs = merge(ocean_kwargs, (; barotropic_forcing = false))

# Run #1 (default)
run_one_degree_diagnostic("no_barotropic_forcing", GPU(); grid, ocean_kwargs)

grid = default_one_degree_grid()
ocean_kwargs = default_ocean_configuration(grid)
ocean_kwargs = merge(ocean_kwargs, (; bottom_drag_coefficients = (μ = 0.003, U = 0.1)))

run_one_degree_diagnostic("higher_drag", GPU(); grid, ocean_kwargs)
