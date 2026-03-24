include("diagnostic_runs.jl")

static_grid = default_one_degree_grid(; mutable=false, active_cells_map=false)
static_ocean_kwargs = default_ocean_configuration(static_grid)
static_ocean_kwargs = merge(static_ocean_kwargs, (momentum_advection=nothing, tracer_advection=nothing))
# Run #1 (default)
run_one_degree_diagnostic("static_noadvection_noactivecells", GPU(); grid = static_grid, ocean_kwargs = static_ocean_kwargs)
