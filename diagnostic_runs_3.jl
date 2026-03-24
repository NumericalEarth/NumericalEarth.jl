include("diagnostic_runs.jl")

using Oceananigans.Units

grid = default_one_degree_grid(mutable=false)

@inline νz(x, y, z, t) = ifelse(z < -50, 1e-4, 1e-2)
@inline κz(x, y, z, t) = ifelse(z < -20, 1e-5, 1e-2)

ocean_kwargs = default_ocean_configuration(grid)
ocean_kwargs = merge(ocean_kwargs, (; closure = (VerticalScalarDiffusivity(κ=κz, ν=νz), ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1.0))))
sim = run_one_degree_diagnostic("no_atmosphere_no_catke", GPU(); grid, ocean_kwargs, atmosphere = nothing, stop_time = 350days) 
