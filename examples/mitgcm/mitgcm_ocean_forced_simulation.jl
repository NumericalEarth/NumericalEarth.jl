# # A MITgcm Ocean Simulation Forced by JRA55 Reanalysis
#
# This example demonstrates how to run a coupled ocean–atmosphere simulation
# using the MITgcm ocean model (loaded as a shared library) forced by the
# JRA55 reanalysis data through NumericalEarth's coupling framework.
#
# The MITgcm experiment is based on the `global_oce_latlon` configuration:
# a 4° × 4° lat-lon global ocean with 15 vertical levels, spanning 80°S to 80°N.
#
# The custom configuration in `mitgcm_config/` adds:
# - KPP vertical mixing (not in the stock experiment)
# - Disables EXF (surface forcing comes from Julia/NumericalEarth instead)
#
# ## Prerequisites
#
# Either provide a pre-built shared library:
#   ```
#   export MITGCM_LIB=/path/to/libmitgcm.dylib   # or .so on Linux
#   export MITGCM_RUN=/path/to/run                 # the MITgcm run directory
#   ```
#
# Or provide the MITgcm source directory (the library will be built automatically):
#   ```
#   export MITGCM_DIR=/path/to/MITgcm
#   ```
#
# If neither is set, MITgcm source will be downloaded automatically (requires git and gfortran).

using NumericalEarth
using MITgcm
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

# ## Load the MITgcm ocean
#
# We try several strategies in order:
# 1. Explicit library/run paths from environment variables
# 2. Build from MITgcm source with KPP-enabled custom configuration

# Path to the custom MITgcm configuration (KPP + no EXF)
example_dir = @__DIR__
config_dir  = joinpath(example_dir, "mitgcm_config")
code_dir    = joinpath(config_dir, "code")
input_dir   = joinpath(config_dir, "input")

# Directory for the built library and run files
build_output_dir = joinpath(example_dir, "build")

function load_ocean(; build_output_dir, code_dir, input_dir)
    # Strategy 1: explicit environment variables
    lib_env = get(ENV, "MITGCM_LIB", "")
    run_env = get(ENV, "MITGCM_RUN", "")
    if isfile(lib_env) && isdir(run_env)
        @info "Using MITGCM_LIB / MITGCM_RUN environment variables" lib_env run_env
        return MITgcmOceanSimulation(lib_env, run_env; verbose=false)
    end

    # Strategy 2: build from source with custom KPP configuration
    mitgcm_dir = get(ENV, "MITGCM_DIR", "")
    if !isdir(mitgcm_dir)
        @info "Downloading MITgcm source (this only happens once)..."
        mitgcm_dir = MITgcm.download_mitgcm_source()
    end
    @info "Building MITgcm from source with KPP" mitgcm_dir code_dir input_dir
    return MITgcmOceanSimulation(mitgcm_dir;
                                  output_dir = build_output_dir,
                                  code_dir,
                                  input_dir,
                                  verbose=true)
end

ocean = load_ocean(; build_output_dir, code_dir, input_dir)

Nx, Ny, Nr = ocean.library.dims.Nx, ocean.library.dims.Ny, ocean.library.dims.Nr
@info "MITgcm ocean loaded" Nx Ny Nr

# ## Build a JRA55 prescribed atmosphere
#
# We force the ocean with the JRA55 reanalysis data, which includes:
# - 10-meter wind velocity (u, v)
# - 2-meter temperature and specific humidity
# - Downwelling longwave and shortwave radiation
# - Freshwater fluxes (rain, snow)
#
# The `JRA55NetCDFBackend(41)` loads 41 time snapshots at a time into memory.

atmos = JRA55PrescribedAtmosphere()

# The coupled ocean--atmosphere model.
# We use the default radiation model and we do not couple an ice model for simplicity.

radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)
coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere=atmos, radiation)
simulation = Simulation(coupled_model; Δt = 1200, stop_time = 60days)

# ## Progress callback
#
# Print diagnostics every 10 days: SST/SSS ranges and wall-clock time.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    lib   = ocean.library
    niter = MITgcm.get_niter(lib)
    mtime = MITgcm.get_time(lib)

    mask = view(ocean.hfacc, :, :, 1)
    sst  = view(ocean.theta, :, :, 1)
    sss  = view(ocean.salt,  :, :, 1)

    ocean_sst = sst[mask .> 0]
    ocean_sss = sss[mask .> 0]

    elapsed = 1e-9 * (time_ns() - wall_time[])

    @printf("iter %5d | MITgcm iter %6d | day %7.1f | ", iteration(sim), niter, mtime / 86400)
    @printf("SST: [%6.2f, %5.2f] | SSS: [%5.2f, %5.2f] | wall: %s\n",
            minimum(ocean_sst), maximum(ocean_sst),
            minimum(ocean_sss), maximum(ocean_sss),
            prettytime(elapsed))

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

# We also set up a callback to collect the surface velocities (u, v) for later visualization.

u  = []
v  = []
η  = []
fu = []
fv = []

function save_variables(sim)
    push!(u,  deepcopy(sim.model.interfaces.exchanger.ocean.state.u))
    push!(v,  deepcopy(sim.model.interfaces.exchanger.ocean.state.v))
    push!(η,  deepcopy(sim.model.ocean.etan))
    push!(fu, deepcopy(sim.model.ocean.fu))
    push!(fv, deepcopy(sim.model.ocean.fv))
end

add_callback!(simulation, save_variables, IterationInterval(50))

# ## Run!

@info "Running 1-year simulation with JRA55 forcing..." Δt stop_time=365days

run!(simulation)

# ## Visualize surface velocities
#
# We produce an animation of the surface zonal and meridional velocities.

iter = Observable(1)
ui = @lift begin
    ut = interior(u[$iter], :, :, 1)
    ut[ut .== 0] .= NaN
    ut
end

vi = @lift begin
    ut = interior(v[$iter], :, :, 1)
    ut[ut .== 0] .= NaN
    ut
end

ηi = @lift begin
    ηt = η[$iter]
    ηt[ηt .== 0] .= NaN
    ηt
end

fui = @lift begin
    ηt = fu[$iter]
    ηt[ηt .== 0] .= NaN
    ηt
end

fvi = @lift begin
    ηt = fv[$iter]
    ηt[ηt .== 0] .= NaN
    ηt
end

Nt = length(u)

fig = Figure(resolution = (1000, 1000))
ax1 = Axis(fig[1, 1]; title = "Surface zonal velocity (m/s)", xlabel = "", ylabel = "Latitude")
ax2 = Axis(fig[2, 1]; title = "Surface meridional velocity (m/s)", xlabel = "", ylabel = "Latitude")
ax3 = Axis(fig[3, 1]; title = "Sea surface height (m)", xlabel = "", ylabel = "Latitude")
# ax4 = Axis(fig[4, 1]; title = "Zonal wind stress", xlabel = "", ylabel = "Latitude")
# ax5 = Axis(fig[5, 1]; title = "Meridional wind stress", xlabel = "", ylabel = "Latitude")

grid = coupled_model.interfaces.exchanger.grid

λ = λnodes(grid, Center())
φ = φnodes(grid, Center())

hm1 = heatmap!(ax1, λ, φ, ui,  nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))
hm2 = heatmap!(ax2, λ, φ, vi,  nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))
hm2 = heatmap!(ax3, λ, φ, ηi,  nan_color = :grey, colormap = :bwr, colorrange = (-1.5, 1.5))
# hm2 = heatmap!(ax4, λ, φ, fui, nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))
# hm2 = heatmap!(ax5, λ, φ, fvi, nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))

Colorbar(fig[1, 2], hm1)
Colorbar(fig[2, 2], hm2)

CairoMakie.record(fig, "mitgcm_ocean_surface.mp4", 1:Nt, framerate = 8) do nn
    iter[] = nn
end
nothing #hide

# ![](mitgcm_ocean_surface.mp4)

# ## Finalize

@info "Simulation complete, finalizing MITgcm..."
MITgcm.finalize!(ocean.library)
@info "Done!"
