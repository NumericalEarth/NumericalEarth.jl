# # MITgcm Ocean Forced Simulation
#
# This script runs MITgcm (as a shared library) on the `global_oce_latlon`
# grid: 90×40×15, 4°×4° lat-lon, 80S–80N, with JRA55 atmospheric forcing
# applied through NumericalEarth's coupling framework.
#
# Initial conditions and bathymetry come from the MITgcm tutorial binary files.
# Physics: KPP vertical mixing, GM/Redi, CD scheme.
#
# Run for 2 years to compare against `numerical_earth_example.jl`.

using NumericalEarth
using MITgcm
using Oceananigans
using Oceananigans.Units
using Printf

# ## Build and load MITgcm

example_dir = @__DIR__
config_dir  = joinpath(example_dir, "mitgcm_config")
code_dir    = joinpath(config_dir, "code")
input_dir   = joinpath(config_dir, "input")
build_dir   = joinpath(example_dir, "build_jra55")

mitgcm_dir = get(ENV, "MITGCM_DIR", "")
if !isdir(mitgcm_dir)
    @info "Downloading MITgcm source..."
    mitgcm_dir = MITgcm.download_mitgcm_source()
end

@info "Building MITgcm..." mitgcm_dir code_dir input_dir
ocean = MITgcmOceanSimulation(mitgcm_dir;
                              output_dir = build_dir,
                              code_dir,
                              input_dir,
                              verbose = false)

lib = ocean.library
Nx, Ny, Nr = lib.dims.Nx, lib.dims.Ny, lib.dims.Nr
@info "MITgcm loaded" Nx Ny Nr

# ## Build coupled model with JRA55 forcing

atmos     = JRA55PrescribedAtmosphere()
radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)

coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere=atmos, radiation)

Δt        = 1200
stop_time = 2 * 365days
simulation = Simulation(coupled_model; Δt, stop_time)

# ## Progress callback

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    lib   = ocean.library
    mtime = MITgcm.get_time(lib)

    refresh_state!(ocean)
    mask = view(ocean.hfacc, :, :, 1)
    ocean_sst = ocean.theta[:, :, 1][mask .> 0]
    ocean_sss = ocean.salt[:, :, 1][mask .> 0]
    eta       = ocean.etan[mask .> 0]

    elapsed = 1e-9 * (time_ns() - wall_time[])

    @printf("iter %6d | day %7.1f | SST: [%6.2f, %5.2f] | SSS: [%5.2f, %5.2f] | η: [%.4f, %.4f] mean=%.4e | wall: %s\n",
            iteration(sim), mtime / 86400,
            minimum(ocean_sst), maximum(ocean_sst),
            minimum(ocean_sss), maximum(ocean_sss),
            minimum(eta), maximum(eta), mean(eta),
            prettytime(elapsed))

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

# ## Run

@info "Running MITgcm + JRA55 simulation..." Δt stop_time
wall_time[] = time_ns()
run!(simulation)

@info "Simulation complete. Finalizing..."
MITgcm.finalize!(lib)
@info "Done!"
