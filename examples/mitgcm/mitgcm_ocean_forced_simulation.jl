# # Comparing JRA55 and Original MITgcm Forcing
#
# This example runs two ocean simulations using the MITgcm ocean model
# (loaded as a shared library) on the `global_oce_latlon` configuration:
# a 4deg x 4deg lat-lon global ocean with 15 vertical levels, spanning 80S to 80N.
#
# **Simulation 1 (JRA55):** Surface forcing from the JRA55 reanalysis,
# applied through NumericalEarth's coupling framework (bulk formulae computed in Julia).
#
# **Simulation 2 (Original MITgcm):** Surface forcing from the original
# `global_oce_latlon` verification experiment: Trenberth wind stress, NCEP net
# heat flux, and Levitus SST/SSS relaxation, read by MITgcm's EXF package.
#
# Both simulations use the same physics (KPP vertical mixing, GM/Redi)
# and the same grid/initial conditions. The comparison video shows how
# different atmospheric forcing products lead to different ocean states.
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

# ## Configuration paths

example_dir = @__DIR__

# JRA55 simulation config (KPP, no EXF — forcing from Julia)
jra55_config_dir = joinpath(example_dir, "mitgcm_config")
jra55_code_dir   = joinpath(jra55_config_dir, "code")
jra55_input_dir  = joinpath(jra55_config_dir, "input")
jra55_build_dir  = joinpath(example_dir, "build_jra55")

# Original MITgcm forcing config (KPP + EXF — forcing from binary files)
exf_config_dir = joinpath(example_dir, "mitgcm_config_exf")
exf_code_dir   = joinpath(exf_config_dir, "code")
exf_input_dir  = joinpath(exf_config_dir, "input")
exf_build_dir  = joinpath(example_dir, "build_exf")

# Simulation parameters
Δt        = 1200        # seconds
stop_time = 60days      # total integration time
save_interval = 50      # save every 50 iterations

nsteps = Int(stop_time / Δt)

# ## Helper: load a MITgcm ocean from source or pre-built library

function load_ocean(; build_output_dir, code_dir, input_dir)
    # Strategy 1: explicit environment variables
    lib_env = get(ENV, "MITGCM_LIB", "")
    run_env = get(ENV, "MITGCM_RUN", "")
    if isfile(lib_env) && isdir(run_env)
        @info "Using MITGCM_LIB / MITGCM_RUN environment variables" lib_env run_env
        return MITgcmOceanSimulation(lib_env, run_env; verbose=false)
    end

    # Strategy 2: build from source
    mitgcm_dir = get(ENV, "MITGCM_DIR", "")
    if !isdir(mitgcm_dir)
        @info "Downloading MITgcm source (this only happens once)..."
        mitgcm_dir = MITgcm.download_mitgcm_source()
    end
    @info "Building MITgcm from source" mitgcm_dir code_dir input_dir
    return MITgcmOceanSimulation(mitgcm_dir;
                                  output_dir = build_output_dir,
                                  code_dir,
                                  input_dir,
                                  verbose=true)
end

# ## Progress callback

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

# ==========================================
# Simulation 1: JRA55 Forcing
# ==========================================

@info "═══════════════════════════════════════════════"
@info "  Simulation 1: JRA55 Reanalysis Forcing"
@info "═══════════════════════════════════════════════"

ocean_jra = load_ocean(; build_output_dir = jra55_build_dir,
                         code_dir = jra55_code_dir,
                         input_dir = jra55_input_dir)

Nx, Ny, Nr = ocean_jra.library.dims.Nx, ocean_jra.library.dims.Ny, ocean_jra.library.dims.Nr
@info "MITgcm ocean loaded (JRA55 config)" Nx Ny Nr

# Build a JRA55 prescribed atmosphere
atmos = JRA55PrescribedAtmosphere()
radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)

coupled_model = OceanSeaIceModel(ocean_jra, nothing; atmosphere=atmos, radiation)
simulation = Simulation(coupled_model; Δt, stop_time)

add_callback!(simulation, progress, TimeInterval(10days))

# Collect SST and surface velocity snapshots
sst_jra55 = []
u_jra55   = []
v_jra55   = []

function save_jra55(sim)
    ocean = sim.model.ocean
    refresh_state!(ocean)
    land = view(ocean.hfacc, :, :, 1) .== 0

    sst = copy(ocean.theta[:, :, 1])
    sst[land] .= NaN
    push!(sst_jra55, sst)

    uv = copy(ocean.uvel[:, :, 1])
    uv[land] .= NaN
    push!(u_jra55, uv)

    vv = copy(ocean.vvel[:, :, 1])
    vv[land] .= NaN
    push!(v_jra55, vv)
end

add_callback!(simulation, save_jra55, IterationInterval(save_interval))

@info "Running JRA55-forced simulation..." Δt stop_time
wall_time[] = time_ns()
run!(simulation)

@info "JRA55 simulation complete. Finalizing..."
MITgcm.finalize!(ocean_jra.library)

# ==========================================
# Simulation 2: Original MITgcm EXF Forcing
# ==========================================

@info "═══════════════════════════════════════════════"
@info "  Simulation 2: Original MITgcm EXF Forcing"
@info "  (Trenberth wind stress + NCEP heat flux"
@info "   + Levitus SST/SSS relaxation)"
@info "═══════════════════════════════════════════════"

ocean_exf = load_ocean(; build_output_dir = exf_build_dir,
                         code_dir = exf_code_dir,
                         input_dir = exf_input_dir)

@info "MITgcm ocean loaded (EXF config)" Nx=ocean_exf.library.dims.Nx Ny=ocean_exf.library.dims.Ny

# Collect SST and surface velocity snapshots
sst_original = []
u_original   = []
v_original   = []

lib_exf = ocean_exf.library
mask_exf = view(ocean_exf.hfacc, :, :, 1)

@info "Running EXF-forced simulation ($nsteps steps)..."
wall_time[] = time_ns()

for s in 1:nsteps
    # EXF handles forcing internally — just step forward
    step!(lib_exf)
    refresh_state!(ocean_exf)

    # Save snapshots at the same interval as the JRA55 simulation
    if s % save_interval == 0
        land = mask_exf .== 0

        sst = copy(ocean_exf.theta[:, :, 1])
        sst[land] .= NaN
        push!(sst_original, sst)

        uv = copy(ocean_exf.uvel[:, :, 1])
        uv[land] .= NaN
        push!(u_original, uv)

        vv = copy(ocean_exf.vvel[:, :, 1])
        vv[land] .= NaN
        push!(v_original, vv)
    end

    # Print progress every 10 days
    if s % (Int(10days) ÷ Δt) == 0
        mtime = MITgcm.get_time(lib_exf)
        ocean_sst = ocean_exf.theta[:, :, 1][mask_exf .> 0]
        ocean_sss = ocean_exf.salt[:, :, 1][mask_exf .> 0]
        elapsed = 1e-9 * (time_ns() - wall_time[])
        @printf("EXF step %5d | day %7.1f | SST: [%6.2f, %5.2f] | SSS: [%5.2f, %5.2f] | wall: %s\n",
                s, mtime / 86400,
                minimum(ocean_sst), maximum(ocean_sst),
                minimum(ocean_sss), maximum(ocean_sss),
                prettytime(elapsed))
        wall_time[] = time_ns()
    end
end

@info "EXF simulation complete. Finalizing..."
MITgcm.finalize!(lib_exf)

# ==========================================
# Comparison video
# ==========================================

@info "Creating comparison video..."

Nt = min(length(sst_jra55), length(sst_original))
@info "Number of frames: $Nt"

iter = Observable(1)

# SST observables
sst_jra_obs = @lift sst_jra55[$iter]
sst_exf_obs = @lift sst_original[$iter]

# Surface velocity observables
u_jra_obs = @lift u_jra55[$iter]
u_exf_obs = @lift u_original[$iter]
v_jra_obs = @lift v_jra55[$iter]
v_exf_obs = @lift v_original[$iter]

# Day label
day_label = @lift @sprintf("Day %.0f", ($iter - 1) * save_interval * Δt / 86400)

# Grid coordinates (same for both simulations)
λ = ocean_exf.xc[:, 1]
φ = ocean_exf.yc[1, :]

fig = Figure(size = (1200, 1100))

Label(fig[0, 1:2], day_label, fontsize=20, tellwidth=false)

# Row 1: SST comparison
ax1 = Axis(fig[1, 1]; title="SST — JRA55 forcing (°C)", xlabel="", ylabel="Latitude")
ax2 = Axis(fig[1, 2]; title="SST — Original MITgcm forcing (°C)", xlabel="", ylabel="")

hm1 = heatmap!(ax1, λ, φ, sst_jra_obs; nan_color=:grey, colormap=:thermal, colorrange=(-2, 30))
hm2 = heatmap!(ax2, λ, φ, sst_exf_obs; nan_color=:grey, colormap=:thermal, colorrange=(-2, 30))
Colorbar(fig[1, 3], hm2)

# Row 2: Surface zonal velocity comparison
ax3 = Axis(fig[2, 1]; title="Surface u — JRA55 (m/s)", xlabel="", ylabel="Latitude")
ax4 = Axis(fig[2, 2]; title="Surface u — Original MITgcm (m/s)", xlabel="", ylabel="")

hm3 = heatmap!(ax3, λ, φ, u_jra_obs; nan_color=:grey, colormap=:bwr, colorrange=(-0.2, 0.2))
hm4 = heatmap!(ax4, λ, φ, u_exf_obs; nan_color=:grey, colormap=:bwr, colorrange=(-0.2, 0.2))
Colorbar(fig[2, 3], hm4)

# Row 3: Surface meridional velocity comparison
ax5 = Axis(fig[3, 1]; title="Surface v — JRA55 (m/s)", xlabel="Longitude", ylabel="Latitude")
ax6 = Axis(fig[3, 2]; title="Surface v — Original MITgcm (m/s)", xlabel="Longitude", ylabel="")

hm5 = heatmap!(ax5, λ, φ, v_jra_obs; nan_color=:grey, colormap=:bwr, colorrange=(-0.2, 0.2))
hm6 = heatmap!(ax6, λ, φ, v_exf_obs; nan_color=:grey, colormap=:bwr, colorrange=(-0.2, 0.2))
Colorbar(fig[3, 3], hm6)

CairoMakie.record(fig, "mitgcm_forcing_comparison.mp4", 1:Nt, framerate=8) do nn
    iter[] = nn
end

@info "Comparison video saved to mitgcm_forcing_comparison.mp4"
nothing #hide

# ![](mitgcm_forcing_comparison.mp4)

# ## Finalize

@info "Done!"
