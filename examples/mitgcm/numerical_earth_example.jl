# # NumericalEarth Ocean Simulation
#
# This script runs Oceananigans on the same grid as the MITgcm `global_oce_latlon`
# tutorial: 90×40×15, 4°×4° lat-lon, 80S–80N, with JRA55 atmospheric forcing.
#
# Initial conditions and bathymetry are read from the MITgcm tutorial binary files
# (linked into the MITgcm build directory) to ensure an exact match with
# `mitgcm_ocean_forced_simulation.jl`.
#
# Physics: CATKE vertical mixing, GM/Redi, horizontal viscosity.
# Run for 2 years.

using NumericalEarth
using MITgcm
using Oceananigans
using Oceananigans.Units
using Printf
using Statistics

# ## Grid — matches MITgcm global_oce_latlon exactly

Nx, Ny, Nz = 90, 40, 15

Δz = [50., 70., 100., 140., 190., 240., 290., 340., 390., 440., 490., 540., 590., 640., 690.]

z_faces = zeros(Nz + 1)
z_faces[Nz + 1] = 0.0
for k in Nz:-1:1
    z_faces[k] = z_faces[k + 1] - Δz[Nz - k + 1]
end

grid = LatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                             latitude = (-80, 80),
                             longitude = (0, 360),
                             z = z_faces,
                             halo = (5, 5, 5))

# ## Bathymetry and initial conditions from MITgcm binary files
#
# Build MITgcm first to get the tutorial binary files in the run directory.

example_dir = @__DIR__
config_dir  = joinpath(example_dir, "mitgcm_config")
code_dir    = joinpath(config_dir, "code")
input_dir   = joinpath(config_dir, "input")
build_dir   = joinpath(example_dir, "build_jra55")

# Build MITgcm (or reuse existing build) to get binary data files
mitgcm_dir = get(ENV, "MITGCM_DIR", "")
if !isdir(mitgcm_dir)
    @info "Downloading MITgcm source..."
    mitgcm_dir = MITgcm.download_mitgcm_source()
end

run_dir = joinpath(build_dir, "run")
if !isdir(run_dir) || !isfile(joinpath(run_dir, "bathymetry.bin"))
    @info "Building MITgcm to obtain binary data files..."
    MITgcm.build_mitgcm_library(mitgcm_dir;
                                output_dir = build_dir,
                                code_dir,
                                input_dir)
end

function read_bin(filepath; dims)
    array = zeros(Float32, prod(dims))
    read!(filepath, array)
    array = bswap.(array)
    array = Float64.(array)
    return reshape(array, dims...)
end

bathy = read_bin(joinpath(run_dir, "bathymetry.bin"); dims = (Nx, Ny))
Tini  = reverse(read_bin(joinpath(run_dir, "lev_t.bin"); dims = (Nx, Ny, Nz)), dims = 3)
Sini  = reverse(read_bin(joinpath(run_dir, "lev_s.bin"); dims = (Nx, Ny, Nz)), dims = 3)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathy))

@info "Grid and bathymetry loaded" Nx Ny Nz

# ## Ocean model — physics matching MITgcm config

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

catke_closure   = NumericalEarth.Oceans.default_ocean_closure()
eddy_closure    = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3,
                                                     skew_flux_formulation=AdvectiveFormulation())
vert_viscosity  = VerticalScalarDiffusivity(ν=1e-3, κ=3e-5)
horiz_viscosity = HorizontalScalarDiffusivity(ν=5e5)

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = VectorInvariant(; vorticity_scheme = Oceananigans.Advection.EnergyConserving())
tracer_advection   = UpwindBiased(order=3)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         reference_density = 1035,
                         closure = (catke_closure, eddy_closure, horiz_viscosity, vert_viscosity))

set!(ocean.model, T=Tini, S=Sini)

# ## JRA55 forcing — same as MITgcm simulation

atmos     = JRA55PrescribedAtmosphere()
radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)

coupled_model = OceanOnlyModel(ocean; atmosphere=atmos, radiation)

Δt        = 1200
stop_time = 2 * 365days
simulation = Simulation(coupled_model; Δt, stop_time)

# ## Progress callback

wall_time = Ref(time_ns())

function progress(sim)
    model = sim.model.ocean.model
    T = model.tracers.T
    S = model.tracers.S

    Tmin, Tmax = minimum(T), maximum(T)
    Smin, Smax = minimum(S), maximum(S)
    ηm = mean(model.free_surface.displacement)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    @printf("iter %6d | day %7.1f | SST: [%6.2f, %5.2f] | SSS: [%5.2f, %5.2f] | mean(η): %.4e | wall: %s\n",
            iteration(sim), time(sim) / 86400,
            Tmin, Tmax, Smin, Smax, ηm,
            prettytime(elapsed))

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

# ## Run

@info "Running Oceananigans + JRA55 simulation..." Δt stop_time
wall_time[] = time_ns()
run!(simulation)

@info "Done!"
