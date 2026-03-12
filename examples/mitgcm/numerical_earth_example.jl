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
# Numerics matched to MITgcm where possible: AB2 timestepper,
# centered 2nd-order tracer advection, POLY3 EOS (matching MITgcm exactly),
# implicit free surface (PCG, matching MITgcm CG2D),
# no penetrating shortwave (all SW applied at surface).
# Run for 2 years.

using NumericalEarth
using MITgcm
using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials: AbstractSeawaterPolynomial, BoussinesqEquationOfState
import SeawaterPolynomials: ρ′, thermal_sensitivity, haline_sensitivity, with_float_type
using Printf
using Statistics
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Utils: launch!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans: fields
using Oceananigans.MultiRegion: @apply_regionally
import Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_ab2_step!
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    compute_momentum_flux_bcs!, ab2_step_velocities!, step_free_surface!,
    compute_transport_velocities!, compute_tracer_tendencies!,
    ab2_step_grid!, correct_barotropic_mode!, ab2_step_tracers!,
    compute_momentum_tendencies!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.Models: surface_kernel_parameters
using Oceananigans.Advection: EnergyConserving

# ## POLY3 Equation of State — exact match with MITgcm
#
# MITgcm's POLY3 EOS uses a 3rd-order polynomial in (T - T_ref, S - S_ref)
# with different coefficients at each vertical level, read from POLY3.COEFFS.
# ρ = rhoConst + eosSig0[k] + c₁·tp + c₂·sp + c₃·tp² + c₄·tp·sp + c₅·sp²
#                            + c₆·tp³ + c₇·tp²·sp + c₈·tp·sp² + c₉·sp³
# where tp = T - T_ref[k], sp = S - S_ref[k].

struct POLY3SeawaterPolynomial{N, M, FT} <: AbstractSeawaterPolynomial
    z_faces :: NTuple{M, FT}   # N+1 z-face positions, bottom to top
    T_ref   :: NTuple{N, FT}   # reference temperature per level
    S_ref   :: NTuple{N, FT}   # reference salinity per level
    ρ_ref   :: NTuple{N, FT}   # reference density anomaly (eosSig0) per level
    c₁ :: NTuple{N, FT}
    c₂ :: NTuple{N, FT}
    c₃ :: NTuple{N, FT}
    c₄ :: NTuple{N, FT}
    c₅ :: NTuple{N, FT}
    c₆ :: NTuple{N, FT}
    c₇ :: NTuple{N, FT}
    c₈ :: NTuple{N, FT}
    c₉ :: NTuple{N, FT}
end

Base.eltype(::POLY3SeawaterPolynomial{N, M, FT}) where {N, M, FT} = FT
Base.summary(::POLY3SeawaterPolynomial{N, M, FT}) where {N, M, FT} = "POLY3SeawaterPolynomial{$N levels, $FT}"

function with_float_type(FT, p::POLY3SeawaterPolynomial{N, M}) where {N, M}
    cvt(t) = NTuple{length(t), FT}(FT.(t))
    return POLY3SeawaterPolynomial{N, M, FT}(
        cvt(p.z_faces), cvt(p.T_ref), cvt(p.S_ref), cvt(p.ρ_ref),
        cvt(p.c₁), cvt(p.c₂), cvt(p.c₃), cvt(p.c₄), cvt(p.c₅),
        cvt(p.c₆), cvt(p.c₇), cvt(p.c₈), cvt(p.c₉))
end

# Map geopotential height Z to the Oceananigans vertical level index (k=1 is bottom).
@inline function find_poly3_level(Z, z_faces)
    N = length(z_faces) - 1
    k = 1
    for i in 2:N
        k = ifelse(Z >= z_faces[i], i, k)
    end
    return k
end

@inline function ρ′(Θ, Sᴬ, Z, eos::BoussinesqEquationOfState{<:POLY3SeawaterPolynomial})
    p = eos.seawater_polynomial
    k = find_poly3_level(Z, p.z_faces)
    tp = Θ  - p.T_ref[k]
    sp = Sᴬ - p.S_ref[k]
    Δσ = p.c₁[k]*tp + p.c₂[k]*sp +
         p.c₃[k]*tp^2 + p.c₄[k]*tp*sp + p.c₅[k]*sp^2 +
         p.c₆[k]*tp^3 + p.c₇[k]*tp^2*sp + p.c₈[k]*tp*sp^2 + p.c₉[k]*sp^3
    return p.ρ_ref[k] + Δσ
end

# thermal_sensitivity = -∂ρ/∂Θ
@inline function thermal_sensitivity(Θ, Sᴬ, Z, eos::BoussinesqEquationOfState{<:POLY3SeawaterPolynomial})
    p = eos.seawater_polynomial
    k = find_poly3_level(Z, p.z_faces)
    tp = Θ  - p.T_ref[k]
    sp = Sᴬ - p.S_ref[k]
    return -(p.c₁[k] + 2p.c₃[k]*tp + p.c₄[k]*sp +
             3p.c₆[k]*tp^2 + 2p.c₇[k]*tp*sp + p.c₈[k]*sp^2)
end

# haline_sensitivity = ∂ρ/∂S
@inline function haline_sensitivity(Θ, Sᴬ, Z, eos::BoussinesqEquationOfState{<:POLY3SeawaterPolynomial})
    p = eos.seawater_polynomial
    k = find_poly3_level(Z, p.z_faces)
    tp = Θ  - p.T_ref[k]
    sp = Sᴬ - p.S_ref[k]
    return p.c₂[k] + p.c₄[k]*tp + 2p.c₅[k]*sp +
           p.c₇[k]*tp^2 + 2p.c₈[k]*tp*sp + 3p.c₉[k]*sp^2
end

function read_poly3_coeffs(filepath, z_faces)
    lines = readlines(filepath)
    Nr = parse(Int, strip(lines[1]))

    T_ref = zeros(Nr)
    S_ref = zeros(Nr)
    ρ_ref = zeros(Nr)
    for k in 1:Nr
        vals = parse.(Float64, split(strip(lines[1 + k])))
        T_ref[k], S_ref[k], ρ_ref[k] = vals
    end

    C = zeros(9, Nr)
    for k in 1:Nr
        vals = parse.(Float64, split(strip(lines[1 + Nr + k])))
        C[:, k] .= vals
    end

    # MITgcm k=1 is top, Oceananigans k=1 is bottom → reverse
    rev(v) = Tuple(reverse(v))
    return POLY3SeawaterPolynomial(
        Tuple(z_faces),
        rev(T_ref), rev(S_ref), rev(ρ_ref),
        rev(C[1,:]), rev(C[2,:]), rev(C[3,:]),
        rev(C[4,:]), rev(C[5,:]), rev(C[6,:]),
        rev(C[7,:]), rev(C[8,:]), rev(C[9,:]))
end

# ## Grid — matches MITgcm global_oce_latlon exactly

import Oceananigans.Operators: ζ₃ᶠᶠᶜ, Γᶠᶠᶜ
using Oceananigans.Operators
using Oceananigans.Grids: peripheral_node

@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = ifelse(peripheral_node(i, j, k, grid, Face(), Face(), Center()), zero(grid),
    Γᶠᶠᶜ(i, j, k, grid, u, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid))

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

grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bathy; minimum_fractional_cell_height=0.05))

@info "Grid and bathymetry loaded" Nx Ny Nz

# ## Ocean model — physics matching MITgcm config

catke_closure   = NumericalEarth.Oceans.default_ocean_closure()
vert_viscosity  = VerticalScalarDiffusivity(ν=1e-3, κ=3e-5)
horiz_closure   = HorizontalScalarDiffusivity(ν=5e5, κ=1e3)

free_surface       = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient,
                                         reltol = 1e-13, maxiter = 500)
momentum_advection = Centered(order=2) 
tracer_advection   = Centered(order=2)

poly3 = read_poly3_coeffs(joinpath(input_dir, "POLY3.COEFFS"), z_faces)
equation_of_state = BoussinesqEquationOfState(poly3, 1035.0)

# NumericalEarth only defines reference_density for TEOS10; extend for our POLY3 EOS
import NumericalEarth.EarthSystemModels: reference_density
reference_density(eos::BoussinesqEquationOfState) = eos.reference_density

coriolis = HydrostaticSphericalCoriolis(scheme=Oceananigans.Coriolis.EENConserving())

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         reference_density = 1035,
                         equation_of_state,
                         coriolis,
                         radiative_forcing = nothing,
                         timestepper = :QuasiAdamsBashforth2,
                         closure = (catke_closure, horiz_closure, vert_viscosity))

set!(ocean.model, T=Tini, S=Sini)

# ## JRA55 forcing — same as MITgcm simulation

atmos     = JRA55PrescribedAtmosphere()
radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)

coupled_model = OceanOnlyModel(ocean; atmosphere=atmos, radiation, ocean_heat_capacity = 3994.0)

Δt        = 1200
stop_time = 360days
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
