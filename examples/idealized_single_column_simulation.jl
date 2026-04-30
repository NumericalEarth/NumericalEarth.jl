using NumericalEarth
using ClimaSeaIce
using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials
using Dates

# Ocean state parameters
T₀ = 0   # Surface temperature, ᵒC
S₀ = 35   # Surface salinity
N² = 1e-5 # Buoyancy gradient due to temperature stratification
f = 0     # Coriolis parameter

# Atmospheric state parameters
Tᵃᵗ = 273.15 - 10 # Kelvin
u₁₀ = 10 # wind at 10 m, m/s
qᵃᵗ = 0.01 # specific humidity
ℐꜜˢʷ = 400 # shortwave radiation (W m⁻², positive means heating right now)

# Build the atmosphere
atmosphere_grid = RectilinearGrid(size=(), topology=(Flat, Flat, Flat))
atmosphere_times = range(0, 1days, length=3)
atmosphere = PrescribedAtmosphere(atmosphere_grid, atmosphere_times)

# Build the radiation component (lives on the same grid + times as the atmosphere
# in this single-column setup) and prescribe a constant shortwave forcing.
radiation = PrescribedRadiation(atmosphere_grid, atmosphere_times;
                                ocean_surface = SurfaceRadiationProperties(0.1, 0.97))

parent(atmosphere.tracers.T) .= Tᵃᵗ     # K
parent(atmosphere.velocities.u) .= u₁₀ # m/s
parent(atmosphere.tracers.q) .= qᵃᵗ     # mass ratio
parent(radiation.downwelling_shortwave) .= ℐꜜˢʷ # W

# Build ocean model at rest with initial temperature stratification
grid = RectilinearGrid(size=20, z=(-100, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid, coriolis=FPlane(; f))

eos = ocean.model.buoyancy.formulation.equation_of_state
g = ocean.model.buoyancy.formulation.gravitational_acceleration
α = SeawaterPolynomials.thermal_expansion(T₀, S₀, 0, eos)
dTdz = N² / (α * g)
Tᵢ(z) = T₀ + dTdz * z
set!(ocean.model, T=Tᵢ, S=S₀)

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(stability_functions=nothing)
interfaces = NumericalEarth.EarthSystemModels.ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes, radiation)
model = OceanOnlyModel(ocean; atmosphere, radiation, interfaces)

𝒬ᵛ  = model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
𝒬ᵀ  = model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
τˣ  = model.interfaces.atmosphere_ocean_interface.fluxes.x_momentum
τʸ  = model.interfaces.atmosphere_ocean_interface.fluxes.y_momentum
Jᵛ  = model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor

# TODO: the total fluxes are defined on _interfaces_ between components:
# atmopshere_ocean, atmosphere_sea_ice, ocean_sea_ice. They aren't defined wrt to
# just one component
Qo = model.interfaces.net_fluxes.ocean_surface.Q
