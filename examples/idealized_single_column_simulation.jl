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
Tₐ = 273.15 - 10 # Kelvin
u₁₀ = 10 # wind at 10 m, m/s
qₐ = 0.01 # specific humidity
Qs = 400 # shortwave radiation (W m⁻², positive means heating right now)

# Build the atmosphere
radiation = Radiation(ocean_albedo=0.1)
atmosphere_grid = RectilinearGrid(size=(), topology=(Flat, Flat, Flat))
atmosphere_times = range(0, 1days, length=3)
atmosphere = PrescribedAtmosphere(atmosphere_grid, atmosphere_times)

parent(atmosphere.tracers.T) .= Tₐ     # K
parent(atmosphere.velocities.u) .= u₁₀ # m/s
parent(atmosphere.tracers.q) .= qₐ     # mass ratio
parent(atmosphere.downwelling_radiation.shortwave) .= Qs # W

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
interfaces = NumericalEarth.EarthSystemModels.ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes)
model = OceanOnlyModel(ocean; atmosphere, interfaces)

𝒬ᵛ  = model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
𝒬ᵀ  = model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
τx  = model.interfaces.atmosphere_ocean_interface.fluxes.x_momentum
τy  = model.interfaces.atmosphere_ocean_interface.fluxes.y_momentum
Jᵛ  = model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor

# TODO: the total fluxes are defined on _interfaces_ between components:
# atmopshere_ocean, atmosphere_sea_ice, ocean_sea_ice. They aren't defined wrt to
# just one component
Qo = model.interfaces.net_fluxes.ocean_surface.Q
