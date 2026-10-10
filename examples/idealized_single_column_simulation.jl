using NumericalEarth
using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials

# Ocean state parameters

T₀ = 0    # surface temperature (°C)
S₀ = 35   # surface salinity (g kg⁻¹)
N² = 1e-5 # buoyancy frequency squared due to temperature stratification (s⁻²)
f = 0     # Coriolis parameter (s⁻¹)

# Atmospheric state parameters

Tᵃᵗ = 273.15 - 10 # air temperature (K)
u₁₀ = 10          # wind speed at 10 m (m s⁻¹)
qᵃᵗ = 0.01        # specific humidity (kg kg⁻¹)
ℐꜜˢʷ = 400        # downwelling shortwave radiation (W m⁻²)

# A spatially uniform atmosphere, prescribed at three times over one day

atmosphere_grid = RectilinearGrid(size=(), topology=(Flat, Flat, Flat))
atmosphere_times = range(0, 1days, length=3)
atmosphere = PrescribedAtmosphere(atmosphere_grid, atmosphere_times)

# The radiation shares the atmosphere's grid and times. We override the default
# ocean albedo (0.05) and keep the default emissivity (0.97).

radiation = PrescribedRadiation(atmosphere_grid, atmosphere_times;
                                ocean_surface = SurfaceRadiationProperties(albedo=0.1))

for n in eachindex(atmosphere_times)
    set!(atmosphere.temperature[n], Tᵃᵗ)
    set!(atmosphere.velocities.u[n], u₁₀)
    set!(atmosphere.specific_humidity[n], qᵃᵗ)
    set!(radiation.downwelling_shortwave[n], ℐꜜˢʷ)
end

# An ocean column at rest with a uniform temperature stratification

grid = RectilinearGrid(size=20, z=(-100, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid, coriolis=FPlane(; f))

equation_of_state = ocean.model.buoyancy.formulation.equation_of_state
g = ocean.model.buoyancy.formulation.gravitational_acceleration
α = SeawaterPolynomials.thermal_expansion(T₀, S₀, 0, equation_of_state)
dTdz = N² / (α * g)
Tᵢ(z) = T₀ + dTdz * z
set!(ocean.model, T=Tᵢ, S=S₀)

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(stability_functions=nothing)
interfaces = ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes, radiation)
model = OceanOnlyModel(ocean; atmosphere, radiation, interfaces)

# Atmosphere–ocean turbulent fluxes and the net ocean surface temperature flux

𝒬ᵛ  = model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
𝒬ᵀ  = model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
ρτˣ = model.interfaces.atmosphere_ocean_interface.fluxes.x_momentum
ρτʸ = model.interfaces.atmosphere_ocean_interface.fluxes.y_momentum
Jᵛ  = model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor
Jᵀ  = model.interfaces.net_fluxes.ocean.T
