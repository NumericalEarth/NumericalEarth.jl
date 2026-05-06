# # Atmospheric convection over a slab land surface (3D)
#
# 3D companion to `breeze_over_slab_ocean_3d.jl`. Couples a Breeze
# atmospheric large eddy simulation (LES) to a `RucSlabLand` (USGS class 7,
# grassland) through the `AtmosphereLandModel` constructor, which plugs
# the slab into `EarthSystemModel`'s similarity-theory turbulent-flux
# pipeline.
#
# Compared to the constant-`C_d` manual coupling shown in earlier drafts,
# the fluxes here come from the same `SimilarityTheoryFluxes` machinery
# used by `AtmosphereOceanModel`. The slab's `mavail` (β-factor) is
# consumed by the surface-humidity formulation as
# `q_s = β · q_sat(T_g)`, and roughness defaults to a representative
# grassland value (z₀_m = 0.1 m). All RUC slab physics (snow, canopy
# water, soil moisture, β, r_s, r_g, vilka, etc.) remains active —
# see `src/Lands/ruc_slab_land.jl`.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Printf

# ## Grid setup
#
# 10 km × 10 km × 5 km doubly-periodic 3D atmosphere, with a 2D
# `Flat`-z grid for the slab land on the same horizontal layout.

arch = CPU()

Nxᵃᵗ = 32
Nyᵃᵗ = 32
Nzᵃᵗ = 32

grid = RectilinearGrid(arch,
                       size = (Nxᵃᵗ, Nyᵃᵗ, Nzᵃᵗ), halo = (5, 5, 5),
                       x = (-5kilometers, 5kilometers),
                       y = (-5kilometers, 5kilometers),
                       z = (0, 5kilometers),
                       topology = (Periodic, Periodic, Bounded))

land_grid = RectilinearGrid(arch,
                            size = (grid.Nx, grid.Ny),
                            halo = (grid.Hx, grid.Hy),
                            x = (-5kilometers, 5kilometers),
                            y = (-5kilometers, 5kilometers),
                            topology = (Periodic, Periodic, Flat))

# ## Atmosphere
#
# Warm ground, cool air aloft → unstable PBL with upward sensible and
# latent heat fluxes that cool the ground toward equilibrium and drive
# convective overturning in the atmosphere. Similarity theory correctly
# captures the strong-stability regime as well, but the convective case
# produces a clearer signal for a short integration.

Tᵍ₀ = 305.0    # K, initial ground temperature
θᵃᵗ = 295.0    # K, atmospheric potential temperature
U₀  = 5.0      # m/s, geostrophic wind
coriolis = FPlane(latitude = 33)

atmos = atmosphere_simulation(grid; potential_temperature = θᵃᵗ, coriolis)

reference_state = atmos.dynamics.reference_state
θᵢ(x, y, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)
set!(atmos, θ = θᵢ, u = U₀)

# ## Slab land
#
# Uniform USGS grassland (class 7); apply the lookup table to populate
# `vegfrac`, `lai`, `albedo_veg`, `emissivity_veg`, `z0_veg`, `r_smin`.

slab_land = RucSlabLand(land_grid;
                     parameters = RucSlabLandParameters(Float64;
                                                     depth = 0.10,
                                                     density = 1500,
                                                     heat_capacity = 1480,
                                                     soil_depth = 1.0))

vegtype = fill(7, grid.Nx, grid.Ny)
registry = usgs_land_classifications(Float64)
apply_land_classifications!(slab_land, vegtype, registry)

set!(slab_land, T = Tᵍ₀, Tc = Tᵍ₀, θ = 0.30)

slab_land.forcings.solar_irradiance .= 600.0    # W m⁻², bright midday
slab_land.forcings.air_temperature  .= θᵃᵗ
slab_land.forcings.air_humidity     .= 0.005

# ## Coupled model
#
# `AtmosphereLandModel` wires the atmosphere to the slab through
# similarity theory. The default flux formulation uses scalar roughness
# lengths representative of short grass; spatially-varying roughness is
# a follow-up.

model = AtmosphereLandModel(atmos, slab_land)

Δt = 5seconds
stop_time = 30minutes
simulation = Simulation(model; Δt, stop_time)

function progress(sim)
    a = sim.model.atmosphere
    u, v, w = a.velocities
    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    Tg   = sim.model.land.temperature
    msg = @sprintf("Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, T_g: (%.3f, %.3f)",
                   iteration(sim), prettytime(sim), umax, wmax,
                   minimum(Tg), maximum(Tg))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(60))

@info "Running coupled Breeze + RucSlabLand simulation via AtmosphereLandModel..."
run!(simulation)
@info "Done."
