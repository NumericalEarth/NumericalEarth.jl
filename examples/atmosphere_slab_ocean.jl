# # Atmosphere–slab ocean coupling
#
# This example demonstrates coupling a Breeze atmospheric LES with a slab ocean model.
# Unlike the prescribed SST case in Breeze's PrescribedSST example, the sea surface
# temperature (SST) here evolves in response to the atmospheric surface heat fluxes.
#
# The slab ocean model represents a well-mixed ocean layer of fixed depth H with the
# SST tendency equation:
#
# ```math
# ρ cₚ H \frac{∂T}{∂t} = -Q_\text{net}
# ```
#
# where ``Q_\text{net}`` is the net upward surface heat flux (sensible + latent).
# For a 50 m mixed layer, typical surface fluxes of order 100 W/m² produce SST changes
# of order 0.01–0.1 K over a 4-hour simulation.
#
# The atmospheric setup is identical to Breeze's PrescribedSST example: 2D moist
# convection driven by a top-hat SST pattern, with wind and stability-dependent
# bulk exchange coefficients.

using NumericalEarth
using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Oceananigans
using Oceananigans.Units
using Printf

# ## Grid setup
#
# Same 2D domain as the PrescribedSST example: 20 km × 10 km with 128×128 resolution.

grid = RectilinearGrid(size = (128, 128), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Atmosphere model formulation
#
# Anelastic dynamics with warm-phase saturation adjustment microphysics,
# using the same reference state as the PrescribedSST example.

p₀, θ₀ = 101325, 285 # Pa, K
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)

microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

momentum_advection = WENO(order=9)
scalar_advection = WENO(order=5)

# ## Sea surface temperature as a mutable Field
#
# Unlike the PrescribedSST example which uses a function `T₀(x)`, here we create
# an Oceananigans `Field` for SST. This field is shared between the atmosphere
# boundary conditions and the slab ocean model, so updates to SST are automatically
# seen by the atmosphere on the next timestep.

ΔT = 4 # K
T₀_func(x) = θ₀ + ΔT / 2 * sign(cos(2π * x / grid.Lx))

sst_grid = RectilinearGrid(grid.architecture,
                           size = grid.Nx,
                           halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

SST = CenterField(sst_grid)
set!(SST, T₀_func)

# ## Boundary conditions
#
# Wind and stability-dependent polynomial exchange coefficients, identical to
# the PrescribedSST example, but using the SST Field instead of a function.

Uᵍ = 1e-2  # Gustiness (m/s)
coef = PolynomialCoefficient(roughness_length = 1.5e-4)

ρu_surface_flux = ρv_surface_flux = BulkDrag(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST)
ρe_surface_flux = BulkSensibleHeatFlux(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST)
ρqᵗ_surface_flux = BulkVaporFlux(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST)

ρu_bcs = FieldBoundaryConditions(bottom=ρu_surface_flux)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_surface_flux)
ρe_bcs = FieldBoundaryConditions(bottom=ρe_surface_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_surface_flux)

# ## Atmosphere model construction

atmosphere = AtmosphereModel(grid; momentum_advection, scalar_advection, microphysics, dynamics,
                             boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))

# ## Slab ocean
#
# A 50 m deep slab ocean with standard seawater properties.

ocean = SlabOcean(SST, mixed_layer_depth=50, density=1025, heat_capacity=4000)

# ## Coupled model
#
# The `AtmosphereOceanModel` constructor (provided by the Breeze extension) sets up
# the flux extraction operations and creates the coupled model.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Initial conditions

set!(atmosphere, θ=reference_state.potential_temperature, u=1)

# ## Simulation setup
#
# Run for 4 hours with adaptive time stepping, same as PrescribedSST.

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Diagnostic fields

T = atmosphere.temperature
θ = liquid_ice_potential_temperature(atmosphere)
qˡ = atmosphere.microphysical_fields.qˡ
u, v, w = atmosphere.velocities

# ## Surface flux diagnostics

Q_sensible = model.ocean_surface_fluxes.Q_sensible
Q_latent = model.ocean_surface_fluxes.Q_latent
Q_net = model.ocean_surface_fluxes.Q_net

# ## Progress callback

function progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities
    qˡ = atmos.microphysical_fields.qˡ

    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    qˡmax = maximum(qˡ)

    sst = sim.model.ocean.sea_surface_temperature
    sst_min = minimum(sst)
    sst_max = maximum(sst)

    msg = @sprintf("Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, max(qˡ): %.2e, SST: (%.2f, %.2f)",
                    iteration(sim), prettytime(sim), umax, wmax, qˡmax, sst_min, sst_max)

    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# Save atmosphere fields and SST evolution.

output_filename = "atmosphere_slab_ocean.jld2"

s = sqrt(u^2 + w^2)
ξ = ∂z(u) - ∂x(w)
qᵗ = atmosphere.specific_moisture

outputs = (; s, ξ, T, θ, qˡ, qᵗ, Q_sensible, Q_latent, Q_net, SST)

ow = JLD2Writer(model, outputs;
                filename = output_filename,
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

# ## Run

@info "Running atmosphere–slab ocean simulation..."
run!(simulation)

@info "Simulation complete."
@info "Final SST range: $(minimum(SST)) to $(maximum(SST)) K"
