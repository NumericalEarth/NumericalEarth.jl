# # Atmosphere–slab ocean coupling
#
# This example demonstrates coupling a Breeze atmospheric LES with a slab ocean model
# using NumericalEarth's `EarthSystemModel` framework. The sea surface temperature (SST)
# evolves in response to the atmospheric surface heat fluxes computed by the ESM's
# similarity theory.
#
# The slab ocean model represents a well-mixed ocean layer of fixed depth ``H`` with the
# SST tendency equation:
#
# ```math
# \frac{∂T}{∂t} = -\frac{J^T}{H}
# ```
#
# where ``J^T`` is the temperature flux (in K m/s) assembled by the coupling framework.
#
# The atmospheric setup is similar to Breeze's PrescribedSST example: 2D moist
# convection driven by a top-hat SST pattern, with wind- and stability-dependent
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

# ## Atmosphere setup
#
# Anelastic dynamics with warm-phase saturation adjustment microphysics.

p₀, θ₀ = 101325, 285 # Pa, K
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

momentum_advection = WENO(order=9)
scalar_advection = WENO(order=5)

# ## Sea surface temperature
#
# The SST is a mutable `Field` on a 1D horizontal grid. It is shared between the
# atmosphere boundary conditions and the slab ocean, so updates to SST are automatically
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

# ## Atmosphere boundary conditions
#
# We use Breeze's polynomial bulk exchange coefficients. The same SST field
# is passed to the BCs and to the slab ocean.

Uᵍ = 1e-2
coef = PolynomialCoefficient(roughness_length = 1.5e-4)

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST))
ρe_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=coef, gustiness=Uᵍ, surface_temperature=SST))

atmosphere = AtmosphereModel(grid; momentum_advection, scalar_advection, microphysics, dynamics,
                             boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))

# ## Slab ocean
#
# A 50 m deep slab ocean with standard seawater properties.

ocean = SlabOcean(SST, mixed_layer_depth=50, density=1025, heat_capacity=4000)

# ## Coupled model
#
# `AtmosphereOceanModel` returns an `EarthSystemModel` that couples the atmosphere
# and ocean through the similarity theory turbulent flux computation.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Initial conditions

set!(atmosphere, θ=reference_state.potential_temperature, u=1)

# ## Simulation
#
# Run for 4 hours with a fixed timestep.

simulation = Simulation(model, Δt=10, stop_time=4hours)

# ## Diagnostics

T = atmosphere.temperature
θ = liquid_ice_potential_temperature(atmosphere)
qˡ = atmosphere.microphysical_fields.qˡ
u, v, w = atmosphere.velocities

# ## Progress callback

function progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities

    umax = maximum(abs, u)
    wmax = maximum(abs, w)

    sst = sim.model.ocean.sea_surface_temperature
    sst_min = minimum(sst)
    sst_max = maximum(sst)

    msg = @sprintf("Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, SST: (%.3f, %.3f)",
                    iteration(sim), prettytime(sim), umax, wmax, sst_min, sst_max)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

# ## Output
#
# Save atmosphere fields, SST, and computed interface fluxes.

output_filename = "atmosphere_slab_ocean"

s = sqrt(u^2 + w^2)
ξ = ∂z(u) - ∂x(w)
qᵗ = atmosphere.specific_moisture

atmo_outputs = (; s, ξ, T, θ, qˡ, qᵗ)

ow = JLD2Writer(model, atmo_outputs;
                filename = output_filename * "_atmo.jld2",
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:atmo] = ow

# Save SST separately on its own grid
sst_ow = JLD2Writer(model, (; SST);
                     filename = output_filename * "_sst.jld2",
                     schedule = TimeInterval(2minutes),
                     overwrite_existing = true)

simulation.output_writers[:sst] = sst_ow

# Save the ESM-computed interface fluxes
ao_interface = model.interfaces.atmosphere_ocean_interface
if !isnothing(ao_interface)
    Q_sensible = ao_interface.fluxes.sensible_heat
    Q_latent = ao_interface.fluxes.latent_heat
    flux_outputs = (; Q_sensible, Q_latent)

    flux_ow = JLD2Writer(model, flux_outputs;
                         filename = output_filename * "_fluxes.jld2",
                         schedule = TimeInterval(2minutes),
                         overwrite_existing = true)

    simulation.output_writers[:fluxes] = flux_ow
end

# ## Run

@info "Running atmosphere–slab ocean coupled simulation..."
run!(simulation)
@info "Simulation complete. Final SST range: $(minimum(SST)) to $(maximum(SST)) K"

# ## Animation
#
# Create an animation showing the atmospheric state and SST evolution.

using CairoMakie

atmo_ds = FieldDataset(output_filename * "_atmo.jld2")
sst_ds = FieldDataset(output_filename * "_sst.jld2")

times = atmo_ds["s"].times
Nt = length(times)

fig = Figure(size = (1200, 800), fontsize = 14)

ax_s = Axis(fig[1, 1], title="Wind speed (m/s)", xlabel="x (km)", ylabel="z (km)")
ax_ξ = Axis(fig[1, 2], title="Vorticity (1/s)", xlabel="x (km)", ylabel="z (km)")
ax_θ = Axis(fig[2, 1], title="θₗᵢ (K)", xlabel="x (km)", ylabel="z (km)")
ax_q = Axis(fig[2, 2], title="Cloud water (g/kg)", xlabel="x (km)", ylabel="z (km)")
ax_sst = Axis(fig[3, 1:2], title="Sea surface temperature (K)", xlabel="x (km)", ylabel="SST (K)")

n = Observable(1)

x_atmo = grid.xᶜᵃᵃ[1:grid.Nx] ./ 1e3
z_atmo = grid.zᵃᵃᶜ[1:grid.Nz] ./ 1e3
x_sst = sst_grid.xᶜᵃᵃ[1:sst_grid.Nx] ./ 1e3

sn  = @lift interior(atmo_ds["s"][$n],  :, 1, :)
ξn  = @lift interior(atmo_ds["ξ"][$n],  :, 1, :)
θn  = @lift interior(atmo_ds["θ"][$n],  :, 1, :)
qˡn = @lift interior(atmo_ds["qˡ"][$n], :, 1, :) .* 1e3  # Convert to g/kg

sstn = @lift interior(sst_ds["SST"][$n], :, 1, 1)

heatmap!(ax_s, x_atmo, z_atmo, sn; colormap=:speed, colorrange=(0, 5))
heatmap!(ax_ξ, x_atmo, z_atmo, ξn; colormap=:balance, colorrange=(-0.05, 0.05))
heatmap!(ax_θ, x_atmo, z_atmo, θn; colormap=:thermal, colorrange=(θ₀ - 1, θ₀ + 3))
heatmap!(ax_q, x_atmo, z_atmo, qˡn; colormap=:dense, colorrange=(0, 1))
lines!(ax_sst, x_sst, sstn; color=:red, linewidth=2)
ylims!(ax_sst, θ₀ - ΔT/2 - 0.5, θ₀ + ΔT/2 + 0.5)

title = @lift "Atmosphere–slab ocean coupling, t = " * prettytime(times[$n])
Label(fig[0, 1:2], title, fontsize=18)

@info "Rendering animation..."
record(fig, output_filename * ".mp4", 1:Nt; framerate=12) do i
    n[] = i
end

@info "Animation saved to $(output_filename).mp4"
nothing #hide

# ![](atmosphere_slab_ocean.mp4)
