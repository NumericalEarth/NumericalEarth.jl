# # Coupling components on different horizontal grids
#
# Earth system components rarely want the same resolution. An atmosphere with a large eddy turnover length is
# often coarser than the ocean beneath it, and a land model has its own grid again. `EarthSystemModel` handles this
# by computing the surface fluxes on one exchange grid, the ocean's here, and remapping every component that lives
# on a different horizontal grid. Remapping is first-order conservative, so the energy that leaves the ocean is the
# energy that enters the atmosphere.
#
# In this example a coarse [Breeze](https://github.com/NumericalEarth/Breeze.jl) atmosphere sits over a slab ocean
# four times finer, with a sea surface temperature front that only the ocean can resolve. We compare the heat flux on
# the two grids and check that their global integrals agree.
#
# Loading ConservativeRegridding enables the remapping:

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Breeze
using ConservativeRegridding
using CairoMakie

# ## The two grids
#
# Both grids span the same doubly periodic domain. The atmosphere has 16 × 16 columns and the ocean 64 × 64 cells.

extent = 80kilometers

atmosphere_grid = RectilinearGrid(size = (16, 16, 32),
                                  halo = (5, 5, 5),
                                  x = (0, extent),
                                  y = (0, extent),
                                  z = (0, 4kilometers),
                                  topology = (Periodic, Periodic, Bounded))

ocean_grid = RectilinearGrid(size = (64, 64),
                             halo = (5, 5),
                             x = (0, extent),
                             y = (0, extent),
                             topology = (Periodic, Periodic, Flat))

# ## The components
#
# The atmosphere starts with a uniform potential temperature and a steady wind. The ocean has warm and cold bands that
# are sharper than an atmospheric grid cell.

θ₀ = 285

reference_state = ReferenceState(atmosphere_grid, ThermodynamicConstants(); potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)

atmosphere = atmosphere_simulation(atmosphere_grid; dynamics)
set!(atmosphere.model, θ=θ₀, u=5)

ocean = SlabOcean(ocean_grid, depth=50, density=1025, heat_capacity=4000)

sea_surface_temperature(x, y) = θ₀ + 3 + 6tanh(4sinpi(2x / extent)) * cospi(2y / extent)
set!(ocean, T=sea_surface_temperature)

# The model picks the regridder: the two grids have the same type but different horizontal cell faces,
# so it remaps. On a horizontally identical grid it would share the fields directly.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Running and saving the fluxes
#
# We save the sensible heat flux on the exchange grid, where it is computed, and on the atmosphere's grid, where it is
# applied, together with their global integrals.

exchange_grid_flux = model.interfaces.surface_partition.fluxes.sensible_heat
atmosphere_grid_flux = model.interfaces.net_fluxes.atmosphere.ρE

simulation = Simulation(model; Δt = 2seconds, stop_time = 30minutes)

filename = "coupling_across_horizontal_grids"

simulation.output_writers[:fluxes] = JLD2Writer(model, (; exchange_grid_flux, atmosphere_grid_flux);
                                                filename = filename * "_fluxes",
                                                schedule = TimeInterval(30minutes),
                                                overwrite_files = true)

simulation.output_writers[:integrals] = JLD2Writer(model, (; exchange_grid_integral = Integral(exchange_grid_flux),
                                                             atmosphere_grid_integral = Integral(atmosphere_grid_flux));
                                                   filename = filename * "_integrals",
                                                   schedule = TimeInterval(2minutes),
                                                   overwrite_files = true)

run!(simulation)

# ## The same flux on two grids
#
# The ocean resolves the front, and each atmospheric column receives the area average of the cells below it.

exchange_grid_series = FieldTimeSeries(filename * "_fluxes.jld2", "exchange_grid_flux")
atmosphere_grid_series = FieldTimeSeries(filename * "_fluxes.jld2", "atmosphere_grid_flux")

fig = Figure(size = (900, 400))
ax1 = Axis(fig[1, 1]; title = "Sensible heat flux on the exchange grid (W m⁻²)", xlabel = "x (m)", ylabel = "y (m)")
ax2 = Axis(fig[1, 2]; title = "Sensible heat flux on the atmosphere grid (W m⁻²)", xlabel = "x (m)")
hm = heatmap!(ax1, exchange_grid_series[end]; colormap = :balance, colorrange = (-60, 60))
heatmap!(ax2, atmosphere_grid_series[end]; colormap = :balance, colorrange = (-60, 60))
Colorbar(fig[1, 3], hm)
save("coupling_across_horizontal_grids_fluxes.png", fig)
nothing #hide

# ![](coupling_across_horizontal_grids_fluxes.png)
#
# ## The energy budget closes
#
# The global integrals on the two grids follow each other to round-off. Heat is exchanged across the grids
# without being created or lost.

exchange_grid_integral = FieldTimeSeries(filename * "_integrals.jld2", "exchange_grid_integral")
atmosphere_grid_integral = FieldTimeSeries(filename * "_integrals.jld2", "atmosphere_grid_integral")

times = exchange_grid_integral.times
Q̄ᵉ = [only(interior(exchange_grid_integral[n])) for n in eachindex(times)]
Q̄ᵃ = [only(interior(atmosphere_grid_integral[n])) for n in eachindex(times)]

fig = Figure(size = (800, 350))
ax = Axis(fig[1, 1]; xlabel = "Time (minutes)", ylabel = "∫ Q dA (W)")
lines!(ax, times ./ minutes, Q̄ᵉ; label = "exchange grid", linewidth = 4)
lines!(ax, times ./ minutes, Q̄ᵃ; label = "atmosphere grid", linestyle = :dash, color = :black)
axislegend(ax)
save("coupling_across_horizontal_grids_budget.png", fig)
nothing #hide

# ![](coupling_across_horizontal_grids_budget.png)
