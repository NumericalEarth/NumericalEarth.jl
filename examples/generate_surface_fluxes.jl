# # Surface fluxes from a prescribed ocean and atmosphere
#
# NumericalEarth estimates the surface exchange of momentum, heat, and water vapor
# between the atmosphere and the ocean with bulk formulae.
#
# This example computes these turbulent surface fluxes for an ocean initialized
# from ECCO data under a prescribed JRA55 atmosphere.
#
# Besides NumericalEarth, we need Oceananigans for the grid and `Field` utilities,
# and CairoMakie for plotting.

using NumericalEarth
using Oceananigans
using Dates
using CairoMakie

# ## Computing fluxes on the ECCO grid
#
# We start from the native ECCO grid between 80°S and 80°N, and add bathymetry
# regridded onto it with `regrid_bathymetry`.

ecco_temperature = Metadatum(:temperature; dataset=ECCO4Monthly(), region=BoundingBox(latitude=(-80, 80)))
underlying_grid = native_grid(ecco_temperature; halo=(7, 7, 7))
bottom_height = regrid_bathymetry(underlying_grid)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# We plot the bottom height of the ECCO grid.

fig, ax, hm = heatmap(bottom_height)
Colorbar(fig[1, 2], hm, height = Relative(3/4), label = "Bottom height (m)")

save("ECCO_continents.png", fig)

# ![](ECCO_continents.png)

# Next, we construct the atmosphere and the ocean.
#
# The atmosphere is prescribed from the JRA55 reanalysis. It contains
# - the zonal wind `u`,
# - the meridional wind `v`,
# - the surface air temperature `T`,
# - the surface specific humidity `q`, and
# - the surface pressure `p`.
#
# With `time_indices_in_memory = 2`, only the first two snapshots, January 1st
# at 00:00 and 03:00, are loaded into memory.

atmosphere = JRA55PrescribedAtmosphere(; time_indices_in_memory = 2)
ocean = ocean_simulation(grid, closure=nothing)

# We then set the ocean temperature and salinity from ECCO data. First we create
# the temperature and salinity metadata,

ecco_set = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = DateTime(1993, 1, 1))

# (without a `date`, the metadata default to the first date of the dataset) and then
# `set!` the ECCO state into `ocean.model`.

set!(ocean.model, ecco_set)

# Finally, we construct the coupled model, which computes the fluxes upon construction.
# We omit `sea_ice`, so the model is ocean-only, and pair the JRA55 atmosphere with a
# matching `JRA55PrescribedRadiation` that supplies the downwelling shortwave and
# longwave radiation and the radiative properties of the ocean surface.

radiation = JRA55PrescribedRadiation(; time_indices_in_memory = 2)
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

# With the surface fluxes computed, we extract and plot them. The turbulent fluxes
# are stored in `coupled_model.interfaces.atmosphere_ocean_interface.fluxes`.

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

fig = Figure(size = (800, 800), fontsize = 15)

ax = Axis(fig[1, 1], title = "Sensible heat flux (W m⁻²)", ylabel = "Latitude")
heatmap!(ax, fluxes.sensible_heat; colormap = :bwr)

ax = Axis(fig[1, 2], title = "Latent heat flux (W m⁻²)")
heatmap!(ax, fluxes.latent_heat; colormap = :bwr)

ax = Axis(fig[2, 1], title = "Zonal wind stress (N m⁻²)", ylabel = "Latitude")
heatmap!(ax, fluxes.x_momentum; colormap = :bwr)

ax = Axis(fig[2, 2], title = "Meridional wind stress (N m⁻²)", xlabel = "Longitude")
heatmap!(ax, fluxes.y_momentum; colormap = :bwr)

ax = Axis(fig[3, 1], title = "Water vapor flux (kg m⁻² s⁻¹)", xlabel = "Longitude", ylabel = "Latitude")
heatmap!(ax, fluxes.water_vapor; colormap = :bwr)

save("surface_fluxes.png", fig)

# ![](surface_fluxes.png)
