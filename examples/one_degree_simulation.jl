# # [One-degree global ocean--sea ice simulation](@id one-degree-ocean-seaice)
#
# This example configures a global ocean--sea ice simulation at 1ᵒ horizontal resolution with
# realistic bathymetry and a few closures including the "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`.
# The simulation is forced by the repeat-year JRA55 atmospheric reanalysis
# and initialized with temperature, salinity, sea ice concentration, and sea ice thickness
# from the ECCO state estimate.
#
# For this example, we need Oceananigans, NumericalEarth, Dates, CUDA, and
# CairoMakie to visualize the simulation.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates
using Printf
using Statistics
using CUDA

# ### Grid and bathymetry

# We start by constructing an underlying `TripolarGrid` at about 1ᵒ resolution,

arch = GPU()
Nx = 360
Ny = 180
Nz = 50

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

# Next, we regrid the bathymetry onto this grid, using interpolation passes to smooth it.
# Keeping 2 major basins retains the Mediterranean Sea:

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)

# We then incorporate the bathymetry into an `ImmersedBoundaryGrid`,

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map=true)

# ### Closures
#
# We parameterize the mesoscale eddy fluxes with a Gent-McWilliams isopycnal diffusivity,
# and vertical mixing in the upper-ocean boundary layer with CATKE.

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=AdvectiveFormulation())
vertical_mixing = NumericalEarth.Oceans.default_ocean_closure()

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use split-explicit time stepping with 70 substeps for the barotropic mode.

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(eddy_closure, vertical_mixing))

ocean.model

# ### Sea ice simulation
#
# We also build a sea ice simulation with the default configuration:
# EVP rheology and a zero-layer thermodynamic model that evolves ice thickness
# and concentration.

sea_ice = sea_ice_simulation(grid, ocean; advection=tracer_advection)

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
ecco_variables = (:temperature, :salinity, :sea_ice_thickness, :sea_ice_concentration)
ecco_set = MetadataSet(ecco_variables; dataset = ECCO4Monthly(), date)

# A single `MetadataSet` initializes both components; each model picks up only
# the variables it knows about.

set!(ocean.model,   ecco_set)   # :temperature, :salinity → T, S
set!(sea_ice.model, ecco_set)   # :sea_ice_thickness, :sea_ice_concentration → h, ℵ

# ### JRA55-based atmosphere, radiation, and land
#
# We force the simulation with the JRA55-do atmospheric reanalysis, which provides
# the atmospheric state and radiative fluxes, as well as land-based freshwater fluxes
# from rivers and icebergs.
#
# The radiation component uses the latitude-dependent ocean albedo of
# [large2009global](@citet).

land = JRA55PrescribedLand(grid)
atmosphere = JRA55PrescribedAtmosphere(arch)

ocean_surface = SurfaceRadiationProperties(albedo = LatitudeDependentAlbedo())
radiation = JRA55PrescribedRadiation(arch; ocean_surface)

# ### Coupled simulation
#
# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`. We use a time step of 20 minutes.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)
simulation = Simulation(coupled_model; Δt=20minutes, stop_time=90days)

# ### A progress messenger
#
# We write a function that prints a progress message while the simulation runs.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    e = ocean.model.tracers.e
    Tmin, Tmax, Tmean = minimum(T), maximum(T), mean(view(T, :, :, ocean.model.grid.Nz))
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", extrema(To): (%.1f, %.1f) ᵒC, mean(To(z=0)): %.1f ᵒC", Tmin, Tmax, Tmean)
    msg4 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg5 = @sprintf(", wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg5

    wall_time[] = time_ns()

    return nothing
end

# We add it as a callback to the simulation.

add_callback!(simulation, progress, TimeInterval(5days))

# ### Output
#
# We are almost there! We need to save some output. We save _only the surface_ values
# of all velocity and tracer components using the `indices` keyword argument.
# Besides temperature and salinity, the tracers include the prognostic turbulent kinetic
# energy, `e`, that CATKE uses to diagnose the vertical mixing length.

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
η = ocean.model.free_surface.displacement
sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration,
                         T = sea_ice.model.ice_thermodynamics.top_surface_temperature),
                         sea_ice.model.velocities)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(1days),
                                            filename = "ocean_one_degree_surface_fields",
                                            indices = (:, :, grid.Nz),
                                            overwrite_files = true)

ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η);
                                                 schedule = TimeInterval(1days),
                                                 filename = "ocean_one_degree_free_surface",
                                                 overwrite_files = true)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              schedule = TimeInterval(1days),
                                              filename = "sea_ice_one_degree_surface_fields",
                                              overwrite_files = true)

# ### Ready to run

# We are ready to press the big red button and run the simulation.

run!(simulation)

# ### A movie
#
# We load the saved output and make a movie of the simulation. First we plot a snapshot.

using CairoMakie

# We suffix the ocean fields with "o",

uo_ts = FieldTimeSeries("ocean_one_degree_surface_fields.jld2", "u"; backend = OnDisk())
vo_ts = FieldTimeSeries("ocean_one_degree_surface_fields.jld2", "v"; backend = OnDisk())
To_ts = FieldTimeSeries("ocean_one_degree_surface_fields.jld2", "T"; backend = OnDisk())
eo_ts = FieldTimeSeries("ocean_one_degree_surface_fields.jld2", "e"; backend = OnDisk())
ηo_ts = FieldTimeSeries("ocean_one_degree_free_surface.jld2",   "η"; backend = OnDisk())

# and the sea ice fields with "i":

ui_ts = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "u"; backend = OnDisk())
vi_ts = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "v"; backend = OnDisk())
hi_ts = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "h"; backend = OnDisk())
ℵi_ts = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "ℵ"; backend = OnDisk())

times = uo_ts.times
Nt = length(times)
n = Observable(Nt)

Toₙ = @lift To_ts[$n]
eoₙ = @lift eo_ts[$n]
ηoₙ = @lift ηo_ts[$n]

# The surface speeds and the effective ice thickness, `h ℵ`, are fields computed from
# snapshot fields that we `set!` at every frame. The sea ice speed is set to zero
# where there is no ice.

uoₙ = uo_ts[1]
voₙ = vo_ts[1]
uiₙ = ui_ts[1]
viₙ = vi_ts[1]
hiₙ = hi_ts[1]
ℵiₙ = ℵi_ts[1]

so = Field(sqrt(uoₙ^2 + voₙ^2))
si = Field(sqrt(uiₙ^2 + viₙ^2) * (hiₙ * ℵiₙ > 1e-7))
he = Field(hiₙ * ℵiₙ)

soₙ = @lift begin
    set!(uoₙ, uo_ts[$n])
    set!(voₙ, vo_ts[$n])
    so
end

siₙ = @lift begin
    set!(uiₙ, ui_ts[$n])
    set!(viₙ, vi_ts[$n])
    set!(hiₙ, hi_ts[$n])
    set!(ℵiₙ, ℵi_ts[$n])
    si
end

heₙ = @lift begin
    set!(hiₙ, hi_ts[$n])
    set!(ℵiₙ, ℵi_ts[$n])
    he
end

# Finally, we plot a snapshot of the ocean surface speed, sea surface height, temperature,
# and the turbulent kinetic energy from CATKE, as well as the sea ice speed and the
# effective sea ice thickness.

fig = Figure(size = (900, 750))

title = @lift string("Global 1ᵒ ocean simulation after ", prettytime(times[$n] - times[1]))

axso = Axis(fig[1, 1])
axηo = Axis(fig[1, 3])
axTo = Axis(fig[2, 1])
axeo = Axis(fig[2, 3])
axsi = Axis(fig[3, 1])
axhi = Axis(fig[3, 3])

hm = heatmap!(axso, soₙ, colorrange = (0, 0.5), colormap = :deep, nan_color = :lightgray)
Colorbar(fig[1, 2], hm, label = "Ocean surface speed (m s⁻¹)")
hm = heatmap!(axηo, ηoₙ, colorrange = (-1.2, 1.2), colormap = :balance, nan_color = :lightgray)
Colorbar(fig[1, 4], hm, label = "Sea surface height (m)")
hm = heatmap!(axTo, Toₙ, colorrange = (-1, 32), colormap = :magma, nan_color = :lightgray)
Colorbar(fig[2, 2], hm, label = "Surface temperature (ᵒC)")
hm = heatmap!(axeo, eoₙ, colorrange = (0, 1e-3), colormap = :solar, nan_color = :lightgray)
Colorbar(fig[2, 4], hm, label = "Turbulent kinetic energy (m² s⁻²)")
hm = heatmap!(axsi, siₙ, colorrange = (0, 0.5), colormap = :greys, nan_color = :lightgray)
Colorbar(fig[3, 2], hm, label = "Sea ice speed (m s⁻¹)")
hm = heatmap!(axhi, heₙ, colorrange = (0, 4), colormap = :blues, nan_color = :lightgray)
Colorbar(fig[3, 4], hm, label = "Effective ice thickness (m)")

for ax in (axso, axηo, axTo, axeo, axsi, axhi)
    hidedecorations!(ax)
end

Label(fig[0, :], title)

save("global_snapshot.png", fig)
nothing #hide

# ![](global_snapshot.png)

# And now a movie:

CairoMakie.record(fig, "one_degree_global_ocean_surface.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](one_degree_global_ocean_surface.mp4)
