# # [One-degree global ocean--sea ice simulation](@id one-degree-ocean-seaice)
#
# This example configures a global ocean--sea ice simulation at 1ᵒ horizontal resolution with
# realistic bathymetry and a few closures including the "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`.
# The simulation is forced by repeat-year JRA55 atmospheric reanalysis
# and initialized by temperature, salinity, sea ice concentration, and sea ice thickness
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

# ### Grid and Bathymetry
#
# We start by constructing an underlying TripolarGrid at ~1 degree horizontal resolution
# and with a 50-level exponentially-stretched vertical grid,

arch = GPU()
Nx = 360
Ny = 180
Nz = 50

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

# Next, we build bathymetry on this grid, using interpolation passes to smooth the bathymetry.
# With 2 major basins, we keep the Mediterranean (though we need to manually open the Gibraltar
# Strait to connect it to the Atlantic):

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)

# We then incorporate the bathymetry into an ImmersedBoundaryGrid,

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map=true)

# ### Closures
#
# We include a Gent-McWilliams isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization.

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=AdvectiveFormulation())
vertical_mixing = NumericalEarth.Oceans.default_ocean_closure()

# ### Ocean simulation
#
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 70 substeps for the barotropic mode.

free_surface = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection = WENO(order=5)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(eddy_closure, vertical_mixing))

@info "We've built an ocean simulation with model:"
@show ocean.model

# ### Sea Ice simulation
#
# We also build a sea ice simulation. We use the default configuration:
# EVP rheology and a zero-layer thermodynamic model that advances thickness
# and concentration.

sea_ice = sea_ice_simulation(grid, ocean; advection=tracer_advection)

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
ecco_variables = (:temperature, :salinity, :sea_ice_thickness, :sea_ice_concentration)
ecco_set = MetadataSet(ecco_variables; dataset = ECCO4Monthly(), date)

# A single MetadataSet drives both components; variables not in
# `variable_glossary` for a given model fall through silently.
set!(ocean.model,   ecco_set)   # picks up :temperature, :salinity → T, S
set!(sea_ice.model, ecco_set)   # picks up :sea_ice_thickness, :sea_ice_concentration → h, ℵ

# ### JRA55-based atmospheric state, radiation, and land model
#
# We force the simulation with data derived from the JRA55-do atmospheric reanalysis,
# which include the atmospheric state and radiative fluxes, as well as
# land-based freshwater fluxes from rivers and icebergs.
#
# In the radiation component we prescribed a latitude-dependent ocean albedo due to
# Large & Yeager 2009.

land = JRA55PrescribedLand(arch)
atmosphere = JRA55PrescribedAtmosphere(arch)

ocean_surface = SurfaceRadiationProperties(albedo = LatitudeDependentAlbedo())
radiation = JRA55PrescribedRadiation(arch; ocean_surface)

# ### Coupled simulation
#
# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.
#
# With Runge-Kutta 3rd order time-stepping we can safely use a timestep of 20 minutes.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)
simulation = Simulation(coupled_model; Δt=20minutes, stop_time=20days)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    e = ocean.model.tracers.e
    Tmin, Tmax, Tavg = minimum(T), maximum(T), mean(view(T, :, :, ocean.model.grid.Nz))
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uᵒᶜ|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", extrema(Tᵒᶜ): (%.1f, %.1f) ᵒC, mean(Tᵒᶜ(z=0)): %.1f ᵒC", Tmin, Tmax, Tavg)
    msg4 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg5 = @sprintf(", wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg5

    wall_time[] = time_ns()

     return nothing
end

# And add it as a callback to the simulation.
add_callback!(simulation, progress, TimeInterval(5days))

# ### Output
#
# We are almost there! We need to save some output. Below we choose to save _only surface_
# fields using the `indices` keyword argument. We save all the velocity and tracer components.
# Note, that besides temperature and salinity, the CATKE vertical mixing parameterization
# also uses a prognostic turbulent kinetic energy, `e`, to diagnose the vertical mixing length.

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
free_surface = ocean.model.free_surface.displacement
sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration,
                         T = sea_ice.model.ice_thermodynamics.top_surface_temperature),
                         sea_ice.model.velocities)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(1days),
                                            filename = "ocean_one_degree_surface_fields",
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η = free_surface);
                                                 schedule = TimeInterval(1days),
                                                 filename = "ocean_one_degree_free_surface",
                                                 overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              schedule = TimeInterval(1days),
                                              filename = "sea_ice_one_degree_surface_fields",
                                              overwrite_existing = true)

# ### Ready to run

# We are ready to press the big red button and run the simulation.
run!(simulation)

# ### A movie
#
# We load the saved output and make a movie of the simulation. First we plot a snapshot:
using CairoMakie

# Ocean surface fields (superscript `ᵒᶜ`):
uᵒᶜ = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "u"; backend = OnDisk())
vᵒᶜ = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "v"; backend = OnDisk())
Tᵒᶜ = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "T"; backend = OnDisk())
eᵒᶜ = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "e"; backend = OnDisk())
ηᵒᶜ = FieldTimeSeries("ocean_one_degree_free_surface.jld2",    "η"; backend = OnDisk())

# Sea-ice fields (superscript `ˢⁱ`):
uˢⁱ = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "u"; backend = OnDisk())
vˢⁱ = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "v"; backend = OnDisk())
hˢⁱ = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "h"; backend = OnDisk())
ℵˢⁱ = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "ℵ"; backend = OnDisk())
Tˢⁱ = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "T"; backend = OnDisk())

times = uᵒᶜ.times
Nt = length(times)
n = Observable(Nt)

# The Oceananigans Makie extension fills immersed (land) cells with `NaN` automatically
# when a `Field`, `AbstractOperation`, or `Observable` thereof is passed to `heatmap!`,
# so we just lift the relevant fields and operations at each frame.
Tᵒᶜₙ = @lift Tᵒᶜ[$n]
eᵒᶜₙ = @lift eᵒᶜ[$n]
ηᵒᶜₙ = @lift ηᵒᶜ[$n]
heₙ  = @lift hˢⁱ[$n] * ℵˢⁱ[$n]
sᵒᶜₙ = @lift sqrt(uᵒᶜ[$n]^2 + vᵒᶜ[$n]^2)

# Sea-ice speed: zero out where ice is too thin; the extension masks land via NaN.
uˢⁱₙ = Field{Face, Center, Nothing}(uˢⁱ.grid)
vˢⁱₙ = Field{Center, Face, Nothing}(vˢⁱ.grid)
sˢⁱ  = Field(sqrt(uˢⁱₙ^2 + vˢⁱₙ^2))

sˢⁱₙ = @lift begin
    parent(uˢⁱₙ) .= parent(uˢⁱ[$n])
    parent(vˢⁱₙ) .= parent(vˢⁱ[$n])
    compute!(sˢⁱ)
    hₙ = interior(hˢⁱ[$n])
    ℵₙ = interior(ℵˢⁱ[$n])
    interior(sˢⁱ)[hₙ .* ℵₙ .< 1e-7] .= 0
    sˢⁱ
end

# Finally, we plot a snapshot of the surface speed, temperature, and the turbulent
# eddy kinetic energy from the CATKE vertical mixing parameterization as well as the
# sea ice speed and the effective sea ice thickness.
fig = Figure(size=(1200, 1000))

title = @lift string("Global 1ᵒ ocean simulation after ", prettytime(times[$n] - times[1]))

axsᵒᶜ = Axis(fig[1, 1])
axηᵒᶜ = Axis(fig[1, 3])
axTᵒᶜ = Axis(fig[2, 1])
axeᵒᶜ = Axis(fig[2, 3])
axsˢⁱ = Axis(fig[3, 1])
axhˢⁱ = Axis(fig[3, 3])

hm = heatmap!(axsᵒᶜ, sᵒᶜₙ, colorrange = (0, 0.5), colormap = :deep,  nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Ocean Surface speed (m s⁻¹)")
hm = heatmap!(axηᵒᶜ, ηᵒᶜₙ, colorrange = (-1.2, 1.2), colormap = :balance, nan_color=:lightgray)
Colorbar(fig[1, 4], hm, label = "Sea Surface Height (m)")
hm = heatmap!(axTᵒᶜ, Tᵒᶜₙ, colorrange = (-1, 32), colormap = :magma, nan_color=:lightgray)
Colorbar(fig[2, 2], hm, label = "Surface Temperature (ᵒC)")
hm = heatmap!(axeᵒᶜ, eᵒᶜₙ, colorrange = (0, 1e-3), colormap = :solar, nan_color=:lightgray)
Colorbar(fig[2, 4], hm, label = "Turbulent Kinetic Energy (m² s⁻²)")

hm = heatmap!(axsˢⁱ, sˢⁱₙ, colorrange = (0, 0.5), colormap = :greys, nan_color=:lightgray)
Colorbar(fig[3, 2], hm, label = "Sea ice speed (m s⁻¹)")
hm = heatmap!(axhˢⁱ, heₙ, colorrange =  (0, 4),  colormap = :blues, nan_color=:lightgray)
Colorbar(fig[3, 4], hm, label = "Effective ice thickness (m)")


for ax in (axsᵒᶜ, axsˢⁱ, axTᵒᶜ, axhˢⁱ, axeᵒᶜ)
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
