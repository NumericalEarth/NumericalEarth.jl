# # [Nested regional ocean near Bouvet Island](@id bouvet-nested-ocean)
#
# This example nests a regional ocean in the GLORYS daily reanalysis. The domain covers
# the Antarctic Circumpolar Current around Bouvet Island, 5ᵒW to 20ᵒE and 57ᵒS to 50ᵒS,
# at 1/8ᵒ horizontal resolution, and all four sides are open to the parent.
#
# `nested_ocean_model(grid, GLORYSDaily(); dates)` builds the whole nest:
#
# - GLORYS strips along each open side, streamed from disk and sampled onto narrow bands
#   once per child step;
# - open boundary conditions on the velocities, tracers and the barotropic transport;
# - a band of relaxation toward GLORYS along the open sides;
# - a seafloor blended toward GLORYS' own bathymetry at the open sides, so that the boundary
#   transport fits the water column the parent was run on;
# - an initial state interpolated from GLORYS at the first date.
#
# ERA5 drives the surface fluxes.
#
# For this example, we need Oceananigans, NumericalEarth, CopernicusMarine (for GLORYS),
# CopernicusClimateDataStore (for ERA5), CUDA, and CairoMakie to visualize the simulation.

using NumericalEarth
using NumericalEarth.DataWrangling.IBCSO: IBCSOv2
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using CopernicusMarine
using CopernicusClimateDataStore
using Dates
using Printf
using CUDA

# ### Grid and bathymetry
#
# We use a z★ vertical coordinate, which lets the cells stretch with the free surface.

arch = GPU()

λ₁, λ₂ = -5, 20
φ₁, φ₂ = -57, -50

Nx = 8 * (λ₂ - λ₁)
Ny = 8 * (φ₂ - φ₁)
Nz = 40

depth = 5300meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        z,
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂))

# The bathymetry comes from IBCSO v2, the International Bathymetric Chart of the Southern Ocean,
# which covers everything south of 50ᵒS. Partial cells represent the seafloor more faithfully
# than full cells on the steep flanks of the Southwest Indian and Mid-Atlantic ridges,
# and the active cells map skips the cells under the seafloor.

bottom_height = regrid_bathymetry(underlying_grid;
                                  dataset = IBCSOv2(),
                                  minimum_depth = 5,
                                  interpolation_passes = 5,
                                  major_basins = 3)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# ### The nested ocean
#
# We nest in GLORYS for the first half of January 2000. The open sides, the width of the
# relaxation band (2/3ᵒ), the relaxation rate (1/6 hours), and the width of the bathymetry
# blend (1/3ᵒ) all take their defaults. The remaining keyword arguments go to
# `hydrostatic_ocean_model`.

dates = (DateTime(2000, 1, 1), DateTime(2000, 1, 16))

nested = nested_ocean_model(grid, GLORYSDaily(); dates)

# The child is an ordinary `HydrostaticFreeSurfaceModel` on the blended bathymetry.

child = nested.child

# ### Atmosphere and coupled model
#
# ERA5 hourly single-level fields over the domain force the ocean through bulk formulae.

region = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1))

atmosphere = ERA5PrescribedAtmosphere(arch; start_date = first(dates), end_date = last(dates), region)
radiation  = ERA5PrescribedRadiation(arch;  start_date = first(dates), end_date = last(dates), region)

# A `NestedModel` goes into a `Simulation` and an `OceanOnlyModel` like any other ocean model.

Δt = 10minutes
ocean = Simulation(nested; Δt)
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time = 15days)

# We add a callback to print a message about how the simulation is going.

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = child.velocities
    T = child.tracers.T
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|u|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", extrema(T): (%.1f, %.1f) ᵒC", minimum(T), maximum(T))
    msg4 = @sprintf(", wall time: %s", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, TimeInterval(1days))

# ### Output
#
# We save the surface velocities, temperature and salinity every 6 hours. The free surface
# lives on its own level above the top cell, so it gets a writer of its own.

outputs = (; u = child.velocities.u, v = child.velocities.v,
             T = child.tracers.T, S = child.tracers.S)

ocean.output_writers[:surface] = JLD2Writer(child, outputs;
                                            schedule = TimeInterval(6hours),
                                            filename = "bouvet_nested_surface_fields",
                                            indices = (:, :, Nz),
                                            overwrite_files = true)

ocean.output_writers[:free_surface] = JLD2Writer(child, (; η = child.free_surface.displacement);
                                                 schedule = TimeInterval(6hours),
                                                 filename = "bouvet_nested_free_surface",
                                                 overwrite_files = true)

# ### Ready to run

run!(simulation)

# ### A movie
#
# We load the surface fields and animate the temperature and the surface speed.

using CairoMakie

u = FieldTimeSeries("bouvet_nested_surface_fields.jld2", "u"; backend = OnDisk())
v = FieldTimeSeries("bouvet_nested_surface_fields.jld2", "v"; backend = OnDisk())
T = FieldTimeSeries("bouvet_nested_surface_fields.jld2", "T"; backend = OnDisk())
η = FieldTimeSeries("bouvet_nested_free_surface.jld2", "η"; backend = OnDisk())

times = T.times
Nt = length(times)

λ, φ, _ = nodes(T.grid, Center(), Center(), Center())

# Land and the seafloor above the surface cell are masked out.

blended_bottom = on_architecture(CPU(), child.grid.immersed_boundary.bottom_height)[1:Nx, 1:Ny, 1]
land = blended_bottom .>= 0

function masked(a)
    a[land] .= NaN
    return a
end

surface_speed(n) = masked(sqrt.(interior(u[n], 1:Nx, :, 1) .^ 2 .+ interior(v[n], :, 1:Ny, 1) .^ 2))
surface_temperature(n) = masked(Array(interior(T[n], :, :, 1)))
free_surface(n) = masked(Array(interior(η[n], :, :, 1)))

# The color ranges span the last snapshot.

colorrange(a) = extrema(filter(!isnan, a))

n = Observable(1)

sn = @lift surface_speed($n)
Tn = @lift surface_temperature($n)
ηn = @lift free_surface($n)

fig = Figure(size = (1000, 900))

title = @lift string("Nested in GLORYS, ", prettytime(times[$n]))

axs = Axis(fig[1, 1], xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")
axT = Axis(fig[2, 1], xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")
axη = Axis(fig[3, 1], xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")

hms = heatmap!(axs, λ, φ, sn; colorrange = colorrange(surface_speed(Nt)), colormap = :deep, nan_color = :lightgray)
hmT = heatmap!(axT, λ, φ, Tn; colorrange = colorrange(surface_temperature(Nt)), colormap = :magma, nan_color = :lightgray)
hmη = heatmap!(axη, λ, φ, ηn; colorrange = colorrange(free_surface(Nt)), colormap = :balance, nan_color = :lightgray)

Colorbar(fig[1, 2], hms, label = "Surface speed (m s⁻¹)")
Colorbar(fig[2, 2], hmT, label = "Surface temperature (ᵒC)")
Colorbar(fig[3, 2], hmη, label = "Free surface (m)")

Label(fig[0, :], title, fontsize = 18)

n[] = Nt
save("bouvet_nested_surface.png", fig)
nothing #hide

# ![](bouvet_nested_surface.png)

CairoMakie.record(fig, "bouvet_nested_surface.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](bouvet_nested_surface.mp4)
