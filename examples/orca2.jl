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

# We start by constructing an underlying TripolarGrid at ~1 degree resolution,

const data_path = has_cuda_gpu() ? "/cephfs/home/js2430/store/Global/data" : "data" #"/home/js2430/rds/hpc-work/GlobalOceanBioME/data"#

arch = GPU()
Nz = 30
z = ExponentialDiscretization(Nz, -5500, 0; scale = 1240)
grid = ORCATripolarGrid(arch; dataset = ORCA2(), z, Nz, remove_closed_basins = true)# ORCATripolarGrid

# ### Closures
#
# We include a Gent-McWilliams isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization.

@inline νhb(i, j, k, grid, λ) = Oceananigans.Operators.Az(i, j, k, grid, Center(), Center(), Center())^2 / λ
νh = CenterField(grid)
set!(νh, KernelFunctionOperation{Center, Center, Center}(νhb, grid, 40days))
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νh) 

@inline henyey_diffusivity(x, y, z) = max(2e-6, 3e-5 * abs(sind(y)))
ν_henyey = CenterField(grid)
set!(ν_henyey, henyey_diffusivity)
vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=ν_henyey, ν=3e-5)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=500, κ_symmetric=500)

vertical_mixing = NumericalEarth.Oceans.default_ocean_closure()

free_surface       = SplitExplicitFreeSurface(grid; substeps=20*3)
momentum_advection = VectorInvariant()
tracer_advection   = WENO(order=5)

dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 11, 1)
mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90), z=(-100, 0))
salinity = Metadata(:salinity;  dates, dataset=ECCO4DarwinMonthly(), dir = data_path)
rate = 1/6days
FS = DatasetRestoring(salinity, grid; mask, rate)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(eddy_closure, vertical_mixing, horizontal_viscosity, vertical_diffusivity),
                         forcing = (; S = FS))

sea_ice = sea_ice_simulation(grid, ocean; dynamics = nothing)

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
dataset = ECCO4Monthly()
ecco_temperature           = Metadatum(:temperature; date, dataset, dir = data_path)
ecco_salinity              = Metadatum(:salinity; date, dataset, dir = data_path)
ecco_sea_ice_thickness     = Metadatum(:sea_ice_thickness; date, dataset, dir = data_path)
ecco_sea_ice_concentration = Metadatum(:sea_ice_concentration; date, dataset, dir = data_path)

set!(ocean.model, T=ecco_temperature, S=ecco_salinity)
set!(sea_ice.model, h=ecco_sea_ice_thickness, ℵ=ecco_sea_ice_concentration)

# ### Atmospheric forcing

# We force the simulation with a JRA55-do atmospheric reanalysis.
radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(2920),
                                       include_rivers_and_icebergs = true,
                                       dir = data_path)

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# With Runge-Kutta 3rd order time-stepping we can safely use a timestep of 20 minutes.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=60minutes, stop_time=10*365days)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    e = ocean.model.tracers.e
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg4 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg5 = @sprintf(", wall time: %s", prettytime(step_time))
    msg6 = @sprintf(", Δt = %s, %.1f SYPD\n", prettytime(sim.Δt), Units.day / (step_time * 365 / 10))

    @info msg1 * msg2 * msg4 * msg5 * msg6

    wall_time[] = time_ns()

    return nothing
end

# And add it as a callback to the simulation.
add_callback!(simulation, progress, TimeInterval(10days))

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration,
                         T = sea_ice.model.ice_thermodynamics.top_surface_temperature))

u, v, w = ocean.model.velocities

diagnostics = (drake_transport = Field(Integral(view(u, 108, 20:33, :), dims = (2, 3))),
               mean_T = Field(Average(ocean.model.tracers.T)),
               mean_S = Field(Average(ocean.model.tracers.S)),
               mean_SSH = Field(Average(ocean.model.free_surface.displacement)),
               total_TKE = Field(Integral(ocean.model.tracers.e)))

vertical_slices = 
    (atlantic = view(ocean.model.tracers.T, 125, :, :),
     pacific = view(ocean.model.tracers.T, 66, :, :))

fname_suffix = "orca2"

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(365days/12),
                                            filename = "surface_"*fname_suffix,
                                            indices = (:, :, grid.Nz))

ocean.output_writers[:diagnostics] = JLD2Writer(ocean.model, diagnostics;
                                                schedule = TimeInterval(365days/12),
                                                filename = "diagnostics_"*fname_suffix)

ocean.output_writers[:vslices] = JLD2Writer(ocean.model, vertical_slices;
                                            schedule = TimeInterval(365days/12),
                                            filename = "basin_slice_"*fname_suffix)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              schedule = TimeInterval(365days/12),
                                              filename = "sea_ice_"*fname_suffix)

# ### Ready to run

# We are ready to press the big red button and run the simulation.
run!(simulation)

using CairoMakie, GeoMakie

fds_surface     = FieldDataset("surface_"*fname_suffix*".jld2", backend = OnDisk())
fds_ice         = FieldDataset("sea_ice_"*fname_suffix*".jld2", backend = OnDisk())
fds_diagnostics = FieldDataset("diagnostics_"*fname_suffix*".jld2")
fds_slice       = FieldDataset("basin_slice_"*fname_suffix*".jld2")

grid = fds_surface["T"].grid

land = interior(grid.immersed_boundary.bottom_height) .≥ 0

Nt = length(fds_surface["u"])

uoₙ = Field{Face, Center, Nothing}(grid)
voₙ = Field{Center, Face, Nothing}(grid)

uiₙ = Field{Face, Center, Nothing}(grid)
viₙ = Field{Center, Face, Nothing}(grid)

so = Field(sqrt(uoₙ^2 + voₙ^2))
si = Field(sqrt(uiₙ^2 + viₙ^2))

n = Observable(Nt)

spd_plt = @lift begin
    parent(uoₙ) .= parent(fds_surface["u"][$n])
    parent(voₙ) .= parent(fds_surface["v"][$n])
    compute!(so)
    soₙ = interior(so)
    soₙ[land] .= NaN
    [view(soₙ, :, :, 1)...]
end

T_plt = @lift begin
    T = interior(fds_surface["T"][$n])
    T[land] .= NaN
    [view(T, :, :, 1)...]
end

ice_plt = @lift begin
    hₙ = interior(fds_ice["h"][$n])
    ℵₙ = interior(fds_ice["ℵ"][$n])
    he = hₙ .* ℵₙ
    he[he .< 1e-7] .= NaN
    he[land] .= NaN
    [view(he, :, :, 1)...]
end

ice_T_plt = @lift begin
    T  = interior(fds_ice["T"][$n])
    h  = interior(fds_ice["h"][$n])
    ℵ  = interior(fds_ice["ℵ"][$n])
    he = h .* ℵ
    T[he .< 1e-7] .= NaN
    T[land] .= NaN
    [view(T, :, :, 1)...]
end

times = fds_diagnostics["drake_transport"].times

title = @lift string(floor(times[$n]./365days))*" years, "*string(mod(times[$n]/(365days/12), 12))*" months"

λ, φ = nodes(so)

fig = Figure(size=(1200, 1200))

ax = GeoAxis(fig[1:2, 1:4]; dest="+proj=moll", limits = (-180, 180, -80, 80), title)
ax2 = GeoAxis(fig[3:4, 1:4]; dest="+proj=moll", limits = (-180, 180, -80, 80))

ax3 = GeoAxis(fig[1, 5:6]; dest="+proj=stere +lat_0=90", limits = (-180, 180, 40, 90))
ax4 = GeoAxis(fig[2, 5:6]; dest="+proj=stere +lat_0=-90", limits = (-180, 180, -90, -40))

ax5 = GeoAxis(fig[3, 5:6]; dest="+proj=stere +lat_0=90", limits = (-180, 180, 40, 90))
ax6 = GeoAxis(fig[4, 5:6]; dest="+proj=stere +lat_0=-90", limits = (-180, 180, -90, -40))

ax7 = Axis(fig[5, 1:3], title = "Atlantic", ylabel = "Depth (m)", xlabel = "Latitude (°)")
ax8 = Axis(fig[5, 4:6], title = "Pacific", ylabel = "Depth (m)", xlabel = "Latitude (°)")

markersize = [[(8*Δx[i, j]/maximum(Δx), 5*Δy[i, j]/maximum(Δy)) for i in 1:Nx, j in 1:Ny]...];

sco = scatter!(ax, [λ...], [φ...], color=spd_plt, colormap=:deep, nan_color=:lightgray; markersize, colorrange = (0, 0.5), marker = :rect)
sci = scatter!(ax, [λ...], [φ...], color=ice_plt, colormap=:greys; markersize, colorrange = (0, 4), marker = :rect)

scatter!(ax3, [λ...], [φ...], color=spd_plt, colormap=:deep, nan_color=:lightgray, markersize = 3, colorrange = (0, 0.5), marker = :rect)
scatter!(ax3, [λ...], [φ...], color=ice_plt, colormap=:greys; markersize, colorrange = (0, 4), marker = :rect)

scatter!(ax4, [λ...], [φ...], color=spd_plt, colormap=:deep, nan_color=:lightgray, markersize = 3, colorrange = (0, 0.5), marker = :rect)
scatter!(ax4, [λ...], [φ...], color=ice_plt, colormap=:greys; markersize, colorrange = (0, 4), marker = :rect)

Colorbar(fig[1:2, 0], sco, label = "Ocean Surface Speed (m/s)")
Colorbar(fig[1:2, 7], sci, label = "Sea Ice Effective Thickness (m)")

scT = scatter!(ax2, [λ...], [φ...], color=T_plt, colormap=:magma; markersize, colorrange = (-1, 32), marker = :rect)
scatter!(ax5, [λ...], [φ...], color=T_plt, colormap=:magma; markersize, colorrange = (-1, 32), marker = :rect)
scatter!(ax6, [λ...], [φ...], color=T_plt, colormap=:magma; markersize, colorrange = (-1, 32), marker = :rect)
scTi = scatter!(ax2, [λ...], [φ...], color=ice_T_plt, colormap=:vik; markersize, colorrange = (-2, 0), marker = :rect)
scatter!(ax5, [λ...], [φ...], color=ice_T_plt, colormap=:vik; markersize, colorrange = (-2, 0), marker = :rect)
scatter!(ax6, [λ...], [φ...], color=ice_T_plt, colormap=:vik; markersize, colorrange = (-2, 0), marker = :rect)

title = @lift string(floor(times[$n]./365days))*" years, "*string(mod(times[$n]/(365days/12), 12))*" months"
Colorbar(fig[3:4, 0], scT, label = "Sea Surface Temperature (°C)")
Colorbar(fig[3:4, 7], scTi, label = "Ice Surface Temperature (°C)")

λ, φ, z = nodes(grid, Center(), Center(), Center())

heatmap!(ax7, φ[125, :], z, (@lift interior(fds_slice["atlantic"][$n], 1, :, :)), colormap=:magma, colorrange = (-1, 32), nan_color=:lightgray)
heatmap!(ax8, φ[66, :], z, (@lift interior(fds_slice["pacific"][$n], 1, :, :)), colormap=:magma, colorrange = (-1, 32), nan_color=:lightgray)

#=record(fig, "orca2.mp4", 1:Nt, framerate = 10) do i
    n[] = i
end=#
save("orca2_snapshot.png", fig)
nothing #hide

# ![](orca2_snapshot.png)

fig = Figure()

ax = Axis(fig[1, 1], title = "Drake transport (Sverdrops)")
ax2 = Axis(fig[2, 1], ylabel = "Temperature (°C)", title = "Global Mean")
ax3 = Axis(fig[2, 1], yaxisposition = :right, ylabel = "Salinity (psu)")
ax4 = Axis(fig[3, 1], ylabel = "TKE (m²/s)")
ax5 = Axis(fig[3, 1], yaxisposition = :right, ylabel = "SSH (m)", xlabel = "Year")
hidexdecorations!(ax3)
hidexdecorations!(ax5)

lines!(ax, times./(365days), interior(fds_diagnostics["drake_transport"], 1, 1, 1, :)./10^6)
scatter!(ax, (@lift [times[$n]/(365days)]), (@lift [interior(fds_diagnostics["drake_transport"], 1, 1, 1, $n)[]/10^6]))
ylims!(ax, 120, 180)

lines!(ax2, times./(365days), interior(fds_diagnostics["mean_T"], 1, 1, 1, :))
scatter!(ax2, (@lift [times[$n]/(365days)]), (@lift [interior(fds_diagnostics["mean_T"], 1, 1, 1, $n)[]]))
lines!(ax3, times./(365days), interior(fds_diagnostics["mean_S"], 1, 1, 1, :), color = Makie.wong_colors()[2])
scatter!(ax3, (@lift [times[$n]/(365days)]), (@lift [interior(fds_diagnostics["mean_S"], 1, 1, 1, $n)[]]), color = Makie.wong_colors()[2])

lines!(ax4, times./(365days), interior(fds_diagnostics["total_TKE"], 1, 1, 1, :))
scatter!(ax4, (@lift [times[$n]/(365days)]), (@lift [interior(fds_diagnostics["total_TKE"], 1, 1, 1, $n)[]]))
lines!(ax5, times./(365days), interior(fds_diagnostics["mean_SSH"], 1, 1, 1, :), color = Makie.wong_colors()[2])
scatter!(ax5, (@lift [times[$n]/(365days)]), (@lift [interior(fds_diagnostics["mean_SSH"], 1, 1, 1, $n)[]]), color = Makie.wong_colors()[2])

save("orca2_diagnostics.png", fig)
nothing #hide

# ![](orca2_diagnostics.png)