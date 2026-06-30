# # [Coarse global ocean--sea ice simulation](@id coarse-degree-ocean-seaice)
#
# This example configures a global ocean--sea ice simulation on the ["ORCA2" grid](https://www.nemo-ocean.eu/doc/node108.html#Tab_orca_zgr)
# which is tripolar grid which transitions to a Mercator grid in the tropics to better resolve 
# equatorial dynamics (as the sign of the coriolis coefficient switches on the equator).
# The grid nominally has a 2° resolution with meridional refinement to 0.5° in the tropics 
# and near Antarctica, and refinement to ~0.5° in the Mediterranean, Red, Black and Caspian Seas.
# The grid has been refined carefully designed by NEMO to carefully capture important physics
# and maintain anisotropy close to 1 in the ocean, especially in strongly eddying regions such 
# as the gulf stream. They also provide bathymetry which is refined to represent important straits.
#
# The model is forced with by repeat-year JRA55 atmospheric reanalysis and initialized by
# temperature, salinity, sea ice concentration, and sea ice thickness from the ECCO state estimate. 
# It includes a few closures:
# - "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`,
# - `CATKEVerticalDiffusivity` for vertical convective mixing,
# - `HorizontalScalarBiharmonicDiffusivity` to damp grid scale noise,
# - and `VerticalScalarDiffusivity` emulating mixing from internal tides
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
arch = GPU()
Nz = 30
z = ExponentialDiscretization(Nz, -5500, 0; scale = 1240)
grid = ORCATripolarGrid(arch; dataset = ORCA2(), z, Nz, remove_closed_basins = true, halo = (5, 5, 4))

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

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity

vertical_mixing = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(); 
                                           maximum_tracer_diffusivity = 1,
                                           maximum_tke_diffusivity = 1,
                                           maximum_viscosity = 1)

free_surface       = SplitExplicitFreeSurface(grid; substeps=60)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)

@inline restoring_mask(x, y, z, t) = z>-50
rate = 1/50days
dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 11, 1)
salinity = Metadata(:salinity;  dates, dataset=ECCO4DarwinMonthly(), dir = data_path)
FS = DatasetRestoring(salinity, grid; mask = restoring_mask, rate)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(eddy_closure, vertical_mixing, horizontal_viscosity, vertical_diffusivity),
                         forcing = (; S = FS))

sea_ice = sea_ice_simulation(grid, ocean; dynamics = nothing)

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
dataset = ECCO4Monthly()
ecco_temperature           = Metadatum(:temperature; date, dataset)
ecco_salinity              = Metadatum(:salinity; date, dataset)
ecco_sea_ice_thickness     = Metadatum(:sea_ice_thickness; date, dataset)
ecco_sea_ice_concentration = Metadatum(:sea_ice_concentration; date, dataset)

set!(ocean.model, T=ecco_temperature, S=ecco_salinity)
set!(sea_ice.model, h=ecco_sea_ice_thickness, ℵ=ecco_sea_ice_concentration)

# ### Atmospheric forcing

# We force the simulation with a JRA55-do atmospheric reanalysis.
radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(),
                                       include_rivers_and_icebergs = true)

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# With Runge-Kutta 3rd order time-stepping we can safely use a timestep of 90 minutes.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=90minutes, stop_time=2.5*365days)

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

# ### Load and plot results

using CairoMakie

fds_surface     = FieldDataset("surface_"*fname_suffix*".jld2", backend = OnDisk())
fds_ice         = FieldDataset("sea_ice_"*fname_suffix*".jld2", backend = OnDisk())
fds_diagnostics = FieldDataset("diagnostics_"*fname_suffix*".jld2")
fds_slice       = FieldDataset("basin_slice_"*fname_suffix*".jld2")

grid = fds_surface["T"].grid

land = interior(grid.immersed_boundary.bottom_height) .≥ 0

Nt = length(fds_surface["u"])

uoₙ = Field{Face, Center, Nothing}(grid)
voₙ = Field{Center, Face, Nothing}(grid)

so = Field(sqrt(uoₙ^2 + voₙ^2))

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

title = @lift string(floor(times[$n]./365days))*" years, "*string(floor(Int, mod(times[$n]/(365days/12), 12)))*" months"

λ, φ = nodes(so)

Nx, Ny = size(grid)
Δx = grid.Δxᶜᶜᵃ[1:Nx, 1:Ny]
Δy = grid.Δyᶜᶜᵃ[1:Nx, 1:Ny]
markersize = [[(8*Δx[i, j]/maximum(Δx)./cosd(φ[i, j]), 8*Δy[i, j]/maximum(Δy)) for i in 1:Nx, j in 1:Ny]...];
polar_markersize = [[(12*Δx[i, j]/maximum(Δx), 12*Δy[i, j]/maximum(Δy) .* sind(90-φ[i, j])) for i in 1:Nx, j in 1:Ny]...];

fig = Figure(size=(1200, 1200))

ax = Axis(fig[1:2, 1:4], limits = (-180, 180, -80, 80), backgroundcolor = :lightgray, xgridvisible = false, ygridvisible = false; title)
ax2 = Axis(fig[3:4, 1:4], limits = (-180, 180, -80, 80), backgroundcolor = :lightgray, xgridvisible = false, ygridvisible = false)

ax3 = PolarAxis(fig[1, 5:6], rgridvisible = false, thetagridvisible = false)
ax4 = PolarAxis(fig[2, 5:6], rgridvisible = false, thetagridvisible = false)

ax5 = PolarAxis(fig[3, 5:6], rgridvisible = false, thetagridvisible = false)
ax6 = PolarAxis(fig[4, 5:6], rgridvisible = false, thetagridvisible = false)

ax7 = Axis(fig[5, 1:3], title = "Atlantic", ylabel = "Depth (m)", xlabel = "Latitude (°)")
ax8 = Axis(fig[5, 4:6], title = "Pacific", ylabel = "Depth (m)", xlabel = "Latitude (°)")

sco = scatter!(ax, [λ...], [φ...], color=spd_plt, colormap=:deep, nan_color=:lightgray; markersize, colorrange = (0, 0.5), marker = :rect)
sci = scatter!(ax, [λ...], [φ...], color=ice_plt, colormap=:greys; markersize, colorrange = (0, 4), marker = :rect)

scatter!(ax3, π/180 .* [λ...], 90 .-[φ...], color=spd_plt, colormap=:deep, nan_color=:lightgray, markersize = polar_markersize, colorrange = (0, 0.5), marker = :rect, rotation = π/180 .* [λ...])
scatter!(ax3, π/180 .* [λ...], 90 .-[φ...], color=ice_plt, colormap=:greys; markersize = polar_markersize, colorrange = (0, 4), marker = :rect, rotation = π/180 .* [λ...])

scatter!(ax4, π/180 .* [λ...], [φ...] .+ 90, color=spd_plt, colormap=:deep, nan_color=:lightgray, markersize = polar_markersize, colorrange = (0, 0.5), marker = :rect, rotation = π/180 .* [λ...])
scatter!(ax4, π/180 .* [λ...], [φ...] .+ 90, color=ice_plt, colormap=:greys; markersize = polar_markersize, colorrange = (0, 4), marker = :rect, rotation = π/180 .* [λ...])

Colorbar(fig[1:2, 0], sco, label = "Ocean Surface Speed (m/s)")
Colorbar(fig[1:2, 7], sci, label = "Sea Ice Effective Thickness (m)")

lo, hi = -50, 0
center = -1
frac = (center - lo) / (hi - lo)
compressed_roma = cgrad(
    :roma,
    [0.0, frac, 1.0], 
    categorical = false,
    rev = true
)

scT = scatter!(ax2, [λ...], [φ...], color=T_plt, colormap=:magma; markersize, colorrange = (-1, 32), marker = :rect, nan_color=:lightgray)
scatter!(ax5, π/180 .* [λ...], 90 .-[φ...], color=T_plt, colormap=:magma; markersize = polar_markersize, colorrange = (-1, 32), marker = :rect, nan_color=:lightgray)
scatter!(ax6, π/180 .* [λ...], [φ...] .+ 90, color=T_plt, colormap=:magma; markersize = polar_markersize, colorrange = (-1, 32), marker = :rect, nan_color=:lightgray)
scTi = scatter!(ax2, [λ...], [φ...], color=ice_T_plt, colormap=compressed_roma; markersize, colorrange = (-50, 0), marker = :rect)
scatter!(ax5, π/180 .* [λ...], 90 .-[φ...], color=ice_T_plt, colormap=compressed_roma; markersize = polar_markersize, colorrange = (-50, 0), marker = :rect)
scatter!(ax6, π/180 .* [λ...], [φ...] .+ 90, color=ice_T_plt, colormap=compressed_roma; markersize = polar_markersize, colorrange = (-50, 0), marker = :rect)

title = @lift string(floor(times[$n]./365days))*" years, "*string(mod(times[$n]/(365days/12), 12))*" months"
Colorbar(fig[3:4, 0], scT, label = "Sea Surface Temperature (°C)")
Colorbar(fig[3:4, 7], scTi, label = "Ice Surface Temperature (°C)")

λ, φ, z = nodes(grid, Center(), Center(), Center())

heatmap!(ax7, φ[125, :], z, (@lift interior(fds_slice["atlantic"][$n], 1, :, :)), colormap=:magma, colorrange = (-1, 32), nan_color=:lightgray)
heatmap!(ax8, φ[66, :], z, (@lift interior(fds_slice["pacific"][$n], 1, :, :)), colormap=:magma, colorrange = (-1, 32), nan_color=:lightgray)

rlims!(ax3, 0, 40); hiderdecorations!(ax3)
rlims!(ax4, 0, 40); hiderdecorations!(ax4)
rlims!(ax5, 0, 40); hiderdecorations!(ax5)
rlims!(ax6, 0, 40); hiderdecorations!(ax6)

record(fig, "orca2.mp4", 1:Nt, framerate = 10) do i
    n[] = i
end

nothing #hide

# ![](orca2.mp4)

fig = Figure()

ax = Axis(fig[1, 1], title = "Drake transport (Sverdrops)")
ax2 = Axis(fig[2, 1], ylabel = "Temperature (°C)", title = "Global Mean")
ax3 = Axis(fig[2, 1], yaxisposition = :right, ylabel = "Salinity (psu)")
ax4 = Axis(fig[3, 1], ylabel = "TKE (m²/s)")
ax5 = Axis(fig[3, 1], yaxisposition = :right, ylabel = "SSH (m)", xlabel = "Year")
hidexdecorations!(ax3)
hidexdecorations!(ax5)

lines!(ax, times./(365days), interior(fds_diagnostics["drake_transport"], 1, 1, 1, :)./10^6)
ylims!(ax, 120, 180)

lines!(ax2, times./(365days), interior(fds_diagnostics["mean_T"], 1, 1, 1, :))
lines!(ax3, times./(365days), interior(fds_diagnostics["mean_S"], 1, 1, 1, :), color = Makie.wong_colors()[2])
lines!(ax4, times./(365days), interior(fds_diagnostics["total_TKE"], 1, 1, 1, :))
lines!(ax5, times./(365days), interior(fds_diagnostics["mean_SSH"], 1, 1, 1, :), color = Makie.wong_colors()[2])

save("orca2_diagnostics.png", fig)

nothing #hide

# ![](orca2_diagnostics.png)
