# # Global climate simulation
#
# This example configures a global ocean--sea ice simulation at 1ᵒ horizontal resolution with
# realistic bathymetry and a few closures including the "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`.
# The atmosphere is represented by a 4-layer [SpeedyWeather](https://github.com/SpeedyWeather/SpeedyWeather.jl)
# simulation on the T63 spectral grid (this grid has approximately 1.875ᵒ resolution).
#
# The atmosphere is initialized with the [jablonowski2006baroclinic](@citet) initial conditions,
# which consist of a zonal wind centered at mid-latitudes and higher altitudes and a temperature
# profile that is baroclinically unstable. The surface pressure is adjusted by topography for
# approximately globally constant mean-sea level pressure. The initial specific humidity is
# calculated from temperature for a constant relative humidity everywhere.
# The ocean and sea ice are initialized by ocean temperature, salinity, sea ice concentration,
# and sea ice thickness from the ECCO state estimate.
#
# The example couples three models: `Oceananigans.HydrostaticFreeSurfaceModel` (the ocean),
# `ClimaSeaIce.SeaIceModel` (the sea ice), and `SpeedyWeather.PrimitiveWetModel` (the atmosphere).
# `NumericalEarth.EarthSystemModel` orchestrates the coupled system.
#
# ConservativeRegridding.jl regrids fields between the atmosphere and the ocean--sea ice components.
# All components run on a CUDA-enabled GPU.

using Oceananigans, SpeedyWeather, NumericalEarth, ConservativeRegridding
using CUDA
using NCDatasets, CairoMakie
using Oceananigans.Units
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: φnode
using Printf, Statistics, Dates

# ## Ocean and sea-ice model configuration
# The ocean and sea ice components are simplified versions of those in the
# [one-degree ocean--sea ice example](@ref one-degree-ocean-seaice).
#
# The first step is to create the grid.

Nx = 360
Ny = 180
Nz = 60
z = ExponentialDiscretization(Nz, -6000, 0; mutable=true)
grid = TripolarGrid(Oceananigans.GPU(); size=(Nx, Ny, Nz), z, halo=(5, 5, 4))
nothing #hide

# Next, we regrid the bathymetry and build an immersed boundary grid.

bottom_height = regrid_bathymetry(grid; minimum_depth=10, major_basins=2, interpolation_passes=10)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
nothing #hide

# Now we specify the numerical schemes and the closures for the ocean simulation.
# We use a biharmonic horizontal viscosity with a damping timescale of 15 days and a
# latitude-dependent background vertical diffusivity following [Henyey1986](@citet).

momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)
free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
catke_closure      = NumericalEarth.Oceans.default_ocean_closure()
eddy_closure       = Oceananigans.TurbulenceClosures.IsopycnalSkewSymmetricDiffusivity(κ_skew=500, κ_symmetric=200)

@inline νhb(i, j, k, grid, timescale) = Oceananigans.Operators.Azᶜᶜᶜ(i, j, k, grid)^2 / timescale
ν = CenterField(grid)
Oceananigans.set!(ν, KernelFunctionOperation{Center, Center, Center}(νhb, grid, 15days))
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(; ν)

νz = 1e-5
## κz(φ) = max(2e-6, 3e-5 * |sin(φ)|)
@inline henyey_diffusivity(i, j, k, grid) = max(2e-6, 3e-5 * abs(sind(φnode(i, j, k, grid, Center(), Center(), Center()))))
κz = CenterField(grid)
Oceananigans.set!(κz, KernelFunctionOperation{Center, Center, Center}(henyey_diffusivity, grid))
vertical_diffusivity = VerticalScalarDiffusivity(ν=νz, κ=κz)

closures = (catke_closure, eddy_closure, horizontal_viscosity, vertical_diffusivity)
nothing #hide

# We build the ocean simulation and initialize its temperature and salinity from ECCO on January 1, 1992.

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         closure = closures)

ecco_set = MetadataSet(:temperature, :salinity,
                       :sea_ice_thickness, :sea_ice_concentration;
                       dataset = ECCO4Monthly(),
                       date = DateTime(1992, 1, 1))
Oceananigans.set!(ocean.model, ecco_set)   # T, S

# The sea ice simulation is initialized with sea ice thickness and concentration from ECCO.

sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=5))

Oceananigans.set!(sea_ice.model, ecco_set)   # h, ℵ

# ## Atmosphere model configuration
# SpeedyWeather.jl provides the atmosphere. Here, we configure a T63L4 model with a 3-hour output interval.
# `atmosphere_simulation` builds an atmosphere model with the hooks that NumericalEarth needs
# to compute intercomponent fluxes.

nlayers = 4
output_interval = 3hours
stop_time = 30days
spectral_grid = SpeedyWeather.SpectralGrid(; NF=Float64, truncation=64, nlayers, Grid=FullClenshawGrid, architecture=SpeedyWeather.GPU())
time_stepping = SpeedyWeather.Leapfrog(spectral_grid; Δt_at_T32=Minute(40)) # gives Δt = 20 min at truncation=64
atmosphere = atmosphere_simulation(spectral_grid; output_interval, time_stepping, stop_time)
atmosphere.model.feedback.verbose = false  # disable SpeedyWeather's progress bar in favor of the callback defined below
nothing #hide

# The atmosphere model comes with the initial conditions described above:

atmosphere.model.initial_conditions

# ## The coupled model
# We are now ready to blend everything together.
# All components share the atmosphere's time step.

Δt = convert(eltype(grid), atmosphere.model.time_stepping.Δt)
nothing #hide

# We build the coupled `earth_model`. NumericalEarth computes the turbulent (sensible and latent heat)
# fluxes with its own bulk formulae and passes them back to SpeedyWeather.
# The coupled model has no radiation component, since radiation is not yet coupled
# to SpeedyWeather's upwelling longwave radiation.

earth_model = EarthSystemModel(; atmosphere, sea_ice, ocean)

# ## Building and running the simulation
#
# We build the coupled simulation and attach output writers that save to disk every 3 hours.

earth = Oceananigans.Simulation(earth_model; Δt, stop_time)
outputs = merge(ocean.model.velocities, ocean.model.tracers)
sea_ice_fields = merge(sea_ice.model.velocities, sea_ice.model.dynamics.auxiliaries.fields,
                       (; h=sea_ice.model.ice_thickness, ℵ=sea_ice.model.ice_concentration))

ocean.output_writers[:free_surf] = JLD2Writer(ocean.model, (; η=ocean.model.free_surface.displacement);
                                              overwrite_files=true,
                                              schedule=TimeInterval(output_interval),
                                              filename="ocean_free_surface.jld2")

ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                            overwrite_files=true,
                                            schedule=TimeInterval(output_interval),
                                            filename="ocean_surface_fields.jld2",
                                            indices=(:, :, grid.Nz))

sea_ice.output_writers[:fields] = JLD2Writer(sea_ice.model, sea_ice_fields;
                                             overwrite_files=true,
                                             schedule=TimeInterval(output_interval),
                                             filename="sea_ice_fields.jld2")

𝒬ᵀᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
𝒬ᵛᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
τˣᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.x_momentum
τʸᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.y_momentum
𝒬ᵀᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.sensible_heat
𝒬ᵛᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.latent_heat
τˣᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.x_momentum
τʸᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.y_momentum
𝒬ⁱᵒ  = earth.model.interfaces.net_fluxes.sea_ice.bottom.heat
Jˢⁱᵒ = earth.model.interfaces.sea_ice_ocean_interface.fluxes.salt
fluxes = (; 𝒬ᵀᵃᵒ, 𝒬ᵛᵃᵒ, τˣᵃᵒ, τʸᵃᵒ, 𝒬ᵀᵃⁱ, 𝒬ᵛᵃⁱ, τˣᵃⁱ, τʸᵃⁱ, 𝒬ⁱᵒ, Jˢⁱᵒ)

ocean.output_writers[:fluxes] = JLD2Writer(earth.model.ocean.model, fluxes;
                                           overwrite_files=true,
                                           schedule=TimeInterval(output_interval),
                                           filename="intercomponent_fluxes.jld2")

# We also add a callback that prints a progress message while the simulation runs.

wall_time = Ref(time_ns())
model_time = Ref(0.0)

function progress(sim)
    atmosphere = sim.model.atmosphere
    ocean = sim.model.ocean

    ua, va     = atmosphere.variables.dynamics.u_mean_grid, atmos.variables.dynamics.v_mean_grid
    uo, vo, wo = ocean.model.velocities

    ## RingGrids only defines the one-argument `maximum`, so `abs` must be broadcast first to stay on the GPU
    uamax = (maximum(abs.(ua)), maximum(abs.(va)))
    uomax = (maximum(abs, uo), maximum(abs, vo), maximum(abs, wo))

    step_time = 1e-9 * (time_ns() - wall_time[])
    sypd = (time(sim) - model_time[]) / step_time / 365

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|ua|: (%.1e, %.1e) m s⁻¹", uamax...)
    msg3 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", uomax...)
    msg4 = @sprintf(", wall time: %s", prettytime(step_time))
    msg5 = @sprintf(", SYPD: %.3f \n", sypd)

    @info msg1 * msg2 * msg3 * msg4 * msg5

    wall_time[] = time_ns()
    model_time[] = time(sim)

    return nothing
end

add_callback!(earth, progress, TimeInterval(2days))

# Let's run the coupled model!

Oceananigans.run!(earth)

# ## Visualizing the results
#
# We plot the surface speeds in the atmosphere, ocean, and sea ice, as well as the
# atmospheric temperature in the lowest model layer, the sea surface temperature, and the
# sensible and latent heat fluxes at the atmosphere--ocean interface.
# SpeedyWeather writes its output to a NetCDF file in the `run_0001` folder,
# while the ocean and sea ice outputs are JLD2 files that we load as `FieldTimeSeries`.

atmosphere_output = Dataset("run_0001/output.nc")

Ta = reverse(atmosphere_output["temp"][:, :, nlayers, :], dims=2)
ua = reverse(atmosphere_output["u"][:, :, nlayers, :],    dims=2)
va = reverse(atmosphere_output["v"][:, :, nlayers, :],    dims=2)
sa = @. sqrt(ua^2 + va^2)

To_ts = FieldTimeSeries("ocean_surface_fields.jld2", "T")
uo_ts = FieldTimeSeries("ocean_surface_fields.jld2", "u")
vo_ts = FieldTimeSeries("ocean_surface_fields.jld2", "v")

ui_ts = FieldTimeSeries("sea_ice_fields.jld2", "u")
vi_ts = FieldTimeSeries("sea_ice_fields.jld2", "v")
ℵi_ts = FieldTimeSeries("sea_ice_fields.jld2", "ℵ")

𝒬ᵀᵃᵒ_ts = FieldTimeSeries("intercomponent_fluxes.jld2", "𝒬ᵀᵃᵒ")
𝒬ᵛᵃᵒ_ts = FieldTimeSeries("intercomponent_fluxes.jld2", "𝒬ᵛᵃᵒ")

times = 𝒬ᵀᵃᵒ_ts.times
Nt = min(size(sa, 3), length(times))

# The ocean speed and the concentration-weighted sea ice speed are fields computed
# from snapshot fields that we `set!` at every frame.

uoₙ = uo_ts[1]
voₙ = vo_ts[1]
uiₙ = ui_ts[1]
viₙ = vi_ts[1]
ℵiₙ = ℵi_ts[1]

so = Oceananigans.Field(sqrt(uoₙ^2 + voₙ^2))
si = Oceananigans.Field(sqrt(uiₙ^2 + viₙ^2) * ℵiₙ)

n = Observable(1)

saₙ = @lift sa[:, :, $n]

soₙ = @lift begin
    Oceananigans.set!(uoₙ, uo_ts[$n])
    Oceananigans.set!(voₙ, vo_ts[$n])
    so
end

siₙ = @lift begin
    Oceananigans.set!(uiₙ, ui_ts[$n])
    Oceananigans.set!(viₙ, vi_ts[$n])
    Oceananigans.set!(ℵiₙ, ℵi_ts[$n])
    si
end

fig = Figure(size = (666, 1000))

ax1 = Axis(fig[1, 1], title = "Surface speed, atmosphere")
ax2 = Axis(fig[2, 1], title = "Surface speed, ocean")
ax3 = Axis(fig[3, 1], title = "Surface speed, sea ice")

hm1 = heatmap!(ax1, saₙ; colormap = :deep,  nan_color = :lightgray, colorrange = (0, 35))
hm2 = heatmap!(ax2, soₙ; colormap = :magma, nan_color = :lightgray, colorrange = (0, 0.6))
hm3 = heatmap!(ax3, siₙ; colormap = :ice,   nan_color = :lightgray, colorrange = (0, 0.6))

Colorbar(fig[1, 2], hm1, label = "(m s⁻¹)")
Colorbar(fig[2, 2], hm2, label = "(m s⁻¹)")
Colorbar(fig[3, 2], hm3, label = "(m s⁻¹)")

for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
end

title = @lift prettytime(times[$n] - times[1])
Label(fig[0, :], title, fontsize = 18)

CairoMakie.record(fig, "surface_speeds.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](surface_speeds.mp4)

Taₙ = @lift Ta[:, :, $n]
Toₙ = @lift To_ts[$n]
𝒬ᵀᵃᵒₙ = @lift 𝒬ᵀᵃᵒ_ts[$n]
𝒬ᵛᵃᵒₙ = @lift 𝒬ᵛᵃᵒ_ts[$n]

fig = Figure(size = (700, 1400))

ax1 = Axis(fig[1, 1], title = "Lowest-layer air temperature")
ax2 = Axis(fig[2, 1], title = "Sea surface temperature")
ax3 = Axis(fig[3, 1], title = "Sensible heat flux")
ax4 = Axis(fig[4, 1], title = "Latent heat flux")

hm1 = heatmap!(ax1, Taₙ;   colormap = :plasma,  nan_color = :lightgray, colorrange = (-45, 30))
hm2 = heatmap!(ax2, Toₙ;   colormap = :plasma,  nan_color = :lightgray, colorrange = (-2, 32))
hm3 = heatmap!(ax3, 𝒬ᵀᵃᵒₙ; colormap = :balance, nan_color = :lightgray, colorrange = (-200, 200))
hm4 = heatmap!(ax4, 𝒬ᵛᵃᵒₙ; colormap = :balance, nan_color = :lightgray, colorrange = (-200, 200))

Colorbar(fig[1, 2], hm1, label = "(ᵒC)")
Colorbar(fig[2, 2], hm2, label = "(ᵒC)")
Colorbar(fig[3, 2], hm3, label = "(W m⁻²)")
Colorbar(fig[4, 2], hm4, label = "(W m⁻²)")

for ax in (ax1, ax2, ax3, ax4)
    hidedecorations!(ax)
end

Label(fig[0, :], title, fontsize = 18)

CairoMakie.record(fig, "surface_temperature_and_heat_flux.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](surface_temperature_and_heat_flux.mp4)
