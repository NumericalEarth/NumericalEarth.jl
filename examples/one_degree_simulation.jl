# # [One-degree global ocean--sea ice simulation](@id one-degree-ocean-seaice)
#
# This example configures a global ocean--sea ice simulation on the 1ᵒ NEMO eORCA1 grid, in an OMIP-like
# configuration: 100 vertical levels, CATKE vertical mixing over a uniform background diffusivity,
# Gent-McWilliams and Redi isopycnal mixing discretized on Griffies triads, and a biharmonic viscosity.
# The simulation is forced by repeat-year JRA55 atmospheric reanalysis, its surface salinity is restored to the
# World Ocean Atlas monthly climatology, and it is initialized by World Ocean Atlas temperature and salinity and
# by ECCO sea ice concentration and thickness.
#
# For this example, we need Oceananigans, NumericalEarth, ClimaSeaIce, WorldOceanAtlasTools, Dates, CUDA, and
# CairoMakie to visualize the simulation.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using ClimaSeaIce: IncrementalRemapping, ConductiveFlux, UniformThicknessDistribution
using WorldOceanAtlasTools
using Dates
using Printf
using Statistics
using CUDA

# ### Grid and Bathymetry
#
# The horizontal grid and the bathymetry come from the NEMO eORCA1 mesh, of which we keep only the largest
# connected basin. In the vertical we use 100 levels whose spacing grows by 7.49% per level from 1.5 m at
# the surface until it reaches 100 m at about 1400 m depth, so that the 100 levels span 5500 m.

arch = GPU()

z = ReferenceToStretchedDiscretization(extent = 5500,
                                       constant_spacing = 1.5,
                                       constant_spacing_extent = 1.5,
                                       maximum_spacing = 100,
                                       stretching = LinearStretching(0.0749),
                                       rounding_digits = 4,
                                       mutable = true)

grid = ORCAGrid(arch; dataset = ORCAOne(), z, halo = (8, 8, 8), major_basins = 1)

# ### Closures
#
# Vertical mixing is CATKE, on top of a uniform background diffusivity κ = 10⁻⁵ m² s⁻¹ and viscosity
# ν = 3 × 10⁻⁵ m² s⁻¹. The mesoscale eddy fluxes are parameterized with a Gent-McWilliams skew diffusivity
# of 500 m² s⁻¹ and a Redi isopycnal diffusivity of 800 m² s⁻¹, both discretized with Griffies triads.
# Finally, a biharmonic viscosity ν = Az² / λ damps grid-scale motions on a timescale λ = 50 days
# independently of the cell area Az.

using Oceananigans.TurbulenceClosures: TriadIsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength
using Oceananigans.Operators: Az

vertical_mixing = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                           mixing_length = CATKEMixingLength(Cᵇ = 0.01),
                                           maximum_viscosity = 3,
                                           maximum_tracer_diffusivity = 3,
                                           maximum_tke_diffusivity = 3,
                                           negative_tke_damping_time_scale = 10)

background_mixing = VerticalScalarDiffusivity(κ = 1e-5, ν = 3e-5)

eddy_closure = TriadIsopycnalSkewSymmetricDiffusivity(κ_skew = 500, κ_symmetric = 800)

@inline biharmonic_viscosity(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) = Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = biharmonic_viscosity,
                                                             discrete_form = true,
                                                             parameters = 50days)

# ### Land
#
# The river and iceberg runoff of JRA55-do is spread over the eight wet cells closest to each river mouth.
# We build the land before the ocean, so that the ocean can enhance the vertical mixing at the river mouths.

land = JRA55PrescribedLand(grid; spread_radius = nothing, maximum_spread_cells = 8)

# ### Ocean simulation
#
# The penetrating shortwave radiation is absorbed in two bands, whose attenuation follows the SeaWiFS
# monthly chlorophyll climatology, cycled every year. The surface salinity is restored to the World Ocean
# Atlas monthly climatology with a piston velocity of 0.254 m day⁻¹, which over the 1.5 m top cell
# corresponds to a restoring rate of 0.254 / 1.5 day⁻¹. The restoring is applied as a surface salt flux
# whose global mean is removed, so that it neither adds nor removes salt.

using NumericalEarth.DataWrangling: SeaWiFSMonthly, SurfaceFluxRestoring, ConservativeSurfaceFluxRestoring,
                                    ConservativeSurfaceFluxRestoringCallback

chlorophyll_dates = (DateTime(2000, 1, 1), DateTime(2000, 12, 1))
chlorophyll = FieldTimeSeries(Metadata(:chlorophyll; dataset = SeaWiFSMonthly(), dates = chlorophyll_dates), grid)
radiative_forcing = TwoColorRadiation(grid; chlorophyll)

salinity_restoring = DatasetRestoring(Metadata(:salinity; dataset = WOAMonthly()), arch;
                                      rate = 0.254 / 1.5days,
                                      time_indices_in_memory = 12)

salinity_flux = ConservativeSurfaceFluxRestoring(SurfaceFluxRestoring(salinity_restoring), grid)

# We use a split-explicit free surface with 300 barotropic substeps, fifth-order WENO for momentum and
# the default seventh-order WENO for tracers. Both treat vertical advection implicitly wherever the
# vertical Courant number is large.

free_surface       = SplitExplicitFreeSurface(grid; substeps = 300)
momentum_advection = WENOVectorInvariant(order = 5, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5))

ocean = ocean_simulation(grid; momentum_advection, free_surface, radiative_forcing, reference_density = 1026,
                         closure = (vertical_mixing, eddy_closure, horizontal_viscosity, background_mixing),
                         river_routing = land.river_routing,
                         additional_surface_fluxes = (; S = salinity_flux))

# ### Sea Ice simulation
#
# The sea ice has EVP rheology, a snow layer over a slab of ice, and is advected by incremental remapping.
# The ice--ocean drag is computed against the ocean velocity averaged over the top 6 m. Heat is conducted
# through snow and ice as if the thickness were spread uniformly over four sub-grid categories, which
# raises the effective conductivity by 1 + 1/3 + 1/5 + 1/7.

dynamics = NumericalEarth.SeaIces.sea_ice_dynamics(grid, ocean; sea_ice_ocean_drag_reference_depth = 6)
internal_heat_flux = ConductiveFlux(conductivity = 2, itd_shape = UniformThicknessDistribution(categories = 4))

sea_ice = sea_ice_simulation(grid, ocean; dynamics, internal_heat_flux, advection = IncrementalRemapping())

# ### Initial condition
#
# We initialize the ocean with the World Ocean Atlas annual climatology and the sea ice with the ECCO
# state estimate of January 1993.

set!(ocean.model, MetadataSet((:temperature, :salinity); dataset = WOAAnnual()))
set!(sea_ice.model, MetadataSet((:sea_ice_thickness, :sea_ice_concentration); dataset = ECCO4Monthly(),
                                date = DateTime(1993, 1, 1)))

# ### JRA55-based atmospheric state and radiation
#
# We force the simulation with the JRA55-do repeat-year atmospheric state and downwelling radiation.
# The ocean has an albedo of 0.06, and the sea ice albedo depends on the ice thickness, the snow thickness,
# and the surface temperature as in CCSM3.

atmosphere = JRA55PrescribedAtmosphere(arch)

sea_ice_albedo = SeaIceAlbedo(sea_ice.model.ice_thickness,
                              sea_ice.model.snow_thickness,
                              sea_ice.model.snow_thermodynamics.top_surface_temperature)

ocean_surface   = SurfaceRadiationProperties(0.06, 1)
sea_ice_surface = SurfaceRadiationProperties(sea_ice_albedo, 1)
radiation = JRA55PrescribedRadiation(arch; ocean_surface, sea_ice_surface)

# ### Turbulent fluxes
#
# Over the ocean, the turbulent fluxes follow COARE 3.6, with a wind-dependent Charnock parameter,
# scalar roughness lengths that depend on the air viscosity, and a minimum gustiness of 0.5 m s⁻¹.
# Over the sea ice the roughness lengths are fixed at 5 × 10⁻⁴ m for momentum and 5 × 10⁻⁵ m for heat and
# moisture. The saturation specific humidity is enhanced by the moist-air factor of Gill (1982) over both
# the ocean, where it is 98% of the pure-water value, and the sea ice. The ice--ocean heat flux follows the
# three-equation formulation, with a friction velocity computed from the ice--ocean stress.

using NumericalEarth.EarthSystemModels: AtmosphericThermodynamics
using NumericalEarth.EarthSystemModels.InterfaceComputations: COARELogarithmicSimilarityProfile,
                                                              ConvectiveGustiness,
                                                              WindDependentWaveFormulation,
                                                              TemperatureDependentAirViscosity,
                                                              ImpureSaturationSpecificHumidity,
                                                              GillSaturationEnhancement,
                                                              ThreeEquationHeatFlux,
                                                              MomentumBasedFrictionVelocity,
                                                              atmosphere_sea_ice_stability_functions

saturation_enhancement = GillSaturationEnhancement()

atmosphere_ocean_interface_specific_humidity =
    ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Liquid(), 0.98; saturation_enhancement)

atmosphere_sea_ice_interface_specific_humidity =
    ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Ice(); saturation_enhancement)

air_kinematic_viscosity = TemperatureDependentAirViscosity()

atmosphere_ocean_fluxes =
    SimilarityTheoryFluxes(similarity_form = COARELogarithmicSimilarityProfile(),
                           subgrid_velocities = ConvectiveGustiness(minimum_gustiness = 0.5),
                           momentum_roughness_length = MomentumRoughnessLength(; air_kinematic_viscosity,
                                                                               wave_formulation = WindDependentWaveFormulation()),
                           temperature_roughness_length = ScalarRoughnessLength(; air_kinematic_viscosity),
                           water_vapor_roughness_length = ScalarRoughnessLength(; air_kinematic_viscosity))

atmosphere_sea_ice_fluxes =
    SimilarityTheoryFluxes(stability_functions = atmosphere_sea_ice_stability_functions(),
                           similarity_form = COARELogarithmicSimilarityProfile(),
                           subgrid_velocities = ConvectiveGustiness(minimum_gustiness = 0.2),
                           momentum_roughness_length = 5e-4,
                           temperature_roughness_length = 5e-5,
                           water_vapor_roughness_length = 5e-5)

sea_ice_ocean_heat_flux = ThreeEquationHeatFlux(sea_ice; heat_transfer_coefficient = 0.0057,
                                                         friction_velocity = MomentumBasedFrictionVelocity())

# ### Coupled simulation
#
# Now we are ready to build the coupled ocean--sea ice model and bring everything together into a
# `simulation` with a 90-minute time step.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation,
                                 atmosphere_ocean_fluxes, atmosphere_sea_ice_fluxes, sea_ice_ocean_heat_flux,
                                 atmosphere_ocean_interface_specific_humidity,
                                 atmosphere_sea_ice_interface_specific_humidity)

simulation = Simulation(coupled_model; Δt=90minutes, stop_time=90days)

# The salinity restoring flux, and with it its global mean, is recomputed from the ocean state every time step.

add_callback!(simulation, ConservativeSurfaceFluxRestoringCallback(salinity_flux, ocean.model))

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
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", extrema(To): (%.1f, %.1f) ᵒC, mean(To(z=0)): %.1f ᵒC", Tmin, Tmax, Tavg)
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
                                            overwrite_files = true)

ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η = free_surface);
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
# We load the saved output and make a movie of the simulation. First we plot a snapshot:
using CairoMakie

# We suffix the ocean fields with "o":
uo = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "u"; backend = OnDisk())
vo = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "v"; backend = OnDisk())
To = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "T"; backend = OnDisk())
eo = FieldTimeSeries("ocean_one_degree_surface_fields.jld2",  "e"; backend = OnDisk())
ηo = FieldTimeSeries("ocean_one_degree_free_surface.jld2",    "η"; backend = OnDisk())

# and sea ice fields with "i":
ui = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "u"; backend = OnDisk())
vi = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "v"; backend = OnDisk())
hi = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "h"; backend = OnDisk())
ℵi = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "ℵ"; backend = OnDisk())
Ti = FieldTimeSeries("sea_ice_one_degree_surface_fields.jld2", "T"; backend = OnDisk())

times = uo.times
Nt = length(times)
n = Observable(Nt)

# We create a land mask and use it to fill land points with `NaN`s.
# The grid's `bottom_height` is stored as a halo-padded `OffsetArray`,
# so we take a view of its interior to build the mask.
Nx, Ny, _ = size(grid)
land = view(To.grid.immersed_boundary.bottom_height, 1:Nx, 1:Ny, 1:1) .≥ 0

Toₙ = @lift begin
    Tₙ = interior(To[$n])
    Tₙ[land] .= NaN
    view(Tₙ, :, :, 1)
end

eoₙ = @lift begin
    eₙ = interior(eo[$n])
    eₙ[land] .= NaN
    view(eₙ, :, :, 1)
end

ηoₙ = @lift begin
    ηₙ = interior(ηo[$n])
    ηₙ[land] .= NaN
    view(ηₙ, :, :, 1)
end

heₙ = @lift begin
    hₙ = interior(hi[$n])
    ℵₙ = interior(ℵi[$n])
    hₙ[land] .= NaN
    view(hₙ, :, :, 1) .* view(ℵₙ, :, :, 1)
end

# We compute the surface speeds for the ocean and the sea ice.
uoₙ = Field{Face, Center, Nothing}(uo.grid)
voₙ = Field{Center, Face, Nothing}(vo.grid)

uiₙ = Field{Face, Center, Nothing}(ui.grid)
viₙ = Field{Center, Face, Nothing}(vi.grid)

so = Field(sqrt(uoₙ^2 + voₙ^2))
si = Field(sqrt(uiₙ^2 + viₙ^2))

soₙ = @lift begin
    parent(uoₙ) .= parent(uo[$n])
    parent(voₙ) .= parent(vo[$n])
    compute!(so)
    soₙ = interior(so)
    soₙ[land] .= NaN
    view(soₙ, :, :, 1)
end

siₙ = @lift begin
    parent(uiₙ) .= parent(ui[$n])
    parent(viₙ) .= parent(vi[$n])
    compute!(si)
    siₙ = interior(si)
    hₙ = interior(hi[$n])
    ℵₙ = interior(ℵi[$n])
    he = hₙ .* ℵₙ
    siₙ[he .< 1e-7] .= 0
    siₙ[land] .= NaN
    view(siₙ, :, :, 1)
end

# Finally, we plot a snapshot of the surface speed, temperature, and the turbulent
# eddy kinetic energy from the CATKE vertical mixing parameterization as well as the
# sea ice speed and the effective sea ice thickness.
fig = Figure(size=(900, 750))

title = @lift string("Global 1ᵒ ocean simulation after ", prettytime(times[$n] - times[1]))

axso = Axis(fig[1, 1])
axηo = Axis(fig[1, 3])
axTo = Axis(fig[2, 1])
axeo = Axis(fig[2, 3])
axsi = Axis(fig[3, 1])
axhi = Axis(fig[3, 3])

hm = heatmap!(axso, soₙ, colorrange = (0, 0.5), colormap = :deep,  nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Ocean Surface speed (m s⁻¹)")
hm = heatmap!(axηo, ηoₙ, colorrange = (-1.2, 1.2), colormap = :balance, nan_color=:lightgray)
Colorbar(fig[1, 4], hm, label = "Sea Surface Height (m)")
hm = heatmap!(axTo, Toₙ, colorrange = (-1, 32), colormap = :magma, nan_color=:lightgray)
Colorbar(fig[2, 2], hm, label = "Surface Temperature (ᵒC)")
hm = heatmap!(axeo, eoₙ, colorrange = (0, 1e-3), colormap = :solar, nan_color=:lightgray)
Colorbar(fig[2, 4], hm, label = "Turbulent Kinetic Energy (m² s⁻²)")

hm = heatmap!(axsi, siₙ, colorrange = (0, 0.5), colormap = :greys, nan_color=:lightgray)
Colorbar(fig[3, 2], hm, label = "Sea ice speed (m s⁻¹)")
hm = heatmap!(axhi, heₙ, colorrange =  (0, 4),  colormap = :blues, nan_color=:lightgray)
Colorbar(fig[3, 4], hm, label = "Effective ice thickness (m)")


for ax in (axso, axsi, axTo, axhi, axeo)
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
