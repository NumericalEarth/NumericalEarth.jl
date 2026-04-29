using NumericalEarth
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids
using NumericalEarth.Oceans
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using Printf

using CUDA
arch = GPU()

depth = 2000meters
Nz = 30
z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/3, mutable=true)

Nx = 180 # longitudinal direction -> 250 points is about 1.5ᵒ resolution
Ny = 180 # meridional direction -> same thing, 48 points is about 1.5ᵒ resolution
Nz = length(r_faces) - 1

grid = RotatedLatitudeLongitudeGrid(arch, size = (Nx, Ny, Nz),
                                          latitude = (-45, 45),
                                          longitude = (-45, 45),
                                          z,
                                          north_pole = (180, 0),
                                          halo = (5, 5, 4),
                                          topology = (Bounded, Bounded, Bounded))

bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

#####
##### A Propgnostic Ocean model
#####

# A very diffusive ocean
momentum_advection = WENOVectorInvariant(order=3)
tracer_advection   = WENO(order=3)

free_surface = SplitExplicitFreeSurface(grid; cfl=0.7)
closure = NumericalEarth.Oceans.default_ocean_closure()

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         closure)

dataset = ECCO4Monthly()

set!(ocean.model, T=Metadatum(:temperature; dataset),
                  S=Metadatum(:salinity;    dataset))

#####
##### A Prognostic Sea-ice model
#####

using ClimaSeaIce.SeaIceDynamics
using ClimaSeaIce.Rheologies

# Remember to pass the SSS as a bottom bc to the sea ice!
SSS = view(ocean.model.tracers.S.data, :, :, grid.Nz)
bottom_heat_boundary_condition = IceWaterThermalEquilibrium(SSS)

SSU = view(ocean.model.velocities.u, :, :, grid.Nz)
SSV = view(ocean.model.velocities.u, :, :, grid.Nz)
τo  = SemiImplicitStress(uₑ=SSU, vₑ=SSV)
τua = Field{Face, Center, Nothing}(grid)
τva = Field{Center, Face, Nothing}(grid)

dynamics = SeaIceMomentumEquation(grid;
                                  coriolis = ocean.model.coriolis,
                                  top_momentum_stress = (u=τua, v=τva),
                                  bottom_momentum_stress = τo,
                                  ocean_velocities = (u=0.1*SSU, v=0.1*SSV),
                                  rheology = ElastoViscoPlasticRheology(),
                                  solver = SplitExplicitSolver(120))

sea_ice = sea_ice_simulation(grid; bottom_heat_boundary_condition, dynamics, advection=WENO(order=7))

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset),
                    ℵ=Metadatum(:sea_ice_concentration; dataset))

#####
##### A Prescribed Atmosphere model
#####

atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(40))
radiation  = Radiation()

#####
##### Arctic coupled model
#####

arctic = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
arctic = Simulation(arctic, Δt=5minutes, stop_time=365days)

# Sea-ice variables
h = sea_ice.model.ice_thickness
ℵ = sea_ice.model.ice_concentration
u = sea_ice.model.velocities.u
v = sea_ice.model.velocities.v

# Fluxes
Tu = arctic.model.interfaces.atmosphere_sea_ice_interface.temperature
𝒬ᵛ = arctic.model.interfaces.atmosphere_sea_ice_interface.fluxes.latent_heat
𝒬ᵀ = arctic.model.interfaces.atmosphere_sea_ice_interface.fluxes.sensible_heat
𝒬ⁱⁿᵗ = arctic.model.interfaces.sea_ice_ocean_interface.fluxes.interface_heat
𝒬ᶠʳᶻ = arctic.model.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
𝒬ᵗᵒᵖ = arctic.model.interfaces.net_fluxes.sea_ice_top.heat
𝒬ᵇᵒᵗ = arctic.model.interfaces.net_fluxes.sea_ice_bottom.heat
τˣ = arctic.model.interfaces.net_fluxes.sea_ice_top.u
τʸ = arctic.model.interfaces.net_fluxes.sea_ice_top.v

# Output writers
arctic.output_writers[:vars] = JLD2Writer(sea_ice.model, (; h, ℵ, u, v, Tu, 𝒬ᵛ, 𝒬ᵀ, 𝒬ⁱⁿᵗ, 𝒬ᶠʳᶻ, 𝒬ᵗᵒᵖ, 𝒬ᵇᵒᵗ, τˣ, τʸ),
                                          including = [:grid],
                                          filename = "sea_ice_quantities.jld2",
                                          schedule = IterationInterval(12),
                                          overwrite_existing=true)

arctic.output_writers[:averages] = JLD2Writer(sea_ice.model, (; h, ℵ, Tu, 𝒬ᵛ, 𝒬ᵀ, 𝒬ⁱⁿᵗ, 𝒬ᶠʳᶻ, 𝒬ᵗᵒᵖ, 𝒬ᵇᵒᵗ, u, v, τˣ, τʸ),
                                              including = [:grid],
                                              filename = "averaged_sea_ice_quantities.jld2",
                                              schedule = AveragedTimeInterval(1days),
                                              overwrite_existing=true)

wall_time = Ref(time_ns())

using Statistics

function progress(sim)
    sea_ice = sim.model.sea_ice
    ocean   = sim.model.ocean
    hmax  = maximum(sea_ice.model.ice_thickness)
    ℵmax  = maximum(sea_ice.model.ice_concentration)
    uimax = maximum(abs, sea_ice.model.velocities.u)
    vimax = maximum(abs, sea_ice.model.velocities.v)
    uomax = maximum(abs, ocean.model.velocities.u)
    vomax = maximum(abs, ocean.model.velocities.v)

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
    msg3 = @sprintf("max uˢⁱ: (%.2f, %.2f) m s⁻¹, ", uimax, vimax)
    msg4 = @sprintf("max uᵒᶜ: (%.2f, %.2f) m s⁻¹, ", uomax, vomax)
    msg5 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg5

     wall_time[] = time_ns()

     return nothing
end

# And add it as a callback to the simulation.
add_callback!(arctic, progress, IterationInterval(10))

run!(arctic)

#####
##### Comparison to ECCO Climatology
#####

dataset = ECCO4Monthly()
dates   = all_dates(version)[1:12]

h_metadata = Metadata(:sea_ice_thickness;     dataset, dates)
ℵ_metadata = Metadata(:sea_ice_concentration; dataset, dates)

# Montly averaged ECCO data
hE = ECCOFieldTimeSeries(h_metadata, grid; time_indices_in_memory=12)
ℵE = ECCOFieldTimeSeries(ℵ_metadata, grid; time_indices_in_memory=12)

# Daily averaged Model output
h = FieldTimeSeries("averaged_sea_ice_quantities.jld2", "h")
ℵ = FieldTimeSeries("averaged_sea_ice_quantities.jld2", "ℵ")

# Montly average the model output
hm = FieldTimeSeries{Center, Center, Nothing}(grid, hE.times; backend=InMemory())
ℵm = FieldTimeSeries{Center, Center, Nothing}(grid, hE.times; backend=InMemory())

hMe = [mean(h[t]) for t in 1:length(h.times)]
ℵMe = [mean(ℵ[t]) for t in 1:length(ℵ.times)]

