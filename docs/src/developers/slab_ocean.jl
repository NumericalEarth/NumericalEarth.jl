# # Implementing a Slab Ocean Component
#
# This tutorial demonstrates how to implement a new ocean component for `EarthSystemModel`: a **slab ocean model**.
# A slab ocean represents the ocean as a single well-mixed layer with a fixed depth, making it computationally efficient
# while still capturing the essential thermodynamic coupling between the ocean, atmosphere, and sea ice.
# This approach is commonly used in climate sensitivity studies and is detailed by [Garuba2024Slab](@citet).
#
# ## Overview
#
# A slab ocean model simplifies the full 3D ocean by:
# - Representing the ocean as a single temperature (and optionally salinity) per horizontal grid point
# - Using a fixed "slab depth" that determines the heat capacity
# - Evolving temperature based on net heat fluxes from the atmosphere and sea ice
# - Having no ocean dynamics (no velocities)
#
# To integrate a slab ocean into `EarthSystemModel`, we implement the contract in
# [Coupling a new component](@ref):
#
# 1. `grid`: the grid of the ocean component
# 2. `surface_state`: the fields that the coupler reads
# 3. `net_fluxes`: the fields that the coupler writes
# 4. `reference_density`: of the ocean component
# 5. `heat_capacity`: of the ocean component
# 6. `time_step!`: advances the component forward in time
#
# ## Define the Slab Ocean Type
#
# First, we define a struct to hold the slab ocean state and its constructor. We also define a summary and show method for the slab ocean type.

using Oceananigans, Base

struct SlabOcean{G, C, T, F}
    grid :: G
    clock :: C
    temperature :: T
    temperature_flux :: F
end

function SlabOcean(grid; clock =  Clock(time=0))
    temperature = CenterField(grid)
    temperature_flux = CenterField(grid)
    return SlabOcean(grid, clock, temperature, temperature_flux)
end

Base.summary(slab_ocean::SlabOcean) = "SlabOcean with Depth=$(slab_ocean.grid.Lz)"
Base.show(io::IO, slab_ocean::SlabOcean) = print(io, Base.summary(slab_ocean))

# ## Implement the contract
#
# `surface_state` returns the fields that the coupler reads to compute the turbulent fluxes.
# A slab ocean has no velocities and a constant salinity.

using NumericalEarth.EarthSystemModels
using NumericalEarth.EarthSystemModels.InterfaceComputations

Oceananigans.Grids.grid(slab_ocean::SlabOcean) = slab_ocean.grid

EarthSystemModels.surface_state(slab_ocean::SlabOcean) = (; u =Oceananigans.Fields.ZeroField(Float64),
                                                            v = Oceananigans.Fields.ZeroField(Float64),
                                                            T = slab_ocean.temperature,
                                                            S = Oceananigans.Fields.ConstantField(35.0))

# The `net_fluxes` function returns the fields that the coupler writes. For a slab ocean, we return the
# temperature flux stored in the `slab_ocean` and fields for the other fluxes, which go unused because
# the slab ocean has no dynamics and a constant salinity.

function InterfaceComputations.net_fluxes(slab_ocean::SlabOcean)
    grid = slab_ocean.grid
    return (; T = slab_ocean.temperature_flux,
              S = CenterField(grid),
              u = CenterField(grid),
              v = CenterField(grid),
              η = CenterField(grid),
              freshwater_heat_content = CenterField(grid),
              u_coefficient = nothing,
              v_coefficient = nothing)
end

# We also define the thermodynamic properties of the ocean component.

EarthSystemModels.reference_density(::SlabOcean) = 1025.0
EarthSystemModels.heat_capacity(::SlabOcean) = 3990.0

# ## Extend the TimeSteppers.jl module
#
# The `time_step!` method is called by the coupled model to advance the component forward in time.
# For a slab ocean, this method advances temperature through the computed flux. Note the convention
# that the fluxes are positive when they are leaving the ocean component (cooling the ocean).

using Oceananigans.TimeSteppers: tick!

function Oceananigans.TimeSteppers.time_step!(slab_ocean::SlabOcean, Δt)
    tick!(slab_ocean.clock, Δt)
    parent(slab_ocean.temperature) .-= parent(slab_ocean.temperature_flux) .* Δt ./ slab_ocean.grid.Lz
    return nothing
end

# ## Complete Example: Coupling Slab Ocean with JRA55 and Sea Ice
#
# Here's a complete example showing how to use the slab ocean in a coupled simulation.
# We use the JRA55 reanalysis for the atmosphere and the ECCO4Monthly dataset to initialize our slab ocean.
# We also initialize the sea ice with climatological data and see how the sea ice evolves.

using CUDA
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates

arch = GPU()
grid = TripolarGrid(arch, size=(720, 720, 1), z=(-50, 0))
bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=10)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

slab_ocean = SlabOcean(grid)
set!(slab_ocean.temperature, Metadatum(:temperature, dataset=ECCO4Monthly()))

atmosphere = NumericalEarth.JRA55PrescribedAtmosphere(arch)

sea_ice = NumericalEarth.sea_ice_simulation(grid, slab_ocean, advection=WENO(order=7))
set!(sea_ice.model, h=Metadatum(:sea_ice_thickness,     dataset=ECCO4Monthly()),
                    ℵ=Metadatum(:sea_ice_concentration, dataset=ECCO4Monthly()))

interfaces = ComponentInterfaces(atmosphere, slab_ocean, sea_ice; exchange_grid=grid)
coupled_model = NumericalEarth.EarthSystemModel(; atmosphere, sea_ice, ocean=slab_ocean, interfaces)

simulation = Simulation(coupled_model, Δt=60minutes, stop_time=120days)
run!(simulation)

# And now visualize.

using CairoMakie

Oceananigans.ImmersedBoundaries.mask_immersed_field!(slab_ocean.temperature, NaN)
Oceananigans.ImmersedBoundaries.mask_immersed_field!(sea_ice.model.ice_thickness, NaN)
Oceananigans.ImmersedBoundaries.mask_immersed_field!(sea_ice.model.ice_concentration, NaN)

fig = Figure(size = (800, 1000), fontsize = 16)
axT = Axis(fig[1, 1], title="Slab Ocean Temperature")
axh = Axis(fig[2, 1], title="Sea Ice Thickness")
axℵ = Axis(fig[3, 1], title="Sea Ice Concentration")

heatmap!(axT, Array(interior(slab_ocean.temperature, :, :, 1)),          colormap=:thermal)
heatmap!(axh, Array(interior(sea_ice.model.ice_thickness, :, :, 1)),     colormap=:ice)
heatmap!(axℵ, Array(interior(sea_ice.model.ice_concentration, :, :, 1)), colormap=:deep)
hidedecorations!(axT)
hidedecorations!(axh)
hidedecorations!(axℵ)

save("slab_ocean.png", fig)
nothing #hide

# ![](slab_ocean.png)

# ## Summary
#
# The slab ocean example demonstrates how a simplified component can be integrated into the coupling framework while
# maintaining compatibility with the existing atmosphere and sea ice components.
# The key insight is that the coupling system handles flux computations generically;
# your component just needs to provide the right interface methods.
