# # Ocean circulation beneath an ice shelf: ISOMIP experiment 1
#
# This example simulates the ocean circulation in a cavity beneath a floating
# ice shelf. It is experiment 1 of ISOMIP, the Ice Shelf–Ocean Model
# Intercomparison Project (Hunter, 2006): a closed box fully covered by an ice
# shelf whose draft ramps with latitude. ISOMIP is model-independent. Here it
# is set up in the configuration of MITgcm's `verification/isomip` experiment
# (its geometry, partial cells and time step), so that the results can be
# compared with MITgcm directly.
#
# A 900 m deep box of near-freezing water sits under an ice shelf whose base
# slopes from 700 m depth at the southern wall up to 200 m depth, then stays
# flat to the northern wall. Melting at the ice base cools and freshens the
# water there, and the buoyant meltwater drives a circulation around the
# cavity.
#
# The example shows the four ingredients needed to put an ice shelf in an
# Oceananigans model:
#
# 1. An `ImmersedBoundaryGrid` with a `PartialCellBottomAndTop`, which takes both a
#    bed and an ice draft.
# 2. The weight of the ice, a `TopLoad` passed to the immersed boundary as its
#    `top_load`.
# 3. Melting at the ice base from NumericalEarth's `IceShelfOceanInterface`.
# 4. Quadratic drag on the ice base and the seafloor.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, CairoMakie"
# ```

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Coriolis: DualGridScheme
using NumericalEarth.EarthSystemModels: ThreeEquationHeatFlux
using Printf
using Statistics: mean

arch = CPU()

# ## Grid
#
# MITgcm uses 50 × 100 cells of 0.3° × 0.1° at 70-80°S and 30 levels of 30 m.
# Here the domain is a Cartesian box of the same extent, with the zonal width
# evaluated at the central latitude, 75°S.

Nx, Ny, Nz = 50, 100, 30
Lx = 50 * 0.3 * 111.13kilometers * cosd(75)
Ly = 100 * 0.1 * 111.13kilometers
Lz = 900

# WENO(order = 5) on an immersed boundary grid needs a halo of 4.

underlying_grid = RectilinearGrid(arch;
                                  size = (Nx, Ny, Nz),
                                  x = (0, Lx),
                                  y = (0, Ly),
                                  z = (-Lz, 0),
                                  halo = (4, 4, 4),
                                  topology = (Bounded, Bounded, Bounded))

# ## Ice shelf geometry
#
# A cavity is bounded by two surfaces: the seafloor below and the ice base
# (the "draft") above. Water lives between them.
#
# The seafloor is flat at 900 m, except that MITgcm's configuration makes the
# first column and the first row of cells dry land. Here that dry land is a
# bed height of 0 m, which closes those columns.

Δx = Lx / Nx
Δy = Ly / Ny

bed(x, y) = (x < Δx || y < Δy) ? 0 : -Lz

# The ice draft deepens from 200 m at 4° north of the southern wall down to
# 700 m at the wall, a slope of 125 m per degree of latitude. It does not vary
# in x.

ice_draft(x, y) = min(-200, -700 + 125 * y / 111.13kilometers)

# `PartialCellBottomAndTop` lets both the seafloor and the ice base cut through a
# cell, so the sloping draft is represented smoothly rather than as a
# staircase. `minimum_fractional_cell_height = 0.05` matches MITgcm's
# `hFacMin`. `GridFittedBottomAndTop` is the full-cell alternative. The grid is
# built below, once the weight of the ice is defined.

# ## Initial state and equation of state
#
# The water starts at rest, with uniform temperature and salinity.
# MITgcm's reference run uses the JMD95Z equation of state. A linear equation
# of state is used here. At -1.9 °C and 34.4 psu, its haline contraction is
# within 1% of TEOS-10's over the cavity's depths (200-700 m). Its thermal
# expansion lies within TEOS-10's range there, which grows with pressure
# from 3.2 × 10⁻⁵ to 4.7 × 10⁻⁵ K⁻¹.

T₀ = -1.9 # °C
S₀ = 34.4 # psu

equation_of_state = LinearEquationOfState(thermal_expansion = 3.733e-5, haline_contraction = 7.843e-4)
buoyancy = SeawaterBuoyancy(; equation_of_state)

# ## The weight of the ice shelf
#
# A floating ice shelf pushes down on the water beneath it. The water is in
# hydrostatic balance with that weight, so its pressure at the ice base equals
# the weight of the water the ice displaces. Beneath a sloping draft, that
# pressure varies horizontally.
#
# `TopLoad` computes this pressure (divided by the reference density, i.e.
# as a potential) from a reference state, here the initial conditions, when
# the grid is built, and adds it to the hydrostatic pressure of every column
# beneath the ice. The same reference
# state must be passed to `set!` below. If it is not, the water is not at rest
# to begin with.

Tᵢ(x, y, z) = T₀
Sᵢ(x, y, z) = S₀

top_load = TopLoad(buoyancy, (; T = Tᵢ, S = Sᵢ))

grid = ImmersedBoundaryGrid(underlying_grid,
                            PartialCellBottomAndTop(bed, ice_draft; minimum_fractional_cell_height = 0.05, top_load))

# ## Melting at the ice base
#
# `IceShelfOceanInterface` solves the three-equation melt problem in every
# column beneath the ice. The equations balance:
#
# * heat conducted from the ocean against the latent heat of melting;
# * salt against dilution by meltwater;
# * the pressure-dependent freezing point at the ice base.
#
# The ocean temperature and salinity used are averaged over a 30 m boundary
# layer below the ice.
#
# MITgcm's ISOMIP run uses a fixed heat transfer velocity, γᵀ = 10⁻⁴ m/s.
# We reproduce it by giving `ThreeEquationHeatFlux` a constant friction
# velocity u★ with γᵀ = Γᵀ u★. The salt transfer velocity is MITgcm's
# default, γˢ = 5.05 × 10⁻³ γᵀ.
#
# The two models solve different melt problems. MITgcm's ISOMIP melt is a
# two-equation formulation: the freezing point is evaluated at the salinity
# of the ocean next to the ice, and γˢ never enters. In the three-equation
# formulation, meltwater freshens the ice base. That raises the freezing
# point there and limits melting, and with γˢ this small the limit is
# strong. The ice-base temperature and salinity agree closely with MITgcm,
# but the peak melt rate is about 40% of MITgcm's.

Γᵀ = 0.022
u★ = 1e-4 / Γᵀ

flux_formulation = ThreeEquationHeatFlux(heat_transfer_coefficient = Γᵀ,
                                         salt_transfer_coefficient = 5.05e-3 * Γᵀ,
                                         friction_velocity = u★)

interface = IceShelfOceanInterface(grid; flux_formulation,
                                   reference_density = 1030,
                                   heat_capacity = 3974,
                                   latent_heat = 334e3,
                                   boundary_layer_thickness = 30)

# The melt fluxes are applied to T and S as a forcing spread over the same
# 30 m boundary layer. This keeps a thin
# partial cell under the ice from receiving the whole flux.

forcing = ice_shelf_tracer_forcing(interface)

# ## Drag on the ice base and the seafloor
#
# MITgcm applies quadratic drag with Cᴰ = 2.5 × 10⁻³ on both the ice base
# and the seafloor. `ice_shelf_boundary_conditions` puts it on the ice base,
# as the `top` of the immersed boundary condition on u and v, i.e. on the
# face at the top of every water column beneath the ice.
# The seafloor is an ordinary ocean bottom, so it gets Oceananigans' standard
# `BulkDrag`, as in any other simulation. The bed here coincides with the
# bottom of the grid; with immersed bathymetry, also pass
# `bottom_drag_coefficient = Cᴰ` to `ice_shelf_boundary_conditions`.

Cᴰ = 2.5e-3
seafloor_drag = BulkDrag(coefficient = Cᴰ)

boundary_conditions = ice_shelf_boundary_conditions(interface;
                                                    drag_coefficient = Cᴰ,
                                                    u = (; bottom = seafloor_drag),
                                                    v = (; bottom = seafloor_drag))

# ## The model
#
# The rest of the setup is an ordinary hydrostatic model. The viscosities
# and diffusivities are MITgcm's, and the Coriolis parameter is evaluated at
# 75°S.
#
# The Coriolis force uses the C-D grid scheme of Adcroft, Hill and Marshall
# (1999), as MITgcm's ISOMIP run does, with the same 4 × 10⁵ s relaxation time.
# With the plain C-grid average, the steps in the ice draft drive a spurious
# zonal flow in the top wet cells.
#
# The free surface is split-explicit: the barotropic mode is substepped
# within each baroclinic step. It is solved everywhere, including beneath the
# ice. There the surface is the pressure at the ice base, and the free surface
# is masked against the top of each water column rather than the top of the
# grid. Time stepping is third-order Runge-Kutta.

closure = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1e-3, κ = 5e-5),
           HorizontalScalarDiffusivity(ν = 600, κ = 100))

model = HydrostaticFreeSurfaceModel(grid;
                                    tracers = (:T, :S),
                                    buoyancy,
                                    coriolis = FPlane(f = -1.409e-4, scheme = DualGridScheme(grid; relaxation_time = 4e5)),
                                    closure,
                                    boundary_conditions,
                                    forcing,
                                    momentum_advection = WENO(order = 5),
                                    tracer_advection = WENO(order = 5),
                                    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = 30minutes),
                                    timestepper = :SplitRungeKutta3)

set!(model, T = Tᵢ, S = Sᵢ)

# ## Simulation
#
# MITgcm runs this case for 360 days with a 30-minute time step.

simulation = Simulation(model; Δt = 30minutes, stop_time = 360days)

# The melt fluxes depend on the ocean state, so they are recomputed every time step.

add_callback!(simulation, sim -> compute_ice_shelf_fluxes!(interface, sim), IterationInterval(1))

# The melt rate is a mass flux in kg m⁻² s⁻¹. Dividing it by the density of
# fresh water gives metres of water per unit time.

# The mean is taken over the columns beneath the ice, where `k_draft > 0`.

melt_rate = interface.fluxes.melt_rate
ice_covered = Array(interior(interface.k_draft)) .> 0

function progress(sim)
    u, v, w = sim.model.velocities
    mean_melt = mean(Array(interior(melt_rate))[ice_covered]) / 1000 * 365days

    msg = @sprintf("iteration: %d, time: %s, mean melt: %.2f m/yr, max|u|: (%.2e, %.2e, %.2e) m/s",
                   iteration(sim), prettytime(sim), mean_melt,
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))
    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(480))

outputs = merge(model.velocities, model.tracers, (; melt_rate))

simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                filename = "isomip_ice_shelf_cavity.jld2",
                                                schedule = TimeInterval(10days),
                                                overwrite_files = true)

run!(simulation)

# ## Results
#
# The zonal mean of the meridional velocity and temperature shows the
# overturning circulation beneath the ice. The map shows the melt rate at the
# ice base.

using CairoMakie

vt = FieldTimeSeries("isomip_ice_shelf_cavity.jld2", "v")
Tt = FieldTimeSeries("isomip_ice_shelf_cavity.jld2", "T")
mt = FieldTimeSeries("isomip_ice_shelf_cavity.jld2", "melt_rate")

yc = ynodes(Tt.grid, Center()) ./ 1kilometers
yf = ynodes(vt.grid, Face()) ./ 1kilometers
zc = znodes(Tt.grid, Center())
xc = xnodes(Tt.grid, Center()) ./ 1kilometers

n = length(Tt.times)

# Immersed cells hold zero, so they are shown as NaN, i.e. left blank.
wet = interior(Tt[n]) .!= 0
zonal_mean(a, mask) = dropdims(sum(a .* mask, dims=1) ./ max.(sum(mask, dims=1), 1), dims=1)

v̄ = zonal_mean(interior(vt[n])[:, 1:Ny, :], wet)
T̄ = zonal_mean(interior(Tt[n]), wet)
v̄[dropdims(sum(wet, dims=1), dims=1) .== 0] .= NaN
T̄[dropdims(sum(wet, dims=1), dims=1) .== 0] .= NaN

melt = interior(mt[n], :, :, 1) ./ 1000 .* 365days
melt[.!ice_covered[:, :, 1]] .= NaN

fig = Figure(size = (1000, 900))

axv = Axis(fig[1, 1]; title = "Zonal-mean v at $(prettytime(Tt.times[n]))", xlabel = "y (km)", ylabel = "z (m)")
hmv = heatmap!(axv, yf[1:Ny], zc, v̄; colormap = :balance, colorrange = (-0.01, 0.01))
Colorbar(fig[1, 2], hmv; label = "m s⁻¹")

axT = Axis(fig[2, 1]; title = "Zonal-mean temperature", xlabel = "y (km)", ylabel = "z (m)")
hmT = heatmap!(axT, yc, zc, T̄; colormap = :thermal)
Colorbar(fig[2, 2], hmT; label = "°C")

axm = Axis(fig[3, 1]; title = "Melt rate", xlabel = "x (km)", ylabel = "y (km)")
hmm = heatmap!(axm, xc, yc, melt; colormap = :viridis)
Colorbar(fig[3, 2], hmm; label = "m yr⁻¹")

save("isomip_ice_shelf_cavity.png", fig)
nothing #hide

# ![](isomip_ice_shelf_cavity.png)
