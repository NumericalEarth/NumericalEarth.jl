using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

Δz = [50., 70., 100., 140., 190., 240., 290., 340., 390., 440., 490., 540., 590., 640., 690.]

z_faces = [0]
for k in 2:16
    push!(z_faces, z_faces[k-1] - Δz[k-1])
end

z_faces = reverse(z_faces)
grid = LatitudeLongitudeGrid(size = (90, 40, 15), 
                            latitude = (-80, 80), 
                            longitude = (0, 360), 
                            z = z_faces,
                            halo = (5, 5, 5))

function read_bin(filepath; dims = (90, 40, 15))
    array = zeros(Float32, prod(dims))
    read!(filepath, array)
    array = bswap.(array)
    array = array .|> Float64
    return reshape(array, dims...)
end

bat  = read_bin("bathymetry.bin"; dims = (90, 40))
Tini = reverse(read_bin("lev_t.bin"), dims=3)
Sini = reverse(read_bin("lev_s.bin"), dims=3)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bat))

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=AdvectiveFormulation())
catke_closure = NumericalEarth.Oceans.default_ocean_closure()
vertical_mixing = VerticalScalarDiffusivity(ν=1e-3, κ=3e-5)
horiz_viscosity = HorizontalScalarDiffusivity(ν=5e5)

@info "Building an ocean model"

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 70 substeps for the barotropic mode.

free_surface       = ImplicitFreeSurface() # (grid; substeps=40)
momentum_advection = VectorInvariant(; vorticity_scheme = Oceananigans.Advection.EnergyConserving())
tracer_advection   = UpwindBiased(order=3)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface, reference_density=1035,
                         closure=(catke_closure, horiz_viscosity, vertical_mixing), timestepper = :QuasiAdamsBashforth2)

@show ocean.model.coriolis

set!(ocean.model, T=Tini, S=Sini)

atmos = JRA55PrescribedAtmosphere()

# The coupled ocean--atmosphere model.
# We use the default radiation model and we do not couple an ice model for simplicity.

@info "Building a coupled simulation"

radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)
coupled_model = OceanOnlyModel(ocean; atmosphere=atmos, radiation)
nesimulation = Simulation(coupled_model; Δt = 1200, stop_time = 60days)

## Progress callback
# Print diagnostics every 10 days: SST/SSS ranges and wall-clock time.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    model = ocean.model
    niter = model.clock.iteration
    mtime = model.clock.time

    ocean_sst = view(model.tracers.T, :, :, 15)
    ocean_sss = view(model.tracers.S, :, :, 15)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    @printf("iter %5d | MITgcm iter %6d | day %7.1f | ", iteration(sim), niter, mtime / 86400)
    @printf("SST: [%6.2f, %5.2f] | SSS: [%5.2f, %5.2f] | wall: %s\n",
            minimum(ocean_sst), maximum(ocean_sst),
            minimum(ocean_sss), maximum(ocean_sss),
            prettytime(elapsed))

    wall_time[] = time_ns()
    return nothing
end

add_callback!(nesimulation, progress, TimeInterval(10days))

# We also set up a callback to collect the surface velocities (u, v) for later visualization.

u = []
v = []
η = []
fu = []
fv = []

function save_variables(sim)
    push!(u, deepcopy(sim.model.interfaces.exchanger.ocean.state.u))
    push!(v, deepcopy(sim.model.interfaces.exchanger.ocean.state.v))
    push!(η, deepcopy(sim.model.ocean.model.free_surface.displacement))
    push!(fu, deepcopy(sim.model.ocean.model.velocities.u.boundary_conditions.top.condition))
    push!(fv, deepcopy(sim.model.ocean.model.velocities.v.boundary_conditions.top.condition))
end

add_callback!(nesimulation, save_variables, IterationInterval(50))

# ## Run!

@info "Running 1-year simulation with JRA55 forcing..." nesimulation.Δt stop_time=365days

run!(nesimulation)

# ## Visualize surface velocities
#
# We produce an animation of the surface zonal and meridional velocities.

iter = Observable(1)
ui = @lift begin
    ut = interior(u[$iter], :, :, 15)
    ut[ut .== 0] .= NaN
    ut
end

vi = @lift begin
    ut = interior(v[$iter], :, :, 15)
    ut[ut .== 0] .= NaN
    ut
end

ηi = @lift begin
    ηt = interior(η[$iter], :, :, 1)
    ηt[ηt .== 0] .= NaN
    ηt
end

fui = @lift begin
    ηt = interior(fu[$iter], :, :, 1)
    ηt[ηt .== 0] .= NaN
    ηt .* 1020
end

fvi = @lift begin
    ηt = interior(fv[$iter], :, :, 1)
    ηt[ηt .== 0] .= NaN
    ηt .* 1020
end

Nt = length(u)

fig = Figure(resolution = (1000, 1000))
ax1 = Axis(fig[1, 1]; title = "Surface zonal velocity (m/s)", xlabel = "", ylabel = "Latitude")
ax2 = Axis(fig[2, 1]; title = "Surface meridional velocity (m/s)", xlabel = "", ylabel = "Latitude")
ax3 = Axis(fig[3, 1]; title = "Sea surface height (m)", xlabel = "", ylabel = "Latitude")
# ax4 = Axis(fig[4, 1]; title = "Zonal wind stress", xlabel = "", ylabel = "Latitude")
# ax5 = Axis(fig[5, 1]; title = "Meridional wind stress", xlabel = "", ylabel = "Latitude")

grid = coupled_model.interfaces.exchanger.grid

λ = λnodes(grid, Center())
φ = φnodes(grid, Center())

hm1 = heatmap!(ax1, λ, φ, ui, nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))
hm2 = heatmap!(ax2, λ, φ, vi, nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))
hm2 = heatmap!(ax3, λ, φ, ηi, nan_color = :grey, colormap = :bwr, colorrange = (-1.5, 1.5))
# hm2 = heatmap!(ax4, λ, φ, fui, nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))
# hm2 = heatmap!(ax5, λ, φ, fvi, nan_color = :grey, colormap = :bwr, colorrange = (-0.2, 0.2))

Colorbar(fig[1, 2], hm1)
Colorbar(fig[2, 2], hm2)

CairoMakie.record(fig, "numerical_ocean_surface.mp4", 1:Nt, framerate = 8) do nn
    iter[] = nn
end
nothing #hide

# ## Finalize


