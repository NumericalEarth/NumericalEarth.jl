# # Single-column ocean simulation forced by JRA55 re-analysis
#
# In this example, we simulate the evolution of an ocean water column
# forced by an atmosphere derived from the JRA55 re-analysis.
# The simulated column is located at ocean station
# Papa (144.9ᵒ W and 50.1ᵒ N).
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, CairoMakie"
# ```

using NumericalEarth
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Units
using Oceananigans.Models: buoyancy_frequency
using Dates
using Printf

# # Construct the grid
#
# First, we construct a single-column grid with 2 meter spacing
# located at ocean station Papa.

# Ocean station papa location
location_name = "ocean_station_papa"
λ★, φ★ = 35.1, 50.1

grid = RectilinearGrid(size = 200,
                       x = λ★,
                       y = φ★,
                       z = (-400, 0),
                       topology = (Flat, Flat, Bounded))

# # An "ocean simulation"
#
# Next, we use NumericalEarth's `ocean_simulation` constructor to build a realistic
# ocean simulation on the single-column grid,

ocean = ocean_simulation(grid; Δt=10minutes, coriolis=FPlane(latitude = φ★))

# which wraps around the ocean model

ocean.model

# We set initial conditions from ECCO4:

set!(ocean.model, T=Metadatum(:temperature, dataset=ECCO4Monthly()),
                  S=Metadatum(:salinity, dataset=ECCO4Monthly()))

# # A prescribed atmosphere based on JRA55 re-analysis
#
# We build a `JRA55PrescribedAtmosphere` at the same location as the single-colunm grid
# which is based on the JRA55 reanalysis.

atmosphere = JRA55PrescribedAtmosphere(longitude = λ★,
                                       latitude = φ★,
                                       end_date = DateTime(1990, 1, 31), # Last day of the simulation
                                       backend  = InMemory())

# This builds a representation of the atmosphere on the small grid

atmosphere.grid

# Let's take a look at the atmospheric state

ua = interior(atmosphere.velocities.u, 1, 1, 1, :)
va = interior(atmosphere.velocities.v, 1, 1, 1, :)
Ta = interior(atmosphere.tracers.T, 1, 1, 1, :)
qa = interior(atmosphere.tracers.q, 1, 1, 1, :)
t_days = atmosphere.times / days

using CairoMakie

set_theme!(Theme(linewidth=3, fontsize=24))

fig = Figure(size=(800, 1000))
axu = Axis(fig[2, 1]; ylabel="Atmosphere \n velocity (m s⁻¹)")
axT = Axis(fig[3, 1]; ylabel="Atmosphere \n temperature (ᵒK)")
axq = Axis(fig[4, 1]; ylabel="Atmosphere \n specific humidity", xlabel = "Days since Jan 1, 1990")
Label(fig[1, 1], "Atmospheric state over ocean station Papa", tellwidth=false)

lines!(axu, t_days, ua, label="Zonal velocity")
lines!(axu, t_days, va, label="Meridional velocity")
ylims!(axu, -6, 6)
axislegend(axu, framevisible=false, nbanks=2, position=:lb)

lines!(axT, t_days, Ta)
lines!(axq, t_days, qa)

current_figure()

# We continue constructing a simulation.
radiation = Radiation()
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model, Δt=ocean.Δt, stop_time=30days)

wall_clock = Ref(time_ns())

function progress(sim)
    msg = "Ocean Station Papa"
    msg *= string(", iter: ", iteration(sim), ", time: ", prettytime(sim))

    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg *= string(", wall time: ", prettytime(elapsed))
    wall_clock[] = time_ns()

    u, v, w = sim.model.ocean.model.velocities
    msg *= @sprintf(", max|u|: (%.2e, %.2e)", maximum(abs, u), maximum(abs, v))

    T = sim.model.ocean.model.tracers.T
    S = sim.model.ocean.model.tracers.S
    e = sim.model.ocean.model.tracers.e
    ρ = sim.model.interfaces.ocean_properties.reference_density
    c = sim.model.interfaces.ocean_properties.heat_capacity

    τx = first(sim.model.interfaces.net_fluxes.ocean.u)
    τy = first(sim.model.interfaces.net_fluxes.ocean.v)
    Q  = first(sim.model.interfaces.net_fluxes.ocean.T) * ρ * c

    u★ = sqrt(sqrt(τx^2 + τy^2))

    Nz = size(T, 3)
    msg *= @sprintf(", u★: %.2f m s⁻¹", u★)
    msg *= @sprintf(", Q: %.2f W m⁻²",  Q)
    msg *= @sprintf(", T₀: %.2f ᵒC", first(interior(T, 1, 1, Nz)))
    msg *= @sprintf(", extrema(T): (%.2f, %.2f) ᵒC", minimum(T), maximum(T))
    msg *= @sprintf(", S₀: %.2f g/kg", first(interior(S, 1, 1, Nz)))
    msg *= @sprintf(", e₀: %.2e m² s⁻²", first(interior(e, 1, 1, Nz)))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# Build flux outputs
τx = simulation.model.interfaces.net_fluxes.ocean.u
τy = simulation.model.interfaces.net_fluxes.ocean.v
JT = simulation.model.interfaces.net_fluxes.ocean.T
Js = simulation.model.interfaces.net_fluxes.ocean.S
Jᵛ = simulation.model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor
𝒬ᵀ = simulation.model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
𝒬ᵛ = simulation.model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
ρₒ = simulation.model.interfaces.ocean_properties.reference_density
cₚ = simulation.model.interfaces.ocean_properties.heat_capacity

Q = ρₒ * cₚ * JT
ρτx = ρₒ * τx
ρτy = ρₒ * τy
N² = buoyancy_frequency(ocean.model)
κc = ocean.model.closure_fields.κc

fluxes = (; ρτx, ρτy, Jᵛ, Js, 𝒬ᵛ, 𝒬ᵀ)
auxiliary_fields = (; N², κc)
u, v, w = ocean.model.velocities
T, S, e = ocean.model.tracers
fields = merge((; u, v, T, S, e), auxiliary_fields)

# Slice fields at the surface
outputs = merge(fields, fluxes)

filename = "single_column_omip_$(location_name)"

ocean.output_writers[:jld2] = JLD2Writer(ocean.model, outputs; filename,
                                         schedule = TimeInterval(3hours),
                                         overwrite_existing = true)

run!(simulation)

# Now let's load the saved output and visualise.

filename *= ".jld2"

u  = FieldTimeSeries(filename, "u")
v  = FieldTimeSeries(filename, "v")
T  = FieldTimeSeries(filename, "T")
S  = FieldTimeSeries(filename, "S")
e  = FieldTimeSeries(filename, "e")
N² = FieldTimeSeries(filename, "N²")
κ  = FieldTimeSeries(filename, "κc")

𝒬ᵛ = FieldTimeSeries(filename, "𝒬ᵛ")
𝒬ᵀ = FieldTimeSeries(filename, "𝒬ᵀ")
Js = FieldTimeSeries(filename, "Js")
Ev = FieldTimeSeries(filename, "Jᵛ")
ρτx = FieldTimeSeries(filename, "ρτx")
ρτy = FieldTimeSeries(filename, "ρτy")

Nz = size(T, 3)
times = 𝒬ᵀ.times

ua  = atmosphere.velocities.u
va  = atmosphere.velocities.v
Ta  = atmosphere.tracers.T
qa  = atmosphere.tracers.q
Qlw = atmosphere.downwelling_radiation.longwave
Qsw = atmosphere.downwelling_radiation.shortwave
Pr  = atmosphere.freshwater_flux.rain
Ps  = atmosphere.freshwater_flux.snow

Nt   = length(times)
uat  = zeros(Nt)
vat  = zeros(Nt)
Tat  = zeros(Nt)
qat  = zeros(Nt)
Qswt = zeros(Nt)
Qlwt = zeros(Nt)
Pt   = zeros(Nt)

for n = 1:Nt
    t = Oceananigans.Units.Time(times[n])
    uat[n]  =  ua[1, 1, 1, t]
    vat[n]  =  va[1, 1, 1, t]
    Tat[n]  =  Ta[1, 1, 1, t]
    qat[n]  =  qa[1, 1, 1, t]
    Qswt[n] = Qsw[1, 1, 1, t]
    Qlwt[n] = Qlw[1, 1, 1, t]
    Pt[n]   =  Pr[1, 1, 1, t] + Ps[1, 1, 1, t]
end

fig = Figure(size=(1800, 1800))

axτ = Axis(fig[1, 1:3], xlabel="Days since Oct 1 1992", ylabel="Wind stress (N m⁻²)")
axQ = Axis(fig[1, 4:6], xlabel="Days since Oct 1 1992", ylabel="Heat flux (W m⁻²)")
axu = Axis(fig[2, 1:3], xlabel="Days since Oct 1 1992", ylabel="Velocities (m s⁻¹)")
axT = Axis(fig[2, 4:6], xlabel="Days since Oct 1 1992", ylabel="Surface temperature (ᵒC)")
axF = Axis(fig[3, 1:3], xlabel="Days since Oct 1 1992", ylabel="Freshwater volume flux (m s⁻¹)")
axS = Axis(fig[3, 4:6], xlabel="Days since Oct 1 1992", ylabel="Surface salinity (g kg⁻¹)")

axuz = Axis(fig[4:5, 1:2], xlabel="Velocities (m s⁻¹)",                ylabel="z (m)")
axTz = Axis(fig[4:5, 3:4], xlabel="Temperature (ᵒC)",                  ylabel="z (m)")
axSz = Axis(fig[4:5, 5:6], xlabel="Salinity (g kg⁻¹)",                 ylabel="z (m)")
axNz = Axis(fig[6:7, 1:2], xlabel="Buoyancy frequency (s⁻²)",          ylabel="z (m)")
axκz = Axis(fig[6:7, 3:4], xlabel="Eddy diffusivity (m² s⁻¹)",         ylabel="z (m)", xscale=log10)
axez = Axis(fig[6:7, 5:6], xlabel="Turbulent kinetic energy (m² s⁻²)", ylabel="z (m)", xscale=log10)

title = @sprintf("Single-column simulation at %.2f, %.2f", φ★, λ★)
Label(fig[0, 1:6], title)

n = Observable(1)

times = (times .- times[1]) ./days
Nt = length(times)
tn = @lift times[$n]

colors = Makie.wong_colors()

ρₒ = coupled_model.interfaces.ocean_properties.reference_density
τx = interior(ρτx, 1, 1, 1, :) ./ ρₒ
τy = interior(ρτy, 1, 1, 1, :) ./ ρₒ
u★ = @. (τx^2 + τy^2)^(1/4)

lines!(axu, times, interior(u, 1, 1, Nz, :), color=colors[1], label="Zonal")
lines!(axu, times, interior(v, 1, 1, Nz, :), color=colors[2], label="Meridional")
lines!(axu, times, u★, color=colors[3], label="Ocean-side u★")
vlines!(axu, tn, linewidth=4, color=(:black, 0.5))
axislegend(axu)

lines!(axτ, times, interior(ρτx, 1, 1, 1, :), label="Zonal")
lines!(axτ, times, interior(ρτy, 1, 1, 1, :), label="Meridional")
vlines!(axτ, tn, linewidth=4, color=(:black, 0.5))
axislegend(axτ)

lines!(axT, times, Tat[1:Nt] .- 273.15,      color=colors[1], linewidth=2, linestyle=:dash, label="Atmosphere temperature")
lines!(axT, times, interior(T, 1, 1, Nz, :), color=colors[2], linewidth=4, label="Ocean surface temperature")
vlines!(axT, tn, linewidth=4, color=(:black, 0.5))
axislegend(axT)

lines!(axQ, times, interior(𝒬ᵛ, 1, 1, 1, 1:Nt),    color=colors[2], label="Latent",    linewidth=2)
lines!(axQ, times, interior(𝒬ᵀ, 1, 1, 1, 1:Nt),    color=colors[3], label="Sensible",  linewidth=2)
lines!(axQ, times, - interior(Qsw, 1, 1, 1, 1:Nt), color=colors[4], label="Shortwave", linewidth=2)
lines!(axQ, times, - interior(Qlw, 1, 1, 1, 1:Nt), color=colors[5], label="Longwave",  linewidth=2)
vlines!(axQ, tn, linewidth=4, color=(:black, 0.5))
axislegend(axQ)

lines!(axF, times, Pt[1:Nt], label="Prescribed freshwater flux")
lines!(axF, times, - interior(Ev, 1, 1, 1, 1:Nt), label="Evaporation")
vlines!(axF, tn, linewidth=4, color=(:black, 0.5))
axislegend(axF)

lines!(axS, times, interior(S, 1, 1, Nz, :))
vlines!(axS, tn, linewidth=4, color=(:black, 0.5))

un  = @lift u[$n]
vn  = @lift v[$n]
Tn  = @lift T[$n]
Sn  = @lift S[$n]
κn  = @lift κ[$n]
en  = @lift e[$n]
N²n = @lift N²[$n]

scatterlines!(axuz, un, label="u")
scatterlines!(axuz, vn, label="v")
scatterlines!(axTz, Tn)
scatterlines!(axSz, Sn)
scatterlines!(axez, en)
scatterlines!(axNz, N²n)
scatterlines!(axκz, κn)

axislegend(axuz)

ulim = max(maximum(abs, u), maximum(abs, v))
xlims!(axuz, -ulim, ulim)

Tmin, Tmax = extrema(T)
xlims!(axTz, Tmin - 0.1, Tmax + 0.1)

Nmax = maximum(N²)
xlims!(axNz, -Nmax/10, Nmax * 1.05)

κmax = maximum(κ)
xlims!(axκz, 1e-9, κmax * 1.1)

emax = maximum(e)
xlims!(axez, 1e-11, emax * 1.1)

Smin, Smax = extrema(S)
xlims!(axSz, Smin - 0.2, Smax + 0.2)

CairoMakie.record(fig, "single_column_profiles.mp4", 1:Nt, framerate=24) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
nothing #hide

# ![](single_column_profiles.mp4)
