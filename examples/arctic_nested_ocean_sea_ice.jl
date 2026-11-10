# 1° regional Arctic ocean + sea ice nested in GLORYS, on a rotated latitude-longitude grid whose
# equator passes through the North Pole. The domain includes Bering Strait and the Atlantic passages
# (Fram Strait, Barents Sea Opening, Denmark Strait, Faroe-Shetland, Davis Strait).
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using CopernicusMarine   # loads the GLORYS download extension
using CUDA
using Dates, Printf

arch = get(ENV, "ARCH", "GPU") == "GPU" ? GPU() : CPU()
years = parse(Float64, get(ENV, "YEARS", "2"))
outdir = get(ENV, "OUTDIR", "arctic_output")
datadir = get(ENV, "DATADIR", "arctic_data")
mkpath(outdir)

start_date = DateTime(2018, 1, 1)
end_date = start_date + Day(round(Int, 365years)) + Day(31)   # bracket the run with the parent's monthly means
dates = (start_date, end_date)

#####
##### Grid: 1°, ±35° about the North Pole
#####

Nz = 40
z = ExponentialDiscretization(Nz, -5000, 0; scale = 1200)
north_pole = (180, 0)   # rotated pole on the geographic equator: the geographic North Pole sits at the box centre
underlying_grid = RotatedLatitudeLongitudeGrid(arch; size = (70, 70, Nz), halo = (7, 7, 7),
                                               longitude = (145, 215), latitude = (-35, 35),
                                               z, north_pole)

bottom_height = regrid_bathymetry(underlying_grid; minimum_depth = 10, major_basins = 1,
                                  interpolation_passes = 10)
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height); active_cells_map = true)

#####
##### Ocean (nested in GLORYS) and sea ice (through the open sides)
#####

parent_dataset = GLORYSMonthly()
ocean = nested_ocean_simulation(grid, parent_dataset; dates, dir = joinpath(datadir, "GLORYS"),
                                relaxation_width = 4, bathymetry_blend_width = 3)
sea_ice = nested_sea_ice_simulation(ocean, parent_dataset; dates, dir = joinpath(datadir, "GLORYS"))

jra55_kw = (dataset = MultiYearJRA55(), start_date, end_date = min(end_date, DateTime(2019, 12, 31, 21)),
            time_indices_in_memory = 40, dir = joinpath(datadir, "JRA55"))
atmosphere = JRA55PrescribedAtmosphere(arch; jra55_kw...)
radiation = JRA55PrescribedRadiation(arch; jra55_kw...)

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt = 10minutes, stop_time = 365days * years)

# Adaptive Δt at advective CFL 0.7
conjure_time_step_wizard!(simulation; cfl = 0.7, max_Δt = 90minutes, max_change = 1.1)

#####
##### Diagnostics and output
#####

child = ocean.model.child
wall = Ref(time_ns())
function progress(sim)
    u, v, w = child.velocities
    T = child.tracers.T
    h = sea_ice.model.ice_thickness
    ℵ = sea_ice.model.ice_concentration
    @info @sprintf("iter %d | t = %s | Δt = %s | max|u,v| = %.2e, %.2e | T ∈ [%.2f, %.2f] | max h = %.2f m | ice area frac %.3f | wall %.1f min",
                   iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), minimum(T), maximum(T),
                   maximum(h), sum(interior(ℵ)) / length(interior(ℵ)), (time_ns() - wall[]) / 60e9)
    return nothing
end
add_callback!(simulation, progress, TimeInterval(10days))

Nz = size(grid, 3)
u, v, w = child.velocities
outputs = (T = view(child.tracers.T, :, :, Nz),
           S = view(child.tracers.S, :, :, Nz),
           u = view(u, :, :, Nz), v = view(v, :, :, Nz),
           η = child.free_surface.displacement,
           h = sea_ice.model.ice_thickness,
           ℵ = sea_ice.model.ice_concentration)

simulation.output_writers[:surface] = JLD2Writer(child, outputs; schedule = TimeInterval(2days),
                                                filename = joinpath(outdir, "arctic_surface"),
                                                overwrite_existing = true)

run!(simulation)
@info "done" time(simulation)
