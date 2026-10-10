# # Mixed layer depth from ECCO
#
# We compute the mixed layer depth from ECCO temperature and salinity, month by month,
# and animate its evolution over ten years.

using NumericalEarth
using NumericalEarth.Diagnostics: MixedLayerDepthField
using Oceananigans
using Oceananigans.Models: buoyancy_operation
using CairoMakie
using Dates

using SeawaterPolynomials: TEOS10EquationOfState

arch = CPU()
Nx = 360
Ny = 160

dataset = ECCO4Monthly()
z = NumericalEarth.DataWrangling.z_interfaces(dataset)
z = z[20:end]
Nz = length(z) - 1

grid = LatitudeLongitudeGrid(arch; z,
                             size = (Nx, Ny, Nz),
                             latitude  = (-80, 80),
                             longitude = (0, 360))

bottom_height = regrid_bathymetry(grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

dates = (DateTime(1993, 1, 1), DateTime(2003, 1, 1))
temperature_metadata = Metadata(:temperature; dataset, dates)
salinity_metadata    = Metadata(:salinity;    dataset, dates)

T_ts = FieldTimeSeries(temperature_metadata, grid; time_indices_in_memory=2)
S_ts = FieldTimeSeries(salinity_metadata,    grid; time_indices_in_memory=2)
h_ts = FieldTimeSeries{Center, Center, Nothing}(grid, T_ts.times)

# The mixed layer depth `h` is diagnosed from the buoyancy of the temperature
# and salinity fields `T` and `S`, which we update every month.

T = CenterField(grid)
S = CenterField(grid)

buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
h = MixedLayerDepthField(buoyancy, grid, (; T, S))

Nt = length(h_ts)

for n = 1:Nt-1
    set!(T, T_ts[n])
    set!(S, S_ts[n])
    compute!(h)
    set!(h_ts, h, n)
end

# We plot the mixed layer depth,

fig = Figure(size=(1500, 800))
ax = Axis(fig[2, 1], xlabel="Longitude", ylabel="Latitude")
n = Observable(1)

title = @lift "ECCO mixed layer depth in " * Dates.format(temperature_metadata.dates[$n], "U yyyy")
Label(fig[1, 1], title, tellwidth=false)

hn = @lift h_ts[$n]
hm = heatmap!(ax, hn, colorrange=(0, 500), colormap=:magma, nan_color=:lightgray)
Colorbar(fig[2, 2], hm, label="Mixed layer depth (m)")

fig

# and record a movie.

CairoMakie.record(fig, "ecco_mld.mp4", 1:Nt-1, framerate=4) do nn
    n[] = nn
end
nothing #hide

# ![](ecco_mld.mp4)
