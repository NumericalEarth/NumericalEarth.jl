# # Inspecting ECCO temperature and salinity
#
# An interactive figure that shows ECCO temperature, salinity, and the buoyancy
# difference between the surface and a chosen depth. Sliders select the month and the depth.

using NumericalEarth
using Oceananigans
using Oceananigans.Models: buoyancy_operation
using CairoMakie
using Dates

using SeawaterPolynomials: TEOS10EquationOfState

arch = CPU()
Nx = 360 ÷ 4
Ny = 160 ÷ 4

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

dates = (DateTime(1993, 1, 1), DateTime(1999, 1, 1))
temperature_metadata = Metadata(:temperature; dataset, dates)
salinity_metadata    = Metadata(:salinity;    dataset, dates)
Nt = length(temperature_metadata)

T_ts = FieldTimeSeries(temperature_metadata, grid; time_indices_in_memory=Nt)
S_ts = FieldTimeSeries(salinity_metadata,    grid; time_indices_in_memory=Nt)

# The buoyancy `b` is computed from the temperature and salinity fields `T` and `S`,
# which we set to the selected month. `Δb` is the difference between the surface
# buoyancy `bₛ` and the buoyancy at every depth.

T = CenterField(grid)
S = CenterField(grid)

buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
b = Field(buoyancy_operation(buoyancy, grid, (; T, S)))
bₛ = Field{Center, Center, Nothing}(grid)
Δb = Field(bₛ - b)

fig = Figure(size=(900, 1050))

axT = Axis(fig[1, 1])
axS = Axis(fig[2, 1])
axb = Axis(fig[3, 1])

kslider = Slider(fig[1:3, 0], range=1:Nz, startvalue=Nz, horizontal=false)
nslider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
k = kslider.value
n = nslider.value

Tk = @lift view(T_ts[$n], :, :, $k)
Sk = @lift view(S_ts[$n], :, :, $k)

Δbk = @lift begin
    set!(T, T_ts[$n])
    set!(S, S_ts[$n])
    compute!(b)
    set!(bₛ, view(b, :, :, Nz))
    view(Δb, :, :, $k)
end

hmT = heatmap!(axT, Tk, nan_color=:lightgray, colorrange=(-2, 30), colormap=:thermal)
hmS = heatmap!(axS, Sk, nan_color=:lightgray, colorrange=(31, 37), colormap=:haline)
hmb = heatmap!(axb, Δbk, nan_color=:lightgray, colorrange=(0, 1e-3), colormap=:magma)

Colorbar(fig[1, 2], hmT, label="Temperature (ᵒC)")
Colorbar(fig[2, 2], hmS, label="Salinity (g kg⁻¹)")
Colorbar(fig[3, 2], hmb, label="Buoyancy difference (m s⁻²)")

fig
