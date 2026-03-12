# # Comparing Inpainting Methods
#
# This example compares the two inpainting algorithms available in NumericalEarth
# for filling missing values in ocean climatology data:
#
# 1. **`NearestNeighborInpainting`**: Simple iterative nearest-neighbor averaging.
# 2. **`DiffusiveInpainting`**: MOM6-inspired ICE-9 method — nearest-neighbor fill
#    followed by Laplacian smoothing (∇²ϕ = 0) for a smooth extrapolation.
#
# We use EN4 January 1993 salinity on a 1° grid as the test case.

using NumericalEarth
using NumericalEarth.DataWrangling: NearestNeighborInpainting, DiffusiveInpainting
using Oceananigans
using CairoMakie
using Dates

# ## Build grid and load EN4 salinity

grid = TripolarGrid(size = (360, 160, 50), z = (-5000, 0), halo = (5, 5, 5))
bottom = regrid_bathymetry(grid, major_basins=2, interpolation_passes = 25, minimum_depth = 25)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
date = DateTime(1993, 1, 1)
dataset = EN4Monthly()
metadata = Metadatum(:salinity; date, dataset)

# ## Inpaint with NearestNeighborInpainting

S_nn = CenterField(grid)
set!(S_nn, metadata; inpainting = NearestNeighborInpainting(Inf))

# ## Inpaint with DiffusiveInpainting

S_diff = CenterField(grid)
set!(S_diff, metadata; inpainting = DiffusiveInpainting())

# ## Visualize the comparison at the surface

fig = Figure(size = (1400, 900))

k_surface = size(grid, 3) - 10

Oceananigans.ImmersedBoundaries.mask_immersed_field!(S_nn, NaN)
Oceananigans.ImmersedBoundaries.mask_immersed_field!(S_diff, NaN)

## Surface salinity
s_nn  = interior(S_nn,  :, :, k_surface)
s_diff = interior(S_diff, :, :, k_surface)

ax1 = Axis(fig[1, 1]; title = "NearestNeighborInpainting — Surface", xlabel = "Longitude", ylabel = "Latitude")
ax2 = Axis(fig[1, 2]; title = "DiffusiveInpainting — Surface", xlabel = "Longitude", ylabel = "")
ax3 = Axis(fig[2, 1:2]; title = "Difference (Diffusive − NearestNeighbor) — Surface", xlabel = "Longitude", ylabel = "Latitude")

hm1 = heatmap!(ax1, s_nn;  colormap = :viridis, colorrange = (32, 37))
hm2 = heatmap!(ax2, s_diff; colormap = :viridis, colorrange = (32, 37))
Colorbar(fig[1, 3], hm2; label = "Salinity (PSU)")

diff_surface = s_diff .- s_nn
hm3 = heatmap!(ax3, diff_surface; colormap = :bwr, colorrange = (-0.5, 0.5))
Colorbar(fig[2, 3], hm3; label = "ΔPSU")

save("compare_inpainting_surface.png", fig)
@info "Saved compare_inpainting_surface.png"

# ## Visualize a zonal section at the equator

fig2 = Figure(size = (1400, 600))

i_eq = 130
z = znodes(grid, Center())

s_nn_section  = interior(S_nn,  i_eq, :, :)
s_diff_section = interior(S_diff, i_eq, :, :)

ax4 = Axis(fig2[1, 1]; title = "NearestNeighborInpainting — Equatorial section", xlabel = "Longitude", ylabel = "Depth (m)")
ax5 = Axis(fig2[1, 2]; title = "DiffusiveInpainting — Equatorial section", xlabel = "Longitude", ylabel = "")

hm4 = heatmap!(ax4, 1:180, z, s_nn_section;  colormap = :viridis, colorrange = (34, 36))
hm5 = heatmap!(ax5, 1:180, z, s_diff_section; colormap = :viridis, colorrange = (34, 36))
Colorbar(fig2[1, 3], hm5; label = "Salinity (PSU)")

save("compare_inpainting_section.png", fig2)
@info "Saved compare_inpainting_section.png"

nothing #hide
