# # ASTER GED surface emissivity
#
# We map the ASTER GED v3 broadband emissivity and its uncertainty at 100 m over the
# Grand Canyon, and the emissivity over the Congo basin, where cloud gaps are inpainted.

using NumericalEarth
using Oceananigans
using ArchGDAL # activates the ASTER GED extension
using CairoMakie

dataset = ASTERGEDv3(resolution = ASTERGEDHigh100m)

function emissivity_field(dataset, name; longitude, latitude, size = (512, 512))
    grid = LatitudeLongitudeGrid(CPU(); size, longitude, latitude,
                                 topology = (Bounded, Bounded, Flat))
    region = BoundingBox(; longitude, latitude)
    return Field(Metadatum(name; dataset, region), grid)
end

grand_canyon = (longitude = (-112.8, -111.2), latitude = (35.2, 36.8))
congo_basin  = (longitude = (18, 20), latitude = (-1, 1))

grand_canyon_emissivity  = emissivity_field(dataset, :emissivity; grand_canyon...)
grand_canyon_uncertainty = emissivity_field(dataset, :emissivity_uncertainty; grand_canyon...)
congo_basin_emissivity   = emissivity_field(dataset, :emissivity; congo_basin...)

fig = Figure(size = (1650, 520), backgroundcolor = :white)
Label(fig[0, 1:6], "ASTER GED v3 broadband emissivity (100 m)"; fontsize = 19, font = :bold)

ax1 = Axis(fig[1, 1]; title = "Emissivity ε, Grand Canyon",
           xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = 1)
hm1 = heatmap!(ax1, grand_canyon_emissivity; colormap = :viridis, colorrange = (0.90, 0.98))
Colorbar(fig[1, 2], hm1; label = "ε")

ax2 = Axis(fig[1, 3]; title = "Emissivity uncertainty σ(ε), Grand Canyon",
           xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = 1)
hm2 = heatmap!(ax2, grand_canyon_uncertainty; colormap = :viridis, colorrange = (0, 0.02))
Colorbar(fig[1, 4], hm2; label = "σ(ε)")

ax3 = Axis(fig[1, 5]; title = "Emissivity ε, Congo basin (cloud gaps inpainted)",
           xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = 1)
hm3 = heatmap!(ax3, congo_basin_emissivity; colormap = :viridis, colorrange = (0.90, 0.98))
Colorbar(fig[1, 6], hm3; label = "ε")

save("asterged_emissivity_map_100m.png", fig)
