using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox
using Oceananigans
using ArchGDAL      # activates the ASTER GED extension
using CairoMakie

dataset = ASTERGEDv3(resolution = ASTERGEDHigh100m)

function emissivity_field(dataset, name, longitude, latitude; size = (512, 512))
    grid = LatitudeLongitudeGrid(CPU(); size, longitude, latitude,
                                 topology = (Bounded, Bounded, Flat))
    region = BoundingBox(; longitude, latitude)
    return Field(Metadatum(name; dataset, region), grid)
end

gc_emissivity  = emissivity_field(dataset, :emissivity,             (-112.8, -111.2), (35.2, 36.8))
gc_uncertainty = emissivity_field(dataset, :emissivity_uncertainty, (-112.8, -111.2), (35.2, 36.8))
congo_emissivity = emissivity_field(dataset, :emissivity, (18, 20), (-1, 1))

fig = Figure(size = (1650, 520), backgroundcolor = :white)
Label(fig[0, 1:6], "ASTER GED v3 broadband emissivity (100 m)"; fontsize = 19, font = :bold)

ax1 = Axis(fig[1, 1]; title = "Emissivity — Grand Canyon",
           xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = 1)
hm1 = heatmap!(ax1, gc_emissivity; colormap = :viridis, colorrange = (0.90, 0.98))
Colorbar(fig[1, 2], hm1; label = "broadband ε")

ax2 = Axis(fig[1, 3]; title = "Uncertainty σ(ε)",
           xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = 1)
hm2 = heatmap!(ax2, gc_uncertainty; colormap = :viridis, colorrange = (0, 0.02))
Colorbar(fig[1, 4], hm2; label = "broadband σ(ε)")

ax3 = Axis(fig[1, 5]; title = "Emissivity — Congo basin (cloud gaps inpainted)",
           xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = 1)
hm3 = heatmap!(ax3, congo_emissivity; colormap = :viridis, colorrange = (0.90, 0.98))
Colorbar(fig[1, 6], hm3; label = "broadband ε")

save("asterged_emissivity_map_100m.png", fig)
