# OpenLandMap-soilDB: 30 m soil texture & bulk density

[`OpenLandMapSoilDB`](@ref) provides global soil properties at **30 m** resolution
predicted by spatiotemporal machine learning
([Hengl et al., 2026, *ESSD* 18:989](https://doi.org/10.5194/essd-18-989-2026),
CC-BY 4.0). It delivers the *inputs* a pedotransfer function needs to derive the
soil-hydraulic closure of [`VariablySaturatedHydrology`](@ref) — texture mass
fractions (sand, silt, clay) and fine-earth bulk density — over three depth
intervals (0–30, 30–60, 60–100 cm).

At 30 m it resolves the surface heterogeneity a ~100 m atmospheric LES actually
sees, unlike the 250 m [`SoilGrids2`](@ref) sibling. The data are plain geographic
(EPSG:4326), so no reprojection is needed; the global grid is ~1.44M × 528k cells,
so it is read in regional windows via a bounding box. Reading the
cloud-optimized GeoTIFFs is anonymous (no credentials) and requires ArchGDAL:

```@example openlandmap
using NumericalEarth
using Oceananigans
using Oceananigans.Grids: λnodes, φnodes
using ArchGDAL          # activates the windowed cloud-optimized-GeoTIFF reader
using CairoMakie

# A texturally heterogeneous test window over the Grand Canyon.
region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))

clay = Field(Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region), CPU())
nothing # hide
```

The result is a three-dimensional `Field`: the horizontal window on the native
30 m grid, with the three depth intervals stacked on the vertical axis (deepest
first, increasing upward). Masked cells — permanent ice, sand deserts, water, and
the incised canyon drainage here — carry `NaN`.

## Multi-variable map (topsoil, 0–30 cm)

```@example openlandmap
variables = (:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)
titles = Dict(:sand_fraction => "sand fraction", :silt_fraction => "silt fraction",
              :clay_fraction => "clay fraction", :bulk_density => "bulk density [kg m⁻³]")

fields = Dict(v => Field(Metadatum(v; dataset = OpenLandMapSoilDB(), region), CPU())
              for v in variables)

grid = clay.grid
λ = Array(λnodes(grid, Center()))
φ = Array(φnodes(grid, Center()))
topsoil = size(interior(clay), 3)   # 0–30 cm is the top (last) layer

fig = Figure(size = (1000, 850))
for (n, v) in enumerate(variables)
    i, j = fldmod1(n, 2)
    col = 2j - 1
    ax = Axis(fig[i, col]; title = titles[v], xlabel = "lon", ylabel = "lat", aspect = DataAspect())
    data = Array(interior(fields[v]))[:, :, topsoil]
    hm = heatmap!(ax, λ, φ, data; colormap = v == :bulk_density ? :viridis : :turbo)
    Colorbar(fig[i, col + 1], hm)
end
fig
```

The texture fractions are physically consistent — sand and silt are anti-correlated,
and `sand + silt + clay ≈ 1` per pixel:

```@example openlandmap
s  = Array(interior(fields[:sand_fraction]))[:, :, topsoil]
si = Array(interior(fields[:silt_fraction]))[:, :, topsoil]
c  = Array(interior(fields[:clay_fraction]))[:, :, topsoil]
total = filter(!isnan, s .+ si .+ c)
extrema(total)
```

## Texture across depth

Clay increases with depth here (illuviation), while the topsoil is sandier:

```@example openlandmap
clay3 = Array(interior(clay))
depth_labels = ("60–100 cm", "30–60 cm", "0–30 cm")
crange = extrema(filter(!isnan, clay3))

figd = Figure(size = (1200, 430))
for k in 1:size(clay3, 3)
    ax = Axis(figd[1, k]; title = "clay, $(depth_labels[k])", xlabel = "lon",
              ylabel = k == 1 ? "lat" : "", aspect = DataAspect())
    hm = heatmap!(ax, λ, φ, clay3[:, :, k]; colormap = :turbo, colorrange = crange)
    k == size(clay3, 3) && Colorbar(figd[1, k + 1], hm)
end
figd
```

## Resolution: native 30 m vs a ~100 m LES grid

Interpolating the metadatum onto a coarser target grid is a mild downsample that
preserves the covariate-driven pattern at the LES scale:

```@example openlandmap
les = LatitudeLongitudeGrid(CPU(); size = (400, 400, 3),
                            longitude = region.longitude, latitude = region.latitude,
                            z = [-1.0, -0.6, -0.3, 0.0], halo = (3, 3, 3))
clay_les = Field(Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region), les)
λl = Array(λnodes(les, Center()))
φl = Array(φnodes(les, Center()))

figr = Figure(size = (1000, 430))
ax1 = Axis(figr[1, 1]; title = "native 30 m", xlabel = "lon", ylabel = "lat", aspect = DataAspect())
hm1 = heatmap!(ax1, λ, φ, clay3[:, :, topsoil]; colormap = :turbo, colorrange = crange)
Colorbar(figr[1, 2], hm1)
ax2 = Axis(figr[1, 3]; title = "≈100 m LES grid", xlabel = "lon", aspect = DataAspect())
hm2 = heatmap!(ax2, λl, φl, Array(interior(clay_les))[:, :, topsoil]; colormap = :turbo, colorrange = crange)
Colorbar(figr[1, 4], hm2)
figr
```

!!! note "Pedotransfer function deferred"
    This page delivers the raw 30 m soil fields only. Converting texture and bulk
    density into the van Genuchten parameters (`ν`, `θʳ`, `Kₛ`, `α`, `n`) that
    [`VariablySaturatedHydrology`](@ref) consumes requires a pedotransfer function
    (Rosetta / Cosby), which is a separate step.
