# # Real ETH canopy in the DSM bare-earth workflow — central Amazon
#
# This is the `bare_earth_terrain.jl` scene (Rio Negro–Solimões confluence near Manaus),
# but with the **measured ETH canopy height** in place of the synthetic elevation-gated
# canopy that example stands in with. The same canopy field then feeds two consumers:
#
#   1. **Bare-earth terrain** — `bare_earth_elevation(z_DSM, h_c)` = `max(z_DSM − h_c, 0)`.
#   2. **Aerodynamic roughness** — `compute_canopy_roughness!` (Raupach drag partition).
#
# The canopy the DSM overstates the ground by is exactly the canopy that sets the surface
# roughness — the terrain subtraction and the roughness closure share one measured field.
#
# DSM stand-in: ETOPO 2022 (token-free; `GLO30()` is the commercial-use 30 m surface model).
# The ETH product is 10 m; here it is aggregated to the ~1 km land grid on read (its COG
# overviews make that cheap). Needs `using ArchGDAL` and `using CairoMakie`.

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.DataWrangling.CanopyHeight: eth_tile_urls, ETH_BROWSER_USER_AGENT, ETH_LIBDRIVE_TOKEN
using NumericalEarth.DataWrangling.CanopyRoughness: compute_canopy_roughness!
using Oceananigans
using Oceananigans.Fields: set!, interior
using ArchGDAL
using ArchGDAL.GDAL: cplsetconfigoption
using CairoMakie

# ## Domain and DSM (identical to `bare_earth_terrain.jl`)
latitude  = -3.5, -2.4
longitude = -60.5, -59.0
grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (1500, 1100),
                             topology = (Bounded, Bounded, Flat))   # ~110 m across the basin
region = BoundingBox(; longitude, latitude)

z_dsm = regrid_topography(grid; dataset = ETOPO2022())

# ## Real canopy height on the model grid
#
# Aggregate the ETH 10 m tiles straight onto the ~1 km grid with an area-weighted mean
# (`-r average`), skipping the no-data byte. Reading through the COG overviews keeps this to
# a few seconds even across four 3° tiles. (In the library this belongs behind a grid-aware
# `Field(metadatum, grid)`; inlined here so the demo is self-contained.)
function eth_canopy_on_grid(grid, region)
    ext = Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt)
    ext.configure_vsicurl!()
    cplsetconfigoption("GDAL_HTTP_USERAGENT", ETH_BROWSER_USER_AGENT)
    cplsetconfigoption("GDAL_HTTP_USERPWD", ETH_LIBDRIVE_TOKEN * ":")
    cplsetconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    Δλ = (λ₂ - λ₁) / size(grid, 1)
    Δφ = (φ₂ - φ₁) / size(grid, 2)
    datasets = [ArchGDAL.read(s) for s in eth_tile_urls(region, :canopy_height)]
    raw = try
        ArchGDAL.gdalwarp(datasets,
            ["-t_srs", "EPSG:4326", "-te", string(λ₁), string(φ₁), string(λ₂), string(φ₂),
             "-tr", string(Δλ), string(Δφ), "-r", "average", "-srcnodata", "255", "-ot", "Float32"]) do w
            reverse(Float32.(ArchGDAL.read(w, 1)), dims = 2)
        end
    finally
        for d in datasets; ArchGDAL.destroy(d); end
    end

    h = Field{Center, Center, Nothing}(grid)
    interior(h, :, :, 1) .= ifelse.(raw .== 255, NaN32, raw)   # no-data byte → NaN
    return h
end

canopy_height = eth_canopy_on_grid(grid, region)

# ## (1) Bare-earth terrain — DSM minus the measured canopy
z_bare  = bare_earth_elevation(z_dsm, canopy_height)
removed = compute!(Field(z_dsm - z_bare))      # the canopy lift removed from the terrain

# ## (2) Roughness — the same canopy through the Raupach closure
# The basin is evergreen broadleaf forest (IGBP 2); a uniform dense canopy (LAI 5) here.
lai = Field{Center, Center, Nothing}(grid); set!(lai, 5)
land_cover = Field{Center, Center, Nothing}(grid); set!(land_cover, 2)
z0, d0 = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0, d0, lai, land_cover, canopy_height, grid)

# ## Figures
outdir = joinpath(@__DIR__, "eth_canopy_bare_earth_figures"); mkpath(outdir)
finite(f) = filter(isfinite, vec(Array(interior(f))))
zmax = maximum(finite(z_dsm))

function heat!(fig, pos, title, field, crange, cmap)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, field; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; width = 11)
end

fig = Figure(size = (1500, 950))
Label(fig[0, 1:4], "Real ETH canopy in the DSM bare-earth workflow — Rio Negro–Solimões, Amazon"; fontsize = 19, font = :bold)
heat!(fig, (1, 1), "DSM elevation (ETOPO stand-in)", z_dsm,         (0, zmax), :terrain)
heat!(fig, (1, 3), "canopy height h_c (ETH)",        canopy_height, (0, maximum(finite(canopy_height))), :YlGn)
heat!(fig, (2, 1), "bare-earth DTM = DSM − canopy",  z_bare,        (0, zmax), :terrain)
heat!(fig, (2, 3), "z₀ roughness from the same h_c", z0,            (0, maximum(finite(z0))), :viridis)
save(joinpath(outdir, "fig1_canopy_dsm_roughness.png"), fig)

# A west–east transect at 3.0°S: the DSM rides a canopy height above the bare-earth line.
jrow = size(grid, 2) ÷ 2
x = 1:size(grid, 1)
fig = Figure(size = (1200, 440))
ax = Axis(fig[1, 1]; xlabel = "west → east (grid cells)", ylabel = "elevation (m)",
          title = "DSM vs bare-earth along 3°S: the gap is the ETH canopy")
lines!(ax, x, Array(interior(z_dsm,  :, jrow, 1)); linewidth = 2, label = "DSM")
lines!(ax, x, Array(interior(z_bare, :, jrow, 1)); linewidth = 2, label = "bare-earth")
lines!(ax, x, Array(interior(removed, :, jrow, 1)); linewidth = 2, color = :seagreen, label = "removed canopy")
axislegend(ax; position = :rt)
save(joinpath(outdir, "fig2_transect.png"), fig)

@info "canopy + DSM (PR #465 region)" h_c_max = round(maximum(finite(canopy_height)), digits = 1) dsm_range = round.((minimum(finite(z_dsm)), zmax), digits = 1) removed_canopy_max = round(maximum(finite(removed)), digits = 1) z0_mean = round(sum(finite(z0)) / length(finite(z0)), digits = 2)
