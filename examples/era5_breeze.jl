# # ERA5 downscaling with Breeze and NestedSimulation
#
# This is a limited-area model (LAM) example that couples the Breeze
# compressible solver to forthcoming SlabLand and SlabOcean components.
#
# It downloads ERA5 reanalysis restricted to a bounding box, regrids it onto a
# terrain-following `LatitudeLongitudeGrid` sized for ~9 km horizontal cells (1/12°,
# exactly ERA5's 0.25° / 3) at the domain center latitude, builds a compressible Breeze
# atmosphere, and drives it through a `NestedSimulation`: an ERA5-forced parent supplies
# the lateral boundary conditions and an interior Davies relaxation fringe.
#
# In progress:
# - [x] Breeze model construction
# - [x] initial state setting (set! the model from ingested fields)
# - [x] open boundary conditions (parent-driven OBC + Davies fringe relaxation)
# - [x] test with GPU
# - [x] terrain (ETOPO 2022 + `TerrainFollowingVerticalDiscretization`)
# - [ ] dynamical initialization
# - [x] acoustic substepping (Breeze split-explicit, PR #712)
# - [ ] land/ocean coupling

using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.ERA5
using CDSAPI  # activates NumericalEarthCDSAPIExt
using Oceananigans
using Oceananigans: location
using Oceananigans.Fields: interpolate!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: znode
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Coriolis: SphericalCoriolis
using Breeze
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization, materialize_terrain!
using Statistics: mean
using Dates
using Printf

# Set `ARCH=GPU` in the environment to run on CUDA.
if get(ENV, "ARCH", "CPU") == "GPU"
    using CUDA
    const arch = GPU(CUDA.CUDABackend(always_inline = true))
else
    const arch = CPU()
end

# ## Configuration

# ### Domain
#
# Domain centered on the U.S. Department of Energy's Atmospheric Radiation
# Measurement (ARM) Climate Research Facility's Southern Great Plains (SGP)
# site in Lamont, OK. We match the 9 km middle domain (Domain 2) of the WRF
# nest used by [Fan2017](@citet) for this MC3E case: a 27 → 9 → 3 km telescoping
# nest. Starting at D2 sets the telescope up cleanly — ERA5's native 0.25° step
# divides exactly by 3 to 1/12° here, and again by 3 to 1/36° (~3 km) for the
# Domain 3 child we nest down to next via a `NestedSimulation`.
#
# [Fan2017](@citet)'s Domain 2 carries 181 × 166 WRF grid points. Those count staggered
# (cell-edge) locations, so they map to 180 × 165 Breeze *cells* (cells = points − 1).
# We use a uniform 1/12° angular step (ERA5 0.25° / 3), so the physical cells are ~9 km —
# anisotropic at this latitude, using R = 6,371 km:
#   Δx = R·cos(φ₀)·Δλ ≈ 7.44 km
#   Δy = R·Δφ         ≈ 9.27 km

φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)

Δλ = Δφ = 1/12              # uniform 1/12° step (ERA5 0.25° / 3 — clean 3:1 telescoping)
Nx, Ny = 180, 165           # Fan et al. (2017) Domain 2: 181 × 166 points − 1

# From these inputs, we determine the `BoundingBox` corners.

λ_west  = λ₀ - Nx * Δλ / 2
λ_east  = λ₀ + Nx * Δλ / 2
φ_south = φ₀ - Ny * Δφ / 2
φ_north = φ₀ + Ny * Δφ / 2

# Vertical grid matched to [Fan2017](@citet)'s WRF nest: 51 staggered levels → `Nz = 50`
# cells, a constant 60 m surface cell, and a 490 m maximum spacing. Fan publishes only
# those three numbers (60 m near-surface, 490 m max, 51 levels) — no stretching ratio and
# no model top, since WRF uses a terrain-following hydrostatic-pressure (η) coordinate. We
# realize them with a 1.15× geometric stretch (`extent = 19525`), which lands the top at
# Lz ≈ 20 km (~50 hPa, WRF's usual model top) — above the ~16 km jet, so the rigid lid and
# the Rayleigh sponge sit in the quiescent lower stratosphere. 25 of the 50 cells ride the
# 490 m cap; the 60 m surface layer that resolves the boundary layer is unchanged.

z_discretization = ReferenceToStretchedDiscretization(
    extent                  = 19525.0,
    bias                    = :left,
    bias_edge               = 0.0,
    constant_spacing        = 60.0,
    constant_spacing_extent = 60.0,
    maximum_spacing         = 490.0,
    stretching              = LinearStretching(0.15))

Nz = length(z_discretization)
@assert Nz == 50  # Fan et al. (2017): 51 staggered levels → 50 cells; 60 m → 490 m cap, top ~20 km

# ### Initial conditions
#
# We target the 20 May 2011 squall-line MCS from the Midlatitude Continental
# Convective Clouds Experiment (MC3E) at the ARM SGP site, the case studied by
# [Fan2017](@citet). A NE–SW oriented quasi-linear mesoscale convective system
# developed over the Southern Great Plains overnight, peaking in size around
# 1100 UTC with leading deep convection and trailing stratiform precipitation.
# Following the paper, we initialize at 0000 UTC and force for 18 h, spanning
# the convective development (~0600–1000 UTC) and the mature line's passage
# over SGP.

start_date = DateTime(2011, 05, 20, 0)  # 7 pm LT (previous day)
end_date   = DateTime(2011, 05, 20, 18) # 1 pm LT

dates = start_date:Hour(1):end_date

# ### ERA5 reanalysis
#
# Pressure-level variables are regridded onto the parent grid as `FieldTimeSeries`
# (and onto the child grid for the initial condition) further below.

era5_datadir = "era5"   # Where data will be saved locally

# ERA5 forcing region: the LAM footprint padded outward by `era5_pad` and snapped to
# ERA5's native 0.25° grid, so the parent strictly encloses the 1/12° child (the
# Interpolated lateral BCs and the 5-cell Davies fringe need parent data beyond the child
# edge). At 0.25° (~28 km here) ERA5 stands in for Fan's 27 km Domain 1, completing the
# telescope ERA5 → D2 (1/12°) → D3 (1/36°, the next nest-down).

era5_pad = 1.0   # deg; wider than the 5·(1/12°) ≈ 0.42° Davies fringe

snap_out(lo, hi; d = 0.25) = (floor(lo / d) * d, ceil(hi / d) * d)
era5_region = BoundingBox(longitude = snap_out(λ_west - era5_pad, λ_east + era5_pad),
                          latitude  = snap_out(φ_south - era5_pad, φ_north + era5_pad))

@info @sprintf("D2 (9 km LAM): λ ∈ [%.3f, %.3f], φ ∈ [%.3f, %.3f]; Δλ=Δφ=%.4f°",
               λ_west, λ_east, φ_south, φ_north, Δλ)
@info @sprintf("ERA5 parent (D1 role, padded + snapped to 0.25°): λ ∈ [%.2f, %.2f], φ ∈ [%.2f, %.2f]",
               era5_region.longitude[1], era5_region.longitude[2],
               era5_region.latitude[1],  era5_region.latitude[2])

# We use hourly dataset on both single levels and pressure levels.

ds_pl = ERA5HourlyPressureLevels()
ds_sl = ERA5HourlySingleLevel()

# ## Setup LAM grid
#
# Terrain-following `LatitudeLongitudeGrid` with `Bounded` horizontal topologies
# (LAM-style). The vertical coordinate is a `TerrainFollowingVerticalDiscretization`
# built from the stretched reference profile; `materialize_terrain!` (below) fills its
# terrain components from the ETOPO 2022 surface elevation, deforming the coordinate
# surfaces to follow the ground (a Gal-Chen–Somerville σ coordinate via the default
# `LinearDecay` formulation). The bottom surface sits at the local terrain height;
# the top stays flat. `znode` heights are true heights above sea level — the
# coordinate the #241 ERA5 ingest below interpolates onto.

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z         = TerrainFollowingVerticalDiscretization(z_discretization),
                             size      = (Nx, Ny, Nz),
                             halo      = (5, 5, 5),
                             topology  = (Bounded, Bounded, Bounded))

# ETOPO 2022 surface elevation (≥ 0; ocean clamped to sea level) regridded onto
# the LAM horizontal grid — ETOPO's 60″ (~1.85 km) relief is finer than the ~9 km
# cells. `materialize_terrain!` fills the grid's terrain-following coordinate from it
# in place; `CompressibleDynamics` then builds the slope metrics it needs directly
# from the grid (no `terrain_metrics` argument required).

elevation = regrid_topography(grid; dataset = ETOPO2022())

# Terrain taper across the lateral relaxation fringe. The lateral BCs feed the child the
# smooth ERA5 parent state, which assumes the surface sits at the parent orography; at the
# west inflow edge the child ETOPO is up to +713 m above it, so the boundary-supplied
# hydrostatic pressure is inconsistent with the child surface and discharges as a spurious
# near-surface horizontal pressure-gradient force (the cold-start blow-up). Blend ETOPO →
# parent orography over the first `N_taper` cells of every lateral edge so the boundary
# terrain matches the parent (weight 0) and ramps to full ETOPO by the inner fringe edge.
g_accel = Oceananigans.defaults.gravitational_acceleration
orography_grid = LatitudeLongitudeGrid(longitude = (λ_west,  λ_east),
                                       latitude  = (φ_south, φ_north),
                                       z = (0, 1), size = (Nx, Ny, 1),
                                       halo = (5, 5, 3), topology = (Bounded, Bounded, Bounded))
Φ_sfc = CenterField(orography_grid)
set!(Φ_sfc, Metadatum(:geopotential; dataset = ds_sl, date = start_date,
                      region = era5_region, dir = era5_datadir))
era5_orography = Array(interior(Φ_sfc))[:, :, 1] ./ g_accel

N_taper = 5
etopo_full = Array(interior(elevation))[:, :, 1]
blended = similar(etopo_full)
for j in 1:Ny, i in 1:Nx
    weight = clamp(min(i - 1, Nx - i, j - 1, Ny - j) / N_taper, 0, 1)
    blended[i, j] = weight * etopo_full[i, j] + (1 - weight) * era5_orography[i, j]
end
set!(elevation, reshape(blended, size(interior(elevation))))

materialize_terrain!(grid, elevation)

# Outer time step — defined here so the Davies relaxation timescale below can be tied to
# it. Split-explicit substepping integrates the acoustic modes, so Δt is set by advection,
# vertical-binding on the 60 m surface cells (C_z = w·Δt/Δz).
U_horizontal = 60   # m/s — bounds the jet
W_vertical   = 25   # m/s — bounds convective updrafts
Δt = 0.5 * min(min(minimum_xspacing(grid), minimum_yspacing(grid)) / U_horizontal,
               minimum_zspacing(grid) / W_vertical)

# ## Nested domains
#
# Visualize the nesting before stepping the model: the ERA5 forcing region that
# supplies the parent state (lateral BCs + Davies fringe) and the 9 km LAM —
# Fan (2017)'s Domain 2, the `NestedSimulation` child — over ETOPO terrain with
# Natural Earth state/country boundaries, centered on ARM SGP. Drawn here (not
# with the profile plots below) so the domain geometry is written even if the run
# is cut short.

using CairoMakie
using NaturalEarth
import GeoInterface as GI

# Flatten a Natural Earth (Multi)LineString feature collection to NaN-separated
# lon/lat vectors that CairoMakie's `lines!` draws as disjoint border segments.
append_border!(lons, lats, geom) = append_border!(lons, lats, GI.geomtrait(geom), geom)
function append_border!(lons, lats, ::GI.LineStringTrait, line)
    for p in GI.getpoint(line)
        push!(lons, GI.x(p)); push!(lats, GI.y(p))
    end
    push!(lons, NaN); push!(lats, NaN)
end
append_border!(lons, lats, ::GI.MultiLineStringTrait, multiline) =
    foreach(line -> append_border!(lons, lats, line), GI.getgeom(multiline))
append_border!(lons, lats, ::Any, geom) = nothing

function natural_earth_lines(name)
    lons, lats = Float64[], Float64[]
    for feature in naturalearth(name, 50)
        geom = GI.geometry(feature)
        isnothing(geom) || append_border!(lons, lats, geom)
    end
    return lons, lats
end

# A 2.5° buffer around the ERA5 box leaves the nest well inside the map edge;
# the basemap grid samples ETOPO at ~0.03° (≈ 3 km).
map_buffer = 2.5
map_lon = (era5_region.longitude[1] - map_buffer, era5_region.longitude[2] + map_buffer)
map_lat = (era5_region.latitude[1]  - map_buffer, era5_region.latitude[2]  + map_buffer)
map_grid = LatitudeLongitudeGrid(CPU();
                                 longitude = map_lon, latitude = map_lat,
                                 z         = (0, 1),
                                 size      = (round(Int, (map_lon[2] - map_lon[1]) / 0.03),
                                              round(Int, (map_lat[2] - map_lat[1]) / 0.03), 1),
                                 topology  = (Bounded, Bounded, Bounded))
# Full ETOPO relief (negative over ocean) for the basemap, so the map shows true
# bathymetry as well as topography. The land–sea mask is just its sign — `regrid_topography`
# (used above for the model's terrain) clamps the ocean to 0 and loses it. The mask is what
# a SlabLand/ocean surface-BC split would key on; here only the Gulf corner of D2 is ocean.
map_bathymetry = regrid_bathymetry(map_grid; dataset = ETOPO2022())
relief   = Array(interior(map_bathymetry))[:, :, 1]   # m; negative over ocean
is_ocean = relief .< 0                                # land–sea mask (true = ocean)

# Closed rectangle path from (λ, φ) bounds.
domain_box(λ₁, λ₂, φ₁, φ₂) = ([λ₁, λ₂, λ₂, λ₁, λ₁], [φ₁, φ₁, φ₂, φ₂, φ₁])

fig_map = Figure(size = (840, 760), fontsize = 13)
ax_map  = Axis(fig_map[1, 1]; xlabel = "longitude (°)", ylabel = "latitude (°)",
               title  = "ERA5 → 9 km LAM nest (MC3E squall line, ARM SGP)",
               aspect = DataAspect())

# Two-sided normalization onto `:topo`: the full bathymetry range fills the lower (blue)
# half and the full land range the upper (green→yellow→brown→white) half, with z=0 pinned
# to the colormap's sea-level break (0.5). Bake it into a custom colormap so a *linear*
# colorrange keeps the colorbar in physical metres. (Assumes the domain straddles sea level.)
zmin, zmax = extrema(relief)
g0   = -zmin / (zmax - zmin)                 # fraction of the linear range at z = 0
topo = cgrad(:topo)
remap(g) = g ≤ g0 ? 0.5 * (g / g0) : 0.5 + 0.5 * (g - g0) / (1 - g0)
topo_centered = [topo[remap(g)] for g in range(0, 1; length = 512)]

hm_map = heatmap!(ax_map,
                  collect(λnodes(map_grid, Center(), Center(), Center())),
                  collect(φnodes(map_grid, Center(), Center(), Center())),
                  relief; colormap = topo_centered, colorrange = (zmin, zmax))
Colorbar(fig_map[1, 2], hm_map; label = "elevation / depth (m)")

# US state lines and country borders (the topo/bathy coloring renders the coastline itself).
for (name, color, linewidth) in (("admin_1_states_provinces_lines", (:gray20, 0.55), 0.7),
                                 ("admin_0_boundary_lines_land",     (:black,  0.75), 1.4))
    lon, lat = natural_earth_lines(name)
    lines!(ax_map, lon, lat; color, linewidth)
end

lines!(ax_map, domain_box(era5_region.longitude..., era5_region.latitude...)...;
       color = :dodgerblue, linewidth = 3, label = "ERA5 parent — Fan Domain 1 role")
lines!(ax_map, domain_box(λ_west, λ_east, φ_south, φ_north)...;
       color = :crimson, linewidth = 3, label = "9 km LAM — Fan Domain 2 (child)")
scatter!(ax_map, [λ₀], [φ₀]; color = :black, marker = :star5, markersize = 18, label = "ARM SGP")

axislegend(ax_map; position = :rt, framevisible = true, backgroundcolor = (:white, 0.85))

# Clip to the map region — the Natural Earth lines span the globe.
xlims!(ax_map, map_lon...)
ylims!(ax_map, map_lat...)

save("era5_breeze_domains.png", fig_map)
@info "Wrote era5_breeze_domains.png"

# ## Thermodynamic constants
#
# All thermodynamic parameters used downstream (moist gas law, liquid-ice
# potential temperature, virtual temperature) come from Breeze's
# `ThermodynamicConstants`.

constants = ThermodynamicConstants()

Rᵈ   = dry_air_gas_constant(constants)
Rᵛ   = vapor_gas_constant(constants)
cₚᵈ  = constants.dry_air.heat_capacity
κ    = Rᵈ / cₚᵈ
pˢᵗ  = 1e5  # Pa
εfac = Rᵛ / Rᵈ - 1   # for virtual-temperature correction: Tᵛ = T·(1 + εfac·qᵛ)
# (latent heats Lᵥ, Lₛ now live inside `breeze_prognostic_state`.)

# ## Interpolate ERA5 onto the LAM grid
#
# `Field(metadatum, grid)` and `set!(field, metadatum)` regrid ERA5 pressure-level
# data onto an arbitrary target grid, using the per-column geopotential height
# z = Φ(λ, φ, p)/g as the vertical coordinate and clipping sub-surface levels at
# the local surface (NumericalEarth's `PressureLevelGrid`, NumericalEarth/
# NumericalEarth.jl#241). The interpolation is driven by the *target* grid's own
# node heights, so the terrain-following child is sampled at its true physical
# heights — no sigma-z workaround, no custom column interpolation.
#
# These regrids interpolate linearly in height between ERA5 levels for T, qᵛ, qᶜ, qⁱ. Pressure
# is NOT interpolated — over high terrain the sub-surface ERA5 levels clamp and corrupt the
# near-surface state. Instead it is built by hydrostatic integration from the ERA5 surface
# pressure (see `hydrostatic_pressure_from_surface`), keeping it in discrete hydrostatic balance.

# --- Parent grid: ERA5 native (λ, φ), regular true-height z (no terrain) ---
#
# The parent drives the child's lateral boundaries and Davies fringe. It stays on
# a regular (non-terrain-following) grid; the #241 ingest regrids ERA5 onto it by
# true Φ/g — the same vertical coordinate as the terrain-following child — so the
# `Interpolated` lateral BCs and Davies relaxation sample a consistent state when
# they interpolate the parent to the child's nodes.
#
# Mirror NumericalEarth's `native_grid`/`restrict` behavior: it pads the
# requested bbox outward by Δ/2 on each side so cell *centers* land on the
# ERA5 native grid. Without that adjustment, `parent_grid`'s cell centers
# would be offset from the actual ERA5 data positions (up to ~0.12° at the
# bbox edges), which biases the horizontal `interpolate!` onto the LAM.
#
# A single ERA5 fetch (snapshot 1's geopotential metadata) gives us the
# native node coordinates and pressure levels needed to size the grid.

const meta_common_snap1 = (date = start_date, region = era5_region, dir = era5_datadir)
const ϕ_field_snap1     = Field(Metadatum(:geopotential; dataset=ds_pl, meta_common_snap1...))

λ_centers_era5 = collect(λnodes(ϕ_field_snap1.grid, Center(), Center(), Center()))
φ_centers_era5 = collect(φnodes(ϕ_field_snap1.grid, Center(), Center(), Center()))
Nλ_e, Nφ_e = length(λ_centers_era5), length(φ_centers_era5)

# ERA5 returns longitudes in the [0°, 360°] convention; the LAM uses
# [-180°, 180°]. Shift the parent grid labels to match. The FTS data is
# array-indexed and unaffected — only the (λ, φ) labels change.
λ_centers_era5 .= ifelse.(λ_centers_era5 .> 180, λ_centers_era5 .- 360, λ_centers_era5)

Δλ_e = (λ_centers_era5[end] - λ_centers_era5[1]) / (Nλ_e - 1)
Δφ_e = (φ_centers_era5[end] - φ_centers_era5[1]) / (Nφ_e - 1)

parent_grid = LatitudeLongitudeGrid(arch;
                                    longitude = (λ_centers_era5[1]   - Δλ_e/2,
                                                 λ_centers_era5[end] + Δλ_e/2),
                                    latitude  = (φ_centers_era5[1]   - Δφ_e/2,
                                                 φ_centers_era5[end] + Δφ_e/2),
                                    z         = z_discretization,
                                    size      = (Nλ_e, Nφ_e, Nz),
                                    halo      = (5, 5, 5),
                                    topology  = (Bounded, Bounded, Bounded))

# --- PrescribedAtmosphere on the parent grid ---
# Times in seconds since the first snapshot. PrescribedAtmosphere allocates
# default Center-located FTSs for velocities, tracers (T, q), pressure.
# qᶜ, qⁱ aren't standard slots; we own those alongside.

parent_times = [Float64(Dates.value(d - start_date)) / 1000 for d in dates]
parent = PrescribedAtmosphere(parent_grid, parent_times; two_dimensional = false, freshwater_flux = nothing, thermodynamics_parameters = nothing)

# Parent-side `FieldTimeSeries` that drive the child, kept alongside the
# `PrescribedAtmosphere` (which owns u, v, T, q, p). All are Center-located
# regardless of BC stagger — `Interpolated` converts location at boundary-fill
# time. The bundle holds:
#   - qᶜ, qⁱ             raw ERA5 cloud water/ice (inputs to the derivation),
#   - ρ, ρu, ρv, ρθ, ρqᵉ density-weighted, drive the lateral BCs,
#   - θ, qᵗ              specific, Davies-relaxation targets (Breeze PR #708's
#                        `SpecificForcing` applies the ρ multiply at kernel time).
parent_series = NamedTuple(name => FieldTimeSeries{Center, Center, Center}(parent_grid, parent_times)
                           for name in (:qᶜ, :qⁱ, :ρ, :ρu, :ρv, :ρθ, :ρqᵉ, :θ, :qᵗ))

# --- ERA5 pressure-level primitives on the parent grid ---
#
# `FieldTimeSeries(metadata, parent_grid)` regrids the whole window at once. Its
# per-file `set!` reads each snapshot with that snapshot's own geopotential, so the
# Φ/g height mapping is per-snapshot (#241, highest fidelity). All times are held
# resident so we can index by snapshot in the derivation loop below.
parent_pl_series(name) =
    FieldTimeSeries(Metadata(name; dataset = ds_pl, dates = dates,
                             region = era5_region, dir = era5_datadir),
                    parent_grid; time_indices_in_memory = length(dates))

u_series  = parent_pl_series(:eastward_velocity)
v_series  = parent_pl_series(:northward_velocity)
T_series  = parent_pl_series(:temperature)
qᵛ_series = parent_pl_series(:specific_humidity)
qᶜ_series = parent_pl_series(:specific_cloud_liquid_water_content)
qⁱ_series = parent_pl_series(:specific_cloud_ice_water_content)

# ERA5 surface pressure + orography on the parent horizontal, for the hydrostatic balance below.
# Orography is time-constant; surface pressure is re-set per snapshot in the loop.
parent_surface_grid = LatitudeLongitudeGrid(longitude = (λ_centers_era5[1]   - Δλ_e/2,
                                                         λ_centers_era5[end] + Δλ_e/2),
                                            latitude  = (φ_centers_era5[1]   - Δφ_e/2,
                                                         φ_centers_era5[end] + Δφ_e/2),
                                            z = (0, 1), size = (Nλ_e, Nφ_e, 1),
                                            halo = (5, 5, 3), topology = (Bounded, Bounded, Bounded))
Φ_sfc_parent = CenterField(parent_surface_grid)
set!(Φ_sfc_parent, Metadatum(:geopotential; dataset = ds_sl, date = start_date,
                             region = era5_region, dir = era5_datadir))
parent_orography = Array(interior(Φ_sfc_parent))[:, :, 1] ./ g_accel
p₀_parent = CenterField(parent_surface_grid)

# Derive (ρ, θˡⁱ, qᵗ) per snapshot via `breeze_prognostic_state`, storing the specific
# (Davies-target) and density-weighted (lateral-BC) forms. The pressure is built by hydrostatic
# integration from the ERA5 surface pressure (`hydrostatic_pressure_from_surface`) rather than
# interpolated — interpolation clamps the sub-surface levels over high terrain and yields a
# spurious too-dense near-surface state that the lateral BCs would inject into the child.
# TODO: this holds all parent snapshots resident; for production-length runs, recompute the
# balance on a 2-snapshot streaming FieldTimeSeries (DatasetBackend pattern) to cut memory.
for n in eachindex(dates)
    @info @sprintf("Deriving parent snapshot %d/%d at %s", n, length(dates), dates[n])
    set!(p₀_parent, Metadatum(:surface_pressure; dataset = ds_sl, date = dates[n],
                              region = era5_region, dir = era5_datadir))
    p_p = hydrostatic_pressure_from_surface(T_series[n], Array(interior(p₀_parent))[:, :, 1],
                                            parent_orography;
                                            qᵛ = qᵛ_series[n], qᶜ = qᶜ_series[n], qⁱ = qⁱ_series[n],
                                            dry_gas_constant = Rᵈ, vapor_gas_constant = Rᵛ,
                                            gravitational_acceleration = g_accel)
    state = breeze_prognostic_state(constants, T_series[n], qᵛ_series[n],
                                    qᶜ_series[n], qⁱ_series[n], p_p)

    interior(parent.velocities.u, :, :, :, n) .= interior(u_series[n])
    interior(parent.velocities.v, :, :, :, n) .= interior(v_series[n])
    interior(parent.tracers.T,    :, :, :, n) .= interior(T_series[n])
    interior(parent.tracers.q,    :, :, :, n) .= interior(qᵛ_series[n])
    interior(parent.pressure,     :, :, :, n) .= interior(p_p)
    interior(parent_series.qᶜ,    :, :, :, n) .= interior(qᶜ_series[n])
    interior(parent_series.qⁱ,    :, :, :, n) .= interior(qⁱ_series[n])

    interior(parent_series.ρ,   :, :, :, n) .= interior(state.ρ)
    interior(parent_series.ρu,  :, :, :, n) .= interior(state.ρ) .* interior(u_series[n])
    interior(parent_series.ρv,  :, :, :, n) .= interior(state.ρ) .* interior(v_series[n])
    interior(parent_series.ρθ,  :, :, :, n) .= interior(state.ρ) .* interior(state.θˡⁱ)
    interior(parent_series.ρqᵉ, :, :, :, n) .= interior(state.ρ) .* interior(state.qᵗ)
    interior(parent_series.θ,   :, :, :, n) .= interior(state.θˡⁱ)
    interior(parent_series.qᵗ,  :, :, :, n) .= interior(state.qᵗ)
end

# --- LAM-grid IC fields: regrid snapshot 1 of ERA5 directly onto the child ---
# `set!(field, metadatum)` regrids each ERA5 field onto the terrain-following
# child grid by true Φ/g (#241), staggering to the field's own location
# (velocities to faces, scalars to centers). No parent → child step is needed.

initial_metadatum(name) = Metadatum(name; dataset = ds_pl, meta_common_snap1...)

u  = XFaceField(grid);  set!(u,  initial_metadatum(:eastward_velocity))
v  = YFaceField(grid);  set!(v,  initial_metadatum(:northward_velocity))
T  = CenterField(grid); set!(T,  initial_metadatum(:temperature))
qᵛ = CenterField(grid); set!(qᵛ, initial_metadatum(:specific_humidity))
qᶜ = CenterField(grid); set!(qᶜ, initial_metadatum(:specific_cloud_liquid_water_content))
qⁱ = CenterField(grid); set!(qⁱ, initial_metadatum(:specific_cloud_ice_water_content))

# Calculate virtual temperature: Tᵛ = T·(1 + (1 − ε)/ε·qᵛ), ε = Rᵈ/Rᵛ.
# Vapor only by convention — the qᶜ, qⁱ terms belong to the density temperature Tρ.

Tᵛ = Field(T * (1 + εfac * qᵛ))
compute!(Tᵛ)

# Surface pressure: horizontal field on a 3-D grid with Nz=1. (A `Flat` z
# topology trips Oceananigans' interpolation kernel — `_fractional_indices`
# unconditionally destructures `at_node` as `(x, y, z)`; mirrors the pattern
# used in examples/ERA5_hourly_data.jl.)

surface_grid = LatitudeLongitudeGrid(longitude = (λ_west,  λ_east),
                                     latitude  = (φ_south, φ_north),
                                     z         = (0, 1),
                                     size      = (Nx, Ny, 1),
                                     halo      = (5, 5, 3),
                                     topology  = (Bounded, Bounded, Bounded))

p₀ = CenterField(surface_grid)
set!(p₀, Metadatum(:surface_pressure; dataset=ds_sl, meta_common_snap1...))

# Hydrostatically-balanced initial pressure. Interpolating ERA5 pressure to the node heights
# clamps the sub-surface levels over high terrain, leaving the cold-start IC out of the model's
# discrete hydrostatic balance (a ~40 g vertical residual). Build `p` by integrating up from the
# ERA5 surface pressure instead — anchored at each column's terrain surface, with the moist Rᵐ.
p = hydrostatic_pressure_from_surface(T, Array(interior(p₀))[:, :, 1], era5_orography;
                                      qᵛ = qᵛ, qᶜ = qᶜ, qⁱ = qⁱ,
                                      dry_gas_constant = Rᵈ, vapor_gas_constant = Rᵛ,
                                      gravitational_acceleration = g_accel)

# ## Lateral boundary conditions and Davies relaxation
#
# Drive the LAM's lateral boundaries from the parent FTSs:
#   - `ρu`, `ρv` get `NormalFlowBoundaryCondition(Interpolated(fts))` (Face-stagger).
#   - `ρ`, `ρθ`, `ρqᵉ` get `ValueBoundaryCondition(Interpolated(fts))` —
#     `NormalFlowBC` on Center-located fields silently overwrites the first interior
#     cell on the W/S walls (validated against vortex-transit tests).
#
# Davies fringe relaxation toward the same parent state via `parent_forcings`,
# which wraps each parent `FieldTimeSeries` target in an Oceananigans
# `Relaxation` (space/time-interpolated). We key them under specific names
# (`u`, `v`, `θ`, `qᵉ`) so Breeze's `SpecificForcing` (PR #708) applies the ρ
# multiply at kernel time at the right face stagger.

bcs = parent_boundary_conditions(grid;
    variables = (ρu  = parent_series.ρu,
                 ρv  = parent_series.ρv,
                 ρ   = parent_series.ρ,
                 ρe  = parent_series.ρθ,    # `atmosphere_simulation` already sets bottom :ρe
                                  # flux; Breeze converts the merged :ρe BCs to :ρθ
                                  # at model-build time (ValueBC values pass through).
                 ρqᵉ = parent_series.ρqᵉ),
    sides     = (:west, :east, :south, :north),
    bc_types  = (ρ   = ValueBoundaryCondition,
                 ρe  = ValueBoundaryCondition,
                 ρqᵉ = ValueBoundaryCondition))

# Surface-BC placeholders, pending SlabLand wiring. Override `atmosphere_simulation`'s
# coupling Jᵉ/Jᵛ bottom-flux BCs with Dirichlet ValueBCs at constant placeholder
# surface T and qᵛ. Keeping the coupling Jᵉ would route the bottom flux through
# Breeze's `EnergyFluxBoundaryCondition` → `𝒬_to_Jᶿ`, which can't evaluate until
# the bulk-flux state (and qᵛ at the surface) is populated by the land model.

const T_surface_placeholder   = 290.0
const qᵛ_surface_placeholder  = 0.0
const ρ_surface_placeholder   = 1.2                                   # kg/m³ at p₀=10⁵ Pa, T≈290 K
const ρθ_surface_placeholder  = ρ_surface_placeholder * T_surface_placeholder
const ρqᵉ_surface_placeholder = ρ_surface_placeholder * qᵛ_surface_placeholder

bcs = merge(bcs, (; ρe  = FieldBoundaryConditions(west   = bcs.ρe.west,
                                                  east   = bcs.ρe.east,
                                                  south  = bcs.ρe.south,
                                                  north  = bcs.ρe.north,
                                                  bottom = ValueBoundaryCondition(ρθ_surface_placeholder)),
                   ρqᵉ = FieldBoundaryConditions(west   = bcs.ρqᵉ.west,
                                                  east   = bcs.ρqᵉ.east,
                                                  south  = bcs.ρqᵉ.south,
                                                  north  = bcs.ρqᵉ.north,
                                                  bottom = ValueBoundaryCondition(ρqᵉ_surface_placeholder))))

# Fringe geometry: 5 cells deep in each lateral direction. The mask is a
# cosine ramp in degree-distance to the nearest wall — Davies is a numerical
# smoother, so the precise ramp shape isn't physics-critical.

FRINGE_N = 5
fringe_deg = FRINGE_N * max(Δλ, Δφ)

# Capture domain extents + fringe width into closure-local bindings so the
# resulting function is type-stable (required for GPU kernel compilation —
# non-const module globals produce dynamic-dispatch IR).
lateral_mask = let λ_w = λ_west, λ_e = λ_east, φ_s = φ_south, φ_n = φ_north, fringe = fringe_deg
    (λ, φ, z) -> begin
        dW = λ - λ_w
        dE = λ_e - λ
        dS = φ - φ_s
        dN = φ_n - φ
        d  = min(dW, dE, dS, dN)
        d >= fringe && return zero(λ)
        return 0.5 * (1 + cos(π * d / fringe))
    end
end

# Relaxation timescale = 10 outer steps, so the fringe pulls the boundary toward the
# parent within ~10 Δt. (The advective-crossing estimate τ ≈ 5·Δx/U was O(50–700) Δt — far too weak.)
τ_relax = 10 * Δt                                   # s

davies = parent_forcings(; rate = 1/τ_relax,
                         mask = lateral_mask,
                         variables = (u  = parent.velocities.u,
                                      v  = parent.velocities.v,
                                      θ  = parent_series.θ,
                                      qᵉ = parent_series.qᵗ))

# ## Build the Breeze model
#
# Piggyback on the `atmosphere_simulation` helper from
# `ext/NumericalEarthBreezeExt/`, which also pre-wires bottom flux fields
# (ρτˣ, ρτʸ, Jᵉ, Jᵛ) ready for the forthcoming SlabLand / SlabOcean coupling.
# We override the helper's default `AnelasticDynamics` (which dispatches on
# `RectilinearGrid` only) with `CompressibleDynamics`, whose prognostic-density
# / diagnostic-pressure formulation needs no FFT-based Poisson solve and works
# directly on the LAM `LatitudeLongitudeGrid`. On the `TerrainFollowingVerticalDiscretization`
# grid, `CompressibleDynamics` activates the terrain-following physics automatically —
# contravariant vertical velocity, corrected horizontal pressure gradient, terrain-aware
# divergence — so no `terrain_metrics` argument is needed. The `SplitExplicitTimeDiscretization`
# (Breeze PR #712) integrates the acoustic modes with inner substeps, freeing the outer
# step to run at the advection CFL (see Δt below). Its `UpperSponge` adds a 5 km-deep
# Rayleigh layer that damps the vertical momentum (ρw)′ toward the ~26.5 km rigid lid
# (5 s timescale), absorbing vertically-propagating modes so they don't reflect.
#
# TODO: pass `reference_potential_temperature = θ_ref(z)` to `CompressibleDynamics`.
# A reference state lets Breeze compute the horizontal pressure gradient in
# perturbation form (p′ = p − p_ref), which cuts the terrain-following PGF
# cancellation error (Klemp 2011). Without it (current), the full-pressure gradient
# is used — fine for the gentle SGP terrain, but worth adding for steeper domains.
# Generate θ_ref(z) as the ERA5 domain-/time-mean potential-temperature profile
# (regrid θ onto a column and average over (λ, φ) and snapshots).
#
# `atmosphere_simulation` returns an Oceananigans `Simulation`; we drive the
# child through `NestedSimulation` below, so unwrap the underlying
# `AtmosphereModel`. The skeleton `CoupledRadiation` it carries is a no-op
# (radiatively decoupled) until materialized inside an `EarthSystemModel`.

p̄₀ = mean(interior(p₀))

# Add a Rayleigh damping layer. 3 km deep below the ~20 km lid (sponge spans ~17–20 km),
# keeping it in the lower stratosphere above the jet now that the top is shallower.
damping_timescale = 5    # (s)
damping_depth     = 3000 # (m)
rayleigh_damping = UpperSponge(; damping_rate = 1/damping_timescale, depth = damping_depth)

# Advection uses `atmosphere_simulation`'s defaults — WENO(9) for momentum, WENO(5) for
# scalars — higher order than [Fan2017](@citet)'s 5th-order horizontal / 3rd-order vertical.
# Matching Fan's per-direction orders (a `FluxFormAdvection` of WENO(5)/WENO(5)/WENO(3)) was
# tested and left the dynamics essentially unchanged, so the higher-order default is kept.

# Coriolis: a synoptic-scale LAM forced by ERA5 needs the rotating-frame balance, else the ERA5
# pressure field accelerates the interior winds with no geostrophic restoring force. `SphericalCoriolis`
# gives the latitude-varying f on the lat-lon grid. (Disable with CORIOLIS=off for comparison.)
coriolis_scheme = get(ENV, "CORIOLIS", "on") == "on" ? SphericalCoriolis() : nothing

model = atmosphere_simulation(grid;
                              thermodynamic_constants = constants,
                              coriolis            = coriolis_scheme,
                              dynamics            = CompressibleDynamics(SplitExplicitTimeDiscretization(sponge = rayleigh_damping);
                                                                         surface_pressure = p̄₀),
                              boundary_conditions = bcs,
                              forcing             = davies).model

# ## Set initial state from ERA5
#
# Derive Breeze's `CompressibleDynamics` prognostic variables (ρ, θˡⁱ, qᵗ) from
# the per-column-interpolated ERA5 fields via the shared `breeze_prognostic_state`
# helper — the same conversion used to populate the parent FTSs above. (qᵗ is
# repartitioned into vapor/condensate by saturation adjustment on the first
# `update_state!`.)

(; ρ, θˡⁱ, qᵗ) = breeze_prognostic_state(constants, T, qᵛ, qᶜ, qⁱ, p)

set!(model; ρ = ρ, u = u, v = v, qᵗ = qᵗ, θˡⁱ = θˡⁱ)

# Consistent-w IC. The analysis cold-starts Cartesian w = 0, but the terrain-following
# coordinate wants the contravariant w̃ = w − slopeₓ·u − slopeᵧ·v ≈ 0 (flow follows the
# ground). With w = 0 the IC carries w̃ = −slopeₓ·u − slopeᵧ·v ≈ −u·∇h, which the surface
# kinematic BC then discharges as a spurious near-surface w. Make ρw terrain-consistent by
# grafting ρw ← ρw − ρw̃: after `update_state!` populates ρw̃ = ρw − slope·ρu = −slope·ρu
# (since ρw=0), this sets ρw = slope·ρu — exactly the contravariant-zeroing momentum, using
# Breeze's own slope. A second `update_state!` confirms w̃ ≈ 0.
update_state!(model)
interior(model.momentum.ρw) .-= interior(model.dynamics.contravariant_vertical_momentum)
update_state!(model)

# ## NestedSimulation
#
# Wrap the child model in a `NestedSimulation` paired with the parent
# `PrescribedAtmosphere`. `NestedModel.time_step!` syncs the parent clock
# each iteration so the FTS-driven BCs and forcings get the correct
# interpolation time.
#
# Δt is defined with the grid above; the Davies fringe relaxes on a 10·Δt timescale.

nested = NestedSimulation(parent, model; Δt, stop_iteration = 100)

function progress(sim)
    m = sim.model
    @info @sprintf("iter=%3d  t=%.3f s  max|u|=%.3f  max|v|=%.3f  max|w|=%.2e  ρ∈[%.4f, %.4f]",
                   m.clock.iteration, m.clock.time,
                   maximum(abs, m.velocities.u),
                   maximum(abs, m.velocities.v),
                   maximum(abs, m.velocities.w),
                   minimum(interior(m.dynamics.density)),
                   maximum(interior(m.dynamics.density)))
end
add_callback!(nested, progress, IterationInterval(10))

# ## Run
#
# 100-iteration smoke run at the advection-CFL outer Δt (split-explicit substepping).
# The ERA5 analysis cold-starts w = 0, so the IC is not in discrete nonhydrostatic
# balance; at this Δt the substepper unmasks that imbalance as a large spurious
# adjustment (the old acoustic-CFL Δt only damped it slowly). A physical run needs
# dynamical initialization next — `balance_adiabatically!` (Breeze #764) on a stripped
# (no-microphysics, no-forcing, frozen-BC) model, then transfer (ρ, ρu, ρv, ρθ, ρqᵉ, ρw).

@info @sprintf("Δt = %.4f s (advection CFL, split-explicit); running %d iterations", Δt, nested.stop_iteration)
run!(nested)
@info "Done."

# ## Report

@info "Interior field ranges:"
@info @sprintf("  u  ∈ [%+.2f, %+.2f] m/s", minimum(interior(u)),  maximum(interior(u)))
@info @sprintf("  v  ∈ [%+.2f, %+.2f] m/s", minimum(interior(v)),  maximum(interior(v)))
@info @sprintf("  T  ∈ [%+.2f, %+.2f] K",   minimum(interior(T)),  maximum(interior(T)))
@info @sprintf("  Tᵛ ∈ [%+.2f, %+.2f] K",   minimum(interior(Tᵛ)), maximum(interior(Tᵛ)))
@info @sprintf("  qᵛ ∈ [%.2e, %.2e] g/kg",  1000*minimum(interior(qᵛ)), 1000*maximum(interior(qᵛ)))
@info @sprintf("  qᶜ ∈ [%.2e, %.2e] g/kg",  1000*minimum(interior(qᶜ)), 1000*maximum(interior(qᶜ)))
@info @sprintf("  qⁱ ∈ [%.2e, %.2e] g/kg",  1000*minimum(interior(qⁱ)), 1000*maximum(interior(qⁱ)))
@info @sprintf("  p₀ ∈ [%.1f, %.1f] Pa",    minimum(interior(p₀)), maximum(interior(p₀)))

# ## Profile plots
#
# Plot ρ, u, v, θ, qᵗ at three sites spanning the domain's terrain range,
# comparing the initial state (blue) with the post-run state (red). The vertical
# coordinate is the true physical height of the terrain-following grid, so each
# profile's lowest marker sits at the local ETOPO surface elevation.

sites = [("East TX",     -93.5,   34.0),
         ("SGP",         -97.485, 36.605),
         ("High Plains", -101.5,  35.0)]

# Initial-state LAM arrays (from `set!` above); θ is the dry potential temperature.
θ_lam = compute!(Field(T * (pˢᵗ / p)^κ))

ρ_arr  = Array(interior(ρ))
u_arr  = Array(interior(u))
v_arr  = Array(interior(v))
θ_arr  = Array(interior(θ_lam))
qᵗ_arr = Array(interior(qᵗ))

# Post-run LAM state. Specific quantities (θ, qᵗ) are derived from the
# prognostic ρθ, ρqᵉ divided by ρ.
ρ_final_arr  = Array(interior(model.dynamics.density))
u_final_arr  = Array(interior(model.velocities.u))
v_final_arr  = Array(interior(model.velocities.v))
ρθ_final     = Array(interior(model.formulation.potential_temperature_density))
ρqᵉ_final    = Array(interior(model.moisture_density))
θ_final_arr  = ρθ_final  ./ ρ_final_arr
qᵗ_final_arr = ρqᵉ_final ./ ρ_final_arr

# Terrain-following heights vary by column; read them from a host copy of the
# grid (`znode` applies the σ scaling and η displacement per column).
cpu_grid    = on_architecture(CPU(), grid)
elevation_m = Array(interior(elevation))[:, :, 1]
λ_c = collect(λnodes(grid, Center(), Center(), Center()))
φ_c = collect(φnodes(grid, Center(), Center(), Center()))
λ_f = collect(λnodes(grid, Face(),   Center(), Center()))
φ_f = collect(φnodes(grid, Center(), Face(),   Center()))

column_height(i, j) = [znode(i, j, k, cpu_grid, Center(), Center(), Center()) for k in 1:Nz]

vars = [(:ρ,  ρ_arr,  ρ_final_arr,  "ρ (kg/m³)",  :center),
        (:u,  u_arr,  u_final_arr,  "u (m/s)",    :xface),
        (:v,  v_arr,  v_final_arr,  "v (m/s)",    :yface),
        (:θ,  θ_arr,  θ_final_arr,  "θ (K)",      :center),
        (:qᵗ, qᵗ_arr, qᵗ_final_arr, "qᵗ (kg/kg)", :center)]

fig = Figure(size=(1600, 1000), fontsize=12)

Nrows = length(sites)
Ncols = length(vars)
axs   = Matrix{Axis}(undef, Nrows, Ncols)

for (row, (label, λ_site, φ_site)) in enumerate(sites)
    # Site header; elevation from the regridded ETOPO field at the nearest cell.
    i_site = argmin(abs.(λ_c .- λ_site))
    j_site = argmin(abs.(φ_c .- φ_site))
    elev_m = round(Int, elevation_m[i_site, j_site])
    Label(fig[2*row - 1, 1:Ncols], "$label (elevation: $elev_m m)";
          fontsize=15, font=:bold, halign=:center, tellwidth=false)

    for (col, (vname, lam_arr, lam_final_arr, xlab, stagger)) in enumerate(vars)
        i_lam = stagger == :xface ? argmin(abs.(λ_f .- λ_site)) :
                                    argmin(abs.(λ_c .- λ_site))
        j_lam = stagger == :yface ? argmin(abs.(φ_f .- φ_site)) :
                                    argmin(abs.(φ_c .- φ_site))

        z_km = column_height(i_lam, j_lam) ./ 1000

        ax = Axis(fig[2*row, col]; xlabel=xlab,
                  ylabel       = col == 1 ? "height (km)" : "",
                  xlabelsize   = 14,
                  ylabelsize   = 14)
        axs[row, col] = ax

        # LAM profile at the chosen point — markers at cell centers so the
        # discretization is explicit (no implied between-cell behavior).
        scatter!(ax, lam_arr[i_lam, j_lam, :], z_km;
                 color=:steelblue, markersize=6, label="t=0")
        scatter!(ax, lam_final_arr[i_lam, j_lam, :], z_km;
                 color=:crimson, markersize=6, label=@sprintf("t=%.2f s", model.clock.time))

        ylims!(ax, 0, 15)
        vname === :θ && xlims!(ax, 280, 400)
    end
end

# Share y-axis behavior across each site's row; only the leftmost column
# shows ticks and label.
for r in 1:Nrows
    linkyaxes!(axs[r, :]...)
    for c in 2:Ncols
        hideydecorations!(axs[r, c]; grid=false)
    end
end

axislegend(axs[1, end]; position = :rb)

Label(fig[0, 1:Ncols], "ERA5 → terrain-following LAM profiles";
      fontsize=20, font=:bold, tellwidth=false)

save("era5_breeze_profiles.png", fig)
@info "Wrote era5_breeze_profiles.png"

# ## Horizontal cut-plane comparison
#
# Compare u, v, w on a horizontal plane 80 m above ground level between the ERA5
# forcing (~0.25°, the parent) and the 9 km Breeze child (Fan Domain 2). At the
# smoke horizon the ERA5 row is the downscaled initial state and the model row is
# the child after a few acoustic steps — so they differ by resolution and a brief
# transient, not yet by hours of distinct evolution (that needs the multi-hour run).
# The 3 km row arrives with the Domain 3 nest-down.
#
# `cut_plane` interpolates each field's column to the target height above the
# *local terrain surface*, honoring the field's stagger (u on λ-faces, v on
# φ-faces, w on z-faces) and the terrain-following node heights — so it works
# unchanged on the ERA5 ingest grid and the live child model.
#
# TODO: promote `cut_plane` to `NumericalEarth.Diagnostics` once stabilized — it's
# a generic terrain-following AGL slice, useful well beyond this example.

function interp_to_height(zcol, vals, z_target)
    n = length(zcol)
    z_target <= zcol[1] && return vals[1]
    z_target >= zcol[n] && return vals[n]
    k = searchsortedlast(zcol, z_target)
    t = (z_target - zcol[k]) / (zcol[k+1] - zcol[k])
    return (1 - t) * vals[k] + t * vals[k+1]
end

function cut_plane(field, height_agl)
    host_grid = on_architecture(CPU(), field.grid)
    LX, LY, LZ = location(field)
    data = Array(interior(field))
    Nx_f, Ny_f, Nz_f = size(data)
    slice = Matrix{eltype(data)}(undef, Nx_f, Ny_f)
    for j in 1:Ny_f, i in 1:Nx_f
        z_surface = znode(i, j, 1, host_grid, LX(), LY(), Face())  # deformed bottom interface
        z_target  = z_surface + height_agl
        zcol      = [znode(i, j, k, host_grid, LX(), LY(), LZ()) for k in 1:Nz_f]
        slice[i, j] = interp_to_height(zcol, view(data, i, j, :), z_target)
    end
    λ = collect(λnodes(host_grid, LX(), LY(), LZ()))
    φ = collect(φnodes(host_grid, LX(), LY(), LZ()))
    return λ, φ, slice
end

# ERA5 snapshot-1 winds on a terrain-following grid at the requested resolution
# over the D2 window. u, v ingest directly (#241); w is reconstructed from ERA5 ω
# (Pa/s) via w ≈ -ω/(ρ g), with ρ = p/(Rᵈ T) (vapor correction on ρ < 1%, neglected).
g_earth = Oceananigans.defaults.gravitational_acceleration

function era5_winds_on_grid(nx, ny)
    g = LatitudeLongitudeGrid(arch;
                              longitude = (λ_west,  λ_east),
                              latitude  = (φ_south, φ_north),
                              z         = TerrainFollowingVerticalDiscretization(z_discretization),
                              size      = (nx, ny, Nz),
                              halo      = (5, 5, 5),
                              topology  = (Bounded, Bounded, Bounded))
    materialize_terrain!(g, regrid_topography(g; dataset = ETOPO2022()))

    ug = XFaceField(g);  set!(ug, initial_metadatum(:eastward_velocity))
    vg = YFaceField(g);  set!(vg, initial_metadatum(:northward_velocity))
    Tg = CenterField(g); set!(Tg, initial_metadatum(:temperature))
    ωg = CenterField(g); set!(ωg, initial_metadatum(:vertical_velocity))

    # Hydrostatic pressure from the ERA5 surface pressure (dry — the ρ below already neglects the
    # <1% vapor correction), anchored on g's terrain; gives a physical near-surface ρ over terrain.
    sfc_grid_g = LatitudeLongitudeGrid(longitude = (λ_west, λ_east), latitude = (φ_south, φ_north),
                                       z = (0, 1), size = (nx, ny, 1),
                                       halo = (5, 5, 3), topology = (Bounded, Bounded, Bounded))
    psfc_g = CenterField(sfc_grid_g); set!(psfc_g, Metadatum(:surface_pressure; dataset = ds_sl, meta_common_snap1...))
    Φg     = CenterField(sfc_grid_g); set!(Φg,     Metadatum(:geopotential;     dataset = ds_sl, meta_common_snap1...))
    pg = hydrostatic_pressure_from_surface(Tg, Array(interior(psfc_g))[:, :, 1],
                                           Array(interior(Φg))[:, :, 1] ./ g_earth;
                                           dry_gas_constant = Rᵈ, vapor_gas_constant = Rᵛ,
                                           gravitational_acceleration = g_earth)
    wg = compute!(Field(-ωg / (pg / (Rᵈ * Tg) * g_earth)))
    return ug, vg, wg
end

# ERA5 row at native 0.25°; model row is the live 9 km D2 child.
u_e, v_e, w_e = era5_winds_on_grid(round(Int, (λ_east - λ_west) / 0.25),
                                   round(Int, (φ_north - φ_south) / 0.25))
u_d, v_d, w_d = model.velocities.u, model.velocities.v, model.velocities.w

height_agl = 80.0
rows = [("ERA5 ~0.25°",                                u_e, v_e, w_e),
        (@sprintf("D2 9 km (t=%.1f s)", model.clock.time), u_d, v_d, w_d)]
cols = ("u (m/s)", "v (m/s)", "w (m/s)")

fig_cut = Figure(size = (1500, 1300), fontsize = 13)

for (r, (rlabel, fu, fv, fw)) in enumerate(rows)
    for (c, fld) in enumerate((fu, fv, fw))
        λ, φ, slice = cut_plane(fld, height_agl)

        ax = Axis(fig_cut[r, 2c - 1];
                  aspect = DataAspect(),
                  title  = r == 1            ? cols[c]          : "",
                  ylabel = c == 1            ? rlabel           : "",
                  xlabel = r == length(rows) ? "longitude (°)"  : "")

        finite = filter(isfinite, vec(slice))
        m      = isempty(finite) ? one(eltype(slice)) : maximum(abs, finite)
        m      = m == 0 ? one(m) : m

        hm = heatmap!(ax, λ, φ, slice; colormap = :balance, colorrange = (-m, m))
        Colorbar(fig_cut[r, 2c], hm)
        scatter!(ax, [λ₀], [φ₀]; color = :black, marker = :star5, markersize = 12)

        r != length(rows) && hidexdecorations!(ax; grid = false)
        c != 1            && hideydecorations!(ax; grid = false)
    end
end

Label(fig_cut[0, 1:6], @sprintf("Winds at %g m AGL — ERA5 → 9 km D2 (MC3E, ARM SGP)", height_agl);
      fontsize = 18, font = :bold, tellwidth = false)

save("era5_breeze_cutplanes.png", fig_cut)
@info "Wrote era5_breeze_cutplanes.png"
