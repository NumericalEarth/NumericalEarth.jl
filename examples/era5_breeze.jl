# # ERA5 → 3 km convection-permitting hindcast (Breeze + NestedSimulation)
#
# A limited-area model (LAM) example that downscales ERA5 reanalysis to a 3 km Breeze compressible
# atmosphere over the U.S. Southern Great Plains, for the MC3E 20 May 2011 squall-line case
# ([Fan2017](@citet)). ERA5 (the "parent", ~0.25°) drives a single 3 km child through a
# `NestedSimulation`: parent-interpolated open lateral boundary conditions + an interior Davies
# relaxation fringe.
#
# ## What this example does
# - Downloads ERA5 (pressure + single levels) for a fixed parent region and regrids it onto a
#   terrain-following `LatitudeLongitudeGrid` (ETOPO2022 orography, tapered to the parent at the edge).
# - Initializes the prognostic state from ERA5 with a hydrostatic-from-surface pressure and a
#   terrain-consistent `w̃ ≈ 0`; a dynamical-initialization (DFI) pass then spins `ρw` into balance.
# - Integrates the compressible equations with split-explicit acoustic substepping (adaptive
#   substeps + an adaptive outer-Δt wizard), 1-moment mixed-phase microphysics, Coriolis, a
#   reference-θ perturbation-form pressure-gradient, bulk surface drag, and an `UpperSponge` + ρw
#   Rayleigh lid sponge. Open BCs + the Davies fringe track the ERA5 parent.
# - Captures 2-level horizontal slices (saved to `era5_breeze_slices.jld2`) and renders the 2-row
#   ERA5-parent-vs-3 km-child cascade animation.
#
# ## What it does NOT do (yet)
# - No data assimilation — one-way-nested downscaling, not an analysis.
# - No cumulus parameterization: 3 km is convection-*permitting*, so deep convection is resolved on
#   the grid. At this grey-zone-adjacent resolution the resolved convection can be over-vigorous.
# - No land/ocean coupling (surface stress is a bulk-drag stand-in; the SlabLand MOST link is unwired).
# - Single nest only (ERA5 → 3 km). The window here is 2 h to keep the example short; the full MC3E
#   case is 18 h.
#
# ## What we attempted / known issues
# - First set up as a telescope ERA5 → 9 km → 3 km. The 9 km middle nest sits in the convective grey
#   zone, where under-resolved convection runs away (a vertical-mixing band-aid bounds but never makes
#   it physical). Reducing to ERA5 → 3 km direct sidesteps the 9 km grey zone.
# - The split-explicit cold start disables `ThermalDivergenceDamping` (`NoDivergenceDamping`): its
#   (ρθ)′-proxy damper injects a spurious force on the unbalanced start (Breeze #793).
# - The resolved 3 km convection is still vigorous (deep updrafts, locally high max|w|); a physically
#   robust multi-hour run needs explicit convective treatment / horizontal mixing — deferred. This
#   example demonstrates the wired stack runs end-to-end; physical validity of the convection is
#   future work.
# - A DFI cycle-count sensitivity study (1/2/4/8 adiabatic balance cycles) found the post-DFI max|w|
#   is already ~0.6 m/s after a single cycle and the subsequent max|w| growth is insensitive to the
#   cycle count — the deep updrafts are CAPE-driven at grey-zone resolution, not seeded by the
#   initialization transient. A single balance cycle therefore suffices (`balance_cycles = 1`).
# - Near-surface initialization transient: the ERA5 winds are set on the terrain-following grid as-is,
#   in balance with neither surface drag nor the model's pressure field. The lowest cell samples the
#   free-atmosphere ERA5 wind — strongest over high terrain, where k=1 sits ~1 km above sea level — so
#   the near-surface |U| sheds ~25% in the first ~0.5 h as surface drag spins up and the flow
#   geostrophically adjusts (DFI balances ρw, not the horizontal momentum). A balanced / terrain-aware
#   wind initialization, or a DFI that also balances horizontal momentum, would reduce it; motivates
#   further development of the initialization routine.

using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.ERA5
using CDSAPI  # activates NumericalEarthCDSAPIExt
using Oceananigans
using Oceananigans: location
using Oceananigans.Fields: interpolate!
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Units: Time
using Oceananigans.Grids: znode
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Coriolis: SphericalCoriolis
using Breeze
using CloudMicrophysics  # triggers BreezeCloudMicrophysicsExt (OneMomentCloudMicrophysics)
using Breeze.Microphysics: SaturationAdjustment, MixedPhaseEquilibrium
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization, materialize_terrain!
using Statistics: mean, quantile
using JLD2: jldsave
using Dates
using Printf

# This 3 km LAM (300×270×50 ≈ 4M cells, split-explicit) targets a CUDA GPU; switch to `CPU()` only
# for a small smoke test.
using CUDA
const arch = GPU(CUDA.CUDABackend(always_inline = true))

# Single precision (f32): the LAM is memory-bandwidth-bound on the GPU; f32 roughly halves
# the footprint and step cost at no meaningful accuracy cost here. Sets Oceananigans' global
# default float type, cascading to all grids, fields, FieldTimeSeries, constants, and dynamics.
Oceananigans.defaults.FloatType = Float32

# ## Configuration

# ### Domain
#
# Domain centered on the U.S. Department of Energy's Atmospheric Radiation Measurement (ARM) Climate
# Research Facility's Southern Great Plains (SGP) site in Lamont, OK. We take the innermost 3 km
# domain (Domain 3) of the WRF 27 → 9 → 3 km telescoping nest used by [Fan2017](@citet) for this MC3E
# case, driven directly by ERA5 (the parent). ERA5's native 0.25° step divides by 9 to 1/36° (~3 km)
# here, so the child cells align cleanly with the reanalysis grid.
#
# We use a uniform 1/36° angular step, so the physical cells are ~3 km — anisotropic at this
# latitude, using R = 6,371 km:
#   Δx = R·cos(φ₀)·Δλ ≈ 2.48 km
#   Δy = R·Δφ         ≈ 3.09 km

φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)

Δλ = Δφ = 1/36              # uniform 1/36° step (ERA5 0.25° / 9 — ~3 km, convection-permitting)
Nx, Ny = 300, 270           # Fan et al. (2017) Domain 3 (3 km) footprint

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

start_date = DateTime(2011, 05, 20, 0)  # 0000 UTC (7 pm LT previous day)
end_date   = DateTime(2011, 05, 20, 2)  # 2 h here to keep the example short; the full MC3E case is 18 h

dates = start_date:Hour(1):end_date

# ### ERA5 reanalysis
#
# Pressure-level variables are regridded onto the parent grid as `FieldTimeSeries`
# (and onto the child grid for the initial condition) further below.

era5_datadir = "era5"   # Where data will be saved locally

# ERA5 forcing region: the LAM footprint padded outward by `era5_pad` and snapped to
# ERA5's native 0.25° grid, so the parent strictly encloses the child (the
# Interpolated lateral BCs and the 5-cell Davies fringe need parent data beyond the child
# edge). At 0.25° (~28 km here) ERA5 stands in for Fan's 27 km Domain 1, completing the
# telescope ERA5 → D2 (1/12°) → D3 (1/36°, the next nest-down).

era5_pad = 1.0   # deg; wider than the 5·(1/12°) ≈ 0.42° Davies fringe

snap_out(lo, hi; d = 0.25) = (floor(lo / d) * d, ceil(hi / d) * d)
# ERA5 parent region anchored to the original 9 km Fan Domain-2 footprint (180×165 @ 1/12°,
# SGP-centered) + buffer — NOT the 3 km child. One ERA5 retrieval then serves the 3 km child now
# and a 9 km D2 outer nest later (ERA5 → D2 → D3) without re-downloading, and gives the animation's
# parent row the wider synoptic context. Snapped outward to ERA5's native 0.25° grid.
D2_Nx, D2_Ny, D2_Δ = 180, 165, 1/12
era5_region = BoundingBox(longitude = snap_out(λ₀ - D2_Nx * D2_Δ / 2 - era5_pad, λ₀ + D2_Nx * D2_Δ / 2 + era5_pad),
                          latitude  = snap_out(φ₀ - D2_Ny * D2_Δ / 2 - era5_pad, φ₀ + D2_Ny * D2_Δ / 2 + era5_pad))

@info @sprintf("Breeze child (3 km): λ ∈ [%.3f, %.3f], φ ∈ [%.3f, %.3f]; Δλ=Δφ=%.4f°",
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

# Initial outer time step for the adaptive wizard (configured after the simulation, below).
# Split-explicit substepping integrates the acoustic modes, so the outer step is advection-limited;
# we start gentle at 1 s — which also avoids amplifying the cold-start transient — and let the wizard
# ramp it toward `max_Δt`. The Davies fringe relaxation timescale `τ_relax` is tied to this Δt.
Δt = 1.0

# ## Nested domains
#
# Visualize the nesting before stepping the model: the ERA5 forcing region that supplies the parent
# state (lateral BCs + Davies fringe) and the 3 km LAM — Fan (2017)'s Domain 3, the `NestedSimulation`
# child — over ETOPO terrain with Natural Earth state/country boundaries, centered on ARM SGP.
# Drawn here, before the run, so the domain geometry is written even if the run is cut short.

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
               title  = "ERA5 → 3 km LAM nest (MC3E squall line, ARM SGP)",
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
       color = :crimson, linewidth = 3, label = "3 km LAM — Fan Domain 3 (child)")
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

# The ERA5-parent slices (row 1 of the cascade animation) are derived after the run, in the
# "Cascade animation" section below — sampled from the resident hourly parent FTS at the child's
# frame times, so no separate extraction pass is needed here.

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

# ## Set initial state from ERA5
#
# Derive Breeze's `CompressibleDynamics` prognostic variables (ρ, θˡⁱ, qᵗ) from
# the per-column-interpolated ERA5 fields via the shared `breeze_prognostic_state`
# helper — the same conversion used to populate the parent FTSs above. (qᵗ is
# repartitioned into vapor/condensate by saturation adjustment on the first
# `update_state!`.)

(; ρ, θˡⁱ, qᵗ) = breeze_prognostic_state(constants, T, qᵛ, qᶜ, qⁱ, p)

# ## Build the production model
#
# The actual simulation: real (live, parent-driven `Interpolated`) lateral BCs, microphysics,
# Coriolis, and the Davies fringe forcing. The initial pressure is hydrostatically balanced
# from the surface (above), and an optional dynamical-initialization pass (DFI=on, below) spins
# ρw into nonhydrostatic balance before the run.

# Time discretization: split-explicit acoustic substepping. Adaptive substeps handle the acoustic
# CFL, letting the outer step run at the (slower) advection CFL (so the adaptive wizard below can use
# a large Δt). `ThermalDivergenceDamping` is disabled (`NoDivergenceDamping`) — its (ρθ)′-proxy
# divergence damper injects a spurious force on this unbalanced cold start (Breeze #793).
time_discretization = SplitExplicitTimeDiscretization(sponge = rayleigh_damping, damping = NoDivergenceDamping())

# Momentum advection: WENO(9), higher-order than Fan's 5th/3rd; scalars keep the WENO(5) default.
momentum_advection_scheme = WENO(order = 9)

# Microphysics: 1-moment bulk mixed-phase precipitation (rain + snow) with saturation-adjustment
# cloud formation, so the prognostic moisture is `ρqᵉ` and the precip categories `ρqʳ`, `ρqˢ` are
# added (initialized to zero).
const OneMomentCloudMicrophysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics
microphysics_scheme = OneMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(equilibrium = MixedPhaseEquilibrium()))

# Coriolis: a synoptic-scale LAM forced by ERA5 needs the rotating-frame balance, else the
# ERA5 pressure field accelerates the interior winds with no geostrophic restoring force (the
# unbounded-wind drift). `SphericalCoriolis` gives the latitude-varying f on the lat-lon grid.
coriolis_scheme = SphericalCoriolis()

# Lid sponge: in addition to the in-substepper `UpperSponge`, apply a Rayleigh damping of ρw over the
# top `damping_depth` (cubic ramp, `damping_timescale`) as an interior forcing, so vertically-
# propagating energy is absorbed at the rigid lid rather than reflected.
w_sponge_mask = let z_top = z_discretization.faces[end], depth = float(damping_depth)
    (λ, φ, z) -> (s = clamp((z - (z_top - depth)) / depth, zero(z), one(z)); s * s * (3 - 2s))
end
model_forcing = merge(davies, (ρw = Relaxation(rate = 1/damping_timescale, mask = w_sponge_mask, target = 0.0),))

# Reference potential-temperature profile θ_ref(z) = ERA5 domain/time-mean liquid-ice θ, passed to
# `CompressibleDynamics` so the horizontal pressure-gradient force is taken in perturbation form
# (p′ = p − p_ref). This cuts the terrain-following PGF cancellation error (Klemp 2011) that otherwise
# spuriously accelerates the near-surface winds in the lowest cells over the high western terrain.
reference_θ = let zc = collect(0.5 .* (z_discretization.faces[1:end-1] .+ z_discretization.faces[2:end])),
                  θ̄  = vec(mean(Array(interior(parent_series.θ)), dims = (1, 2, 4)))
    z -> begin
        z <= zc[1]   && return θ̄[1]
        z >= zc[end] && return θ̄[end]
        k = searchsortedlast(zc, z); f = (z - zc[k]) / (zc[k+1] - zc[k])
        (1 - f) * θ̄[k] + f * θ̄[k+1]
    end
end

model = atmosphere_simulation(grid;
                              thermodynamic_constants = constants,
                              momentum_advection  = momentum_advection_scheme,
                              microphysics        = microphysics_scheme,
                              coriolis            = coriolis_scheme,
                              dynamics            = CompressibleDynamics(time_discretization; surface_pressure = p̄₀, reference_potential_temperature = reference_θ),
                              boundary_conditions = bcs,
                              forcing             = model_forcing).model

set!(model; ρ = ρ, u = u, v = v, qᵗ = qᵗ, θˡⁱ = θˡⁱ)

# Consistent-w IC: graft ρw ← ρw − ρw̃ so the contravariant w̃ ≈ 0 (flow follows the ground),
# then re-sync diagnostics.
update_state!(model)
interior(model.momentum.ρw) .-= interior(model.dynamics.contravariant_vertical_momentum)
update_state!(model)
@info @sprintf("IC ready (hydrostatic-balanced p + consistent-w): max|u|=%.2f max|w|=%.2f ρ∈[%.4f,%.4f]",
               maximum(abs, interior(model.velocities.u)), maximum(abs, interior(model.velocities.w)),
               minimum(interior(model.dynamics.density)), maximum(interior(model.dynamics.density)))
flush(stdout); flush(stderr)

# ## Dynamical initialization (DFI / FV3 `na_init`)
#
# ERA5 cold-starts w = 0 (hydrostatic), out of nonhydrostatic balance. Spin ρw into
# balance on a stripped adiabatic twin — no microphysics, sponge, or forcing, frozen
# lateral BCs — then graft the balanced dynamics subset (ρ, ρu, ρv, ρθ, ρw) into the
# production model. `balance_adiabatically!` requires the stripped model: production
# physics/forcing/sponge would corrupt the reversible forward/backward excursion.
let
    # The DFI twin uses `ExplicitTimeStepping`, so its balance step must satisfy the vertical
    # acoustic CFL on the 60 m surface cells (Δz/c ≈ 0.17 s) — independent of the (much larger)
    # split-explicit outer Δt the production run uses.
    Δt_balance     = 0.15
    balance_cycles = 1   # one cycle suffices — see the DFI sensitivity note in the header
    twin = atmosphere_simulation(grid;
                                 thermodynamic_constants = constants,
                                 momentum_advection = momentum_advection_scheme,
                                 dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure = p̄₀),
                                 microphysics = nothing,
                                 boundary_conditions = bcs).model
    set!(twin; ρ = ρ, u = u, v = v, qᵛ = qᵛ, θˡⁱ = θˡⁱ)
    update_state!(twin)
    Breeze.balance_adiabatically!(twin; Δt = Δt_balance, cycles = balance_cycles)
    ρθ_production = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    ρθ_balanced   = Breeze.AtmosphereModels.thermodynamic_density(twin.formulation)
    for (field, balanced) in ((model.dynamics.density, twin.dynamics.density),
                              (model.momentum.ρu, twin.momentum.ρu),
                              (model.momentum.ρv, twin.momentum.ρv),
                              (model.momentum.ρw, twin.momentum.ρw),
                              (ρθ_production, ρθ_balanced))
        interior(field) .= interior(balanced)
    end
    update_state!(model)
    @info @sprintf("DFI done (cycles=%d, Δt=%.3f s): max|u|=%.2f max|w|=%.2f ρ∈[%.4f,%.4f]",
                   balance_cycles, Δt_balance,
                   maximum(abs, interior(model.velocities.u)), maximum(abs, interior(model.velocities.w)),
                   minimum(interior(model.dynamics.density)), maximum(interior(model.dynamics.density)))
    flush(stdout); flush(stderr)
end

# ## Surface drag (bulk Monin–Obukhov-style stress)
#
# `atmosphere_simulation` pre-wires ρτˣ/ρτʸ bottom-flux BC fields for the SlabLand/ocean coupling;
# with no land model attached they stay zero (free-slip). Until the SlabLand coupling is wired here —
# its MOST solve scalar-reads Δz[1] and currently crashes on a GPU stretched terrain grid — fill them
# each step with a bulk neutral surface stress ρτ = −ρ Cᵈ |U| U, per-column log-law Cᵈ = (κ/ln(z₁/z₀))²
# (z₀ = 0.1 m; z₁ = first-cell-center height AGL): the dominant near-surface momentum sink. GPU-safe —
# Cᵈ is precomputed host-side, so there is no per-step scalar Δz read.
let κ_vk = 0.4, z₀_mom = 0.1
    cpu_grid_drag = on_architecture(CPU(), grid)
    z₁_drag = Float64[znode(i, j, 1, cpu_grid_drag, Center(), Center(), Center()) -
                      znode(i, j, 1, cpu_grid_drag, Center(), Center(), Face()) for i in 1:Nx, j in 1:Ny]
    Cd_drag  = on_architecture(arch, @. (κ_vk / log(z₁_drag / z₀_mom))^2)
    ρτx_drag = model.momentum.ρu.boundary_conditions.bottom.condition
    ρτy_drag = model.momentum.ρv.boundary_conditions.bottom.condition
    global function surface_drag!(sim)
        uf = view(interior(model.velocities.u), :, :, 1)
        vf = view(interior(model.velocities.v), :, :, 1)
        ρc = view(interior(model.dynamics.density), :, :, 1)
        uc = 0.5 .* (view(uf, 1:Nx, :) .+ view(uf, 2:Nx+1, :))
        vc = 0.5 .* (view(vf, :, 1:Ny) .+ view(vf, :, 2:Ny+1))
        Um = sqrt.(uc .^ 2 .+ vc .^ 2 .+ 1e-12)
        interior(ρτx_drag) .= reshape(.-(ρc .* Cd_drag .* Um .* uc), size(interior(ρτx_drag)))
        interior(ρτy_drag) .= reshape(.-(ρc .* Cd_drag .* Um .* vc), size(interior(ρτy_drag)))
        return nothing
    end
end

# ## NestedSimulation
#
# Wrap the child model in a `NestedSimulation` paired with the parent
# `PrescribedAtmosphere`. `NestedModel.time_step!` syncs the parent clock
# each iteration so the FTS-driven BCs and forcings get the correct
# interpolation time.
#
# Δt is defined with the grid above; the Davies fringe relaxes on a 10·Δt timescale.

# `NestedSimulation` pairs the prescribed ERA5 parent with the Breeze child; `NestedModel.time_step!`
# advances the child then ticks the parent clock so the FTS-driven BCs/forcing interpolate at the
# right time. To telescope further (ERA5 → 9 km → 3 km) you nest a NestedModel inside another —
# `child = NestedModel(d2_model, d3_model)` — out of scope for this single-nest example.
nested = NestedSimulation(parent, model; Δt, stop_time = 7200.0)   # 2 h (matches end_date above)
add_callback!(nested, surface_drag!, IterationInterval(1))   # bulk surface stress → ρτˣ/ρτʸ each step

# Adaptive outer Δt: the acoustic modes are substepped, so the outer step is bounded by the (slower)
# advective CFL — the wizard floats Δt up to `max_Δt` when the flow is calm and pulls it back during
# active convection (Δz/|w| binding).
conjure_time_step_wizard!(nested, IterationInterval(1); cfl = 0.7, max_Δt = 30.0)

# Terrain-following AGL slice (linear interpolation in physical znode height per column).
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
        z_surface = znode(i, j, 1, host_grid, LX(), LY(), Face())
        z_target  = z_surface + height_agl
        zcol      = [znode(i, j, k, host_grid, LX(), LY(), LZ()) for k in 1:Nz_f]
        slice[i, j] = interp_to_height(zcol, view(data, i, j, :), z_target)
    end
    λ = collect(λnodes(host_grid, LX(), LY(), LZ()))
    φ = collect(φnodes(host_grid, LX(), LY(), LZ()))
    return λ, φ, slice
end

# Two-level horizontal slices for the cascade animation, accumulated in memory by the progress
# callback every `slice_stride` iterations: near-surface (k=1) u, v, θ, qᵛ for the surface wind
# speed + virtual potential temperature, and `slice_height` AGL w, qᵛ, qʳ aloft.
slice_frames = NamedTuple[]
slice_stride = 20          # iterations between slice captures
slice_height = 2000.0      # m AGL for the upper-level slice (w, qᵛ, qʳ)
output_dir   = "."

function progress(sim)
    m  = sim.model
    ρ  = interior(m.dynamics.density)
    pf = Oceananigans.prognostic_fields(m)
    qᵉ = interior(m.moisture_density) ./ ρ                          # specific humidity (vapor + cloud)
    qʳ = haskey(pf, :ρqʳ) ? interior(pf[:ρqʳ]) ./ ρ : zero(ρ)       # rain mixing ratio
    @info @sprintf("iter=%4d t=%6.1fs Δt=%5.2f  max|u|=%7.2f max|v|=%7.2f max|w|=%6.2f  ρ∈[%.4f,%.4f]  qᵉ∈[%.4g,%.4g] qʳ∈[%.2g,%.2g]",
                   m.clock.iteration, m.clock.time, sim.Δt,
                   maximum(abs, interior(m.velocities.u)), maximum(abs, interior(m.velocities.v)),
                   maximum(abs, interior(m.velocities.w)), minimum(ρ), maximum(ρ),
                   minimum(qᵉ), maximum(qᵉ), minimum(qʳ), maximum(qʳ))

    ## near-surface (k=1) u, v, θ, qᵛ (velocities averaged faces→centers) + `slice_height`-AGL w, qᵛ, qʳ
    k1(field) = Array(interior(field))[:, :, 1]
    ρ_k1   = k1(m.dynamics.density)
    θ_sfc  = k1(Breeze.AtmosphereModels.thermodynamic_density(m.formulation)) ./ ρ_k1
    qᵛ_sfc = k1(m.moisture_density) ./ ρ_k1
    uf, vf = k1(m.velocities.u), k1(m.velocities.v)
    u_sfc  = size(uf, 1) > size(ρ_k1, 1) ? 0.5 .* (uf[1:end-1, :] .+ uf[2:end, :]) : uf
    v_sfc  = size(vf, 2) > size(ρ_k1, 2) ? 0.5 .* (vf[:, 1:end-1] .+ vf[:, 2:end]) : vf
    λs, φs, w_up = cut_plane(m.velocities.w, slice_height)
    ρ_up   = cut_plane(m.dynamics.density, slice_height)[3]
    qᵛ_up  = cut_plane(m.moisture_density, slice_height)[3] ./ ρ_up
    qʳ_up  = haskey(pf, :ρqʳ) ? cut_plane(pf[:ρqʳ], slice_height)[3] ./ ρ_up : zero(w_up)
    push!(slice_frames, (t = m.clock.time, λ = λs, φ = φs,
                         u_sfc = u_sfc, v_sfc = v_sfc, θ_sfc = θ_sfc, qᵛ_sfc = qᵛ_sfc,
                         w = w_up, qᵛ = qᵛ_up, qʳ = qʳ_up))
    flush(stdout); flush(stderr)  # Julia bypasses libc buffering — flush so SLURM streams live
end
add_callback!(nested, progress, IterationInterval(slice_stride))

# ## Run
#
# Step the nest to `stop_time`; the progress callback accumulates the cascade-animation slices in memory.

@info @sprintf("Δt₀ = %.2f s; running ERA5 → 3 km Breeze to t = %.0f s", Δt, nested.stop_time)
flush(stdout); flush(stderr)
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

# ## Cascade animation
#
# The headline deliverable: a 2-row × 5-column animation of the downscaling. Row 1 is the ERA5 parent
# (dashed rectangle = the 3 km child extent); row 2 is the Breeze child. Columns are the near-surface
# wind speed `|U|` and the virtual potential temperature perturbation θᵥ′, then `w`, `qᵛ`, `qʳ` at 2 km
# AGL. θᵥ′ is referenced to the initial state — θᵥ′ = θᵥ − θᵥ(t=0), pointwise — so the terrain and
# stratification background (which would swamp an anomaly-from-domain-mean over this terrain) cancels,
# leaving the evolving cold pool. Row 1 samples the resident hourly parent FTS at the child's frame
# times, at the same two levels as the child (`w ≈ 0` and `qʳ` blank — reanalysis carries no model rain).

parent_frames = let zc_p = 0.5 .* (z_discretization.faces[1:end-1] .+ z_discretization.faces[2:end]),
                    λ_p = collect(λnodes(parent_grid, Center(), Center(), Center())),
                    φ_p = collect(φnodes(parent_grid, Center(), Center(), Center()))
    nx_p, ny_p = length(λ_p), length(φ_p)
    function at2km(fts, t)
        a = Array(interior(fts[Time(t)]))
        out = Matrix{Float32}(undef, nx_p, ny_p)
        @inbounds for j in 1:ny_p, i in 1:nx_p
            zt   = parent_orography[i, j] + slice_height
            k    = clamp(searchsortedlast(zc_p, zt), 1, length(zc_p) - 1)
            frac = clamp((zt - zc_p[k]) / (zc_p[k+1] - zc_p[k]), 0, 1)
            out[i, j] = Float32((1 - frac) * a[i, j, k] + frac * a[i, j, k+1])
        end
        return out
    end
    k1(fts, t) = Array(interior(fts[Time(t)]))[:, :, 1]
    cx(a) = size(a, 1) > nx_p ? 0.5 .* (a[1:end-1, :] .+ a[2:end, :]) : a
    cy(a) = size(a, 2) > ny_p ? 0.5 .* (a[:, 1:end-1] .+ a[:, 2:end]) : a
    [(t = f.t, λ = λ_p, φ = φ_p,
      u_sfc = cx(k1(parent.velocities.u, f.t)), v_sfc = cy(k1(parent.velocities.v, f.t)),
      θ_sfc = k1(parent_series.θ, f.t), qᵛ_sfc = k1(parent_series.qᵗ, f.t),
      w = at2km(parent.velocities.w, f.t), qᵛ = at2km(parent_series.qᵗ, f.t),
      qʳ = zeros(Float32, nx_p, ny_p)) for f in slice_frames]
end

# Persist both rows' slices so the animation can be regenerated — and the fields analyzed — offline.
jldsave(output_dir * "/era5_breeze_slices.jld2"; child = slice_frames, parent = parent_frames, height_agl = slice_height)
@info @sprintf("wrote %d child + %d parent slice frames → %s/era5_breeze_slices.jld2",
               length(slice_frames), length(parent_frames), output_dir)

θᵥ(f) = f.θ_sfc .* (1 .+ 0.61f0 .* f.qᵛ_sfc)   # virtual potential temperature θᵥ ≈ θ(1 + 0.61 qᵛ)
cascade_fields(f, θᵥ₀) = (; U = sqrt.(f.u_sfc .^ 2 .+ f.v_sfc .^ 2),
                          w = f.w, θvp = θᵥ(f) .- θᵥ₀, qv = f.qᵛ .* 1f3, qr = f.qʳ .* 1f3)
child_fields  = [cascade_fields(f, θᵥ(slice_frames[1]))  for f in slice_frames]
parent_fields = [cascade_fields(f, θᵥ(parent_frames[1])) for f in parent_frames]

cascade_range(key, hi) = quantile(filter(isfinite, abs.(reduce(vcat,
                            [vec(getproperty(d, key)) for d in vcat(child_fields, parent_fields)]))), hi)
Umax  = max(cascade_range(:U,   0.995), 5)
wmax  = max(cascade_range(:w,   0.995), 1)
θmax  = max(cascade_range(:θvp, 0.99),  0.5)
qvmax = max(cascade_range(:qv,  0.995), 1)
qrmax = max(cascade_range(:qr,  0.999), 0.1)
cascade_columns = [(:U,   "|U|ₛ (m s⁻¹)",   :speed,   (0, Umax)),
                   (:w,   "w₂ₖₘ (m s⁻¹)",   :balance, (-wmax, wmax)),
                   (:θvp, "θᵥ′ₛ (K)",       :balance, (-θmax, θmax)),
                   (:qv,  "qᵛ₂ₖₘ (g kg⁻¹)", :dense,   (0, qvmax)),
                   (:qr,  "qʳ₂ₖₘ (g kg⁻¹)", :dense,   (0, qrmax))]

λbox = extrema(slice_frames[1].λ); φbox = extrema(slice_frames[1].φ)
boxλ = [λbox[1], λbox[2], λbox[2], λbox[1], λbox[1]]
boxφ = [φbox[1], φbox[1], φbox[2], φbox[2], φbox[1]]

fig_cascade = Figure(size = (1500, 640))
cascade_n   = Observable(1)
Label(fig_cascade[0, 1:5],
      (@lift @sprintf("MC3E 20 May 2011 — ERA5 → 3 km Breeze — t = %.1f h", slice_frames[$cascade_n].t / 3600)),
      fontsize = 20, tellwidth = false)
for (ci, (key, label, cmap, crange)) in enumerate(cascade_columns)
    parent_ax = Axis(fig_cascade[1, ci]; title = label, aspect = DataAspect())
    child_ax  = Axis(fig_cascade[2, ci]; aspect = DataAspect())
    heatmap!(parent_ax, parent_frames[1].λ, parent_frames[1].φ,
             (@lift getproperty(parent_fields[$cascade_n], key)); colormap = cmap, colorrange = crange)
    lines!(parent_ax, boxλ, boxφ; color = :black, linestyle = :dash, linewidth = 1.5)
    hm = heatmap!(child_ax, slice_frames[1].λ, slice_frames[1].φ,
                  (@lift getproperty(child_fields[$cascade_n], key)); colormap = cmap, colorrange = crange)
    Colorbar(fig_cascade[3, ci], hm; vertical = false, flipaxis = false, height = 10)
    hidedecorations!(parent_ax); hidedecorations!(child_ax)
    if ci == 1
        text!(parent_ax, 0.03, 0.97; text = "ERA5 (D1)",   space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
        text!(child_ax,  0.03, 0.97; text = "Breeze 3 km", space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
    end
end
CairoMakie.record(fig_cascade, output_dir * "/era5_cascade_2row.mp4", 1:length(slice_frames); framerate = 8) do nn
    cascade_n[] = nn   # CairoMakie.record: `record` is also exported by CUDA, so qualify it
end
@info @sprintf("wrote %s/era5_cascade_2row.mp4 (%d frames)", output_dir, length(slice_frames))
