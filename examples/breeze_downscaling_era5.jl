# # ERA5 → 3 km convection-permitting hindcast (Breeze + NestedSimulation)
#
# A limited-area model (LAM) example that downscales ERA5 reanalysis to a 3 km Breeze compressible
# atmosphere over the U.S. Southern Great Plains, for the Midlatitude Continental
# Convective Clouds Experiment (MC3E) 20 May 2011 squall-line case ([Fan2017](@citet)).
# `nested_atmosphere_model(parent, child_grid; …)` builds a `NestedModel`, which pairs a "parent"
# `PrescribedAtmosphere` with a "child" Breeze `AtmosphereModel`. The parent here is an ERA5
# `PrescribedAtmosphere` (on its native 0.25° pressure-level grid), driving a ~3 km Breeze child
# through open lateral boundary conditions + interior Davies relaxation — both derived on the fly from
# the parent's raw state. A plain `Simulation(model)` then steps it (the `NestedModel`'s `time_step!`
# advances the child and ticks the parent clock).
#
# ## What this example does
# - Downloads ERA5 (pressure + single levels) for a fixed parent region and regrids it onto a
#   terrain-following `LatitudeLongitudeGrid` (ETOPO2022 orography, tapered to the parent at the edge).
# - Initializes the prognostic state from ERA5 with a hydrostatic-from-surface pressure and a
#   terrain-consistent `w̃ ≈ 0`; a dynamical-initialization (DFI) pass then spins `ρw` into balance.
# - Integrates the compressible equations with split-explicit acoustic substepping (adaptive
#   substeps + an adaptive outer-Δt wizard), 1-moment mixed-phase microphysics, Coriolis, a
#   reference-θ perturbation-form pressure-gradient, bulk surface drag, and Rayleigh damping.
# - Writes and animates horizontal slices.
#
# ## What it does NOT do (yet)
# - Single nest only (ERA5 → 3 km). The window here is 2 h to keep the example short; the full MC3E
#   study in [Fan2017](@citet) was run for 18 h, with most analyses performed between 6 and 12 h.
# - No land/ocean coupling (surface stress is a bulk-drag stand-in; the SlabLand MOST link is unwired).
# - No boundary layer parameterization: diffusion is provided by numerical dissipation.
# - No cumulus parameterization: 3 km is convection-*permitting*, so deep convection is resolved on
#   the grid.
# - No `RectilinearGrid` (constant Δx, Δy) with map projection.
#
# ## What we attempted / known issues
# - First set up as a telescoping nest: ERA5 → 9 km → 3 km. The 9 km middle nest sits in the convective
#   grey zone, where under-resolved convection runs away (a vertical-mixing band-aid bounds but never
#   makes it physical). Reducing to ERA5 → 3 km direct sidesteps the 9 km grey zone.
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
using Breeze
using CloudMicrophysics  # loaded so nested_atmosphere_model's default microphysics is 1-moment mixed-phase
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization
using Statistics: mean, quantile
using JLD2: jldsave
using Dates
using Printf

# This 12 km LAM (150×136×50 ≈ 1.0M cells, split-explicit) targets a CUDA GPU; switch to `CPU()` only
# for a small smoke test. (Coarsened 4× from Fan's 3 km Domain 3, over a 2×-expanded domain.)
using CUDA
const arch = GPU(CUDA.CUDABackend(always_inline = true))

# Set Oceananigans' global default float type, cascading to all grids, fields, FieldTimeSeries, constants, and dynamics.
Oceananigans.defaults.FloatType = Float32

# ## Configuration

# ### Domain
#
# Domain centered on the U.S. Department of Energy's Atmospheric Radiation Measurement (ARM) Climate
# Research Facility's Southern Great Plains (SGP) site in Lamont, OK. We take the 3 km domain (Domain 3)
# of the WRF 27 → 9 → 3 → 1 km telescoping nest used by [Fan2017](@citet) for this MC3E
# case, driven directly by ERA5 (the parent), but coarsened 4× to ~12 km for a fast configuration.
#
# Note that the Breeze cells are anisotropic at this latitude, using R = 6,371 km:
#   Δx = R·cos(φ₀)·Δλ ≈ 9.9 km
#   Δy = R·Δφ         ≈ 12.4 km

φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)

Δλ = Δφ = 1/9               # uniform 1/9° step (~12 km; 4× coarser than Fan's 3 km Domain 3)
Nx, Ny = 150, 136           # ~12 km cells over a 2×-expanded domain (vs Fan Domain 3 footprint)

# From these inputs, we determine the `BoundingBox` corners.

λ_west  = λ₀ - Nx * Δλ / 2
λ_east  = λ₀ + Nx * Δλ / 2
φ_south = φ₀ - Ny * Δφ / 2
φ_north = φ₀ + Ny * Δφ / 2

# Relaxation zone geometry: 5 cells deep in each lateral direction

relax_width = 5
relax_width_deg = relax_width * max(Δλ, Δφ)

# Vertical grid matched to [Fan2017](@citet)'s WRF nest with 51 staggered levels → `Nz = 50`
# cells, a constant 60 m surface cell, and a 490 m maximum spacing. Stretching ratio is estimated
# to give a model top at Lz ≈ 20 km (~50 hPa, WRF's default model top).

z_discretization = ReferenceToStretchedDiscretization(
    extent                  = 19525.0,
    bias                    = :left,
    bias_edge               = 0.0,
    constant_spacing        = 60.0,
    constant_spacing_extent = 60.0,
    maximum_spacing         = 490.0,
    stretching              = LinearStretching(0.15))

Nz = length(z_discretization)
@assert Nz == 50

# ### Simulation time control

start_date = DateTime(2011, 05, 20, 0)
end_date   = DateTime(2011, 05, 20, 12) # 12 h window (the full MC3E case in Fan2017 was 18 h)

dates = start_date:Hour(1):end_date

# Initial outer time step for the adaptive wizard (configured after the simulation, below).
# Split-explicit substepping integrates the acoustic modes, so the outer step is advection-limited;
# we start gentle at 1 s — which also avoids amplifying the cold-start transient — and let the wizard
# ramp it toward `max_Δt`. The Davies relaxation timescale `τ_relax` is tied to this Δt.

Δt = 1.0

# ### ERA5 reanalysis
#
# Used for initial and boundary conditions.
# Pressure-level variables are regridded onto the parent grid as `FieldTimeSeries`
# (and onto the child grid for the initial condition) further below.
# Hourly datasets define metadata for data retrieval.

era5_datadir = "era5"   # Where data will be saved locally

ds_pl = ERA5HourlyPressureLevels()
ds_sl = ERA5HourlySingleLevel()

# ERA5 forcing region: the LAM footprint padded outward by `era5_pad`, so the parent encloses the
# child with room for the lateral BCs + the 5-cell Davies relaxation zone. `BoundingBox` snaps the
# requested region outward onto ERA5's native 0.25° grid when the data is read (the native grid is
# restricted by `floor`/`ceil` to the enclosing cells), so no manual snapping is needed. Downloaded on
# demand; at 0.25°, ERA5 stands in for Fan's 27 km Domain 1.

era5_pad = 1.0  # deg; wider than the 5·(1/12°) ≈ 0.42° Davies relaxation zone width

era5_region = BoundingBox(longitude = (λ_west  - era5_pad, λ_east  + era5_pad),
                          latitude  = (φ_south - era5_pad, φ_north + era5_pad))

@info @sprintf("Breeze child (~12 km): λ ∈ [%.3f, %.3f], φ ∈ [%.3f, %.3f]; Δλ=Δφ=%.4f°",
               λ_west, λ_east, φ_south, φ_north, Δλ)
@info @sprintf("ERA5 parent (D1 role, padded + snapped to 0.25°): λ ∈ [%.2f, %.2f], φ ∈ [%.2f, %.2f]",
               era5_region.longitude[1], era5_region.longitude[2],
               era5_region.latitude[1],  era5_region.latitude[2])

# ## Setup LAM grid
#
# We create a bounded terrain-following `LatitudeLongitudeGrid` with a
# `TerrainFollowingVerticalDiscretization` built from our custom stretching profile

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z         = TerrainFollowingVerticalDiscretization(z_discretization),
                             size      = (Nx, Ny, Nz),
                             halo      = (5, 5, 5),
                             topology  = (Bounded, Bounded, Bounded))

# Get the parent terrain: ERA5 surface geopotential regridded onto the child's horizontal grid as a
# 2-D `(Center, Center, Nothing)` field — no separate orography grid needed.

g_accel = Oceananigans.defaults.gravitational_acceleration
Φ_sfc = Field{Center, Center, Nothing}(grid)
set!(Φ_sfc, Metadatum(:geopotential; dataset = ds_sl, date = start_date,
                      region = era5_region, dir = era5_datadir))
parent_elevation = Array(interior(Φ_sfc))[:, :, 1] ./ g_accel

# ETOPO 2022, with 60" (~1.85 km) relief, is finer than the our LAM grid cells. The
# surface elevation (≥ 0; ocean clamped to sea level) is regridded onto
# the LAM horizontal grid.

elevation = regrid_topography(grid; dataset = ETOPO2022())

# Now blend the terrain across the lateral relaxation zone such that the terrain along the boundary
# matches the parent elevation and at the interior edge matches the child elevation.

etopo_full = Array(interior(elevation))[:, :, 1]
blended = similar(etopo_full)
for j in 1:Ny, i in 1:Nx
    weight = clamp(min(i - 1, Nx - i, j - 1, Ny - j) / relax_width, 0, 1)
    blended[i, j] = weight * etopo_full[i, j] + (1 - weight) * parent_elevation[i, j]
end
set!(elevation, reshape(blended, size(interior(elevation))))

# `elevation` is materialized onto the grid's terrain-following coordinate inside
# `nested_atmosphere_model` (its `terrain` keyword), deforming the coordinate surfaces to follow the
# ground (a Gal-Chen–Somerville σ coordinate, default `LinearDecay`): the bottom sits at the local
# terrain height, the top stays flat. `CompressibleDynamics` then builds the slope metrics it needs
# directly from the grid, and the #241 ERA5 ingest below interpolates onto the true `znode` heights
# above sea level.

# ## Nested domains
#
# Visualize the nesting before stepping the model: the ERA5 forcing region that supplies the parent
# state (lateral BCs + Davies relaxation) and the 3 km LAM — Fan (2017)'s Domain 3, the `NestedSimulation`
# child — over ETOPO terrain with Natural Earth state/country boundaries, centered on ARM SGP.
# Drawn here, before the run, so the domain geometry is written even if the run is cut short.

using CairoMakie
using NaturalEarth   # with its transitive GeoInterface, triggers NumericalEarthNaturalEarthExt → `natural_earth_lines`

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
               title  = "ERA5 → 12 km LAM nest (MC3E squall line, ARM SGP)",
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
                                 ("admin_0_boundary_lines_land",    (:black,  0.75), 1.4))
    lon, lat = natural_earth_lines(name)
    lines!(ax_map, lon, lat; color, linewidth)
end

lines!(ax_map, domain_box(era5_region.longitude..., era5_region.latitude...)...;
       color = :dodgerblue, linewidth = 3, label = "ERA5 parent — Fan Domain 1 role")
lines!(ax_map, domain_box(λ_west, λ_east, φ_south, φ_north)...;
       color = :crimson, linewidth = 3, label = "12 km LAM (child)")
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

# ## Interpolate ERA5 onto the LAM grid
#
# ### Parent grid
#
# The parent grid is in ERA5 native coordinates: (λ, φ), regular true-height z
# (non-terrain-following).
# `Field(metadatum, grid)` and `set!(field, metadatum)` regrid ERA5 pressure-level
# data onto an arbitrary target grid, using the true per-column geopotential height
# z = Φ(λ, φ, p)/g as the vertical coordinate and clipping sub-surface levels at
# the local surface (through NumericalEarth's `PressureLevelGrid`).
# The interpolation is driven by the *target* grid's own node heights, so the
# terrain-following child is sampled at its true physical heights.
#
# These regrids interpolate linearly in height between ERA5 levels for T, qᵛ, qᶜ, qⁱ. Pressure
# is not interpolated but instead built by hydrostatic integration from the ERA5 surface
# pressure (see `hydrostatic_pressure_from_surface`), keeping it in discrete hydrostatic balance.

# `meta_common_snap1` (snapshot 1) is reused by the initial-condition regrids below.
const meta_common_snap1 = (date = start_date, region = era5_region, dir = era5_datadir)

# ### Prescribed parent atmosphere
#
# `ERA5PrescribedAtmosphere(bounding_box, dates)` loads the parent state (u, v, T, qᵛ and the
# cloud/precip species) onto ERA5's *native* pressure-level grid (geopotential-height aware). The
# nested child below interpolates this parent on the fly for its lateral BCs and Davies relaxation —
# no materialized parent prognostic series, no `breeze_prognostic_state` derivation loop.
parent = ERA5PrescribedAtmosphere(era5_region, dates; architecture = arch, dir = era5_datadir)

# ERA5 pressure velocity ω (Pa/s) on the parent's native grid — the animation maps it to w ≈ −ω/(ρg).
ω_series = FieldTimeSeries(Metadata(:vertical_velocity; dataset = ds_pl, dates = dates,
                                    region = era5_region, dir = era5_datadir),
                           arch; time_indices_in_memory = length(dates))

# ## Surface pressure
#
# Surface pressure: horizontal field on a 3-D grid with Nz=1. (A `Flat` z
# topology trips Oceananigans' interpolation kernel — `_fractional_indices`
# unconditionally destructures `at_node` as `(x, y, z)`; mirrors the pattern
# used in examples/ERA5_hourly_data.jl.) Its domain mean `p̄₀` anchors the model's hydrostatic
# reference at construction; `p₀` itself seeds the hydrostatic-from-surface initial pressure below.

surface_grid = LatitudeLongitudeGrid(longitude = (λ_west,  λ_east),
                                     latitude  = (φ_south, φ_north),
                                     z         = (0, 1),
                                     size      = (Nx, Ny, 1),
                                     halo      = (5, 5, 3),
                                     topology  = (Bounded, Bounded, Bounded))

p₀ = CenterField(surface_grid)
set!(p₀, Metadatum(:surface_pressure; dataset=ds_sl, meta_common_snap1...))
p̄₀ = mean(interior(p₀))

# ## Lateral boundary conditions and Davies relaxation
#
# Both are derived on the fly from the parent's raw ERA5 state by `nested_atmosphere_model` below —
# no materialized parent prognostic series. Internally, each density-weighted child prognostic
# (`ρ, ρu, ρv, ρe, ρqᵉ`) gets a `ParentStateBoundary` that interpolates the parent
# `(u, v, T, qᵛ, qᶜˡ, qᶜⁱ, p)` at the boundary face and applies the matching Breeze transform
# (strictly-positive `p`, `T` interpolate in log space); `relaxation_rate`/`relaxation_mask` add the
# interior Davies nudging toward the parent's `(u, v, θˡⁱ, qᵗ)` (Breeze's `SpecificForcing` applies the
# ρ-weight at the right face stagger).

# Surface-BC placeholders, pending SlabLand wiring. We pass *bottom-only* `FieldBoundaryConditions`
# for `ρe`/`ρqᵉ`; `nested_atmosphere_model` merges them per-side with the parent-derived lateral BCs
# (caller wins per side), so these override `atmosphere_model`'s coupling Jᵉ/Jᵛ bottom-flux BCs with
# Dirichlet ValueBCs at constant placeholder surface state. Keeping the coupling Jᵉ would route the
# bottom flux through Breeze's `EnergyFluxBoundaryCondition` → `𝒬_to_Jᶿ`, which can't evaluate until
# the land model populates the bulk-flux state.
const T_surface_placeholder   = 290.0
const qᵛ_surface_placeholder  = 0.0
const ρ_surface_placeholder   = 1.2                                   # kg/m³ at p₀=10⁵ Pa, T≈290 K
const ρθ_surface_placeholder  = ρ_surface_placeholder * T_surface_placeholder
const ρqᵉ_surface_placeholder = ρ_surface_placeholder * qᵛ_surface_placeholder

surface_bcs = (ρe  = FieldBoundaryConditions(bottom = ValueBoundaryCondition(ρθ_surface_placeholder)),
               ρqᵉ = FieldBoundaryConditions(bottom = ValueBoundaryCondition(ρqᵉ_surface_placeholder)))

# The mask is a cosine ramp in degree-distance to the nearest wall — Davies is a
# numerical smoother, so the precise ramp shape isn't physics-critical.
#
# Capture domain extents + relaxation width into closure-local bindings so the
# resulting function is type-stable (required for GPU kernel compilation —
# non-const module globals produce dynamic-dispatch IR).
lateral_mask = let λ_w = λ_west, λ_e = λ_east, φ_s = φ_south, φ_n = φ_north, width = relax_width_deg
    (λ, φ, z) -> begin
        dW = λ - λ_w
        dE = λ_e - λ
        dS = φ - φ_s
        dN = φ_n - φ
        d  = min(dW, dE, dS, dN)
        d >= width && return zero(λ)
        return oftype(λ, 0.5 * (1 + cos(π * d / width)))
    end
end

τ_relax = 10 * Δt  # relaxation timescale (s); passed to `nested_atmosphere_model` as `relaxation_rate = 1/τ_relax`

# ## Build the Breeze model
#
# `nested_atmosphere_model` builds the child `AtmosphereModel` (via the same `atmosphere_model` helper
# that pre-wires the ρτˣ/ρτʸ/Jᵉ/Jᵛ bottom-flux BC fields for the forthcoming SlabLand / SlabOcean
# coupling), derives the parent-driven lateral BCs + Davies relaxation, and returns a `NestedModel` —
# no `.model`/`.child` unpacking. Its skeleton `CoupledRadiation` is a no-op (radiatively decoupled)
# until materialized inside an `EarthSystemModel`.
#
# `nested_atmosphere_model` supplies the nested-LAM physics defaults: compressible split-explicit
# dynamics with an `UpperSponge` + ρw Rayleigh lid sponge (no divergence damping, which would inject
# a spurious force on this unbalanced cold start), `WENO(order = 9)` momentum advection,
# `SphericalCoriolis`, and — since `CloudMicrophysics` is loaded — 1-moment bulk mixed-phase
# (rain + snow) microphysics. We pass only the IC-derived `surface_pressure` /
# `reference_potential_temperature` that anchor the default dynamics (computed next).

# `nested_atmosphere_model` builds the child `AtmosphereModel` over `grid`, derives its lateral BCs +
# Davies relaxation on the fly from `parent`, applies the physics defaults above, materializes the
# blended `terrain` onto the grid's terrain-following coordinate, and wraps the pair in a `NestedModel`
# — whose `time_step!` advances the child then ticks the parent clock so the on-the-fly BCs/relaxation
# sample the parent at the right time. `surface_bcs` merge per-side with the lateral BCs. The
# hydrostatic reference is recomputed from the initial state below (`compute_reference_state = true`),
# so no reference-θ profile need be supplied here.
model = nested_atmosphere_model(parent, grid;
                                thermodynamic_constants = constants,
                                surface_pressure = p̄₀,
                                terrain = elevation,
                                relaxation_rate = 1 / τ_relax,
                                relaxation_mask = lateral_mask,
                                boundary_conditions = surface_bcs)

# ## Initial conditions
#
# With the terrain now materialized on `grid`, regrid snapshot 1 of ERA5 directly onto the child:
# `set!(field, metadatum)` interpolates each ERA5 field by true Φ/g (#241) onto the deformed
# terrain-following coordinate, staggering to the field's own location (velocities to faces, scalars to
# centers). No parent → child step is needed.

initial_metadatum(name) = Metadatum(name; dataset = ds_pl, meta_common_snap1...)

u  = XFaceField(grid);  set!(u,  initial_metadatum(:eastward_velocity))
v  = YFaceField(grid);  set!(v,  initial_metadatum(:northward_velocity))
T  = CenterField(grid); set!(T,  initial_metadatum(:temperature))
qᵛ = CenterField(grid); set!(qᵛ, initial_metadatum(:specific_humidity))
qᶜ = CenterField(grid); set!(qᶜ, initial_metadatum(:specific_cloud_liquid_water_content))
qⁱ = CenterField(grid); set!(qⁱ, initial_metadatum(:specific_cloud_ice_water_content))

# Virtual temperature Tᵛ = T·(1 + (Rᵛ/Rᵈ − 1)·qᵛ) — vapor only by convention (the qᶜ, qⁱ terms belong
# to the density temperature Tρ); used in the IC diagnostics below.
εfac = Rᵛ / Rᵈ - 1
Tᵛ = Field(T * (1 + εfac * qᵛ))
compute!(Tᵛ)

# Hydrostatically-balanced initial pressure. Interpolating ERA5 pressure to the node heights
# clamps the sub-surface levels over high terrain, leaving the cold-start IC out of the model's
# discrete hydrostatic balance (a ~40 g vertical residual). Build `p` by integrating up from the
# ERA5 surface pressure instead — anchored at each column's terrain surface, with the moist Rᵐ.
p = hydrostatic_pressure_from_surface(T, Array(interior(p₀))[:, :, 1], parent_elevation;
                                      qᵛ = qᵛ, qᶜ = qᶜ, qⁱ = qⁱ,
                                      dry_gas_constant = Rᵈ, vapor_gas_constant = Rᵛ,
                                      gravitational_acceleration = g_accel)

# Initial density from the moist ideal-gas law ρ = p / (Rᵐ T), with Rᵐ = (1 − qᵛ − qᶜ − qⁱ)Rᵈ + qᵛRᵛ —
# the same EOS used to build `p`, so the IC is internally consistent. Total water qᵗ = qᵛ + qᶜ + qⁱ.
ρ  = density_from_pressure(T, p; qᵛ = qᵛ, qᶜ = qᶜ, qⁱ = qⁱ,
                           dry_gas_constant = Rᵈ, vapor_gas_constant = Rᵛ)
qᵗ = Field(qᵛ + qᶜ + qⁱ); compute!(qᵗ)

# Initial state from ERA5; `set!` on the `NestedModel` forwards to the child. Temperature `T` is set
# directly (Breeze inverts it to θˡⁱ given the moisture state); `compute_reference_state = true` then
# recomputes the dynamics' Exner reference from the resulting domain-mean state, so the pressure-gradient
# force is taken in perturbation form (p′ = p − p_ref) — cutting the terrain-following PGF cancellation
# error (Klemp 2011) that otherwise spuriously accelerates the near-surface winds over high terrain,
# without a hand-supplied reference-θ profile.
set!(model; ρ = ρ, u = u, v = v, qᵗ = qᵗ, T = T, compute_reference_state = true)

# Consistent-w IC: graft ρw ← ρw − ρw̃ so the contravariant w̃ ≈ 0 (flow follows the ground),
# then re-sync diagnostics.
update_state!(model)
interior(model.child.momentum.ρw) .-= interior(model.child.dynamics.contravariant_vertical_momentum)
update_state!(model)
@info @sprintf("IC ready (hydrostatic-balanced p + consistent-w): max|u|=%.2f max|w|=%.2f ρ∈[%.4f,%.4f]",
               maximum(abs, interior(model.child.velocities.u)), maximum(abs, interior(model.child.velocities.w)),
               minimum(interior(model.child.dynamics.density)), maximum(interior(model.child.dynamics.density)))
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
    ## Frozen lateral BCs matching production: interpolate the same parent-derived child prognostics
    ## (held by the nested model's exchanger) at the twin's boundaries.
    twin_prog = model.exchanger.prognostic
    twin_moisture = Breeze.moisture_prognostic_name(nothing)
    twin_bcs = parent_boundary_conditions(grid;
        variables = merge((; ρᵈ = twin_prog.ρᵈ, ρu = twin_prog.ρu, ρv = twin_prog.ρv, ρe = twin_prog.ρθ),
                          NamedTuple{tuple(twin_moisture)}(tuple(twin_prog.ρqᵛ))),
        bc_types  = merge((; ρᵈ = ValueBoundaryCondition, ρe = ValueBoundaryCondition),
                          NamedTuple{tuple(twin_moisture)}(tuple(ValueBoundaryCondition))))
    twin = atmosphere_model(grid;
                            thermodynamic_constants = constants,
                            momentum_advection = WENO(order = 9),
                            dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure = p̄₀),
                            microphysics = nothing,
                            boundary_conditions = twin_bcs)
    set!(twin; ρ = ρ, u = u, v = v, qᵛ = qᵛ, T = T)
    update_state!(twin)
    Breeze.balance_adiabatically!(twin; Δt = Δt_balance, cycles = balance_cycles)
    ρθ_production = Breeze.AtmosphereModels.thermodynamic_density(model.child.formulation)
    ρθ_balanced   = Breeze.AtmosphereModels.thermodynamic_density(twin.formulation)
    for (field, balanced) in ((model.child.dynamics.density, twin.dynamics.density),
                              (model.child.momentum.ρu, twin.momentum.ρu),
                              (model.child.momentum.ρv, twin.momentum.ρv),
                              (model.child.momentum.ρw, twin.momentum.ρw),
                              (ρθ_production, ρθ_balanced))
        interior(field) .= interior(balanced)
    end
    update_state!(model)
    @info @sprintf("DFI done (cycles=%d, Δt=%.3f s): max|u|=%.2f max|w|=%.2f ρ∈[%.4f,%.4f]",
                   balance_cycles, Δt_balance,
                   maximum(abs, interior(model.child.velocities.u)), maximum(abs, interior(model.child.velocities.w)),
                   minimum(interior(model.child.dynamics.density)), maximum(interior(model.child.dynamics.density)))
    flush(stdout); flush(stderr)
end

# ## Surface drag (bulk Monin–Obukhov-style stress)
#
# `atmosphere_model` pre-wires ρτˣ/ρτʸ bottom-flux BC fields for the SlabLand/ocean coupling;
# with no land model attached they stay zero (free-slip). Until the SlabLand coupling is wired here —
# its MOST solve scalar-reads Δz[1] and currently crashes on a GPU stretched terrain grid — fill them
# each step with a bulk neutral surface stress ρτ = −ρ Cᵈ |U| U, per-column log-law Cᵈ = (κ/ln(z₁/z₀))²
# (z₀ = 0.1 m; z₁ = first-cell-center height AGL): the dominant near-surface momentum sink. GPU-safe —
# Cᵈ is precomputed host-side, so there is no per-step scalar Δz read.
#
# TODO: Wire up SlabLand/ocean coupling
let κ_vk = 0.4, z₀_mom = 0.1
    cpu_grid_drag = on_architecture(CPU(), grid)
    z₁_drag = Float64[znode(i, j, 1, cpu_grid_drag, Center(), Center(), Center()) -
                      znode(i, j, 1, cpu_grid_drag, Center(), Center(), Face()) for i in 1:Nx, j in 1:Ny]
    Cd_drag  = on_architecture(arch, @. (κ_vk / log(z₁_drag / z₀_mom))^2)
    ρτx_drag = model.child.momentum.ρu.boundary_conditions.bottom.condition
    ρτy_drag = model.child.momentum.ρv.boundary_conditions.bottom.condition
    global function surface_drag!(sim)
        uf = view(interior(model.child.velocities.u), :, :, 1)
        vf = view(interior(model.child.velocities.v), :, :, 1)
        ρc = view(interior(model.child.dynamics.density), :, :, 1)
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
# `NestedSimulation` pairs the prescribed ERA5 parent with the Breeze child; `NestedModel.time_step!`
# advances the child then ticks the parent clock so the FTS-driven BCs/forcing interpolate at the
# right time. To telescope further (ERA5 → 9 km → 3 km) you nest a NestedModel inside another —
# `child = NestedModel(d2_model, d3_model)` — out of scope for this single-nest example.

# `model` is a `NestedModel`, so a plain `Simulation` just works: its `time_step!` advances the Breeze
# child then ticks the parent clock. To telescope further (ERA5 → 9 km → 3 km) you nest a NestedModel
# inside another — `nested_atmosphere_model(d2_model, d3_grid; …)` — out of scope for this single nest.
simulation = Simulation(model; Δt, stop_time = 43200.0)   # 12 h (matches end_date above)
add_callback!(simulation, surface_drag!, IterationInterval(1))   # bulk surface stress → ρτˣ/ρτʸ each step

# Adaptive outer Δt: the acoustic modes are substepped, so the outer step is bounded by the (slower)
# _advective_ CFL.

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl = 0.7, max_Δt = 30)

# ## Setup coprocessing

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
add_callback!(simulation, progress, IterationInterval(slice_stride))

# ## Run
#
# Step the nest to `stop_time`; the progress callback accumulates the cascade-animation slices in memory.

@info @sprintf("Δt₀ = %.2f s; running ERA5 → 12 km Breeze to t = %.0f s", Δt, simulation.stop_time)
flush(stdout); flush(stderr)
run!(simulation)
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
# leaving the evolving cold pool. Row 1 reconstructs the parent prognostics on the fly from the ERA5
# `PrescribedAtmosphere` at the child's frame times, at the same two levels as the child. ERA5's `w` is estimated from its pressure velocity ω as
# w ≈ −ω/(ρg) (synoptic-scale, far weaker than the child's resolved convection); `qʳ` is blank (no model rain).

parent_frames = let pg = parent.grid, pˢᵗ = model.child.dynamics.standard_pressure
    λ_p = collect(λnodes(pg, Center(), Center(), Center()))
    φ_p = collect(φnodes(pg, Center(), Center(), Center()))
    nx_p, ny_p = length(λ_p), length(φ_p)
    cx(a) = size(a, 1) > nx_p ? 0.5 .* (a[1:end-1, :] .+ a[2:end, :]) : a
    cy(a) = size(a, 2) > ny_p ? 0.5 .* (a[:, 1:end-1] .+ a[:, 2:end]) : a
    k1(field) = Array(interior(field))[:, :, 1]
    function frame(t)
        ## Reconstruct the parent prognostics (ρ, θˡⁱ, qᵗ) on the fly from the raw ERA5 state at `t` —
        ## the same transform the lateral BCs use — on the parent's native geopotential-height grid,
        ## then `cut_plane` to the surface and `slice_height` AGL exactly as for the child row.
        (; ρ, θˡⁱ, qᵗ) = breeze_prognostic_state(constants, pˢᵗ,
                            parent.temperature[Time(t)], parent.specific_humidity[Time(t)],
                            parent.microphysical_variables.qᶜˡ[Time(t)],
                            parent.microphysical_variables.qᶜⁱ[Time(t)], parent.pressure)
        ρ_up = cut_plane(ρ, slice_height)[3]
        return (t = t, λ = λ_p, φ = φ_p,
                u_sfc = cx(k1(parent.velocities.u[Time(t)])), v_sfc = cy(k1(parent.velocities.v[Time(t)])),
                θ_sfc = k1(θˡⁱ), qᵛ_sfc = k1(qᵗ),
                w = -cut_plane(ω_series[Time(t)], slice_height)[3] ./ (ρ_up .* g_accel),
                qᵛ = cut_plane(qᵗ, slice_height)[3], qʳ = zeros(Float32, nx_p, ny_p))
    end
    [frame(f.t) for f in slice_frames]
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
row_range(fields, key, hi) = quantile(filter(isfinite, abs.(reduce(vcat,
                            [vec(getproperty(d, key)) for d in fields]))), hi)
Umax  = max(cascade_range(:U,   0.995), 5)
# w spans ~10× between the ERA5 parent (~0.1 m/s) and the 3 km child (~2 m/s) — scale each row separately
wmax_parent = max(row_range(parent_fields, :w, 0.995), 0.3)
wmax_child  = max(row_range(child_fields,  :w, 0.995), 1)
θmax  = max(cascade_range(:θvp, 0.99),  0.5)
qvmax = max(cascade_range(:qv,  0.995), 1)
qrmax = max(cascade_range(:qr,  0.999), 0.1)
cascade_columns = [(:θvp, "θᵥ′ₛ (K)",       :balance, (-θmax, θmax)),
                   (:U,   "|U|ₛ (m s⁻¹)",   :speed,   (0, Umax)),
                   (:w,   "w₂ₖₘ (m s⁻¹)",   :balance, (-wmax_child, wmax_child)),
                   (:qv,  "qᵛ₂ₖₘ (g kg⁻¹)", :dense,   (0, qvmax)),
                   (:qr,  "qʳ₂ₖₘ (g kg⁻¹)", :dense,   (0, qrmax))]

λbox = extrema(slice_frames[1].λ); φbox = extrema(slice_frames[1].φ)
boxλ = [λbox[1], λbox[2], λbox[2], λbox[1], λbox[1]]
boxφ = [φbox[1], φbox[1], φbox[2], φbox[2], φbox[1]]

fig_cascade = Figure(size = (1500, 700))
cascade_n   = Observable(1)
Label(fig_cascade[0, 1:5],
      (@lift @sprintf("MC3E 20 May 2011 — ERA5 → 12 km Breeze — t = %.1f h", slice_frames[$cascade_n].t / 3600)),
      fontsize = 20, tellwidth = false)
for (ci, (key, label, cmap, crange)) in enumerate(cascade_columns)
    parent_ax = Axis(fig_cascade[1, ci]; title = label, aspect = DataAspect())
    child_ax  = Axis(fig_cascade[2, ci]; aspect = DataAspect())
    parent_range = key === :w ? (-wmax_parent, wmax_parent) : crange
    hmp = heatmap!(parent_ax, parent_frames[1].λ, parent_frames[1].φ,
                   (@lift getproperty(parent_fields[$cascade_n], key)); colormap = cmap, colorrange = parent_range)
    lines!(parent_ax, boxλ, boxφ; color = :black, linestyle = :dash, linewidth = 1.5)
    hmc = heatmap!(child_ax, slice_frames[1].λ, slice_frames[1].φ,
                   (@lift getproperty(child_fields[$cascade_n], key)); colormap = cmap, colorrange = crange)
    if key === :w   # ERA5 (~0.1 m/s) and 3 km child (~2 m/s) w differ ~10× — one colorbar per row
        Colorbar(fig_cascade[3, ci], hmp; vertical = false, flipaxis = false, height = 10, label = "ERA5")
        Colorbar(fig_cascade[4, ci], hmc; vertical = false, flipaxis = false, height = 10, label = "3 km")
    else
        Colorbar(fig_cascade[3, ci], hmc; vertical = false, flipaxis = false, height = 10)
    end
    hidedecorations!(parent_ax); hidedecorations!(child_ax)
    if ci == 1
        text!(parent_ax, 0.03, 0.97; text = "ERA5 (D1)",   space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
        text!(child_ax,  0.03, 0.97; text = "Breeze 12 km", space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
    end
end
CairoMakie.record(fig_cascade, output_dir * "/era5_cascade_2row.mp4", 1:length(slice_frames); framerate = 8) do nn
    cascade_n[] = nn   # CairoMakie.record: `record` is also exported by CUDA, so qualify it
end
@info @sprintf("wrote %s/era5_cascade_2row.mp4 (%d frames)", output_dir, length(slice_frames))
