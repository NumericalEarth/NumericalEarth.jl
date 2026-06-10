# # ERA5 downscaling with Breeze and NestedSimulation
#
# This is a limited-area model (LAM) example that couples the Breeze
# compressible solver to forthcoming SlabLand and SlabOcean components.
#
# It downloads ERA5 reanalysis restricted to a bounding box, regrids it onto a
# terrain-following `LatitudeLongitudeGrid` sized for ~3 km horizontal cells at
# the domain center latitude, builds a compressible Breeze atmosphere, and drives
# it through a `NestedSimulation`: an ERA5-forced parent supplies the lateral
# boundary conditions and an interior Davies relaxation fringe.
#
# In progress:
# - [x] Breeze model construction
# - [x] initial state setting (set! the model from ingested fields)
# - [x] open boundary conditions (parent-driven OBC + Davies fringe relaxation)
# - [x] test with GPU
# - [x] terrain (ETOPO 2022 + Breeze `follow_terrain!`, terrain-following coordinates)
# - [ ] dynamical initialization
# - [ ] acoustic substepping
# - [ ] land/ocean coupling

using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.ERA5
using CDSAPI  # activates NumericalEarthCDSAPIExt
using Oceananigans
using Oceananigans.Fields: interpolate!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: MutableVerticalDiscretization, znode
using Oceananigans.Architectures: on_architecture
using Breeze
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
# site in Lamont, OK. We match the 3 km inner domain (Domain 3) of the WRF
# nest used by [Fan2017](@citet) for this MC3E case: a 27 → 9 → 3 km telescoping
# nest. Their 1 km Domain 4 ("nest-down") is left out for now — that finer nest
# is the natural next step for a `NestedSimulation` child.
#
# [Fan2017](@citet)'s Domain 3 carries 301 × 271 WRF grid points. Those count staggered
# (cell-edge) locations, so they map to 300 × 270 Breeze *cells* (cells = points − 1).
# Angular grid steps are chosen so that the physical cells are roughly square
# (~3 km) at the center latitude, using R = 6,371 km:
#   Δx = R·cos(φ₀)·Δλ ≈ 3.03 km
#   Δy = R·Δφ         ≈ 3.00 km

φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)

Δλ, Δφ = 0.034, 0.027       # grid spacings (deg)
Nx, Ny = 300, 270           # Fan et al. (2017) Domain 3: 301 × 271 points − 1

# From these inputs, we determine the `BoundingBox` corners.

λ_west  = λ₀ - Nx * Δλ / 2
λ_east  = λ₀ + Nx * Δλ / 2
φ_south = φ₀ - Ny * Δφ / 2
φ_north = φ₀ + Ny * Δφ / 2

# Vertical grid: matched to [Fan2017](@citet), who use 51 vertical levels with
# spacing ~60 m at the lowest levels stretching to ~490 m aloft. WRF's 51 levels
# count staggered (w) interfaces, so they map to 50 Breeze *cells*. Oceananigans'
# `ReferenceToStretchedDiscretization` gives one constant 60 m cell at the
# surface, then a 1.08× stretching per cell until Δz hits the 490 m cap, then
# uniform 490 m cells. `extent = 16250` is tuned so the stretching law lands on
# exactly `Nz = 50` (Lz ≈ 16.5 km, comfortably below the ERA5 1 hPa top).

z_discretization = ReferenceToStretchedDiscretization(
    extent                  = 16250.0,
    bias                    = :left,
    bias_edge               = 0.0,
    constant_spacing        = 60.0,
    constant_spacing_extent = 60.0,
    maximum_spacing         = 490.0,
    stretching              = LinearStretching(0.08))

Nz = length(z_discretization)
@assert Nz == 50  # Fan et al. (2017): 51 staggered levels → 50 cells

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

# ERA5 bounding box: cover the LAM domain plus 1° padding, snapped outward
# to ERA5's native 0.25° grid.

function era5_bbox(; padding_deg = 1.0, snap_deg = 0.25)
    λ_min = floor((λ_west  - padding_deg) / snap_deg) * snap_deg
    λ_max =  ceil((λ_east  + padding_deg) / snap_deg) * snap_deg
    φ_min = floor((φ_south - padding_deg) / snap_deg) * snap_deg
    φ_max =  ceil((φ_north + padding_deg) / snap_deg) * snap_deg

    return BoundingBox(longitude = (λ_min, λ_max),
                       latitude  = (φ_min, φ_max))
end

era5_region = era5_bbox()

@info @sprintf("LAM grid : λ ∈ [%.3f, %.3f], φ ∈ [%.3f, %.3f]; Δλ=%.4f°, Δφ=%.4f°",
               λ_west, λ_east, φ_south, φ_north, Δλ, Δφ)
@info @sprintf("ERA5 bbox: λ ∈ [%.2f, %.2f], φ ∈ [%.2f, %.2f]",
               era5_region.longitude[1], era5_region.longitude[2],
               era5_region.latitude[1],  era5_region.latitude[2])

# We use hourly dataset on both single levels and pressure levels.

ds_pl = ERA5HourlyPressureLevels()
ds_sl = ERA5HourlySingleLevel()

# ## Setup LAM grid
#
# Terrain-following `LatitudeLongitudeGrid` with `Bounded` horizontal topologies
# (LAM-style). The vertical coordinate is a `MutableVerticalDiscretization` built
# from the stretched height profile; `follow_terrain!` then deforms its coordinate
# surfaces to follow the ETOPO 2022 surface elevation (a Gal-Chen–Somerville σ
# coordinate). The bottom surface sits at the local terrain height; the top stays
# flat. Node heights are true heights above sea level — the coordinate the #241
# ERA5 ingest below interpolates onto.

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z         = MutableVerticalDiscretization(z_discretization),
                             size      = (Nx, Ny, Nz),
                             halo      = (5, 5, 5),
                             topology  = (Bounded, Bounded, Bounded))

# `follow_terrain!` sets the topography via `set_topography!(h_field, grid, topo)`,
# but the registered Breeze only defines the `::Function` method (which samples
# `xnode`/`ynode` — RectilinearGrid-only, so it errors on a `LatitudeLongitudeGrid`).
# Provide the generic Field/array method ourselves: it routes the ETOPO `elevation`
# Field straight through `set!`, bypassing the function path. The rest of
# `follow_terrain!` is grid-agnostic, so this is all that's needed on a LatLon grid —
# no Breeze #776 required. (Remove once the env's Breeze ships this method upstream.)
Breeze.TerrainFollowingDiscretization.set_topography!(h_field, grid, topography) =
    (set!(h_field, topography); nothing)

# ETOPO 2022 surface elevation (≥ 0; ocean clamped to sea level) regridded onto
# the LAM horizontal grid — ETOPO's 60″ (~1.85 km) relief is finer than the ~3 km
# cells. `follow_terrain!` imposes it as terrain-following coordinates and returns
# the `TerrainMetrics` (slopes, model top) that `CompressibleDynamics` consumes.

elevation = regrid_topography(grid; dataset = ETOPO2022())
metrics   = follow_terrain!(grid, elevation)

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
# These regrids interpolate linearly in height between ERA5 levels. Within a layer
# the hydrostatic z ↔ ln(p) relation is affine, and linear interpolation is invariant
# under an affine change of the abscissa — so linear-in-z is identical to linear-in-
# log-p for every quantity *except pressure itself*, which we interpolate in log-p
# (see `regrid_pressure`) so it stays log-linear (hydrostatically consistent) in z.

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

# --- Regridded pressure coordinate ---
# Pressure is the ERA5 vertical *coordinate*, not a stored field. `pressure_field`
# returns the exact (time-constant) level values as a `(Nothing, Nothing, Center)`
# field on the native grid. We regrid log(p) — so pressure interpolates log-linearly
# in height, the hydrostatically-consistent choice and the one quantity for which
# linear-in-z and linear-in-log-p differ — then exponentiate back onto `target_grid`.
function regrid_pressure(metadatum, target_grid)
    pf = pressure_field(metadatum, arch)
    lnp_native = CenterField(pf.grid)
    interior(lnp_native) .= log.(interior(pf))
    fill_halo_regions!(lnp_native)
    lnp = CenterField(target_grid)
    interpolate!(lnp, lnp_native)
    p = CenterField(target_grid)
    interior(p) .= exp.(interior(lnp))
    fill_halo_regions!(p)
    return p
end

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

# Derive (ρ, θˡⁱ, qᵗ) per snapshot via `breeze_prognostic_state` and store the
# specific (Davies-target) and density-weighted (lateral-BC) forms. Pressure isn't
# an ERA5 field, so it's regridded per snapshot (same per-snapshot Φ as the series).
for n in eachindex(dates)
    @info @sprintf("Deriving parent snapshot %d/%d at %s", n, length(dates), dates[n])
    p_p = regrid_pressure(Metadatum(:temperature; dataset = ds_pl, date = dates[n],
                                    region = era5_region, dir = era5_datadir), parent_grid)
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
p  = regrid_pressure(initial_metadatum(:temperature), grid)

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

# τ_relax ≈ 5·Δx / U_scale at the domain center latitude, U ~ 20 m/s.
Δx_phys = 6371e3 * cos(deg2rad(φ₀)) * deg2rad(Δλ)   # m
τ_relax = FRINGE_N * Δx_phys / 20.0                 # s

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
# directly on the LAM `LatitudeLongitudeGrid`. Passing the `terrain_metrics` from
# `follow_terrain!` activates the terrain-following physics (contravariant vertical
# velocity, corrected horizontal pressure gradient, terrain-aware divergence).
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

model = atmosphere_simulation(grid;
                              thermodynamic_constants = constants,
                              dynamics            = CompressibleDynamics(; surface_pressure = p̄₀,
                                                                         terrain_metrics = metrics),
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

# ## NestedSimulation
#
# Wrap the child model in a `NestedSimulation` paired with the parent
# `PrescribedAtmosphere`. `NestedModel.time_step!` syncs the parent clock
# each iteration so the FTS-driven BCs and forcings get the correct
# interpolation time.
#
# Δt is set from the acoustic CFL on the vertical grid — Δz_min = 50 m near
# the surface (the binding constraint here, since horizontal Δx ≈ 3 km is
# much larger) and c_sound ≈ 340 m/s at the reference state. Substepping
# would let us bypass the acoustic limit and use an advection-CFL Δt instead;
# that's the next step.

c_sound = sqrt(constants.dry_air.heat_capacity / (constants.dry_air.heat_capacity - Rᵈ) * Rᵈ * 290.0)
Δt = 0.3 * minimum_zspacing(grid) / c_sound

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
# 100-iteration smoke run at acoustic CFL — exercises BC machinery + Davies
# forcing before substepping and any IC-balance work.

@info @sprintf("Δt = %.4f s (acoustic CFL); running %d iterations", Δt, nested.stop_iteration)
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

using CairoMakie

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
