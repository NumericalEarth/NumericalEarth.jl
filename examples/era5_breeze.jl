# # ERA5 downscaling with Breeze and NestedSimulation
#
# This is a limited-area model (LAM) example that couples the Breeze
# compressible solver to forthcoming SlabLand and SlabOcean components.
#
# At the moment, this script does just the data ingest: download ERA5
# reanalysis restricted to a bounding box and interpolate it onto a
# `LatitudeLongitudeGrid` sized for ~3 km horizontal cells at the domain
# center latitude.
#
# In progress:
# - [x] Breeze model construction
# - [x] initial state setting (set! the model from ingested fields)
# - [ ] test with GPU
# - [ ] acoustic substepping
# - [ ] dynamical initialization
# - [ ] open boundary conditions
# - [ ] land/ocean coupling
# - [ ] terrain

using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.ERA5
using CDSAPI  # activates NumericalEarthCDSAPIExt
using Oceananigans
using Oceananigans.Fields: interpolate!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries
using Breeze
using Statistics: mean
using Dates
using Printf

# ## Configuration

# ### Domain
#
# Domain centered on the U.S. Department of Energy's Atmospheric Radiation
# Measurement (ARM) Climate Research Facility's Southern Great Plains (SGP)
# site in Lamont, OK. Angular grid steps are chosen so that the physical cells
# are roughly square (~3 km) at the center latitude, using R = 6,371 km:
#   Δx = R·cos(φ₀)·Δλ ≈ 3.03 km
#   Δy = R·Δφ         ≈ 3.00 km

φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)

Δλ, Δφ = 0.034, 0.027       # grid spacings (deg)
Nx, Ny = 256, 256

# From these inputs, we determine the `BoundingBox` corners.

λ_west  = λ₀ - Nx * Δλ / 2
λ_east  = λ₀ + Nx * Δλ / 2
φ_south = φ₀ - Ny * Δφ / 2
φ_north = φ₀ + Ny * Δφ / 2

# Vertical grid: Oceananigans' `ReferenceToStretchedDiscretization` gives one
# constant 50 m cell at the surface, then a linear 1.08× stretching per cell
# until Δz hits the 1 km cap, then uniform 1 km cells out to Lz ≈ 37 km —
# comfortably below the ERA5 1 hPa top. `Nz` is determined by the stretching
# law (≈ 64 with these parameters).

z_discretization = ReferenceToStretchedDiscretization(
    extent                  = 36000.0,
    bias                    = :left,
    bias_edge               = 0.0,
    constant_spacing        = 50.0,
    constant_spacing_extent = 50.0,
    maximum_spacing         = 1000.0,
    stretching              = LinearStretching(0.08))

Nz = length(z_discretization)

# ### Initial conditions
#
# We've selected a case day from the Holistic Interactions of Shallow Clouds,
# Aerosols and Land Ecosystems (HI-SCALE) campaign at the ARM SGP site.
# This period features clear skies with periods of cirrus.

start_date = DateTime(2016, 09, 10, 12) # 7 am LT
end_date   = DateTime(2016, 09, 10, 18) # 1 pm LT

dates = start_date:Hour(1):end_date

# ### ERA5 reanalysis
#
# TODO: define a `MetadataSet` as per
# https://github.com/NumericalEarth/NumericalEarth.jl/issues/235

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

pl_vars = [:eastward_velocity, :northward_velocity, :temperature,
           :specific_humidity, :geopotential,
           :specific_cloud_liquid_water_content,
           :specific_cloud_ice_water_content]

# ## Setup LAM grid
#
# `LatitudeLongitudeGrid` with `Bounded` horizontal topologies (LAM-style).
# The vertical coordinate is height in meters; the ERA5 pressure-level
# metadata supplies a domain-mean z(p) profile via the time-mean spatial-mean
# geopotential height (the dataset's default `mean_geopotential_height=true`).

grid = LatitudeLongitudeGrid(longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z         = z_discretization,
                             size      = (Nx, Ny, Nz),
                             halo      = (5, 5, 5),
                             topology  = (Bounded, Bounded, Bounded))

# ## Thermodynamic constants
#
# All thermodynamic parameters used downstream (per-column z conversion,
# moist gas law, liquid-ice potential temperature, virtual temperature)
# come from Breeze's `ThermodynamicConstants`.

constants = ThermodynamicConstants()

g   = ERA5.ERA5_gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
Rᵛ  = vapor_gas_constant(constants)
cₚᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cₚᵈ
pˢᵗ = 1e5  # Pa
Lᵥ  = constants.liquid.reference_latent_heat
Lₛ  = constants.ice.reference_latent_heat

# ## Interpolate ERA5 onto the LAM grid
#
# We bypass `set!(field, metadatum)` for two reasons:
#  (a) use ERA5's instantaneous, spatially-varying Φ(λ, φ, p)/g as the
#      z-mapping for each column rather than a single spatial-mean profile;
#  (b) mask levels with p > p_surface(λ, φ).
#
# See NumericalEarth/NumericalEarth.jl#236 for context.
#
# The interpolation is split into two stages:
#  1. Per-column linear-in-z interpolation from the ERA5 native pressure
#     levels onto the LAM z-coordinate, producing a Field on an intermediate
#     grid that shares the ERA5 native (λ, φ) but the LAM's z.
#  2. Horizontal-only `Oceananigans.interpolate!` from intermediate onto the
#     target LAM field — since the two grids share z, only the horizontal
#     bilinear remains.
#
# Terrain workaround (TEMPORARY): we don't yet have terrain in the LAM, so we
# map ERA5's (Φ − Φ₀)/g (height above local surface) → LAM z. This treats the
# LAM z=0 as "at the surface" everywhere, equivalent to a sigma-z coordinate.
# Φ₀ comes from ERA5's `:geopotential_height` on single levels.
#
# TODO: When terrain support lands, swap back to Φ/g.

# Per-column linear interpolation in z, skipping sub-surface levels.
function interp_z_masked(z, z_col, var_col, p_levels, p₀_local)
    k_lo, k_hi = 0, 0
    @inbounds for k in eachindex(p_levels)
        p_levels[k] > p₀_local && continue
        if z_col[k] <= z
            k_lo = k
        else
            k_hi = k
            break
        end
    end
    k_lo == 0 && return var_col[k_hi]                  # below lowest valid level
    k_hi == 0 && return var_col[k_lo]                  # above highest valid level
    α = (z - z_col[k_lo]) / (z_col[k_hi] - z_col[k_lo])
    return (1 - α) * var_col[k_lo] + α * var_col[k_hi]
end

# Stage 1: column-wise z interpolation onto the intermediate grid.
# The intermediate field shares the ERA5 native (λ, φ) but has the LAM z;
# we simply loop over (i, j) of the native grid and linearly interpolate
# each column to the LAM z-centers, applying the sub-surface mask.
function column_interp_z!(inter_field, era5_data;
                          z_above_sfc, p_era5_lev, p₀_arr)
    out   = interior(inter_field)
    z_lam = collect(znodes(inter_field.grid, Center(), Center(), Center()))
    Nλ_e, Nφ_e, _ = size(era5_data)

    for k in eachindex(z_lam), j in 1:Nφ_e, i in 1:Nλ_e
        out[i, j, k] = interp_z_masked(z_lam[k],
                                       @view(z_above_sfc[i, j, :]),
                                       @view(era5_data[i, j, :]),
                                       p_era5_lev,
                                       p₀_arr[i, j])
    end

    fill_halo_regions!(inter_field)
    return inter_field
end

# --- Parent grid: ERA5 native (λ, φ), LAM z ---
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

p_era5_lev = sort(ds_pl.pressure_levels, rev=true)

λ_centers_era5 = collect(λnodes(ϕ_field_snap1.grid, Center(), Center(), Center()))
φ_centers_era5 = collect(φnodes(ϕ_field_snap1.grid, Center(), Center(), Center()))
Nλ_e, Nφ_e = length(λ_centers_era5), length(φ_centers_era5)
Δλ_e = (λ_centers_era5[end] - λ_centers_era5[1]) / (Nλ_e - 1)
Δφ_e = (φ_centers_era5[end] - φ_centers_era5[1]) / (Nφ_e - 1)

parent_grid = LatitudeLongitudeGrid(longitude = (λ_centers_era5[1]   - Δλ_e/2,
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
parent = PrescribedAtmosphere(parent_grid, parent_times; thermodynamics_parameters = nothing)

qᶜ_fts = FieldTimeSeries{Center, Center, Center}(parent_grid, parent_times)
qⁱ_fts = FieldTimeSeries{Center, Center, Center}(parent_grid, parent_times)

# --- Time-invariant: surface geopotential Φ₀ ---
# Φ₀ is terrain elevation × g; load once from snapshot 1.

const Φ₀_arr_snap1 = Array(interior(Field(Metadatum(:geopotential_height;
                                                    dataset=ds_sl,
                                                    meta_common_snap1...))))[:, :, 1]

# --- Per-snapshot ERA5 → parent FTS population ---
#
# For each ERA5 hourly snapshot:
#   1. Pull T, q*, u, v, geopotential (3D), surface-pressure.
#   2. Build height-above-surface z_above_sfc = (Φ − Φ₀)/g (terrain workaround).
#   3. Column-wise linear-in-z interp onto the parent z-grid, masking
#      sub-surface levels (p > p_surface).
#   4. Copy the result into FTS slot n.

function populate_parent_snapshot!(n, date)
    meta = (date = date, region = era5_region, dir = era5_datadir)

    ϕ_field  = Field(Metadatum(:geopotential;     dataset=ds_pl, meta...))
    p₀_field = Field(Metadatum(:surface_pressure; dataset=ds_sl, meta...))

    p₀_arr = Array(interior(p₀_field))[:, :, 1]   # Pa
    z_above_sfc = (Array(interior(ϕ_field)) .-
                   reshape(Φ₀_arr_snap1, size(Φ₀_arr_snap1, 1), size(Φ₀_arr_snap1, 2), 1)) ./ g

    read3d(name) = Array(interior(Field(Metadatum(name; dataset=ds_pl, meta...))))
    u_era5  = read3d(:eastward_velocity)
    v_era5  = read3d(:northward_velocity)
    T_era5  = read3d(:temperature)
    qᵛ_era5 = read3d(:specific_humidity)
    qᶜ_era5 = read3d(:specific_cloud_liquid_water_content)
    qⁱ_era5 = read3d(:specific_cloud_ice_water_content)
    p_era5_3d = repeat(reshape(p_era5_lev, 1, 1, :),
                       size(z_above_sfc, 1), size(z_above_sfc, 2), 1)

    era5_kw = (; z_above_sfc, p_era5_lev, p₀_arr)

    u_p  = CenterField(parent_grid)
    v_p  = CenterField(parent_grid)
    T_p  = CenterField(parent_grid)
    qᵛ_p = CenterField(parent_grid)
    qᶜ_p = CenterField(parent_grid)
    qⁱ_p = CenterField(parent_grid)
    p_p  = CenterField(parent_grid)

    column_interp_z!(u_p,  u_era5;    era5_kw...)
    column_interp_z!(v_p,  v_era5;    era5_kw...)
    column_interp_z!(T_p,  T_era5;    era5_kw...)
    column_interp_z!(qᵛ_p, qᵛ_era5;   era5_kw...)
    column_interp_z!(qᶜ_p, qᶜ_era5;   era5_kw...)
    column_interp_z!(qⁱ_p, qⁱ_era5;   era5_kw...)
    column_interp_z!(p_p,  p_era5_3d; era5_kw...)

    interior(parent.velocities.u, :, :, :, n) .= interior(u_p)
    interior(parent.velocities.v, :, :, :, n) .= interior(v_p)
    interior(parent.tracers.T,    :, :, :, n) .= interior(T_p)
    interior(parent.tracers.q,    :, :, :, n) .= interior(qᵛ_p)
    interior(parent.pressure,     :, :, :, n) .= interior(p_p)
    interior(qᶜ_fts,              :, :, :, n) .= interior(qᶜ_p)
    interior(qⁱ_fts,              :, :, :, n) .= interior(qⁱ_p)

    return nothing
end

for (n, date) in enumerate(dates)
    @info @sprintf("Populating parent snapshot %d/%d at %s", n, length(dates), date)
    populate_parent_snapshot!(n, date)
end

# --- LAM-grid IC fields: horizontal regrid of snapshot 1 from the parent ---
# `interpolate!` does the bilinear-in-(λ, φ) regrid; the vertical coord is
# already on the parent grid so this is purely horizontal.

u  = XFaceField(grid)
v  = YFaceField(grid)
T  = CenterField(grid)
qᵛ = CenterField(grid)
qᶜ = CenterField(grid)
qⁱ = CenterField(grid)
p  = CenterField(grid)

interpolate!(u,  parent.velocities.u[1])
interpolate!(v,  parent.velocities.v[1])
interpolate!(T,  parent.tracers.T[1])
interpolate!(qᵛ, parent.tracers.q[1])
interpolate!(qᶜ, qᶜ_fts[1])
interpolate!(qⁱ, qⁱ_fts[1])
interpolate!(p,  parent.pressure[1])

# Calculate virtual temperature: Tᵛ = T·(1 + (1 − ε)/ε·qᵛ), ε = Rᵈ/Rᵛ.
# Vapor only by convention — the qᶜ, qⁱ terms belong to the density temperature Tρ.

εfac = Rᵛ / Rᵈ - 1

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

# ## Build the Breeze model
#
# Piggyback on the `atmosphere_simulation` helper from
# `ext/NumericalEarthBreezeExt/`, which also pre-wires bottom flux fields
# (ρτˣ, ρτʸ, Jᵉ, Jᵛ) ready for the forthcoming SlabLand / SlabOcean coupling.
# We override the helper's default `AnelasticDynamics` (which dispatches on
# `RectilinearGrid` only) with `CompressibleDynamics`, whose prognostic-density
# / diagnostic-pressure formulation needs no FFT-based Poisson solve and works
# directly on the LAM `LatitudeLongitudeGrid`.

p̄₀ = mean(interior(p₀))

model = atmosphere_simulation(grid;
                              thermodynamic_constants = constants,
                              dynamics = CompressibleDynamics(; surface_pressure = p̄₀))

# ## Set initial state from ERA5
#
# Derive the prognostic variables Breeze's `CompressibleDynamics` needs from
# the per-column-interpolated ERA5 fields:
#
#   ρ   = p / (Rᵈ · Tᵛ)                                  (moist ideal gas law)
#   θˡⁱ = θ · (1 − (Lᵥ qᶜ + Lₛ qⁱ) / (cₚᵈ T))            (Breeze's diagnostic form,
#         with θ = T · (pˢᵗ/p)^κ                          using cₚᵈ ≈ cᵖᵐ)
#   qᵗ  = qᵛ + qᶜ + qⁱ                                   (saturation adjustment
#                                                         partitions on first
#                                                         update_state!)

ρ   = Field(p / (Rᵈ * Tᵛ))
θ   = T * (pˢᵗ / p)^κ
θˡⁱ = Field(θ * (1 - (Lᵥ * qᶜ + Lₛ * qⁱ) / (cₚᵈ * T)))
qᵗ  = Field(qᵛ + qᶜ + qⁱ)

compute!(ρ)
compute!(θˡⁱ)
compute!(qᵗ)

set!(model; ρ = ρ, u = u, v = v, qᵗ = qᵗ, θˡⁱ = θˡⁱ)

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
# Plot ρ, u, v, θ, qᵗ at three locations spanning the domain's terrain range.
# At each site we overlay the four surrounding ERA5 native-grid columns (the
# bilinear stencil) in light gray. The vertical coordinate is height above
# the local surface (Φ − Φ₀)/g — i.e., we strip the terrain offset out of
# both the LAM (which has none) and the ERA5 columns.

using CairoMakie

sites = [("East TX",     -93.5,   34.0),
         ("SGP",         -97.485, 36.605),
         ("High Plains", -101.5,  35.0)]

# Re-fetch the snapshot-1 ERA5 raw arrays for the gray stencil overlay
# (the FTSs hold vertically-interpolated data; the overlay needs the native
# pressure-level columns).

Φ₀_arr = Array(interior(Field(Metadatum(:geopotential_height; dataset=ds_sl, meta_common_snap1...))))[:, :, 1]
p₀_arr = Array(interior(Field(Metadatum(:surface_pressure;    dataset=ds_sl, meta_common_snap1...))))[:, :, 1]
z_above_sfc = (Array(interior(Field(Metadatum(:geopotential;  dataset=ds_pl, meta_common_snap1...)))) .-
               reshape(Φ₀_arr, size(Φ₀_arr, 1), size(Φ₀_arr, 2), 1)) ./ g

read3d_snap1(name) = Array(interior(Field(Metadatum(name; dataset=ds_pl, meta_common_snap1...))))
u_era5  = read3d_snap1(:eastward_velocity)
v_era5  = read3d_snap1(:northward_velocity)
T_era5  = read3d_snap1(:temperature)
qᵛ_era5 = read3d_snap1(:specific_humidity)
qᶜ_era5 = read3d_snap1(:specific_cloud_liquid_water_content)
qⁱ_era5 = read3d_snap1(:specific_cloud_ice_water_content)
p_era5_3d = repeat(reshape(p_era5_lev, 1, 1, :), size(z_above_sfc, 1), size(z_above_sfc, 2), 1)

# Materialize θ (currently an abstract op) and derive ERA5-native counterparts.
θ_lam   = compute!(Field(T * (pˢᵗ / p)^κ))
Tᵛ_era5 = T_era5 .* (1 .+ εfac .* qᵛ_era5)
ρ_era5  = p_era5_3d ./ (Rᵈ .* Tᵛ_era5)
θ_era5  = T_era5 .* (pˢᵗ ./ p_era5_3d) .^ κ
qᵗ_era5 = qᵛ_era5 .+ qᶜ_era5 .+ qⁱ_era5

ρ_arr   = Array(interior(ρ))
u_arr   = Array(interior(u))
v_arr   = Array(interior(v))
θ_arr   = Array(interior(θ_lam))
qᵗ_arr  = Array(interior(qᵗ))

λ_e = collect(λnodes(ϕ_field_snap1.grid, Center(), Center(), Center()))
φ_e = collect(φnodes(ϕ_field_snap1.grid, Center(), Center(), Center()))
λ_c = collect(λnodes(grid, Center(), Center(), Center()))
φ_c = collect(φnodes(grid, Center(), Center(), Center()))
λ_f = collect(λnodes(grid, Face(),   Center(), Center()))
φ_f = collect(φnodes(grid, Center(), Face(),   Center()))
z_c = collect(znodes(grid, Center(), Center(), Center()))

vars = [(:ρ,  ρ_arr,  ρ_era5,  "ρ (kg/m³)",  :center),
        (:u,  u_arr,  u_era5,  "u (m/s)",    :xface),
        (:v,  v_arr,  v_era5,  "v (m/s)",    :yface),
        (:θ,  θ_arr,  θ_era5,  "θ (K)",      :center),
        (:qᵗ, qᵗ_arr, qᵗ_era5, "qᵗ (kg/kg)", :center)]

fig = Figure(size=(1600, 1000), fontsize=12)

Nrows = length(sites)
Ncols = length(vars)
axs   = Matrix{Axis}(undef, Nrows, Ncols)

for (row, (label, λ_site, φ_site)) in enumerate(sites)
    # Site header spanning all 5 columns; elevation read from Φ₀ at the
    # ERA5 cell containing the site.
    i_site = clamp(floor(Int, (λ_site - λ_e[1]) / Δλ_e + 1), 1, length(λ_e) - 1)
    j_site = clamp(floor(Int, (φ_site - φ_e[1]) / Δφ_e + 1), 1, length(φ_e) - 1)
    elev_m = round(Int, Φ₀_arr[i_site, j_site] / g)
    Label(fig[2*row - 1, 1:Ncols], "$label (elevation: $elev_m m)";
          fontsize=15, font=:bold, halign=:center, tellwidth=false)

    for (col, (vname, lam_arr, era5_arr, xlab, stagger)) in enumerate(vars)
        # Pick the LAM cell closest to the site for this variable's stagger,
        # then center the ERA5 bilinear stencil around the LAM cell's actual
        # position so the blue line is exactly the bilinear mix of the
        # plotted gray columns.
        i_lam = stagger == :xface ? argmin(abs.(λ_f .- λ_site)) :
                                    argmin(abs.(λ_c .- λ_site))
        j_lam = stagger == :yface ? argmin(abs.(φ_f .- φ_site)) :
                                    argmin(abs.(φ_c .- φ_site))
        λ_lam = stagger == :xface ? λ_f[i_lam] : λ_c[i_lam]
        φ_lam = stagger == :yface ? φ_f[j_lam] : φ_c[j_lam]

        fi = (λ_lam - λ_e[1]) / Δλ_e + 1
        fj = (φ_lam - φ_e[1]) / Δφ_e + 1
        i₀ = clamp(floor(Int, fi), 1, length(λ_e) - 1)
        j₀ = clamp(floor(Int, fj), 1, length(φ_e) - 1)
        cells = ((i₀, j₀), (i₀+1, j₀), (i₀, j₀+1), (i₀+1, j₀+1))

        ax = Axis(fig[2*row, col]; xlabel=xlab,
                  ylabel       = col == 1 ? "z above surface (km)" : "",
                  xlabelsize   = 14,
                  ylabelsize   = 14)
        axs[row, col] = ax

        # ERA5 columns (light gray) — the LAM cell's actual bilinear stencil
        for (i, j) in cells
            valid = p_era5_lev .<= p₀_arr[i, j]
            lines!(ax, era5_arr[i, j, valid], z_above_sfc[i, j, valid] ./ 1000;
                   color=:gray70, linewidth=1)
        end

        # LAM profile at the chosen point — markers at cell centers so the
        # discretization is explicit (no implied between-cell behavior).
        scatter!(ax, lam_arr[i_lam, j_lam, :], z_c ./ 1000;
                 color=:steelblue, markersize=6)

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

Label(fig[0, 1:Ncols], "ERA5 → LAM profiles  (LAM: blue;  ERA5 stencil: gray)";
      fontsize=20, font=:bold, tellwidth=false)

save("era5_breeze_profiles.png", fig)
@info "Wrote era5_breeze_profiles.png"
