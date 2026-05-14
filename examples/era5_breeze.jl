# # Regional hindcast with ERA5
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
using Breeze
using Breeze.Thermodynamics # TODO: https://github.com/NumericalEarth/Breeze.jl/pull/699
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

# Two-stage interpolation: column-z onto intermediate, then horizontal
# `interpolate!` onto the target LAM field.
function interp_era5_to_lam!(target, era5_data, intermediate_grid;
                             z_above_sfc, p_era5_lev, p₀_arr)
    inter = CenterField(intermediate_grid)
    column_interp_z!(inter, era5_data; z_above_sfc, p_era5_lev, p₀_arr)
    interpolate!(target, inter)
    return target
end

# --- ERA5 raw arrays and per-column metadata ---
meta_common = (date = start_date, region = era5_region, dir = era5_datadir)
ϕ_field     = Field(Metadatum(:geopotential;        dataset=ds_pl, meta_common...))
Φ₀_field    = Field(Metadatum(:geopotential_height; dataset=ds_sl, meta_common...))
p₀_field    = Field(Metadatum(:surface_pressure;    dataset=ds_sl, meta_common...))

# Native ERA5 array shapes: 3-D (Nλ_e, Nφ_e, Np_e); 2-D fields stored as
# (Nλ_e, Nφ_e, 1) — slice them down to plain 2-D.
Φ₀_arr = Array(interior(Φ₀_field))[:, :, 1]  # m²/s² (raw geopotential — `:geopotential_height`
                                              #       is a misnomer; ERA5 stores it without /g)
p₀_arr = Array(interior(p₀_field))[:, :, 1]  # Pa

# Height above local surface for each ERA5 (i, j, k); see "Terrain workaround"
# above. Both Φ and Φ₀ are in m²/s²; broadcast Φ₀ over the vertical axis with
# `reshape` + `..., 1`, then divide once by g at the end.
z_above_sfc = (Array(interior(ϕ_field)) .-
               reshape(Φ₀_arr, size(Φ₀_arr, 1), size(Φ₀_arr, 2), 1)) ./ g

p_era5_lev = sort(ds_pl.pressure_levels, rev=true)

read3d(name) = Array(interior(Field(Metadatum(name; dataset=ds_pl, meta_common...))))
u_era5  = read3d(:eastward_velocity)
v_era5  = read3d(:northward_velocity)
T_era5  = read3d(:temperature)
qᵛ_era5 = read3d(:specific_humidity)
qᶜ_era5 = read3d(:specific_cloud_liquid_water_content)
qⁱ_era5 = read3d(:specific_cloud_ice_water_content)

# Pressure as a 3-D array (constant in λ, φ; pressure-level values broadcast)
p_era5_3d = repeat(reshape(p_era5_lev, 1, 1, :), size(z_above_sfc, 1), size(z_above_sfc, 2), 1)

# --- Intermediate grid: ERA5 native (λ, φ), LAM z ---
# Mirror NumericalEarth's `native_grid`/`restrict` behavior: it pads the
# requested bbox outward by Δ/2 on each side so cell *centers* land on the
# ERA5 native grid. Without that adjustment, `intermediate_grid`'s cell
# centers would be offset from the actual ERA5 data positions (up to ~0.12°
# at the bbox edges), which biases stage-2's `interpolate!` onto the LAM.
Nλ_e, Nφ_e = size(z_above_sfc, 1), size(z_above_sfc, 2)
λ_centers_era5 = collect(λnodes(ϕ_field.grid, Center(), Center(), Center()))
φ_centers_era5 = collect(φnodes(ϕ_field.grid, Center(), Center(), Center()))
Δλ_e = (λ_centers_era5[end] - λ_centers_era5[1]) / (Nλ_e - 1)
Δφ_e = (φ_centers_era5[end] - φ_centers_era5[1]) / (Nφ_e - 1)
intermediate_grid = LatitudeLongitudeGrid(longitude = (λ_centers_era5[1]   - Δλ_e/2,
                                                       λ_centers_era5[end] + Δλ_e/2),
                                          latitude  = (φ_centers_era5[1]   - Δφ_e/2,
                                                       φ_centers_era5[end] + Δφ_e/2),
                                          z         = z_discretization,
                                          size      = (Nλ_e, Nφ_e, Nz),
                                          halo      = (5, 5, 5),
                                          topology  = (Bounded, Bounded, Bounded))

# --- LAM fields populated by stage-1 + stage-2 ---
u  = XFaceField(grid)
v  = YFaceField(grid)
T  = CenterField(grid)
qᵛ = CenterField(grid)
qᶜ = CenterField(grid)
qⁱ = CenterField(grid)
p  = CenterField(grid)

era5_kw = (; z_above_sfc, p_era5_lev, p₀_arr)

interp_era5_to_lam!(u,  u_era5,    intermediate_grid; era5_kw...)
interp_era5_to_lam!(v,  v_era5,    intermediate_grid; era5_kw...)
interp_era5_to_lam!(T,  T_era5,    intermediate_grid; era5_kw...)
interp_era5_to_lam!(qᵛ, qᵛ_era5,   intermediate_grid; era5_kw...)
interp_era5_to_lam!(qᶜ, qᶜ_era5,   intermediate_grid; era5_kw...)
interp_era5_to_lam!(qⁱ, qⁱ_era5,   intermediate_grid; era5_kw...)
interp_era5_to_lam!(p,  p_era5_3d, intermediate_grid; era5_kw...)

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
set!(p₀, Metadatum(:surface_pressure; dataset=ds_sl, meta_common...))

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
