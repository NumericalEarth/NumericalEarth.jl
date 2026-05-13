# # Regional hindcast with ERA5
#
# This is a regional modeling example that couples the Breeze compressible
# solver to forthcoming SlabLand and SlabOcean components.
#
# At the moment, this script does just the data ingest: download ERA5
# reanalysis restricted to a bounding box and interpolate it onto a
# `LatitudeLongitudeGrid` sized for ~3 km horizontal cells at the domain
# center latitude.
#
# In progress:
# - [x] Breeze model construction
# - [ ] initial state setting (set! the model from ingested fields)
# - [ ] acoustic substepping
# - [ ] dynamical initialization
# - [ ] open boundary conditions
# - [ ] land/ocean coupling
# - [ ] terrain

using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.ERA5
using CDSAPI  # activates NumericalEarthCDSAPIExt for download_dataset
using Oceananigans
using Oceananigans.Fields: CenterField, XFaceField, YFaceField
using Breeze
using Breeze.Thermodynamics
using Statistics: mean
using Dates
using Printf

# ## Configuration
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

λ_west  = λ₀ - Nx * Δλ / 2
λ_east  = λ₀ + Nx * Δλ / 2
φ_south = φ₀ - Ny * Δφ / 2
φ_north = φ₀ + Ny * Δφ / 2

# Vertical grid: 40 cells geometrically stretched 1.08× from Δz_surf = 50 m
# (Δz ≈ 1 km at the top of the stretched zone), then 24 cells of 1 km aloft.
# Lz ≈ 37 km — well within the ERA5 1 hPa top.

Nz_strch = 40
Nz_const = 24
Nz       = Nz_strch + Nz_const
Δz_surf  = 50.0
Δz_top   = 1000.0
r_z      = 1.08

function build_z_faces()
    zᶠ = zeros(Nz + 1)
    for k in 1:Nz_strch
        zᶠ[k+1] = zᶠ[k] + Δz_surf * r_z^(k-1)
    end
    for k in (Nz_strch+1):Nz
        zᶠ[k+1] = zᶠ[k] + Δz_top
    end
    return zᶠ
end

# Initial conditions
#
# Case day from the Holistic Interactions of Shallow Clouds, Aerosols and Land
# Ecosystems (HI-SCALE) campaign at the ARM SGP site. Features clear skies with
# periods of cirrus.

start_date = DateTime(2016, 09, 10, 12) # 7 am LT
end_date   = DateTime(2016, 09, 10, 18) # 1 pm LT

dates = start_date:Hour(1):end_date

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

# ## Download ERA5
#
# `download_dataset` bundles a multi-variable CDS request into one round
# trip per calendar day. Files already on disk are skipped on re-run.

# Where data will be saved locally
era5_datadir = "era5"

ds_pl = ERA5HourlyPressureLevels()
ds_sl = ERA5HourlySingleLevel()

pl_vars = [:eastward_velocity, :northward_velocity, :temperature,
           :specific_humidity, :geopotential,
           :specific_cloud_liquid_water_content,
           :specific_cloud_ice_water_content]

meta_common = (region = era5_region, dir = era5_datadir)

@info "Downloading ERA5 pressure-level data..."
download_dataset(pl_vars, ds_pl, dates; meta_common...)

@info "Downloading ERA5 surface pressure..."
download_dataset(:surface_pressure, ds_sl, dates; meta_common...)

# ## LAM grid
#
# `LatitudeLongitudeGrid` with `Bounded` horizontal topologies (LAM-style).
# The vertical coordinate is height in meters; the ERA5 pressure-level
# metadata supplies a domain-mean z(p) profile via the time-mean spatial-mean
# geopotential height (the dataset's default `mean_geopotential_height=true`).

zᶠ = build_z_faces()

grid = LatitudeLongitudeGrid(longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z         = zᶠ,
                             size      = (Nx, Ny, Nz),
                             halo      = (5, 5, 5),
                             topology  = (Bounded, Bounded, Bounded))

# ## Interpolate ERA5 onto the LAM grid
#
# `set!(field, metadatum)` builds the ERA5 metadata Field on its native
# (λ, φ, z=⟨Φ/g⟩) grid, then `Oceananigans.interpolate!` pushes it onto
# the target LAM grid. Both grids live in (lon-deg, lat-deg, z-m), so the
# cross-grid interpolation is well-defined in a single coordinate system.

u  = XFaceField(grid)
v  = YFaceField(grid)
T  = CenterField(grid)
qᵛ = CenterField(grid)
qᶜ = CenterField(grid)
qⁱ = CenterField(grid)

set!(u,  Metadatum(:eastward_velocity;                   dataset=ds_pl, date=start_date, meta_common...))
set!(v,  Metadatum(:northward_velocity;                  dataset=ds_pl, date=start_date, meta_common...))
set!(T,  Metadatum(:temperature;                         dataset=ds_pl, date=start_date, meta_common...))
set!(qᵛ, Metadatum(:specific_humidity;                   dataset=ds_pl, date=start_date, meta_common...))
set!(qᶜ, Metadatum(:specific_cloud_liquid_water_content; dataset=ds_pl, date=start_date, meta_common...))
set!(qⁱ, Metadatum(:specific_cloud_ice_water_content;    dataset=ds_pl, date=start_date, meta_common...))

# Virtual temperature: Tᵛ = T·(1 + ε·qᵛ), ε = Rᵛ/Rᵈ − 1. Vapor only by
# convention — the qᶜ, qⁱ terms belong to the density temperature Tρ.

constants = ThermodynamicConstants()
ε = vapor_gas_constant(constants) / dry_air_gas_constant(constants) - 1

Tᵛ = Field(T * (1 + ε * qᵛ))
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
set!(p₀, Metadatum(:surface_pressure; dataset=ds_sl, date=start_date, meta_common...))

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

atmos = atmosphere_simulation(grid;
                              thermodynamic_constants = constants,
                              dynamics = CompressibleDynamics(; surface_pressure = p̄₀))

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
