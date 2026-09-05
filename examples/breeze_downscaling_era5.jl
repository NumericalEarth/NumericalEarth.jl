# # ERA5 → 12 km coast-to-plains, convection-permitting hindcast (Breeze + SlabLand + PrescribedOcean)
#
# A limited-area model (LAM) example that downscales ERA5 reanalysis to a ~12 km Breeze compressible
# atmosphere over a domain running from the U.S. Southern Great Plains to the Gulf of Mexico, for the
# Midlatitude Continental Convective Clouds Experiment (MC3E) 20 May 2011 squall-line case
# ([Fan2017](@citet)). The lower boundary is part land, part ocean: an interactive `SlabLand` and a
# prescribed-SST Gulf share the surface through a fractional `SurfacePartition`, all-sky RRTMGP
# radiation heats the column and the surface, and the low-level jet that feeds the squall line draws
# its moisture from the resolved Gulf.
#
# `nested_atmosphere_model(grid, dataset; dates, …)` builds the whole nest: an ERA5 "parent"
# `PrescribedAtmosphere` on its native 0.25° pressure-level grid, driving a Breeze "child" through open
# lateral boundary conditions + interior Davies relaxation, both derived on the fly from the parent's
# raw state. The constructor also initializes the child from the reanalysis and spins it into balance,
# so a plain `Simulation(model)` then steps the ready nest.
#
# ## What this example does
# - Downloads ERA5 (pressure + single levels) and regrids it onto a terrain-following
#   `LatitudeLongitudeGrid` (ETOPO2022 orography, blended to the parent's at the boundary).
# - Initializes the prognostic state from ERA5 with a hydrostatic-from-surface pressure, a
#   terrain-consistent `w̃ ≈ 0`, and a dynamical-initialization (DFI) pass that balances `ρw`.
# - Integrates the compressible equations with split-explicit acoustic substepping, 1-moment
#   mixed-phase microphysics, Coriolis, and Rayleigh damping.
# - Couples an interactive `SlabLand` (prognostic skin temperature + bucket hydrology) and a
#   `PrescribedOcean` under the child through Monin–Obukhov similarity fluxes. The
#   `SurfacePartition` — built automatically because the model has both surfaces — blends each
#   cell's land and ocean fluxes by its ETOPO2022-derived `ocean_fraction`, so coastal cells are
#   fractionally wet rather than binary.
# - Runs all-sky RRTMGP radiative transfer inside the child — interior heating rates on the
#   evolving clouds, observed (CGLS 1 km) surface albedo, surface fluxes into the land energy
#   budget, and the partition-blended skin temperature closing the longwave loop.
# - Writes and animates horizontal slices, a vertical section, and the surface response.
#
# ## What it does NOT do (yet)
# - Single nest only (ERA5 → 12 km; coarsened 4× from Fan's 3 km Domain 3 for a fast configuration).
# - The land is a slab: no soil column, no vegetation, and no boundary-layer or cumulus
#   parameterization — diffusion is numerical, and deep convection is resolved on the grid.
# - The ocean is prescribed: the SST follows ERA5's hourly analysis but does not respond to the
#   atmosphere above it.
# - Snow does not reach the bucket yet (the coupler diagnoses the child's surface rain flux,
#   but no snow analog); immaterial for this warm-season case.

using NumericalEarth
using Oceananigans
using Oceananigans.Units          # `minutes` for the output schedule (not re-exported by Oceananigans)
using Oceananigans.OutputReaders: Clamp   # time-indexing scheme for the prescribed SST series
using Breeze
using CopernicusClimateDataStore # activates NumericalEarthCopernicusClimateDataStoreExt (ERA5 + land-albedo downloads)
using CloudMicrophysics          # nested_atmosphere_model's default microphysics → 1-moment mixed-phase
using RRTMGP                     # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using CairoMakie                 # loads Makie → NumericalEarthMakieExt (`visualize_nested_domain`)
using NaturalEarth, GeoInterface # together → NumericalEarthNaturalEarthExt (`natural_earth_lines`)
using CUDA
using Printf
using Dates: DateTime, Second

# This 12 km LAM (162×180×50 ≈ 1.5M cells, split-explicit) targets a CUDA GPU; switch to `CPU()` only
# for a small smoke test.
arch = GPU()

# Set Oceananigans' global default float type, cascading to all grids, fields, and constants.
Oceananigans.defaults.FloatType = Float32

# ## Configuration
#
# The domain covers [Fan2017](@citet)'s 3 km WRF Domain 3 around the DOE Atmospheric Radiation
# Measurement (ARM) Southern Great Plains (SGP) site in Lamont, OK — coarsened 4× and driven
# directly by ERA5 — and extends south past the Texas–Louisiana coast into the open Gulf of
# Mexico, so the surface under the child is genuinely part land, part ocean.

## dates
name = "mc3e"
duration = 3hours
start_date = DateTime(2011, 05, 20, 0)
stop_date = start_date + Second(duration)

## location: ARM SGP mid-domain, open Gulf water in the southern quarter
φ₀, λ₀ = 36.605, -97.485    # ARM SGP site (deg)
λ_west, λ_east   = -104.5, -86.5
φ_south, φ_north = 24.5, 44.5
Lλ = λ_east - λ_west
Lφ = φ_north - φ_south

## horizontal resolution: grid spacing in degrees; the cell counts follow from the extent
## (which stays fixed, so the realized spacing Lλ/Nx only approximates Δλ)
Δλ = Δφ = 1/9   # ≈ 12 km in latitude (ERA5's native 0.25° → 2.25× refinement)
Nx = round(Int, Lλ / Δλ)
Ny = round(Int, Lφ / Δφ)

dates = (start_date, stop_date)

# Vertical grid matched to [Fan2017](@citet)'s WRF nest: `Nz = 50` cells, a constant 60 m surface
# cell, 490 m maximum spacing, and a model top at ~20 km (~50 hPa).

z = ReferenceToStretchedDiscretization(extent = 19525.0,
                                       bias = :left,
                                       bias_edge = 0,
                                       constant_spacing = 60,
                                       constant_spacing_extent = 60,
                                       maximum_spacing = 490,
                                       stretching = LinearStretching(0.15))

Nz = length(z)

# ## LAM grid
#
# A bounded terrain-following `LatitudeLongitudeGrid`. The terrain itself is handled by
# `nested_atmosphere_model` (its `terrain` keyword): ETOPO2022 is regridded onto the grid, blended
# toward the parent's ERA5 orography over the outermost `relax_width` cells (so the terrain at the
# open boundaries matches the parent state driving them), and materialized onto the grid's
# terrain-following coordinate (Gal-Chen–Somerville σ): the bottom follows the ground, the top stays flat.

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z = TerrainFollowingVerticalDiscretization(z),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

# ## Build the nest
#
# `nested_atmosphere_model(grid, dataset; dates, …)` derives the parent region from the child grid
# (padded by two native ERA5 cells), loads the parent `PrescribedAtmosphere` on ERA5's native
# pressure-level grid, anchors the default split-explicit compressible dynamics at the domain-mean
# ERA5 surface pressure, derives the parent-driven lateral BCs + Davies relaxation (cosine ramp over
# `relaxation_width` cells), materializes the blended `terrain`, and wraps parent + child in a
# `NestedModel` whose `time_step!` advances the child then ticks the parent clock.

# Time window: a `(start, end)` tuple — the hourly forcing cadence is the dataset's own. The initial
# outer time step is gentle (the adaptive wizard below ramps it). The Davies relaxation is an *explicit*
# nudge `dφ/dt = −r(φ − φₚₐᵣₑₙₜ)`, stable only for `r·Δt ≲ 2`, so its rate is a fixed `1/300 s⁻¹` — small
# enough that `r·Δt` stays well below 1 even at the wizard's `max_Δt`, rather than tied to Δt (which would
# climb into the over-relaxation regime as Δt grows). Momentum uses a narrower `WENO(5)` stencil, which
# interacts less violently with the open-boundary halo than the default `WENO(9)`.

era5_datadir = "era5"   # where ERA5 data is saved locally
dataset = ERA5HourlyPressureLevels()
relax_width = 5

nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_datadir,
                               terrain = ETOPO2022(),
                               terrain_blend_width = relax_width,
                               relaxation_rate = 1/300,
                               relaxation_width = relax_width,
                               momentum_advection = WENO(order = 5))

# No `bottom_drag_coefficient`: the surface stress (with the heat and moisture fluxes) comes from
# the surface coupling below, so the child's bottom boundary conditions stay the coupler's flux fields.

# The realized parent region (child + padding, snapped to the native 0.25° grid) serves the domain
# map and the ERA5 snapshots below.
parent = nest.parent
era5_region = BoundingBox(parent.grid)

# ## Nested domains
#
# Visualize the nesting before stepping the model — the ERA5 forcing region and the 12 km LAM child,
# over ETOPO relief with Natural Earth state/country boundaries — so the domain geometry is written
# even if the run is cut short.

fig = visualize_nested_domain(grid;
                              parent = era5_region,
                              padding = 2.5,
                              title = "ERA5 → 12 km LAM nest (MC3E squall line, ARM SGP)",
                              label = "12 km LAM (child)",
                              parent_label = "ERA5 parent",
                              landmarks = tuple("ARM SGP" => (λ₀, φ₀)))

save("era5_breeze_domains.png", fig)

fig

# ERA5 pressure velocity ω (Pa/s), loaded raw onto the parent's own grid — so it shares the parent's
# time-varying geopotential vertical and composes in the `w ≈ −ω/(ρg)` operation with the parent state.
ω_metadata = Metadata(:vertical_velocity; dataset, dates, region = era5_region, dir = era5_datadir)
ω_series = FieldTimeSeries(ω_metadata, parent.grid; time_indices_in_memory = length(ω_metadata.dates))

# ## The land–ocean surface
#
# The lower boundary is part land, part Gulf of Mexico, and both surfaces live on the child's
# horizontal grid. The land is an interactive `SlabLand` — prognostic skin temperature plus
# bucket hydrology, driven by the coupled turbulent fluxes, the child's rain, and the radiation
# below; its bucket starts with ≈ 20 mm of water. The ocean is a `PrescribedOcean` whose SST
# follows ERA5's hourly analysis. ERA5's skin temperature supplies both — over open water it
# carries the analyzed sea surface temperature — so land and ocean start from one consistent
# field.

surface_grid = LatitudeLongitudeGrid(arch;
                                     longitude = (λ_west,  λ_east),
                                     latitude  = (φ_south, φ_north),
                                     size = (Nx, Ny),
                                     halo = (5, 5),
                                     topology = (Bounded, Bounded, Flat))

skin_temperature = Metadatum(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                             date = start_date, region = era5_region, dir = era5_datadir)

land = SlabLand(surface_grid)
set!(land.temperature, skin_temperature)
set!(land; M = 20)

# The ocean follows ERA5's hourly skin temperature rather than holding the initial analysis: a
# `FieldTimeSeries` over the same dates as the forcing, which the coupler reads at the ocean clock,
# interpolating linearly between hours. `Clamp` holds the end values outside the record, where the
# dataset default `Cyclical` would wrap May 20 03:00 back onto 00:00 at the final step.

sea_surface_temperature = Metadata(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                                   dates, region = era5_region, dir = era5_datadir)

sst = FieldTimeSeries(sea_surface_temperature, surface_grid;
                      time_indices_in_memory = length(sea_surface_temperature.dates),
                      time_indexing = Clamp())

ocean = PrescribedOcean(surface_grid, sst.times; sea_surface_temperature = sst)

# ## Surface partition
#
# What makes a cell land or ocean is its `ocean_fraction` þ: the fraction of the cell's area
# below sea level in ETOPO2022 — the same dataset that shapes the child's terrain, so the
# coastline and the orography agree. Coastal cells carry genuine fractions rather than a binary
# mask: the coupler blends each cell's land and ocean fluxes as `(1 - þ)` land + `þ` ocean, and
# the radiation sees the same blend of skin temperatures.

ocean_fraction = regrid_ocean_fraction(surface_grid; dataset = ETOPO2022())

# Alongside þ, the sea surface temperature the Gulf supplies. ERA5's skin temperature is set into
# every cell, so the field holds *land* skin temperature inland; masking to þ > 1/2 shows the water
# the partition actually weights. ETOPO2022's below-sea-level test also catches deep continental
# lakes, so Lake Michigan (bed ≈ 105 m below sea level) enters the northeast corner ~20 K colder.

Tᵒᶜ = Field{Center, Center, Nothing}(surface_grid)
set!(Tᵒᶜ, ocean.sea_surface_temperature[1])
interior(Tᵒᶜ) .= ifelse.(interior(ocean_fraction) .> 1/2, interior(Tᵒᶜ) .- 273.15, NaN)

fig_partition = Figure(size = (1320, 560))
ax_þ = Axis(fig_partition[1, 1]; xlabel = "longitude (°)", ylabel = "latitude (°)",
            title = "Ocean fraction þ (ETOPO2022)")
hm_þ = heatmap!(ax_þ, ocean_fraction; colormap = :dense, colorrange = (0, 1))
scatter!(ax_þ, [λ₀], [φ₀]; marker = :star5, color = :orange, markersize = 15)
text!(ax_þ, λ₀ + 0.3, φ₀; text = "ARM SGP", color = :orange, align = (:left, :center))
Colorbar(fig_partition[1, 2], hm_þ)

ax_T = Axis(fig_partition[1, 3]; xlabel = "longitude (°)",
            title = "Initial sea surface temperature where þ > 1/2")
hm_T = heatmap!(ax_T, Tᵒᶜ; colormap = :thermal, colorrange = (5, 30))
Colorbar(fig_partition[1, 4], hm_T; label = "°C")
save("gulf_ocean_fraction.png", fig_partition)

fig_partition

# ## RRTMGP radiation
#
# All-sky RRTMGP: interior heating on the evolving clouds, surface fluxes into the land
# budget. The surface albedo is observed, not constant — passing the dataset materializes
# the CGLS 1 km blue-sky albedo at the ten-day date nearest `epoch`, with water pixels
# (NaN in the land product) defaulting to open water — the Gulf gets its open-water albedo the
# same way. Everything else is a Breeze default — per-column sun angles from the grid,
# climatological ozone, emissivity, effective radii. No surface temperature either: the coupled
# model below binds the partition-blended skin temperature, re-read every solve. Hourly solves
# match the ERA5 cadence.

radiation = RadiativeTransferModel(grid, AllSkyOptics(), nest.child.thermodynamic_constants;
                                   solar_position = ApparentSolarPosition(epoch = start_date),
                                   surface_albedo = CopernicusAlbedo(),
                                   schedule = TimeInterval(1hour))

# ## Coupled model
#
# `EarthSystemModel` wires nest, land, ocean, and radiation together. Because the model has both
# a land and an ocean component, the `SurfacePartition` activates automatically: each coupled
# step computes Monin–Obukhov fluxes over both surfaces from the child's lowest-cell state,
# blends them by the cell's `ocean_fraction`, and writes the blend into the child's bottom flux
# boundary conditions; the land's accumulators receive their per-land-area fluxes. The RTM is
# bound to the partition-weighted skin temperature — SST over the Gulf, the live land skin
# temperature inland. The adaptive outer Δt is bounded by the advective CFL (acoustic modes are
# substepped).

Δt = 10
atmosphere = Simulation(nest; Δt)   ## the coupled model manages Δt; this sets only the initial value
model = EarthSystemModel(; atmosphere, land, ocean, radiation, ocean_fraction)

simulation = Simulation(model; Δt, stop_time=duration)
conjure_time_step_wizard!(simulation, IterationInterval(3); cfl=0.5, max_Δt=Δt)

# ## Output
#
# Three `JLD2Writer`s save 2-D slices of one shared `fields` NamedTuple of online diagnostics —
# never the 3-D fields, which would overflow the disk. Each output is an `AbstractOperation`
# (Breeze's diagnostic accessors) that every writer computes and slices at save time via its own
# `indices`: horizontal slices at the surface and at `k_aloft` (the reference level nearest 2 km —
# on the terrain-following grid a constant reference level ≈ constant height above ground near the
# surface), and a zonal x-z section at `j_section`, the latitude row through the ARM SGP site.
# A fourth writer saves the surface: the land skin temperature, the bucket saturation, and the
# partition-blended skin temperature Tₛ the atmosphere sees (2-D already, no slicing needed —
# the writer stores each output's own grid).

child = nest.child
k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 2000)
φ_centers = Array(φnodes(grid, Center(), Center(), Center()))
j_section = searchsortedfirst(φ_centers, φ₀)

fields = (θᵛ = VirtualPotentialTemperature(child),
          U  = sqrt(child.velocities.u^2 + child.velocities.v^2),
          w  = child.velocities.w,
          qᵛ = specific_humidity(child),
          qʳ = child.microphysical_fields.qʳ)

surface_filename = name * "_surface.jld2"
aloft_filename   = name * "_aloft.jld2"
section_filename = name * "_section.jld2"

schedule = TimeInterval(20minutes)
slice_writer(indices, filename) = JLD2Writer(model, fields; schedule, filename, indices,
                                             overwrite_existing = true)

simulation.output_writers[:surface] = slice_writer((:, :, 1),         surface_filename)
simulation.output_writers[:aloft]   = slice_writer((:, :, k_aloft),   aloft_filename)
simulation.output_writers[:section] = slice_writer((:, j_section, :), section_filename)

land_filename = name * "_land.jld2"
land_fields = (Tˡᵃ = land.temperature,
               𝒮 = land.saturation,
               Tₛ = surface_temperature(model.interfaces))
simulation.output_writers[:land] = JLD2Writer(model, land_fields; schedule,
                                              filename = land_filename,
                                              overwrite_existing = true)

function progress(sim)
    child = sim.model.atmosphere.model.child
    u, v, w = child.velocities
    ρ  = child.dynamics.total_density
    qᵛ = specific_humidity(child)
    qʳ = child.microphysical_fields.qʳ
    @info @sprintf("iter=%4d, t=%s, Δt=%s, max|u|=(%7.2f, %7.2f, %6.2f), ρ ∈ [%.4f, %.4f], qᵛ ∈ [%.4g, %.4g], qʳ ∈ [%.2g, %.2g]",
                   sim.model.clock.iteration, prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), minimum(ρ), maximum(ρ),
                   minimum(qᵛ), maximum(qᵛ), minimum(qʳ), maximum(qʳ))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Run

run!(simulation)

# ## Cascade animation
#
# A 2-row × 5-column animation of the downscaling: row 1 the ERA5 parent (dashed rectangle = child
# extent), row 2 the Breeze child. Columns are near-surface θᵛ′ (referenced pointwise to t=0, so the
# terrain/stratification background cancels and the cold pool stands out) and wind speed |U|, then
# w, qᵛ, qʳ at 2 km. The saved slices load straight back as `FieldTimeSeries` and plot directly —
# `heatmap!` reads each 2-D `Field`'s own coordinates and moves it host-side, so there is no manual
# indexing, and the θᵛ′ anomaly `θᵛ[n] - θᵛ[1]` is a lazy field operation.

θᵛ_series = FieldTimeSeries(surface_filename, "θᵛ")
U_series  = FieldTimeSeries(surface_filename, "U")
w_series  = FieldTimeSeries(aloft_filename, "w")
qᵛ_series = FieldTimeSeries(aloft_filename, "qᵛ")
qʳ_series = FieldTimeSeries(aloft_filename, "qʳ")
times = θᵛ_series.times

# The parent is prescribed (no output file). Its rows are reconstructed at each frame time from the
# parent's own raw ERA5 fields — which are full-in-memory `FieldTimeSeries` — via `breeze_prognostic_state`,
# the exported (T, qᵛ, p) → (ρ, θˡⁱ, qᵗ) transform (the same physics the lateral BCs apply). We do NOT
# read the `StateExchanger` prognostics here: that FTS is a 2-level window, so post-run per-frame reads
# would alias to whatever the last `exchange_state!` left resident. ERA5's w ≈ −ω/(ρg); no rain.

k_parent = searchsortedfirst(Array(znodes(parent.grid, Center())), 2000)
constants = child.thermodynamic_constants
pˢᵗ = child.dynamics.standard_pressure
g = constants.gravitational_acceleration
ε = vapor_gas_constant(constants) / dry_air_gas_constant(constants) - 1   # virtual-temperature coefficient

slice(operation, k) = Field(operation, indices = (:, :, k))

function parent_slices(t)
    hydrometeors = parent.microphysical_variables
    qˡ = hydrometeors.qᶜˡ[Time(t)] + hydrometeors.qʳ[Time(t)]   ## total liquid: cloud + rain
    qⁱ = hydrometeors.qᶜⁱ[Time(t)] + hydrometeors.qˢ[Time(t)]   ## total ice: cloud ice + snow
    (; ρ, θˡⁱ, qᵗ) = breeze_prognostic_state(constants, pˢᵗ,
                        parent.temperature[Time(t)], parent.specific_humidity[Time(t)],
                        qˡ, qⁱ, parent.pressure)
    u, v = parent.velocities.u[Time(t)], parent.velocities.v[Time(t)]
    ## `u` and `v` are staggered, so |U| would inherit `u`'s location; force it to centers, where the
    ## other three frames already live, so every parent panel shares one prototype location.
    speed = @at((Center, Center, Center), sqrt(u^2 + v^2))
    return (θᵛ = slice(θˡⁱ * (1 + ε * qᵗ),           1),
            U  = slice(speed,                        1),
            w  = slice(-ω_series[Time(t)] / (ρ * g), k_parent),
            qᵛ = slice(qᵗ,                           k_parent))
end
parent_frames = [parent_slices(t) for t in times]
nothing #hide

# One column each: title, colormap, the child/parent 2-D fields as functions of the frame `n`, and
# the parent level `k` those frames are sliced at. `nothing` parent ⇒ blank panel (rain). Fixed,
# illustrative color ranges (this is a demo, not a tuned figure); `w` alone differs ~10× between
# parent and child, so it carries a range per row.
g_per_kg(field) = 1f3 * field
columns = [(title = "θᵛ′ₛ (K)",       colormap = :balance, child_range = (-6, 6),  parent_range = (-6, 6),
            k = 1,        child_series = θᵛ_series,
            child = n -> θᵛ_series[n] - θᵛ_series[1], parent = n -> parent_frames[n].θᵛ - parent_frames[1].θᵛ),
           (title = "|U|ₛ (m s⁻¹)",   colormap = :speed,   child_range = (0, 30),  parent_range = (0, 30),
            k = 1,        child_series = U_series,
            child = n -> U_series[n],                 parent = n -> parent_frames[n].U),
           (title = "w₂ₖₘ (m s⁻¹)",   colormap = :balance, child_range = (-3, 3),  parent_range = (-0.3, 0.3),
            k = k_parent, child_series = w_series,
            child = n -> w_series[n],                 parent = n -> parent_frames[n].w),
           (title = "qᵛ₂ₖₘ (g kg⁻¹)", colormap = :dense,   child_range = (0, 15),  parent_range = (0, 15),
            k = k_parent, child_series = qᵛ_series,
            child = n -> g_per_kg(qᵛ_series[n]),      parent = n -> g_per_kg(parent_frames[n].qᵛ)),
           (title = "qʳ₂ₖₘ (g kg⁻¹)", colormap = :dense,   child_range = (0, 2),   parent_range = (0, 2),
            k = k_parent, child_series = qʳ_series,
            child = n -> g_per_kg(qʳ_series[n]),      parent = n -> nothing)]

boxλ = [λ_west, λ_east, λ_east, λ_west, λ_west]
boxφ = [φ_south, φ_south, φ_north, φ_north, φ_south]

fig_cascade = Figure(size = (1500, 700))
cascade_n = Observable(1)

## Each panel plots one plain field (`similar(prototype)`: the prototype's location, grid and indices,
## its own data), refilled per frame and handed back as the *same* object. `heatmap!` reads the field's
## own λ/φ and moves it host-side, so no node arrays and no manual materialization. Reusing one object
## matters twice: Makie's compute graph otherwise compares successive plot inputs elementwise, which on
## the parent's GPU fields is scalar indexing, and `heatmap!` calls `compute!` on what it is handed,
## which on a field carrying an operand would overwrite the frame just copied in.
function panel!(ax, prototype, field_of, colormap, colorrange)
    panel = similar(prototype)
    frame = Observable(panel)
    on(nn -> (set!(panel, compute!(field_of(nn))); frame[] = panel), cascade_n)
    set!(panel, compute!(field_of(1)))
    return heatmap!(ax, frame; colormap, colorrange)
end

Label(fig_cascade[0, 1:5],
      @lift(@sprintf("MC3E 20 May 2011 — ERA5 → 12 km Breeze — t = %.1f h", times[$cascade_n] / 3600)),
      fontsize = 20, tellwidth = false)

for (i, column) in enumerate(columns)
    parent_ax = Axis(fig_cascade[1, i]; title = column.title, aspect = DataAspect())
    child_ax  = Axis(fig_cascade[2, i]; aspect = DataAspect())

    hmc = panel!(child_ax, column.child_series[1], column.child, column.colormap, column.child_range)

    if isnothing(column.parent(1))
        text!(parent_ax, 0.5, 0.5; text = "no rain", space = :relative, align = (:center, :center), color = :gray)
        Colorbar(fig_cascade[3, i], hmc; vertical = false, flipaxis = false, height = 10)
    else
        parent_prototype = CenterField(parent.grid, indices = (:, :, column.k))
        hmp = panel!(parent_ax, parent_prototype, column.parent, column.colormap, column.parent_range)
        lines!(parent_ax, boxλ, boxφ; color = :black, linestyle = :dash, linewidth = 1.5)
        if column.parent_range == column.child_range
            Colorbar(fig_cascade[3, i], hmc; vertical = false, flipaxis = false, height = 10)
        else   ## w differs ~10× parent-to-child — a colorbar per row
            Colorbar(fig_cascade[3, i], hmp; vertical = false, flipaxis = false, height = 10, label = "ERA5")
            Colorbar(fig_cascade[4, i], hmc; vertical = false, flipaxis = false, height = 10, label = "12 km")
        end
    end
    hidedecorations!(parent_ax); hidedecorations!(child_ax)
    if i == 1
        text!(parent_ax, 0.03, 0.97; text = "ERA5",         space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
        text!(child_ax,  0.03, 0.97; text = "Breeze 12 km", space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
    end
end
CairoMakie.record(fig_cascade, "era5_cascade_2row.mp4", eachindex(times); framerate = 8) do nn
    cascade_n[] = nn   ## `record` is also exported by CUDA, so qualify it
end
nothing #hide

# ![](era5_cascade_2row.mp4)

# ## Section animation
#
# The vertical structure along the zonal x-z section through the ARM SGP site: wind speed (the
# nocturnal Great Plains low-level jet near the surface and the upper-level jet aloft), vertical
# velocity (the convective updrafts and the squall line's mesoscale circulation), and water vapor
# (the moist tongue feeding the convection). The section slices load back as `FieldTimeSeries` of
# 2-D fields on the terrain-following grid, and `heatmap!` maps them to physical ``(λ, z)``
# coordinates directly — one `@lift` per panel is the whole animation machinery.

U_section  = FieldTimeSeries(section_filename, "U")
w_section  = FieldTimeSeries(section_filename, "w")
qᵛ_section = FieldTimeSeries(section_filename, "qᵛ")

φ_section = φ_centers[j_section]

## Type-erased observables: the per-frame fields' concrete types can differ in hidden
## parameters, which a `@lift`-typed Observable rejects at the second frame.
section_n = Observable(1)
Uₙ  = Observable{Any}(U_section[1])
wₙ  = Observable{Any}(w_section[1])
qᵛₙ = Observable{Any}(g_per_kg(qᵛ_section[1]))
on(section_n) do n
    Uₙ[]  = U_section[n]
    wₙ[]  = w_section[n]
    qᵛₙ[] = g_per_kg(qᵛ_section[n])
end

fig_section = Figure(size = (1000, 900))

Label(fig_section[0, 1:2],
      @lift(@sprintf("MC3E 20 May 2011 — vertical structure at %.2f°N — t = %.1f h",
                     φ_section, times[$section_n] / 3600)),
      fontsize = 20, tellwidth = false)

ax_U = Axis(fig_section[1, 1]; ylabel = "z (m)", title = "|U| (m s⁻¹)")
ax_w = Axis(fig_section[2, 1]; ylabel = "z (m)", title = "w (m s⁻¹)")
ax_q = Axis(fig_section[3, 1]; ylabel = "z (m)", xlabel = "longitude (°)", title = "qᵛ (g kg⁻¹)")

hm_U = heatmap!(ax_U, Uₙ;  colormap = :speed,   colorrange = (0, 40))
hm_w = heatmap!(ax_w, wₙ;  colormap = :balance, colorrange = (-3, 3))
hm_q = heatmap!(ax_q, qᵛₙ; colormap = :dense,   colorrange = (0, 15))

Colorbar(fig_section[1, 2], hm_U)
Colorbar(fig_section[2, 2], hm_w)
Colorbar(fig_section[3, 2], hm_q)

CairoMakie.record(fig_section, "era5_section_xz.mp4", eachindex(times); framerate = 8) do nn
    section_n[] = nn
end
nothing #hide

# ![](era5_section_xz.mp4)

# ## Surface response
#
# The coupled surface closes the loop. The partition-blended skin temperature Tₛ — the one
# surface the atmosphere and the radiation see — changes over the run only where the land
# fraction lets it: daytime heating plus cooling under the squall line's cold pool and anvil
# shading over the plains, next to nothing over the prescribed Gulf, and a muted response along
# the fractional coast. The final bucket saturation shows where the squall line rained. Both
# load back as 2-D `FieldTimeSeries` and plot directly.

Tₛ_series = FieldTimeSeries(land_filename, "Tₛ")
𝒮_series  = FieldTimeSeries(land_filename, "𝒮")

fig_land = Figure(size = (1100, 420))

ax_T = Axis(fig_land[1, 1]; xlabel = "longitude (°)", ylabel = "latitude (°)",
            title = "ΔTₛ over $(prettytime(duration)) (K)")
ax_𝒮 = Axis(fig_land[1, 3]; xlabel = "longitude (°)", title = "final surface saturation 𝒮")

hm_T = heatmap!(ax_T, Tₛ_series[end] - Tₛ_series[1]; colormap = :balance, colorrange = (-8, 8))
hm_𝒮 = heatmap!(ax_𝒮, 𝒮_series[end]; colormap = :dense, colorrange = (0, 1))

Colorbar(fig_land[1, 2], hm_T)
Colorbar(fig_land[1, 4], hm_𝒮)

save("era5_breeze_surface_response.png", fig_land)

fig_land
