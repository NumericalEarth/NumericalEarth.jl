# # ERA5 → 12 km land-coupled, convection-permitting hindcast (Breeze + NestedModel + SlabLand)
#
# A limited-area model (LAM) example that downscales ERA5 reanalysis to a ~12 km Breeze compressible
# atmosphere over the U.S. Southern Great Plains, for the Midlatitude Continental Convective Clouds
# Experiment (MC3E) 20 May 2011 squall-line case ([Fan2017](@citet)) — with an interactive `SlabLand`
# surface under the child and all-sky RRTMGP radiation heating the column and the surface, so the
# diurnal cycle, the clouds, and the land interact.
#
# `nested_atmosphere_model(grid, dataset; dates, …)` builds the whole nest: an ERA5 "parent"
# `PrescribedAtmosphere` on its native 0.25° pressure-level grid, driving a Breeze "child" through open
# lateral boundary conditions and interior Davies relaxation, both derived on the fly from the parent's
# raw state. The constructor also initializes the child from the reanalysis and spins it into balance,
# so that a plain `Simulation` steps the ready nest.
#
# ## What this example does
# - Downloads ERA5 (pressure and single levels) and regrids it onto a terrain-following
#   `LatitudeLongitudeGrid` (ETOPO2022 orography, blended into the parent's at the boundary).
# - Initializes the prognostic state from ERA5 with a hydrostatic-from-surface pressure, a
#   terrain-consistent `w̃ ≈ 0`, and a dynamical-initialization (DFI) pass that balances `ρw`.
# - Integrates the compressible equations with split-explicit acoustic substepping, 1-moment
#   mixed-phase microphysics, Coriolis, and Rayleigh damping.
# - Couples an interactive `SlabLand` (prognostic skin temperature + bucket hydrology) under the
#   child through Monin–Obukhov similarity fluxes, and the child's rain into the land bucket.
# - Runs all-sky RRTMGP radiative transfer inside the child — interior heating rates on the
#   evolving clouds, observed (CGLS 1 km) surface albedo, surface fluxes into the land energy
#   budget, and the interface skin temperature closing the longwave loop.
# - Writes and animates horizontal slices, a vertical section, and the land-surface response.
#
# ## What it does NOT do (yet)
# - Single nest only (ERA5 → 12 km; coarsened 4× from Fan's 3 km Domain 3 for a fast configuration).
# - The land is a slab: no soil column, no vegetation, and no boundary-layer or cumulus
#   parameterization — diffusion is numerical, and deep convection is resolved on the grid.
# - Snow does not reach the bucket: the coupler diagnoses the child's surface rain flux but has no
#   snow analog, which is immaterial for this warm-season case.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Breeze
using CopernicusClimateDataStore # activates NumericalEarthCopernicusClimateDataStoreExt (ERA5 + land-albedo downloads)
using CloudMicrophysics          # nested_atmosphere_model's default microphysics → 1-moment mixed-phase
using RRTMGP                     # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using CairoMakie                 # loads Makie → NumericalEarthMakieExt (`visualize_nested_domain`)
using NaturalEarth, GeoInterface # together → NumericalEarthNaturalEarthExt (`natural_earth_lines`)
using CUDA
using Printf
using Dates: DateTime, Second

# This 12 km LAM (150×136×50 ≈ 1.0M cells, split-explicit) targets a CUDA GPU; switch to `CPU()` only
# for a small smoke test.
arch = GPU()

# Set Oceananigans' global default float type, cascading to all grids, fields, and constants.
Oceananigans.defaults.FloatType = Float32

# ## Configuration
#
# The domain is centered on the DOE Atmospheric Radiation Measurement (ARM) Southern Great Plains (SGP) site
# in Lamont, OK — the 3 km Domain 3 of the WRF telescoping nest in [Fan2017](@citet), coarsened 4×
# and driven directly by ERA5.

## dates
name = "mc3e"
duration = 3hours
start_date = DateTime(2011, 05, 20, 0)
end_date = start_date + Second(duration)

## location
φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)
Lλ, Lφ = 16.7, 15.1

## horizontal resolution: grid spacing in degrees; the cell counts follow from the extent
## (which stays fixed, so the realized spacing Lλ/Nx only approximates Δλ)
Δλ = Δφ = 1/9   # ≈ 12 km in latitude (ERA5's native 0.25° → 2.25× refinement)
Nx = round(Int, Lλ / Δλ)
Ny = round(Int, Lφ / Δφ)

dates = (start_date, end_date)
λ₁, λ₂ = λ₀ - Lλ / 2, λ₀ + Lλ / 2
φ₁, φ₂ = φ₀ - Lφ / 2, φ₀ + Lφ / 2

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
# toward the parent's ERA5 orography over the outermost `relaxation_width` cells (so the terrain at the
# open boundaries matches the parent state driving them), and materialized onto the grid's
# terrain-following coordinate (Gal-Chen–Somerville σ): the bottom follows the ground, the top stays flat.

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ₁, λ₂),
                             latitude  = (φ₁, φ₂),
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

# The time window is a `(start_date, end_date)` tuple; the hourly forcing cadence is the dataset's own.
# The Davies relaxation is an explicit nudge, `dφ/dt = −r (φ − φᵖ)` toward the parent state `φᵖ`, which is
# stable only for `r Δt ≲ 2`. We therefore fix its rate at `r = 1/300 s⁻¹`, so that `r Δt` stays well below 1
# even at the largest time step the adaptive wizard below allows. Momentum uses the narrower `WENO(order = 5)`
# stencil, which interacts more gently with the open-boundary halo.

era5_directory = "era5"   # where ERA5 data is saved locally
dataset = ERA5HourlyPressureLevels()
relaxation_width = 5

nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_directory,
                               terrain = ETOPO2022(),
                               terrain_blend_width = relaxation_width,
                               relaxation_rate = 1/300,
                               relaxation_width,
                               momentum_advection = WENO(order = 5))

# The surface stress, heat, and moisture fluxes come from the land coupling below. The realized parent
# region (the child plus padding, snapped to the native 0.25° grid) serves the domain map and the ERA5
# snapshots below.
parent_atmosphere = nest.parent
era5_region = BoundingBox(parent_atmosphere.grid)

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

# We load the ERA5 pressure velocity `ω` (Pa s⁻¹) onto the parent's own grid, so that it shares the parent's
# time-varying geopotential heights and combines with the parent state in the operation `w ≈ −ω / (ρ g)`.
vertical_velocity_metadata = Metadata(:vertical_velocity; dataset, dates, region = era5_region, dir = era5_directory)
ω_ts = FieldTimeSeries(vertical_velocity_metadata, parent_atmosphere.grid;
                       time_indices_in_memory = length(vertical_velocity_metadata.dates))

# ## Land surface
#
# The lower boundary is interactive: a `SlabLand` on the child's horizontal grid — prognostic
# skin temperature plus bucket hydrology, driven by the coupled turbulent fluxes, the child's
# rain, and the radiation below. Temperature initializes from ERA5's skin temperature; the
# bucket starts with ≈ 20 mm of water.

land_grid = LatitudeLongitudeGrid(arch;
                                  longitude = (λ₁, λ₂),
                                  latitude  = (φ₁, φ₂),
                                  size = (Nx, Ny),
                                  halo = (5, 5),
                                  topology = (Bounded, Bounded, Flat))

land = SlabLand(land_grid)
skin_temperature = Metadatum(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                             date = start_date, region = era5_region, dir = era5_directory)
set!(land.temperature, skin_temperature)
set!(land; M = 20)

# ## RRTMGP radiation
#
# All-sky RRTMGP: interior heating on the evolving clouds, surface fluxes into the land
# budget. The surface albedo is observed rather than constant: passing the dataset materializes
# the CGLS 1 km blue-sky albedo at the ten-day date nearest `epoch`, and water pixels (NaN in the
# land product) default to open water. Everything else is a Breeze default: per-column sun angles
# from the grid, climatological ozone, emissivity, and effective radii. We do not set a surface
# temperature either, because the coupled model below binds the atmosphere–land interface skin
# temperature, which every solve re-reads. Hourly solves match the ERA5 cadence.

radiation = RadiativeTransferModel(grid, AllSkyOptics(), nest.child.thermodynamic_constants;
                                   solar_position = ApparentSolarPosition(epoch = start_date),
                                   surface_albedo = CopernicusAlbedo(),
                                   schedule = TimeInterval(1hour))

# ## Coupled model
#
# `AtmosphereLandModel` wires nest, land, and radiation together: it binds the interface skin
# temperature into the RTM, materializes the child's radiation proxy, and each step computes
# Monin–Obukhov fluxes from the child's lowest-cell state, writing them into the child's
# bottom flux boundary conditions and the land's accumulators. The adaptive outer Δt is
# bounded by the advective CFL (acoustic modes are substepped).

Δt = 10
atmosphere = Simulation(nest; Δt)   ## the coupled model manages Δt; this sets only the initial value
model = AtmosphereLandModel(atmosphere, land; radiation)

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
# A fourth writer saves the land surface: skin temperature and bucket saturation (2-D already,
# no slicing needed — the writer stores each output's own grid).

child = nest.child
k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 2000)
latitudes = Array(φnodes(grid, Center()))
j_section = searchsortedfirst(latitudes, φ₀)

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
                                             overwrite_files = true)

simulation.output_writers[:surface] = slice_writer((:, :, 1),         surface_filename)
simulation.output_writers[:aloft]   = slice_writer((:, :, k_aloft),   aloft_filename)
simulation.output_writers[:section] = slice_writer((:, j_section, :), section_filename)

land_filename = name * "_land.jld2"
land_fields = (Tˡᵃ = land.temperature, 𝒮 = land.saturation)
simulation.output_writers[:land] = JLD2Writer(model, land_fields; schedule,
                                              filename = land_filename,
                                              overwrite_files = true)

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
# A 2-row × 5-column animation of the downscaling: the top row shows the ERA5 parent (the dashed
# rectangle marks the child's extent) and the bottom row the Breeze child. The columns show the
# near-surface θᵛ′ (the anomaly from `t = 0` at each point, which cancels the terrain and stratification
# background so that the cold pool stands out) and wind speed |U|, then w, qᵛ, and qʳ at 2 km. The saved slices load straight back as `FieldTimeSeries` and plot directly —
# `heatmap!` reads each 2-D `Field`'s own coordinates and moves it host-side, so there is no manual
# indexing, and the θᵛ′ anomaly `θᵛ[n] - θᵛ[1]` is a lazy field operation.

θᵛ_ts = FieldTimeSeries(surface_filename, "θᵛ")
U_ts  = FieldTimeSeries(surface_filename, "U")
w_ts  = FieldTimeSeries(aloft_filename, "w")
qᵛ_ts = FieldTimeSeries(aloft_filename, "qᵛ")
qʳ_ts = FieldTimeSeries(aloft_filename, "qʳ")
times = θᵛ_ts.times

# The parent is prescribed, so it has no output file. We reconstruct its panels at each frame time from its
# raw ERA5 fields, which are held fully in memory, using `breeze_prognostic_state`, the
# `(T, qᵛ, p) → (ρ, θˡⁱ, qᵗ)` transform that the lateral boundary conditions also apply. The parent's
# vertical velocity is `w ≈ −ω / (ρ g)`, and its rain panel is left blank.

k_parent = searchsortedfirst(Array(znodes(parent_atmosphere.grid, Center())), 2000)
constants = child.thermodynamic_constants
pˢᵗ = child.dynamics.standard_pressure
g = constants.gravitational_acceleration
ε = vapor_gas_constant(constants) / dry_air_gas_constant(constants) - 1   # virtual-temperature coefficient

slice(operation, k) = Field(operation, indices = (:, :, k))

function parent_slices(t)
    hydrometeors = parent_atmosphere.microphysical_variables
    qˡ = hydrometeors.qᶜˡ[Time(t)] + hydrometeors.qʳ[Time(t)]   ## total liquid: cloud + rain
    qⁱ = hydrometeors.qᶜⁱ[Time(t)] + hydrometeors.qˢ[Time(t)]   ## total ice: cloud ice + snow
    (; ρ, θˡⁱ, qᵗ) = breeze_prognostic_state(constants, pˢᵗ,
                        parent_atmosphere.temperature[Time(t)], parent_atmosphere.specific_humidity[Time(t)],
                        qˡ, qⁱ, parent_atmosphere.pressure)
    u, v = parent_atmosphere.velocities.u[Time(t)], parent_atmosphere.velocities.v[Time(t)]
    ## `u` and `v` are staggered, so |U| would inherit `u`'s location; force it to centers, where the
    ## other three frames already live, so every parent panel shares one prototype location.
    speed = @at((Center, Center, Center), sqrt(u^2 + v^2))
    return (θᵛ = slice(θˡⁱ * (1 + ε * qᵗ),           1),
            U  = slice(speed,                        1),
            w  = slice(-ω_ts[Time(t)] / (ρ * g), k_parent),
            qᵛ = slice(qᵗ,                           k_parent))
end
parent_frames = [parent_slices(t) for t in times]
nothing #hide

# Each column holds a title, a colormap, the child and parent 2-D fields as functions of the frame `n`, and
# the parent level `k` at which those fields are sliced. A parent that returns `nothing` leaves its panel
# blank (rain). The color ranges are fixed and illustrative; only `w`, which differs by ~10× between parent
# and child, carries a separate range for each row.
grams_per_kilogram(field) = 1f3 * field
columns = [(title = "θᵛ′ₛ (K)",       colormap = :balance, child_range = (-6, 6),  parent_range = (-6, 6),
            k = 1,        child_ts = θᵛ_ts,
            child = n -> θᵛ_ts[n] - θᵛ_ts[1],             parent = n -> parent_frames[n].θᵛ - parent_frames[1].θᵛ),
           (title = "|U|ₛ (m s⁻¹)",   colormap = :speed,   child_range = (0, 30),  parent_range = (0, 30),
            k = 1,        child_ts = U_ts,
            child = n -> U_ts[n],                         parent = n -> parent_frames[n].U),
           (title = "w₂ₖₘ (m s⁻¹)",   colormap = :balance, child_range = (-3, 3),  parent_range = (-0.3, 0.3),
            k = k_parent, child_ts = w_ts,
            child = n -> w_ts[n],                         parent = n -> parent_frames[n].w),
           (title = "qᵛ₂ₖₘ (g kg⁻¹)", colormap = :dense,   child_range = (0, 15),  parent_range = (0, 15),
            k = k_parent, child_ts = qᵛ_ts,
            child = n -> grams_per_kilogram(qᵛ_ts[n]),    parent = n -> grams_per_kilogram(parent_frames[n].qᵛ)),
           (title = "qʳ₂ₖₘ (g kg⁻¹)", colormap = :dense,   child_range = (0, 2),   parent_range = (0, 2),
            k = k_parent, child_ts = qʳ_ts,
            child = n -> grams_per_kilogram(qʳ_ts[n]),    parent = n -> nothing)]

boxλ = [λ₁, λ₂, λ₂, λ₁, λ₁]
boxφ = [φ₁, φ₁, φ₂, φ₂, φ₁]

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
    on(n -> (set!(panel, compute!(field_of(n))); frame[] = panel), cascade_n)
    set!(panel, compute!(field_of(1)))
    return heatmap!(ax, frame; colormap, colorrange)
end

Label(fig_cascade[0, 1:5],
      @lift(@sprintf("MC3E 20 May 2011 — ERA5 → 12 km Breeze — t = %.1f h", times[$cascade_n] / hour)),
      fontsize = 20, tellwidth = false)

for (i, column) in enumerate(columns)
    parent_ax = Axis(fig_cascade[1, i]; title = column.title, aspect = DataAspect())
    child_ax  = Axis(fig_cascade[2, i]; aspect = DataAspect())

    child_heatmap = panel!(child_ax, column.child_ts[1], column.child, column.colormap, column.child_range)

    if isnothing(column.parent(1))
        text!(parent_ax, 0.5, 0.5; text = "no rain", space = :relative, align = (:center, :center), color = :gray)
        Colorbar(fig_cascade[3, i], child_heatmap; vertical = false, flipaxis = false, height = 10)
    else
        parent_prototype = CenterField(parent_atmosphere.grid, indices = (:, :, column.k))
        parent_heatmap = panel!(parent_ax, parent_prototype, column.parent, column.colormap, column.parent_range)
        lines!(parent_ax, boxλ, boxφ; color = :black, linestyle = :dash, linewidth = 1.5)
        if column.parent_range == column.child_range
            Colorbar(fig_cascade[3, i], child_heatmap; vertical = false, flipaxis = false, height = 10)
        else   ## w differs ~10× parent-to-child — a colorbar per row
            Colorbar(fig_cascade[3, i], parent_heatmap; vertical = false, flipaxis = false, height = 10, label = "ERA5")
            Colorbar(fig_cascade[4, i], child_heatmap; vertical = false, flipaxis = false, height = 10, label = "12 km")
        end
    end
    hidedecorations!(parent_ax); hidedecorations!(child_ax)
    if i == 1
        text!(parent_ax, 0.03, 0.97; text = "ERA5",         space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
        text!(child_ax,  0.03, 0.97; text = "Breeze 12 km", space = :relative, align = (:left, :top), color = :white, fontsize = 15, font = :bold)
    end
end
## `record` is also exported by CUDA, so we qualify it
CairoMakie.record(fig_cascade, "era5_cascade_2row.mp4", eachindex(times); framerate = 8) do n
    cascade_n[] = n
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
# coordinates directly, so a single `on(section_n)` callback animates all three panels.

U_section_ts  = FieldTimeSeries(section_filename, "U")
w_section_ts  = FieldTimeSeries(section_filename, "w")
qᵛ_section_ts = FieldTimeSeries(section_filename, "qᵛ")

section_latitude = latitudes[j_section]

## Type-erased observables: the per-frame fields' concrete types can differ in hidden
## parameters, which a `@lift`-typed Observable rejects at the second frame.
section_n = Observable(1)
Uₙ  = Observable{Any}(U_section_ts[1])
wₙ  = Observable{Any}(w_section_ts[1])
qᵛₙ = Observable{Any}(grams_per_kilogram(qᵛ_section_ts[1]))
on(section_n) do n
    Uₙ[]  = U_section_ts[n]
    wₙ[]  = w_section_ts[n]
    qᵛₙ[] = grams_per_kilogram(qᵛ_section_ts[n])
end

fig_section = Figure(size = (1000, 900))

Label(fig_section[0, 1:2],
      @lift(@sprintf("MC3E 20 May 2011 — vertical structure at %.2f°N — t = %.1f h",
                     section_latitude, times[$section_n] / hour)),
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

CairoMakie.record(fig_section, "era5_section_xz.mp4", eachindex(times); framerate = 8) do n
    section_n[] = n
end
nothing #hide

# ![](era5_section_xz.mp4)

# ## Land response
#
# The coupled surface closes the loop: skin-temperature change over the run (daytime heating,
# plus cooling under the squall line's cold pool and anvil shading) and the final bucket
# saturation — wetted where the squall line rained, dried by evaporation elsewhere. Both load
# back as 2-D `FieldTimeSeries` on the land grid and plot directly.

Tˡᵃ_ts = FieldTimeSeries(land_filename, "Tˡᵃ")
𝒮_ts   = FieldTimeSeries(land_filename, "𝒮")

fig_land = Figure(size = (1100, 420))

ax_T = Axis(fig_land[1, 1]; xlabel = "longitude (°)", ylabel = "latitude (°)",
            title = "ΔTˡᵃ over $(prettytime(duration)) (K)")
ax_𝒮 = Axis(fig_land[1, 3]; xlabel = "longitude (°)", title = "final surface saturation 𝒮")

hm_T = heatmap!(ax_T, Tˡᵃ_ts[end] - Tˡᵃ_ts[1]; colormap = :balance, colorrange = (-8, 8))
hm_𝒮 = heatmap!(ax_𝒮, 𝒮_ts[end]; colormap = :dense, colorrange = (0, 1))

Colorbar(fig_land[1, 2], hm_T)
Colorbar(fig_land[1, 4], hm_𝒮)

save("era5_breeze_land_response.png", fig_land)

fig_land
