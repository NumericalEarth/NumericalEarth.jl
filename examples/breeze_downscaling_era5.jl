# # ERA5 → 12 km convection-permitting hindcast (Breeze + NestedModel)
#
# A limited-area model (LAM) example that downscales ERA5 reanalysis to a ~12 km Breeze compressible
# atmosphere over the U.S. Southern Great Plains, for the Midlatitude Continental Convective Clouds
# Experiment (MC3E) 20 May 2011 squall-line case ([Fan2017](@citet)).
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
#   mixed-phase microphysics, Coriolis, bulk surface drag, and Rayleigh damping.
# - Writes and animates horizontal slices.
#
# ## What it does NOT do (yet)
# - Single nest only (ERA5 → 12 km; coarsened 4× from Fan's 3 km Domain 3 for a fast configuration).
# - No land/ocean coupling (surface stress is a bulk-drag stand-in) and no boundary-layer or cumulus
#   parameterization: diffusion is numerical, and deep convection is resolved on the grid.

using NumericalEarth
using Oceananigans
using Oceananigans.Units          # `minutes` for the output schedule (not re-exported by Oceananigans)
using Breeze
using CopernicusClimateDataStore # activates NumericalEarthCopernicusClimateDataStoreExt (ERA5 downloads)
using CloudMicrophysics          # nested_atmosphere_model's default microphysics → 1-moment mixed-phase
using CairoMakie                 # loads Makie → NumericalEarthMakieExt (`visualize_nested_domain`)
using NaturalEarth               # + GeoInterface → NumericalEarthNaturalEarthExt (`natural_earth_lines`)
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
# Domain centered on the DOE Atmospheric Radiation Measurement (ARM) Southern Great Plains (SGP) site
# in Lamont, OK — the 3 km Domain 3 of the WRF telescoping nest in [Fan2017](@citet), coarsened 4×
# and driven directly by ERA5.

## dates
duration = 18hours
start_date = DateTime(2011, 05, 20, 0)
stop_date = start_date + Second(duration)

## location
φ₀, λ₀ = 36.605, -97.485    # center latitude, longitude (deg)
Lλ, Lφ = 16.7, 15.1

## horizontal resolution
Nx, Ny = 150, 136 # grid cells (ERA5 → 12 km)

dates = (start_date, stop_date)
λ_west, λ_east   = λ₀ .+ [-1, 1] .* Lλ / 2
φ_south, φ_north = φ₀ .+ [-1, 1] .* Lφ / 2

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

model = nested_atmosphere_model(grid, dataset;
                                dates,
                                dir = era5_datadir,
                                terrain = ETOPO2022(),
                                terrain_blend_width = relax_width,
                                relaxation_rate = 1/300,
                                relaxation_width = relax_width,
                                momentum_advection = WENO(order = 5))

# The realized parent region (child + padding, snapped to the native 0.25° grid) serves the domain
# map and the ERA5 snapshots below.
parent = model.parent
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

# ## Simulation
#
# A plain `Simulation` steps the `NestedModel`. `bulk_drag` fills the coupling bottom-stress fields
# (pre-wired by `atmosphere_model` for the forthcoming SlabLand coupling) with a bulk neutral log-law
# surface stress — the dominant near-surface momentum sink until a land model is attached. The
# acoustic modes are substepped, so the adaptive outer Δt is bounded by the (slower) advective CFL.

Δt = 10
simulation = Simulation(model; Δt, stop_time=duration)
add_callback!(simulation, bulk_drag(model), IterationInterval(1))
conjure_time_step_wizard!(simulation, IterationInterval(3); cfl=0.5, max_Δt=Δt)

# ## Output
#
# Two `JLD2Writer`s save 2-D horizontal slices of online diagnostics — never the 3-D fields, which
# would overflow the disk. On the terrain-following grid a constant reference level ≈ constant height
# above ground near the surface, so `k_aloft` is the reference level nearest 2 km. Each output is an
# `AbstractOperation` (Breeze's diagnostic accessors) that the writer computes and slices at save
# time via `indices` — so `θᵥ`/`|U|` land as surface fields and `w`/`qᵛ`/`qʳ` as 2-km-AGL fields.

child = model.child
k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 2000)

surface_fields = (θᵥ = VirtualPotentialTemperature(child),
                  U  = sqrt(child.velocities.u^2 + child.velocities.v^2))

aloft_fields   = (w  = child.velocities.w,
                  qᵛ = specific_humidity(child),
                  qʳ = child.microphysical_fields.qʳ)

surface_filename = "era5_breeze_surface.jld2"
aloft_filename   = "era5_breeze_aloft.jld2"
schedule = TimeInterval(20minutes)
simulation.output_writers[:surface] = JLD2Writer(child, surface_fields; schedule,
                                                 filename = surface_filename,
                                                 indices = (:, :, 1),
                                                 overwrite_existing = true)

simulation.output_writers[:aloft]   = JLD2Writer(child, aloft_fields; schedule,
                                                 filename = aloft_filename,
                                                 indices = (:, :, k_aloft),
                                                 overwrite_existing = true)

function progress(sim)
    child = sim.model.child
    u, v, w = child.velocities
    ρ  = child.dynamics.total_density
    qᵛ = specific_humidity(child)
    qʳ = child.microphysical_fields.qʳ
    @info @sprintf("iter=%4d, t=%s, Δt=%s, max|u|=(%7.2f, %7.2f, %6.2f)  ρ ∈ [%.4f, %.4f], qᵛ ∈ [%.4g, %.4g], qʳ ∈ [%.2g, %.2g]",
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
# extent), row 2 the Breeze child. Columns are near-surface θᵥ′ (referenced pointwise to t=0, so the
# terrain/stratification background cancels and the cold pool stands out) and wind speed |U|, then
# w, qᵛ, qʳ at 2 km. The saved slices load straight back as `FieldTimeSeries` and plot directly —
# `heatmap!` reads each 2-D `Field`'s own coordinates and moves it host-side, so there is no manual
# indexing, and the θᵥ′ anomaly `θᵥ[n] - θᵥ[1]` is a lazy field operation.

θᵥ_series = FieldTimeSeries(surface_filename, "θᵥ")
U_series  = FieldTimeSeries(surface_filename, "U")
w_series  = FieldTimeSeries(aloft_filename, "w")
qᵛ_series = FieldTimeSeries(aloft_filename, "qᵛ")
qʳ_series = FieldTimeSeries(aloft_filename, "qʳ")
times = θᵥ_series.times

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
    return (θᵥ = slice(θˡⁱ * (1 + ε * qᵗ),           1),
            U  = slice(sqrt(u^2 + v^2),              1),
            w  = slice(-ω_series[Time(t)] / (ρ * g), k_parent),
            qᵛ = slice(qᵗ,                           k_parent))
end
parent_frames = [parent_slices(t) for t in times]
nothing #hide

# One column each: title, colormap, and the child/parent 2-D fields as functions of the frame `n`.
# `nothing` parent ⇒ blank panel (rain). Fixed, illustrative color ranges (this is a demo, not a
# tuned figure); `w` alone differs ~10× between parent and child, so it carries a range per row.
g_per_kg(field) = 1f3 * field
columns = [(title = "θᵥ′ₛ (K)",       colormap = :balance, child_range = (-6, 6),  parent_range = (-6, 6),
            child = n -> θᵥ_series[n] - θᵥ_series[1], parent = n -> parent_frames[n].θᵥ - parent_frames[1].θᵥ),
           (title = "|U|ₛ (m s⁻¹)",   colormap = :speed,   child_range = (0, 30),  parent_range = (0, 30),
            child = n -> U_series[n],                 parent = n -> parent_frames[n].U),
           (title = "w₂ₖₘ (m s⁻¹)",   colormap = :balance, child_range = (-3, 3),  parent_range = (-0.3, 0.3),
            child = n -> w_series[n],                 parent = n -> parent_frames[n].w),
           (title = "qᵛ₂ₖₘ (g kg⁻¹)", colormap = :dense,   child_range = (0, 15),  parent_range = (0, 15),
            child = n -> g_per_kg(qᵛ_series[n]),      parent = n -> g_per_kg(parent_frames[n].qᵛ)),
           (title = "qʳ₂ₖₘ (g kg⁻¹)", colormap = :dense,   child_range = (0, 2),   parent_range = (0, 2),
            child = n -> g_per_kg(qʳ_series[n]),      parent = n -> nothing)]

boxλ = [λ_west, λ_east, λ_east, λ_west, λ_west]
boxφ = [φ_south, φ_south, φ_north, φ_north, φ_south]

fig_cascade = Figure(size = (1500, 700))
cascade_n = Observable(1)

## CairoMakie locks each heatmap's data to the first frame's concrete array type, but these lazy
## field operations convert to *different* array types across frames (the parent's pressure-level
## slices especially). Materialize every frame to a host matrix on its grid's own λ/φ, so the plotted
## type stays fixed and the axes read geographic coordinates.
host_matrix(field) = interior(Field(field), :, :, 1) |> Array

function panel!(ax, field_of, colormap, colorrange)
    grid = Field(field_of(1)).grid
    λ = Array(λnodes(grid, Center(), Center(), Center()))
    φ = Array(φnodes(grid, Center(), Center(), Center()))
    data = Observable(host_matrix(field_of(1)))
    on(nn -> (data[] = host_matrix(field_of(nn))), cascade_n)
    return heatmap!(ax, λ, φ, data; colormap, colorrange)
end

Label(fig_cascade[0, 1:5],
      @lift(@sprintf("MC3E 20 May 2011 — ERA5 → 12 km Breeze — t = %.1f h", times[$cascade_n] / 3600)),
      fontsize = 20, tellwidth = false)

for (i, column) in enumerate(columns)
    parent_ax = Axis(fig_cascade[1, i]; title = column.title, aspect = DataAspect())
    child_ax  = Axis(fig_cascade[2, i]; aspect = DataAspect())

    hmc = panel!(child_ax, column.child, column.colormap, column.child_range)

    if isnothing(column.parent(1))
        text!(parent_ax, 0.5, 0.5; text = "no rain", space = :relative, align = (:center, :center), color = :gray)
        Colorbar(fig_cascade[3, i], hmc; vertical = false, flipaxis = false, height = 10)
    else
        hmp = panel!(parent_ax, column.parent, column.colormap, column.parent_range)
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
