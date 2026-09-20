#!/usr/bin/env julia
# visualize_omip.jl — OMIP diagnostic figures: set up + render.
#
# This file is designed for two workflows.
#
# REPL (preferred for iteration)
# ------------------------------
#     julia> include("visualize_omip.jl")
#     # nothing renders; `caches`, `labels`, `cases` and all fig01..fig21
#     # are now defined.
#     julia> fig04(caches, labels, cases)        # render just fig 4
#     julia> fig17(caches, labels, cases)        # later, render fig 17
#     julia> fig04(caches, labels, cases)        # rerun: cache hits, fast
#
# Script (batch)
# --------------
#     julia visualize_omip.jl                    # render all 21 figs
#     FIG=4 julia visualize_omip.jl              # render only fig 4
#     FIG=1,4,7 julia visualize_omip.jl          # comma list
#     FIG=14-19 julia visualize_omip.jl          # range
#     FIG=1,3,9-12,21 julia visualize_omip.jl    # mix
#     THEME=dark julia visualize_omip.jl         # transparent bg, white axes
#     julia visualize_omip.jl my_output_dir      # custom output dir
#
# How sharing works
# -----------------
# Each figure declares its data needs implicitly by calling
# `get_field(cache, :sym)`. Loaders form a DAG (e.g. `:sst_bias` ←
# `:sst` ← `:tos_fts`, plus `:woa_temperature`). Per orchestrator
# session, every loader fires at most once per case — so running
# `fig01` then `fig02` reads `tos` and `sos` once each but never
# reloads the WOA file.
#
# Running `fig01(caches, labels, cases)` alone touches only
# `:tos_fts` + `:woa_temperature` + the masks/grid; nothing else is
# loaded. That's the
# "minimum-time, isolation" property the refactor was designed for.
#
# Edit the `cases` list below before the first include.
#
# Each case specifies its averaging window in one of two mutually
# exclusive ways:
#   - absolute: `start_time` and/or `stop_time` in seconds (omit either
#     to default to 0 / Inf).
#   - relative: `years_from_end = N` → average the last N years of the
#     run, computed from the latest snapshot in the surface JLD2 file.

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

# `store.sh` stages every run to $DATA and leaves only the newest part next to this script, so the
# staged copy is the one to read when it exists.
#
# The seven runs that hold a Slurm allocation on 2026-09-08 — the control plus six levers. Three were
# executing when this list was written (`orca_icez01e-4`, `orca_cunb1.0`, `orca_noicedyn_trcwenoz`) and
# four were queued behind them (`orca`, `orca_triad`, `orca_trcwenoz`, `orca_labrest100`); queued and
# running read the same on disk, since a pending job's record is whatever its last segment left staged.
#
# `orca_cunb1.0` was submitted minutes before this edit and has written no surface part yet, so
# `has_surface_record` drops it with a warning until its first part lands. Leave it in the list.
#
# Record lengths, from the contiguous surface-part index (each part is 360 days, so parts ≈ years):
# `icez0 1e-4` 145, `orca` 98, `tracer cwenoz` 53, `noicedyn cwenoz` 42, `triad` 26, `labrest100` 26.
#
# `years_from_end = 5` reads each record's own last five years, so the figures show where each run
# stands now. The records are very unequal — 143 yr for `icez0 1e-4` against 25 for `triad` — so a
# panel compares the late drift of one run against the spin-up of another; read level and shape per
# run, not run-to-run differences, wherever the record lengths differ.
#
# ⚠ `cwenoz+triad+GRID` blew up at 35.11 yr and its output carries the blow-up: iteration 205200 is
# NaN in both `fields` and `surface`. `years_from_end` anchors on the newest snapshot, so its last
# five years average NaN into every panel it appears in. `kskew600+triad+GRID` blew up at 16.03 yr but
# stopped writing one output interval earlier, so its file is finite throughout.
#
# The two May-2026 archive runs (`orca_corrected_snow_bih50days` and its κ1000 twin) are NOT here:
# their serialized grids carry `Tripolar{…, RightCenterFolded}`, which JLD2 reconstructs as an opaque
# object rather than a grid, so every loader that calls an Oceananigans grid function on them fails.
# Read those runs with the `c8_lib.jl` array readers instead.
const DATA_DIR = "/orcd/nese/raffaele/001/ssilvest/OMIP-data"

const YEARS_FROM_END = 5

cases = [
(prefix = "orca",                   label = "orca (control)",  years_from_end=YEARS_FROM_END),  # queued
(prefix = "orca_icez01e-4",         label = "icez0 1e-4",      years_from_end=YEARS_FROM_END),  # running
(prefix = "orca_cunb2.0",           label = "cunb 2.0",        years_from_end=YEARS_FROM_END),  # running, no output yet
(prefix = "orca_noicedyn_trcwenoz", label = "noicedyn cwenoz", years_from_end=YEARS_FROM_END),  # running
(prefix = "orca_trcwenoz",          label = "tracer cwenoz",   years_from_end=YEARS_FROM_END),  # queued
(prefix = "orca_triad",             label = "triad",           years_from_end=YEARS_FROM_END),  # queued
(prefix = "orca_labrest100",        label = "labrest100",      years_from_end=YEARS_FROM_END),  # queued
(prefix = "orca_gmvbp",             label = "gmvbp",           years_from_end=YEARS_FROM_END),  # queued
(prefix = "orca_triad_cf0.25",      label = "triad_cf0.25",    years_from_end=YEARS_FROM_END),  # queued
# The NZ=100 MAXDZ=100 family, 2026-09-11 session.
(prefix = "orca_triad_nz100_maxdz100",           label = "triad+GRID",       years_from_end=YEARS_FROM_END),
(prefix = "orca_kskew350_nz100_maxdz100",        label = "kskew350+GRID",    years_from_end=YEARS_FROM_END),
(prefix = "orca_icez01e-4_triad_nz100_maxdz100", label = "icez0+GRID",       years_from_end=YEARS_FROM_END),
(prefix = "orca_triad_kskew600_nz100_maxdz100",  label = "kskew600+GRID",    years_from_end=YEARS_FROM_END),
# ⚠ its last five years are NaN — see the note above; give it an absolute window instead of
# years_from_end, or every panel it appears in averages NaN.
(prefix = "orca_trcwenoz_triad_nz100_maxdz100",  label = "cwenoz+triad+GRID",
     start_time = 29.4 * 31536000, stop_time = 34.5 * 31536000),
]

output_dir = length(ARGS) >= 1 ? ARGS[1] : "figures"

# ══════════════════════════════════════════════════════════════
# Infrastructure
# ══════════════════════════════════════════════════════════════

const HERE = @__DIR__
include(joinpath(HERE, "visualize", "common.jl"))
include(joinpath(HERE, "visualize", "cache.jl"))

# `common.jl` resolves run directories against the working directory; redefine after the include so
# the staged copies are found, with this script's own directory — not the caller's cwd — as the
# fallback for runs `store.sh` has not drained yet.
#
# `store.sh` creates the staged directory as soon as it archives a young run's checkpoints, before any
# output part has moved, so the test is whether the staged copy holds output — not whether it exists.
holds_output(dir) = isdir(dir) &&
    any(f -> endswith(f, ".jld2") && !contains(f, "checkpoint"), readdir(dir))

run_dir_for(prefix) = let staged = joinpath(DATA_DIR, "$(prefix)_run")
    holds_output(staged) ? staged : joinpath(HERE, "$(prefix)_run")
end

# ══════════════════════════════════════════════════════════════
# Figure registry: (number, file basename, function symbol)
# ══════════════════════════════════════════════════════════════

const FIG_REGISTRY = [
    (n =  1, file = "fig01_sst_bias.jl",                  fn = :fig01),
    (n =  2, file = "fig02_sss_bias.jl",                  fn = :fig02),
    (n =  3, file = "fig03_ssh.jl",                       fn = :fig03),
    (n =  4, file = "fig04_mld.jl",                       fn = :fig04),
    (n =  5, file = "fig05_seaice_conc.jl",               fn = :fig05),
    (n =  6, file = "fig06_seaice_conc_bias.jl",          fn = :fig06),
    (n =  7, file = "fig07_surface_fluxes.jl",            fn = :fig07),
    (n =  8, file = "fig08_wind_stress.jl",               fn = :fig08),
    (n =  9, file = "fig09_ssh_variance.jl",              fn = :fig09),
    (n = 10, file = "fig10_sie.jl",                       fn = :fig10),
    (n = 11, file = "fig11_sia.jl",                       fn = :fig11),
    (n = 12, file = "fig12_arctic_volume.jl",             fn = :fig12),
    (n = 13, file = "fig13_sia_timeseries.jl",            fn = :fig13),
    (n = 14, file = "fig14_arctic_volume_timeseries.jl",  fn = :fig14),
    (n = 15, file = "fig15_ke.jl",                        fn = :fig15),
    (n = 16, file = "fig16_drift.jl",                     fn = :fig16),
    (n = 17, file = "fig17_profiles.jl",                  fn = :fig17),
    (n = 18, file = "fig18_zonal_mean.jl",                fn = :fig18),
    (n = 19, file = "fig19_zonal_drift.jl",               fn = :fig19),
    (n = 20, file = "fig20_mld_zonal_mean.jl",            fn = :fig20),
    (n = 21, file = "fig21_TS_drift_heatmap.jl",          fn = :fig21),
    (n = 22, file = "fig22_strait_transports.jl",         fn = :fig22),
    (n = 23, file = "fig23_amoc.jl",                      fn = :fig23),
    (n = 24, file = "fig24_near_surface_currents.jl",     fn = :fig24),
    (n = 25, file = "fig25_equatorial_undercurrent.jl",   fn = :fig25),
    (n = 26, file = "fig26_amoc_rapid.jl",                fn = :fig26),
    (n = 27, file = "fig27_ssh_drift.jl",                 fn = :fig27),
    (n = 28, file = "fig28_content_conservation.jl",      fn = :fig28),
    (n = 29, file = "fig29_barotropic_streamfunction.jl", fn = :fig29),
    (n = 30, file = "fig30_arctic_freshwater.jl",         fn = :fig30),
    (n = 31, file = "fig31_zonal_rms_drift.jl",           fn = :fig31),
]

# ══════════════════════════════════════════════════════════════
# Selection: parse FIG env var into a sorted, unique Vector{Int}
# Accepts "all" / unset → all figs; "4" → [4]; "1,4,7" → [1,4,7];
# "14-19" → [14..19]; mixed: "1,3,9-12,21".
# ══════════════════════════════════════════════════════════════

function parse_fig_selection(spec::AbstractString, all_ns::Vector{Int})
    s = strip(spec)
    (isempty(s) || lowercase(s) == "all") && return sort(unique(all_ns))
    out = Int[]
    for token in split(s, ',')
        t = strip(token)
        isempty(t) && continue
        if occursin('-', t)
            parts = split(t, '-')
            length(parts) == 2 || error("Invalid FIG range: '$t'")
            lo = parse(Int, strip(parts[1]))
            hi = parse(Int, strip(parts[2]))
            append!(out, lo:hi)
        else
            push!(out, parse(Int, t))
        end
    end
    return sort(unique(out))
end

# ══════════════════════════════════════════════════════════════
# Build per-case caches (cheap — no data loaded yet)
# Pre-include every fig file so `figNN` symbols are always defined.
# ══════════════════════════════════════════════════════════════

# A freshly launched run has no surface file until it writes its first part, and `years_from_end`
# reads that file to place the window, so an unwritten run would raise here and take every other
# case with it. Drop such cases and say which ones.
function has_surface_record(case)
    dir = run_dir_for(case.prefix)
    isdir(dir) || return false
    return any(f -> startswith(f, "$(case.prefix)_surface") && endswith(f, ".jld2"), readdir(dir))
end

let pending = filter(c -> !has_surface_record(c), cases)
    isempty(pending) || @warn "Skipping cases with no surface output yet: " *
                              join(("$(c.label) ($(c.prefix))" for c in pending), ", ")
    global cases = filter(has_surface_record, cases)
end

labels = [c.label for c in cases]
caches = Dict(c.label => CaseCache(c) for c in cases)

const FIGURES_DIR = joinpath(HERE, "visualize", "figures")

for entry in FIG_REGISTRY
    include(joinpath(FIGURES_DIR, entry.file))
end

# Convenience: render a single figure (or a list) by number from the REPL.
#     julia> render_figures(4)
#     julia> render_figures([1, 4, 7])
#     julia> render_figures([1, 4, 7]; theme = :dark)   # transparent bg, white axes
render_figures(n::Integer; kw...)                          = render_figures((n,); kw...)
render_figures(ns::AbstractVector{<:Integer}; kw...)       = render_figures(Tuple(ns); kw...)
render_figures(ns::AbstractRange{<:Integer}; kw...)        = render_figures(Tuple(ns); kw...)
function render_figures(ns::Tuple{Vararg{Integer}}; theme::Symbol = :light)
    with_theme(presentation_theme(theme)) do
        for entry in FIG_REGISTRY
            entry.n in ns || continue
            @info "Figure $(entry.n): $(entry.file)"
            getfield(@__MODULE__, entry.fn)(caches, labels, cases)
        end
    end
end

# ══════════════════════════════════════════════════════════════
# Auto-render only when invoked as a script (not from the REPL).
# ══════════════════════════════════════════════════════════════

if !isinteractive()
    selection = parse_fig_selection(get(ENV, "FIG", ""), [r.n for r in FIG_REGISTRY])
    theme = Symbol(lowercase(get(ENV, "THEME", "light")))
    @info "Rendering figures: $selection (theme = :$theme)"
    render_figures(selection; theme)
    @info "All requested figures saved to $output_dir"
else
    @info """
    visualize_omip.jl loaded.
      cases  = $(length(cases)) cases
      caches = pre-built (no data loaded yet)
    Render a figure with e.g.:  fig04(caches, labels, cases)
    Or render several with:     render_figures([1, 4, 17])
    """
end
