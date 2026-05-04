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

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

cases = [
    (prefix = "orca_corrected_snow_rbvd_bih50days",             label = "ORCA RBVD",           start_time =  0 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_ncar_snow",                                 label = "ORCA New NCAR",       start_time = 35 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_corrected_snow_simple",                     label = "ORCA CADV",           start_time = 15 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_corrected_snow_cb0.12_ksymm500",            label = "ORCA Redi500",        start_time = 40 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_corrected_snow_cb0.15_bih50days",           label = "ORCA LowDiss",        start_time = 20 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_ncar_snow_cb0.15_bih50days",                label = "ORCA NCAR LowDiss",   start_time = 15 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "halfdegree_corrected_snow_cb0.01_kskew0_ksymm0", label = "Half Degree",         start_time = 40 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_corrected_snow_cb0.06_kskew0_ksymm0",       label = "ORCA NOGM",           start_time = 40 * (365 * 24 * 3600), stop_time = Inf),
    (prefix = "orca_corrected_snow_cb0.06_kskew1000_ksymm1000", label = "ORCA GM1000",         start_time = 40 * (365 * 24 * 3600), stop_time = Inf),
]

output_dir = length(ARGS) >= 1 ? ARGS[1] : "figures"

# ══════════════════════════════════════════════════════════════
# Infrastructure
# ══════════════════════════════════════════════════════════════

const HERE = @__DIR__
include(joinpath(HERE, "visualize", "common.jl"))
include(joinpath(HERE, "visualize", "cache.jl"))

# ══════════════════════════════════════════════════════════════
# Figure registry: (number, file basename, function symbol)
# ══════════════════════════════════════════════════════════════

const FIG_REGISTRY = [
    (n =  1, file = "fig01_sst_bias.jl",                  fn = :fig01),
    (n =  2, file = "fig02_sss_bias.jl",                  fn = :fig02),
    (n =  3, file = "fig03_ssh.jl",                       fn = :fig03),
    (n =  4, file = "fig04_mld.jl",                       fn = :fig04),
    (n =  5, file = "fig05_seaice_conc.jl",               fn = :fig05),
    (n =  6, file = "fig06_surface_fluxes.jl",            fn = :fig06),
    (n =  7, file = "fig07_wind_stress.jl",               fn = :fig07),
    (n =  8, file = "fig08_ssh_variance.jl",              fn = :fig08),
    (n =  9, file = "fig09_sie.jl",                       fn = :fig09),
    (n = 10, file = "fig10_sia.jl",                       fn = :fig10),
    (n = 11, file = "fig11_arctic_volume.jl",             fn = :fig11),
    (n = 12, file = "fig12_sia_timeseries.jl",            fn = :fig12),
    (n = 13, file = "fig13_arctic_volume_timeseries.jl",  fn = :fig13),
    (n = 14, file = "fig14_ke.jl",                        fn = :fig14),
    (n = 15, file = "fig15_drift.jl",                     fn = :fig15),
    (n = 16, file = "fig16_profiles.jl",                  fn = :fig16),
    (n = 17, file = "fig17_zonal_mean.jl",                fn = :fig17),
    (n = 18, file = "fig18_zonal_drift.jl",               fn = :fig18),
    (n = 19, file = "fig19_mld_zonal_mean.jl",            fn = :fig19),
    (n = 20, file = "fig20_TS_drift_heatmap.jl",          fn = :fig20),
    (n = 21, file = "fig21_strait_transports.jl",         fn = :fig21),
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

labels = [c.label for c in cases]
caches = Dict(c.label => CaseCache(c) for c in cases)

const FIGURES_DIR = joinpath(HERE, "visualize", "figures")

for entry in FIG_REGISTRY
    include(joinpath(FIGURES_DIR, entry.file))
end

# Convenience: render a single figure (or a list) by number from the REPL.
#     julia> render_figures(4)
#     julia> render_figures([1, 4, 7])
function render_figures(selection)
    nums = collect(selection isa Number ? (selection,) : selection)
    for entry in FIG_REGISTRY
        entry.n in nums || continue
        @info "Figure $(entry.n): $(entry.file)"
        getfield(@__MODULE__, entry.fn)(caches, labels, cases)
    end
end

# ══════════════════════════════════════════════════════════════
# Auto-render only when invoked as a script (not from the REPL).
# ══════════════════════════════════════════════════════════════

if !isinteractive()
    selection = parse_fig_selection(get(ENV, "FIG", ""), [r.n for r in FIG_REGISTRY])
    @info "Rendering figures: $selection"
    render_figures(selection)
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
