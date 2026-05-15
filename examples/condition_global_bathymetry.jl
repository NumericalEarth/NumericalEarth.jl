# # Offline global bathymetry conditioning
#
# This example demonstrates a reproducible preprocessing workflow for patching
# bathymetry around unstable estuaries, straits, shallow cells, or steep local
# jumps. It reads saved diagnostics and bathymetry, writes a corrected
# bathymetry file plus a patch log, and never runs a model internally.
#
# Users should inspect bathymetry differences and transport-sensitive straits
# before using a conditioned bathymetry in production simulations.

using JLD2
using NumericalEarth

if length(ARGS) < 3
    println("""
    Usage:
      julia --project=. examples/condition_global_bathymetry.jl BATHYMETRY.jld2 DIAGNOSTICS.jld2 PATCHED_BATHYMETRY.jld2

    Expected JLD2 keys:
      BATHYMETRY.jld2:
        h             positive bathymetric depth array
        wetmask       optional Boolean wet mask

      DIAGNOSTICS.jld2:
        cfl           optional CFL diagnostic
        u, v          optional horizontal velocity diagnostics
        nan_mask      optional Boolean mask of columns with NaNs in saved output

    Optional environment variables:
      CFL_THRESHOLD=0.8
      VELOCITY_THRESHOLD=
      MIN_DEPTH=
      ROUGHNESS_THRESHOLD=
      DILATION_RADIUS=2
      SMOOTHING_ITERATIONS=1
      MAX_CHANGE=
      MODE=deepen_only
      ROUGHNESS_LIMIT=
    """)
    exit()
end

bathymetry_path, diagnostics_path, output_path = ARGS[1:3]

_parse_optional_float(name) = isempty(get(ENV, name, "")) ? nothing : parse(Float64, ENV[name])

cfl_threshold = parse(Float64, get(ENV, "CFL_THRESHOLD", "0.8"))
velocity_threshold = _parse_optional_float("VELOCITY_THRESHOLD")
min_depth = _parse_optional_float("MIN_DEPTH")
roughness_threshold = _parse_optional_float("ROUGHNESS_THRESHOLD")
dilation_radius = parse(Int, get(ENV, "DILATION_RADIUS", "2"))
iterations = parse(Int, get(ENV, "SMOOTHING_ITERATIONS", "1"))
max_change = _parse_optional_float("MAX_CHANGE")
mode = Symbol(get(ENV, "MODE", "deepen_only"))
roughness_limit = _parse_optional_float("ROUGHNESS_LIMIT")

@info "Loading bathymetry" bathymetry_path
h, wetmask = jldopen(bathymetry_path, "r") do file
    h = file["h"]
    wetmask = haskey(file, "wetmask") ? file["wetmask"] : h .> 0
    h, wetmask
end

@info "Loading saved diagnostics" diagnostics_path
diagnostics = jldopen(diagnostics_path, "r") do file
    (; cfl = haskey(file, "cfl") ? file["cfl"] : nothing,
       u = haskey(file, "u") ? file["u"] : nothing,
       v = haskey(file, "v") ? file["v"] : nothing,
       nan_mask = haskey(file, "nan_mask") ? file["nan_mask"] : nothing)
end

@info "Computing bathymetry roughness"
roughness = compute_bathymetry_roughness(h, wetmask)

@info "Flagging unstable columns"
unstable_mask = flag_unstable_columns(cfl = diagnostics.cfl,
                                      u = diagnostics.u,
                                      v = diagnostics.v,
                                      h = h,
                                      wetmask = wetmask,
                                      cfl_threshold = cfl_threshold,
                                      velocity_threshold = velocity_threshold,
                                      min_depth = min_depth,
                                      roughness = roughness,
                                      roughness_threshold = roughness_threshold)

if !isnothing(diagnostics.nan_mask)
    unstable_mask .|= Bool.(diagnostics.nan_mask) .& Bool.(wetmask)
end

@info "Dilating unstable mask" dilation_radius
repair_mask = dilate_mask(unstable_mask; radius = dilation_radius, wetmask)

@info "Conditioning bathymetry" iterations mode max_change roughness_limit
patched_h, patch_log = smooth_flagged_bathymetry(h, repair_mask;
                                                 wetmask,
                                                 iterations,
                                                 max_change,
                                                 mode,
                                                 roughness_limit)

summary = summarize_bathymetry_patch(h, patched_h; wetmask)

@info "Writing patched bathymetry" output_path changed_cells=summary.changed_cells
jldopen(output_path, "w") do file
    file["h"] = patched_h
    file["wetmask"] = wetmask
    file["repair_mask"] = repair_mask
    file["roughness"] = roughness
    file["patch_records"] = patch_log.records
    file["patch_summary"] = summary
end

@info "Finished offline bathymetry conditioning" summary
