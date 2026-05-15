# Offline Bathymetry Conditioning

NumericalEarth provides deterministic, array-only utilities for offline bathymetry
conditioning around unstable estuaries, straits, shallow cells, and steep local
bathymetric jumps.

This workflow is intended for reproducible preprocessing and repair. It should
not be used as hidden runtime model behavior, and none of the conditioning
functions run a model internally. Inspect bathymetry differences and
transport-sensitive straits before production use.

The core workflow is:

```julia
using NumericalEarth

roughness = compute_bathymetry_roughness(h, wetmask)

unstable = flag_unstable_columns(cfl = saved_cfl,
                                 h = h,
                                 wetmask = wetmask,
                                 cfl_threshold = 0.8,
                                 min_depth = 10,
                                 roughness = roughness,
                                 roughness_threshold = 0.2)

repair_mask = dilate_mask(unstable; radius = 2, wetmask)

patched_h, patch_log = smooth_flagged_bathymetry(h, repair_mask;
                                                 wetmask,
                                                 mode = :deepen_only,
                                                 roughness_limit = 0.2)

summary = summarize_bathymetry_patch(h, patched_h; wetmask)
```

See `examples/condition_global_bathymetry.jl` for a minimal JLD2-based script
that reads saved diagnostics, writes a patched bathymetry file, and stores the
patch log and summary.

Saved checkpoints can also be ingested directly. For example,

```julia
result = condition_bathymetry("../outputs/RYF_sxtdeg_checkpoint";
                              run = "all",
                              min_depth = 20,
                              velocity_threshold = 10,
                              dilation_radius = 2,
                              smoothing_iterations = 3,
                              max_change = 200,
                              roughness_limit = 0.25)
```

This discovers all matching `*iteration*.jld2` checkpoint files, unions the
unstable-column masks from every checkpoint, conditions one bathymetry array,
and writes one `*_conditioned_bathymetry.jld2` file. If a checkpoint does not
store the grid, sibling output files containing `serialized/grid` are searched
for the original bathymetry and the grid's `Hx` and `Hy` are used to skip halos
while reading checkpoint fields. A bathymetry array or JLD2 file can also be
supplied explicitly with `bathymetry = ...`; if so, pass `field_halo = (Hx, Hy)`
when the checkpoint arrays include halos.
