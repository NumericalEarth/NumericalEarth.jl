#!/bin/bash
#SBATCH --job-name=conus_render
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_veg_run/render_%j.log

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/conus_veg_run
SCRIPT=/shared/home/xklee/numericalearth_land/experiments/conus_slab_canopy/render_conus_figures.jl
PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-slab-canopy/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
# Compiled caches per partition (node CPU type), so images built on other node types
# never land in this node's cache slots; packages still come from the shared depot.
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
mkdir -p "$HOME/.julia_${SLURM_JOB_PARTITION}"
cd "$RUNDIR"

echo "=== node: $(hostname)  start: $(date) ==="
JULIA_NUM_PRECOMPILE_TASKS=8 $JULIA --project="$PROJECT" -e 'using Pkg; Pkg.precompile()' > render_precompile.log 2>&1
echo "PRECOMPILE_EXIT=$?"

TAG=${TAG:-conus12km_veg}
$JULIA --project="$PROJECT" --color=no "$SCRIPT" "$TAG" > "render_${TAG}.log" 2>&1
echo "RENDER_EXIT=$?"
echo "=== done: $(date) ==="
