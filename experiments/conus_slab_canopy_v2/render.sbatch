#!/bin/bash
#SBATCH --job-name=conus_v2_render
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/render_%j.log

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run
SCRIPT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/experiments/conus_slab_canopy_v2/render_conus_figures.jl
PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
cd "$RUNDIR"

echo "=== node: $(hostname)  start: $(date) ==="
TAG=${TAG:-conus12km_v2}
PREVIOUS_STATIC=${PREVIOUS_STATIC:-/shared/home/xklee/numericalearth_land_scratch_plans/conus_veg_run/conus12km_vegfix_static.jld2}
$JULIA --project="$PROJECT" --color=no "$SCRIPT" "$TAG" "$PREVIOUS_STATIC" > "render_${TAG}.log" 2>&1
echo "RENDER_EXIT=$?"
echo "=== done: $(date) ==="
