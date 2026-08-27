#!/bin/bash
#SBATCH --job-name=conus_v2_compare
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/compare_%j.log

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run
SCRIPT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/experiments/conus_slab_canopy_v2/compare_landcover.jl
PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
cd "$RUNDIR"

echo "=== node: $(hostname)  start: $(date) ==="
$JULIA --project="$PROJECT" --color=no "$SCRIPT" "${TAG_A:-conus12km_v2}" "${TAG_B:-conus12km_v2_modis}" > "compare_${TAG_A:-conus12km_v2}_${TAG_B:-conus12km_v2_modis}.txt" 2>&1
echo "COMPARE_EXIT=$?"
echo "=== done: $(date) ==="
