#!/bin/bash
#SBATCH --job-name=era5cmp
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --chdir=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/era5cmp_%j.log
set -uo pipefail
WT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export NUMERICALEARTH_DATA_DIRECTORY=$PWD
echo "=== $(hostname) $(date) ==="
/shared/home/xklee/.juliaup/bin/julia --project=$WT/docs $WT/experiments/conus_slab_canopy_v2/compare_era5land.jl
echo "EXIT=$?"
echo "=== done $(date) ==="
