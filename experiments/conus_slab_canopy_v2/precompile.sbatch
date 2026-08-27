#!/bin/bash
#SBATCH --job-name=conus_v2_precompile
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=36
#SBATCH --time=03:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/precompile_%j.log

set -uo pipefail

PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
mkdir -p "$HOME/.julia_${SLURM_JOB_PARTITION}"

echo "=== node: $(hostname)  start: $(date) ==="
JULIA_NUM_PRECOMPILE_TASKS=16 $JULIA --project="$PROJECT" -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
using NumericalEarth, ArchGDAL, CairoMakie, JLD2
println("PRECOMPILE_OK")'
echo "PRECOMPILE_EXIT=$?"
echo "=== done: $(date) ==="
