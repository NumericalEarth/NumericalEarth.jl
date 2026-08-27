#!/bin/bash
#SBATCH --job-name=borneo_env
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/borneo_calibration/logs/env_%j.log

# Resolve, instantiate and precompile the worktree docs environment for one partition:
#   sbatch --partition=cpu-large env.sbatch
#   sbatch --partition=gpu-dev --gres=gpu:t4:1 env.sbatch

set -uo pipefail

PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/borneo-calibration/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
mkdir -p "$HOME/.julia_${SLURM_JOB_PARTITION}"

echo "=== node: $(hostname)  partition: $SLURM_JOB_PARTITION  start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS:-4} $JULIA --project="$PROJECT" -e '
using Pkg
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
using NumericalEarth, Oceananigans, Reactant, Enzyme, CUDA, CDSAPI, CopernicusClimateDataStore, ArchGDAL, CairoMakie, JLD2
for n in ("NumericalEarthReactantExt", "NumericalEarthArchGDALExt", "NumericalEarthMakieExt", "NumericalEarthCDSAPIExt")
    @assert !isnothing(Base.get_extension(NumericalEarth, Symbol(n))) n
end
println("ENV_OK")'
echo "ENV_EXIT=$?"
echo "=== done: $(date) ==="
