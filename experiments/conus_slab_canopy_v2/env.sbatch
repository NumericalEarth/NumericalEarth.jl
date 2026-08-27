#!/bin/bash
#SBATCH --job-name=conus_v2_env
#SBATCH --partition=gpu-prod
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/env_%j.log

set -uo pipefail

PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
mkdir -p "$HOME/.julia_${SLURM_JOB_PARTITION}"

echo "=== node: $(hostname)  start: $(date) ==="
JULIA_NUM_PRECOMPILE_TASKS=16 $JULIA --project="$PROJECT" -e '
using Pkg
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
using NumericalEarth, Breeze, ArchGDAL, CUDA, CairoMakie, NaturalEarth, JLD2, CopernicusClimateDataStore, CloudMicrophysics
for n in ("NumericalEarthArchGDALExt", "NumericalEarthBreezeExt", "NumericalEarthMakieExt")
    @assert !isnothing(Base.get_extension(NumericalEarth, Symbol(n))) n
end
println("ENV_OK")'
echo "ENV_EXIT=$?"
echo "=== done: $(date) ==="
