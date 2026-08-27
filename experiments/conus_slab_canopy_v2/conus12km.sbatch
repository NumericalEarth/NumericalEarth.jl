#!/bin/bash
#SBATCH --job-name=conus_v2
#SBATCH --partition=gpu-prod
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/conus_v2_%j.log

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run
SCRIPT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/experiments/conus_slab_canopy_v2/conus_slab_canopy_v2.jl
PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
set -a; source "$HOME/.earthdata_env" 2>/dev/null || true; set +a

cd "$RUNDIR"
echo "=== node: $(hostname)  start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

export REFINEMENT=${REFINEMENT:-1}
export LANDCOVER=${LANDCOVER:-modis}
export TAG=${TAG:-conus12km_v2}

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "=== [1/2] smoke (20 simulated minutes) ==="
    SMOKE=1 TAG="smoke_${TAG}" $JULIA -t 8 --project="$PROJECT" --color=no "$SCRIPT" > "smoke_${TAG}.log" 2>&1
    SMOKE_EXIT=$?
    echo "SMOKE_EXIT=$SMOKE_EXIT"
    if [ "$SMOKE_EXIT" -ne 0 ]; then
        tail -60 "smoke_${TAG}.log"
        exit "$SMOKE_EXIT"
    fi
fi

echo "=== [2/2] production (96 h simulated) ==="
$JULIA -t 8 --project="$PROJECT" --color=no "$SCRIPT" > "run_${TAG}.log" 2>&1
echo "RUN_EXIT=$?"
echo "=== done: $(date) ==="
