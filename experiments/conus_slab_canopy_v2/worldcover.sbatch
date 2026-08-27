#!/bin/bash
#SBATCH --job-name=conus_v2_worldcover
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=36
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/worldcover_%A_%a.log

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run
SCRIPT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/experiments/conus_slab_canopy_v2/ingest_worldcover.jl
PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-v2/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0

cd "$RUNDIR"
echo "=== node: $(hostname)  array task: $SLURM_ARRAY_TASK_ID  start: $(date) ==="

PER_NODE=${PER_NODE:-8}
NWORKERS=$(( SLURM_ARRAY_TASK_COUNT * PER_NODE ))
for i in $(seq 0 $(( PER_NODE - 1 ))); do
    WORKER=$(( SLURM_ARRAY_TASK_ID * PER_NODE + i ))
    WORKER=$WORKER NWORKERS=$NWORKERS $JULIA --project="$PROJECT" --color=no "$SCRIPT" \
        > "worldcover_worker_${WORKER}.log" 2>&1 &
done
wait
echo "=== done: $(date) ==="
