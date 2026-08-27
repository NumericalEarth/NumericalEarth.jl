#!/bin/bash
#SBATCH --job-name=borneo_ingest
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/borneo_calibration/logs/ingest_%j.log

# Build the surface and forcing caches for one refinement:
#   REFINEMENT=1 sbatch ingest.sbatch

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/borneo_calibration
WORKTREE=/shared/home/xklee/numericalearth_land/.claude/worktrees/borneo-calibration
PROJECT=$WORKTREE/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
set -a; source "$HOME/.earthdata_env" 2>/dev/null || true; set +a
export REFINEMENT=${REFINEMENT:-1}

cd "$RUNDIR"
echo "=== node: $(hostname)  refinement: $REFINEMENT  start: $(date) ==="
$JULIA -t 8 --project="$PROJECT" --color=no "$WORKTREE/experiments/borneo_calibration/borneo_setup.jl" > "ingest_r${REFINEMENT}.log" 2>&1
echo "INGEST_EXIT=$?"
echo "=== done: $(date) ==="
