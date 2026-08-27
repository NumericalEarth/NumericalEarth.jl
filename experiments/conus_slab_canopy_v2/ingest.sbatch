#!/bin/bash
#SBATCH --job-name=conus_v2_ingest
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=36
#SBATCH --time=12:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_v2_run/ingest_%j.log

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
export REFINEMENT=${REFINEMENT:-1}
export LANDCOVER=${LANDCOVER:-modis}
INGEST_ONLY=1 $JULIA -t 16 --project="$PROJECT" --color=no "$SCRIPT" > "ingest_r${REFINEMENT}_${LANDCOVER}.log" 2>&1
echo "INGEST_EXIT=$?"
echo "=== done: $(date) ==="
