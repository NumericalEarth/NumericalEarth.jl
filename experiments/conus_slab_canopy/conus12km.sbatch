#!/bin/bash
#SBATCH --job-name=conus_veg
#SBATCH --partition=gpu-prod
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/conus_veg_run/conus_veg_%j.log

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/conus_veg_run
SCRIPT=/shared/home/xklee/numericalearth_land/experiments/conus_slab_canopy/conus_slab_canopy.jl
PROJECT=/shared/home/xklee/numericalearth_land/.claude/worktrees/conus-slab-canopy/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

# NumericalEarth caches datasets under this directory (ETOPO, ERA5, CopernicusLandAlbedo,
# MODIS, OpenLandMap, ...). Read at load time, so exported before Julia starts.
export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
# Compiled caches per partition (node CPU type), so images built on other node types
# never land in this node's cache slots; packages still come from the shared depot.
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
mkdir -p "$HOME/.julia_${SLURM_JOB_PARTITION}"

# MODIS / ASTER downloads authenticate through NASA Earthdata.
set -a; source "$HOME/.earthdata_env" 2>/dev/null || true; set +a

cd "$RUNDIR"
echo "=== node: $(hostname)  start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Parallel precompile for this node's CPU target (in-load precompilation is serial).
echo "=== precompile ==="
JULIA_NUM_PRECOMPILE_TASKS=16 $JULIA --project="$PROJECT" -e 'using Pkg; Pkg.precompile()' > precompile_node.log 2>&1
echo "PRECOMPILE_EXIT=$?"

# REFINEMENT=1 -> ~12 km, 2 -> ~6 km (same domain extent, so the ERA5 cache is shared).
export REFINEMENT=${REFINEMENT:-1}

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "=== [1/2] smoke (20 simulated minutes at REFINEMENT=$REFINEMENT) ==="
    SMOKE=1 $JULIA --project="$PROJECT" --color=no "$SCRIPT" > "smoke_r${REFINEMENT}.log" 2>&1
    SMOKE_EXIT=$?
    echo "SMOKE_EXIT=$SMOKE_EXIT"
    if [ "$SMOKE_EXIT" -ne 0 ]; then
        echo "smoke failed, not starting the production run"
        tail -60 "smoke_r${REFINEMENT}.log"
        exit "$SMOKE_EXIT"
    fi
fi

echo "=== [2/2] production (96 h simulated at REFINEMENT=$REFINEMENT) ==="
$JULIA --project="$PROJECT" --color=no "$SCRIPT" > "run_r${REFINEMENT}.log" 2>&1
echo "RUN_EXIT=$?"
echo "=== done: $(date) ==="
