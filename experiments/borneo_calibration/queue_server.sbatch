#!/bin/bash
#SBATCH --job-name=borneo_queue
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/shared/home/xklee/numericalearth_land_scratch_plans/borneo_calibration/logs/queue_%j.log

# One warm Julia session per partition, fed by scripts dropped into
# $RUNDIR/queue_<partition suffix>:
#   sbatch --partition=cpu-large queue_server.sbatch
#   sbatch --partition=gpu-dev --gres=gpu:t4:1 queue_server.sbatch

set -uo pipefail

RUNDIR=/shared/home/xklee/numericalearth_land_scratch_plans/borneo_calibration
WORKTREE=/shared/home/xklee/numericalearth_land/.claude/worktrees/borneo-calibration
PROJECT=$WORKTREE/docs
JULIA=/shared/home/xklee/.juliaup/bin/julia

case "$SLURM_JOB_PARTITION" in
    gpu*) QUEUE=$RUNDIR/queue_gpu ;;
    *)    QUEUE=$RUNDIR/queue_cpu ;;
esac

export NUMERICALEARTH_DATA_DIRECTORY="$RUNDIR"
export JULIA_MAX_NUM_PRECOMPILE_FILES=40
export JULIA_DEPOT_PATH="$HOME/.julia_${SLURM_JOB_PARTITION}:$HOME/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
set -a; source "$HOME/.earthdata_env" 2>/dev/null || true; set +a
export DESTINE_ACCESS_TOKEN=$(tr -d '[:space:]' < "$HOME/.destineapirc" 2>/dev/null || true)

cd "$RUNDIR"
rm -f "$QUEUE/STOP"
echo "=== node: $(hostname)  partition: $SLURM_JOB_PARTITION  queue: $QUEUE  start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
$JULIA -t ${JULIA_THREADS:-4} --project="$PROJECT" --color=no "$WORKTREE/experiments/borneo_calibration/queue_server.jl" "$QUEUE" &
JULIA_PID=$!
# Sample the session's memory every minute: heavy XLA compiles have taken whole nodes down.
while kill -0 $JULIA_PID 2>/dev/null; do
    echo "$(date -u +%FT%T) rss_gb=$(awk -v r="$(ps -o rss= -p $JULIA_PID)" 'BEGIN{printf "%.1f", r/1048576}') free_gb=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)" >> "$RUNDIR/logs/mem_${SLURM_JOB_ID}.log"
    sleep 60
done
wait $JULIA_PID
echo "QUEUE_EXIT=$?"
echo "=== done: $(date) ==="
