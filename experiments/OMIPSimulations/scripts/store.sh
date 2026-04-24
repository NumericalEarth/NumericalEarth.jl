#!/bin/bash
# Move completed OMIP outputs from a live run folder to
# $DATA/OMIP-data/<RUN_NAME>_run while a launch.sh job is still running.
#
# Logic:
#   - Part files (*_part<N>.jld2): the highest N per filename group is
#     still being written by the running sim, so it is left in place;
#     all older parts are moved.
#   - Checkpoint files (*_checkpoint_iteration<N>.jld2): the highest
#     iteration is kept locally so `run!(sim; pickup=true)` still works;
#     older checkpoints are moved.
#   - Anything else in the run folder is left untouched.
#
# Must be run from the same directory as launch.sh (i.e. this scripts
# folder) so that <RUN_NAME>_run resolves the same way it does for the
# running simulation.
#
# Usage:
#   ./store.sh orca
#   ./store.sh orca_ncar
#   ./store.sh orca_corrected_snow_cb0.1
#
# The argument is the RUN_NAME (same as the job name from launch.sh).
# DATA must be set in the calling shell (it is propagated to the
# sbatch job via --export=ALL).

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./store.sh <run_name> [extra sbatch args...]

The <run_name> matches the RUN_NAME built by launch.sh, e.g.:
  orca, orca_ncar, orca_corrected_snow, orca_ncar_cb0.1, halfdegree, ...

Examples:
  ./store.sh orca
  ./store.sh orca_ncar
  ./store.sh orca_corrected_snow_cb0.1
USAGE
}

RUN_NAME="${1:-}"
if [[ -z "$RUN_NAME" ]]; then
    usage
    exit 1
fi
shift || true

case "$RUN_NAME" in
    -h|--help)
        usage
        exit 0
        ;;
esac

if [[ -z "${DATA:-}" ]]; then
    echo "Error: DATA environment variable is not set" >&2
    exit 1
fi

RUN_DIR="${RUN_NAME}_run"
DEST_DIR="${DATA}/OMIP-data/${RUN_DIR}"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: run directory '$RUN_DIR' not found in $(pwd)" >&2
    echo "       (store.sh must be run from the same directory as launch.sh)" >&2
    exit 1
fi

JOB_NAME="${JOB_NAME:-store_${RUN_NAME}}"

SBATCH_ARGS=()
SBATCH_ARGS+=(-o "store_${RUN_NAME}.out")
SBATCH_ARGS+=(-e "store_${RUN_NAME}.err")
SBATCH_ARGS+=(-J "$JOB_NAME")
SBATCH_ARGS+=(--export="ALL,RUN_NAME=${RUN_NAME},RUN_DIR=${RUN_DIR},DEST_DIR=${DEST_DIR}")

sbatch "${SBATCH_ARGS[@]}" "$@" <<'EOF'
#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p sched_mit_raffaele
#SBATCH --time=24:00:00
#SBATCH --mem=4GB

set -euo pipefail

echo "Storing ${RUN_NAME} outputs"
echo "  source: $(pwd)/${RUN_DIR}"
echo "  dest:   ${DEST_DIR}"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: run directory '$RUN_DIR' does not exist in $(pwd)" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

shopt -s nullglob

# Infinite loop
while true
do

# ------------------------------------------------------------------
# Part files: *_part<N>.jld2
# The highest N per filename group is still being written, so it is
# left in place; everything older is moved.
# ------------------------------------------------------------------
declare -A max_part
for f in "$RUN_DIR"/*_part[0-9]*.jld2; do
    base=$(basename "$f")
    tail="${base##*_part}"
    n="${tail%.jld2}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    group="${base%_part${n}.jld2}"
    current="${max_part[$group]:-0}"
    if (( n > current )); then
        max_part[$group]=$n
    fi
done

moved_parts=0
kept_parts=0

for f in "$RUN_DIR"/*_part[0-9]*.jld2; do
    base=$(basename "$f")
    tail="${base##*_part}"
    n="${tail%.jld2}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    group="${base%_part${n}.jld2}"
    max="${max_part[$group]:-0}"
    if (( n == max )); then
        echo "skip (active):  ${base}"
        kept_parts=$((kept_parts + 1))
        continue
    fi
    echo "move:           ${base}"
    mv -- "$f" "$DEST_DIR/"
    moved_parts=$((moved_parts + 1))
done

# ------------------------------------------------------------------
# Checkpoint files: *_iteration<N>.jld2
# The latest iteration per group is required for run!(sim; pickup=true)
# so it is kept locally; earlier checkpoints are moved.
# ------------------------------------------------------------------
declare -A max_ckpt
for f in "$RUN_DIR"/*_iteration[0-9]*.jld2; do
    base=$(basename "$f")
    tail="${base##*_iteration}"
    n="${tail%.jld2}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    group="${base%_iteration${n}.jld2}"
    current="${max_ckpt[$group]:-0}"
    if (( n > current )); then
        max_ckpt[$group]=$n
    fi
done

moved_ckpts=0
kept_ckpts=0
for f in "$RUN_DIR"/*_iteration[0-9]*.jld2; do
    base=$(basename "$f")
    tail="${base##*_iteration}"
    n="${tail%.jld2}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    group="${base%_iteration${n}.jld2}"
    max="${max_ckpt[$group]:-0}"
    if (( n == max )); then
        echo "skip (latest):  ${base}"
        kept_ckpts=$((kept_ckpts + 1))
        continue
    fi
    echo "move:           ${base}"
    mv -- "$f" "$DEST_DIR/"
    moved_ckpts=$((moved_ckpts + 1))
done

echo "Done. Moved ${moved_parts} part file(s) (kept ${kept_parts})," \
     "moved ${moved_ckpts} checkpoint file(s) (kept ${kept_ckpts})."

sleep 3600 # sleep for 1 hour

echo "Sleeping for 1 hour"

done
EOF
