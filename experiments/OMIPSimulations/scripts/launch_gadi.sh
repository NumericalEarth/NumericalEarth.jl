#!/bin/bash
# Submit/re-submit sxthdegree OMIP runs on Gadi (PBS).
#
# Usage examples:
#   qsub -v count=1,max=12 /path/to/launch_gadi.sh
#   qsub -v count=1,max=6,RYF=true,SEGMENT_MONTHS=12 /path/to/launch_gadi.sh
#   qsub -v count=1,max=3,KSKEW=500,KSYMM=250,NCAR=true /path/to/launch_gadi.sh

#PBS -P v46
#PBS -q gpuhopper
#PBS -l walltime=15:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n
#PBS -N OMIP_sxthdegree

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$0")"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-sxthdegree}"
if [[ "$CONFIG" != "sxthdegree" ]]; then
    echo "Error: launch_gadi.sh currently supports CONFIG=sxthdegree only" >&2
    exit 1
fi

count="${count:-1}"
max="${max:-$count}"
SEGMENT_MONTHS="${SEGMENT_MONTHS:-12}"
THREADS="${THREADS:-12}"

FORCING_DIR="${FORCING_DIR:-${DATA:-}forcing_data}"
STAGING_DIR="${STAGING_DIR:-./staged_data}"
BACKEND_SIZE="${BACKEND_SIZE:-}"

RYF="${RYF:-false}"
NCAR="${NCAR:-false}"
CORRECTED="${CORRECTED:-false}"
SNOW="${SNOW:-false}"
CB="${CB:-}"

DEFAULT_KSKEW=0
DEFAULT_KSYMM=0
KSKEW="${KSKEW:-$DEFAULT_KSKEW}"
KSYMM="${KSYMM:-$DEFAULT_KSYMM}"
KSKEW_JULIA="$KSKEW"; [[ "$KSKEW" == "0" ]] && KSKEW_JULIA="nothing"
KSYMM_JULIA="$KSYMM"; [[ "$KSYMM" == "0" ]] && KSYMM_JULIA="nothing"

RUN_NAME="$CONFIG"
[[ "$RYF" == "true" ]] && RUN_NAME="${RUN_NAME}_ryf"
[[ "$CORRECTED" == "true" ]] && RUN_NAME="${RUN_NAME}_corrected"
[[ "$NCAR" == "true" ]] && RUN_NAME="${RUN_NAME}_ncar"
[[ "$SNOW" == "true" ]] && RUN_NAME="${RUN_NAME}_snow"
[[ -n "$CB" ]] && RUN_NAME="${RUN_NAME}_cb${CB}"
[[ "$KSKEW" != "$DEFAULT_KSKEW" ]] && RUN_NAME="${RUN_NAME}_kskew${KSKEW}"
[[ "$KSYMM" != "$DEFAULT_KSYMM" ]] && RUN_NAME="${RUN_NAME}_ksymm${KSYMM}"

STAGING_KWARG=""
if [[ -n "$STAGING_DIR" ]]; then
    RUN_STAGING_DIR="${STAGING_DIR}/${RUN_NAME}"
    STAGING_KWARG="staging_dir = \"${RUN_STAGING_DIR}\","
fi

BACKEND_KWARG=""
[[ -n "$BACKEND_SIZE" ]] && BACKEND_KWARG="backend_size = ${BACKEND_SIZE},"

CB_KWARG=""
[[ -n "$CB" ]] && CB_KWARG="Cᵇ = ${CB},"

FLUX_KWARG=""
[[ "$NCAR" == "true" ]] && FLUX_KWARG="flux_configuration = :ncar,"
[[ "$CORRECTED" == "true" ]] && FLUX_KWARG="flux_configuration = :corrected,"

SNOW_KWARG=""
[[ "$SNOW" == "true" ]] && SNOW_KWARG="with_snow = true,"

RYF_KWARG=""
[[ "$RYF" == "true" ]] && RYF_KWARG="repeat_year_forcing = true,"

JULIA="${JULIA:-$HOME/julia-1.12.5/bin/julia}"

JULIA_EXPR="using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA
using Oceananigans.DistributedComputations

sim = omip_simulation(:sxthdegree;
                      arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication = true),
                      Nz = 75,
                      depth = 5500,
                      κ_skew = ${KSKEW_JULIA},
                      κ_symmetric = ${KSYMM_JULIA},
                      biharmonic_timescale = nothing,
                      ${CB_KWARG}
                      ${FLUX_KWARG}
                      ${SNOW_KWARG}
                      ${RYF_KWARG}
                      Δt = 10minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      output_dir = \"${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\")

sim.stop_time = ${count} * ${SEGMENT_MONTHS} * (365/12)days
run!(sim; pickup = :latest, checkpoint_at_end = true)"

STDOUT_LOG="${RUN_NAME}_gadi_${count}.stdout"
STDERR_LOG="${RUN_NAME}_gadi_${count}.stderr"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "[DRY_RUN] CONFIG=${CONFIG} count=${count} max=${max} SEGMENT_MONTHS=${SEGMENT_MONTHS} RYF=${RYF}"
    echo "[DRY_RUN] RUN_NAME=${RUN_NAME}"
    echo "[DRY_RUN] mpiexec --bind-to socket --map-by socket -n 4 $JULIA --project=.. --check-bounds=no -t ${THREADS} -e <JULIA_EXPR>"
else
    mpiexec --bind-to socket --map-by socket -n 4 \
        "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR" \
        > "$STDOUT_LOG" 2> "$STDERR_LOG"
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exit 0
fi

next_count=$((count + 1))
if [[ "$next_count" -le "$max" ]]; then
    echo "Resubmitting: run ${next_count} of ${max}"
    qsub -v count="${next_count}",max="${max}",SEGMENT_MONTHS="${SEGMENT_MONTHS}",THREADS="${THREADS}",FORCING_DIR="${FORCING_DIR}",STAGING_DIR="${STAGING_DIR}",BACKEND_SIZE="${BACKEND_SIZE}",RYF="${RYF}",NCAR="${NCAR}",CORRECTED="${CORRECTED}",SNOW="${SNOW}",CB="${CB}",KSKEW="${KSKEW}",KSYMM="${KSYMM}" "$SCRIPT_PATH"
else
    echo "Completed final run: ${count} of ${max}"
fi
