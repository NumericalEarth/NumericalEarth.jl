#!/bin/bash
# Submit/re-submit OMIP runs on Gadi (PBS), or run interactively on an
# already-allocated interactive node.
#
# Usage examples:
#   ./launch_gadi.sh orca --interactive
#   ./launch_gadi.sh orca --arch cpu --interactive
#   ./launch_gadi.sh orca --dir=/g/data/v46/txs156/ocean-ensembles --interactive
#   ./launch_gadi.sh sxthdegree
#   RYF=true ./launch_gadi.sh sxthdegree --interactive
#   qsub -v CONFIG=sxthdegree,count=1,max=12 /path/to/launch_gadi.sh

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

usage() {
    cat <<'USAGE'
Usage:
  ./launch_gadi.sh <config> [--arch cpu|gpu] [--dir PATH] [--interactive] [extra qsub args...]

Configurations:
  halfdegree      Half-degree TripolarGrid
  orca            ORCA grid
  sxthdegree      1/6-degree TripolarGrid (4 GPUs)
  tenthdegree     1/10-degree TripolarGrid (4 GPUs)

Modes:
  default         Submit to PBS via qsub
  --interactive   Run immediately on current node (no qsub)

Architecture:
  --arch cpu|gpu  Architecture selector

Directories:
  --dir PATH      Sets USERDIR. By default, inputs come from USERDIR/data
                  and output writers write to USERDIR/outputs.

Environment variables:
  USERDIR, OUTPUT_DIR
  count, max, SEGMENT_MONTHS, THREADS
  RYF, NCAR, CORRECTED, SNOW, CB
  KSKEW, KSYMM, BIHARMONIC, BIHVISC
  FORCING_DIR, STAGING_DIR, BACKEND_SIZE
  DRY_RUN=true

Examples:
  ./launch_gadi.sh orca --interactive
  ./launch_gadi.sh orca --arch cpu --interactive
  ./launch_gadi.sh orca --arch=cpu --interactive
  ./launch_gadi.sh orca --dir=/g/data/v46/txs156/ocean-ensembles --interactive
  NCAR=true SNOW=true ./launch_gadi.sh orca --interactive
  ./launch_gadi.sh sxthdegree -q gpuhopper
  qsub -v CONFIG=sxthdegree,count=1,max=12 /path/to/launch_gadi.sh
USAGE
}

IN_PBS_BATCH=false
if [[ "${PBS_ENVIRONMENT:-}" == "PBS_BATCH" ]]; then
    IN_PBS_BATCH=true
fi

if [[ "${IN_PBS_BATCH}" == "true" && -n "${PBS_O_WORKDIR:-}" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$0")"
cd "$SCRIPT_DIR"

INTERACTIVE=false
CONFIG="${CONFIG:-}"
ARCH_KIND="${ARCH_KIND:-${ARCH:-gpu}}"
USERDIR="${USERDIR:-}"
QSUB_ARGS=()

# Parse CLI args unless this is the script running as a PBS batch job payload.
if [[ "${IN_PBS_BATCH}" != "true" ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --interactive)
                INTERACTIVE=true
                ;;
            --arch)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --arch requires a value (cpu|gpu)." >&2
                    usage
                    exit 1
                fi
                ARCH_KIND="$2"
                shift 2
                continue
                ;;
            --arch=*)
                ARCH_KIND="${1#*=}"
                ;;
            --dir)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --dir requires a path value." >&2
                    usage
                    exit 1
                fi
                USERDIR="$2"
                shift 2
                continue
                ;;
            --dir=*)
                USERDIR="${1#*=}"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                QSUB_ARGS+=("$1")
                ;;
            *)
                if [[ -z "$CONFIG" ]]; then
                    CONFIG="$1"
                else
                    QSUB_ARGS+=("$1")
                fi
                ;;
        esac
        shift
    done
fi

if [[ -z "$CONFIG" ]]; then
    CONFIG="sxthdegree"
fi

case "$CONFIG" in
    halfdegree|half_degree)
        CONFIG="halfdegree"
        ;;
    orca|tenthdegree)
        ;;
    sxthdegree|sxtdegree|sixthdegree)
        CONFIG="sxthdegree"
        ;;
    *)
        echo "Error: unknown configuration '$CONFIG'" >&2
        usage
        exit 1
        ;;
esac

ARCH_KIND="${ARCH_KIND,,}"
case "$ARCH_KIND" in
    cpu|gpu)
        ;;
    *)
        echo "Error: unknown architecture selector '$ARCH_KIND' (use --arch cpu or --arch gpu)" >&2
        usage
        exit 1
        ;;
esac

if [[ -n "$USERDIR" ]]; then
    USERDIR="${USERDIR%/}"
fi

# Per-config defaults, architecture, and run command.
case "$CONFIG" in
    halfdegree)
        DEFAULT_KSKEW=250; DEFAULT_KSYMM=100; NZ=70; DEFAULT_DT="25minutes"
        DEFAULT_BIHARMONIC="40days"; ARCH="GPU()"; GPUS_PER_NODE=1; MPI_RANKS=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 300 * 365days
run!(sim; pickup = :latest)"
        ;;
    orca)
        DEFAULT_KSKEW=500; DEFAULT_KSYMM=250; NZ=70; DEFAULT_DT="30minutes"
        DEFAULT_BIHARMONIC="10days"; ARCH="GPU()"; GPUS_PER_NODE=1; MPI_RANKS=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 300 * 365days
run!(sim; pickup = :latest)"
        ;;
    sxthdegree)
        DEFAULT_KSKEW=0; DEFAULT_KSYMM=0; NZ=75; DEFAULT_DT="10minutes"
        DEFAULT_BIHARMONIC="nothing"
        ARCH="Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication = true)"
        GPUS_PER_NODE=4; MPI_RANKS=4
        EXTRA_USING="using Oceananigans.DistributedComputations"; FILE_SPLIT=""
        RUN_CMD="segment_months = parse(Int, get(ENV, \"SEGMENT_MONTHS\", \"12\"))
run_count = parse(Int, get(ENV, \"count\", \"1\"))
sim.stop_time = run_count * segment_months * (365/12)days
run!(sim; pickup = :latest, checkpoint_at_end = true)"
        ;;
    tenthdegree)
        DEFAULT_KSKEW=0; DEFAULT_KSYMM=0; NZ=100; DEFAULT_DT="8minutes"
        DEFAULT_BIHARMONIC="nothing"
        ARCH="Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication = true)"
        GPUS_PER_NODE=4; MPI_RANKS=4
        EXTRA_USING="using Oceananigans.DistributedComputations"
        FILE_SPLIT="file_splitting_interval = 180days,"
        RUN_CMD="sim.stop_time = 91days
run!(sim)

sim.Δt = 15minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
esac

if [[ "$ARCH_KIND" == "cpu" ]]; then
    if [[ "$CONFIG" == "sxthdegree" || "$CONFIG" == "tenthdegree" ]]; then
        ARCH="Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication = true)"
    else
        ARCH="CPU()"
    fi
    GPUS_PER_NODE=0
fi

count="${count:-1}"
max="${max:-$count}"
SEGMENT_MONTHS="${SEGMENT_MONTHS:-12}"
THREADS="${THREADS:-12}"

if [[ -n "$USERDIR" ]]; then
    FORCING_DIR="${FORCING_DIR:-${USERDIR}/data}"
else
    FORCING_DIR="${FORCING_DIR:-${DATA:-}forcing_data}"
fi
STAGING_DIR="${STAGING_DIR:-./staged_data}"
BACKEND_SIZE="${BACKEND_SIZE:-}"

RYF="${RYF:-false}"
NCAR="${NCAR:-false}"
CORRECTED="${CORRECTED:-false}"
SNOW="${SNOW:-false}"
CB="${CB:-}"
BIHVISC="${BIHVISC:-}"

KSKEW="${KSKEW:-$DEFAULT_KSKEW}"
KSYMM="${KSYMM:-$DEFAULT_KSYMM}"
DT="${DT:-$DEFAULT_DT}"
BIHARMONIC="${BIHARMONIC:-$DEFAULT_BIHARMONIC}"
KSKEW_JULIA="$KSKEW"; [[ "$KSKEW" == "0" ]] && KSKEW_JULIA="nothing"
KSYMM_JULIA="$KSYMM"; [[ "$KSYMM" == "0" ]] && KSYMM_JULIA="nothing"

RUN_NAME="$CONFIG"
[[ "$ARCH_KIND" == "cpu" ]] && RUN_NAME="${RUN_NAME}_cpu"
[[ "$RYF" == "true" ]] && RUN_NAME="${RUN_NAME}_ryf"
[[ "$CORRECTED" == "true" ]] && RUN_NAME="${RUN_NAME}_corrected"
[[ "$NCAR" == "true" ]] && RUN_NAME="${RUN_NAME}_ncar"
[[ "$SNOW" == "true" ]] && RUN_NAME="${RUN_NAME}_snow"
[[ -n "$CB" ]] && RUN_NAME="${RUN_NAME}_cb${CB}"
[[ "$KSKEW" != "$DEFAULT_KSKEW" ]] && RUN_NAME="${RUN_NAME}_kskew${KSKEW}"
[[ "$KSYMM" != "$DEFAULT_KSYMM" ]] && RUN_NAME="${RUN_NAME}_ksymm${KSYMM}"
[[ "$BIHARMONIC" != "$DEFAULT_BIHARMONIC" ]] && RUN_NAME="${RUN_NAME}_bih${BIHARMONIC}"
[[ -n "$BIHVISC" ]] && RUN_NAME="${RUN_NAME}_bihvisc${BIHVISC}"

if [[ -n "$USERDIR" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-${USERDIR}/outputs}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-${RUN_NAME}_run}"
fi

STAGING_KWARG=""
if [[ -n "$STAGING_DIR" ]]; then
    RUN_STAGING_DIR="${STAGING_DIR}/${RUN_NAME}"
    STAGING_KWARG="staging_dir = \"${RUN_STAGING_DIR}\","
fi

BACKEND_KWARG=""
[[ -n "$BACKEND_SIZE" ]] && BACKEND_KWARG="backend_size = ${BACKEND_SIZE},"

CB_KWARG=""
[[ -n "$CB" ]] && CB_KWARG="Cᵇ = ${CB},"

BIHVISC_KWARG=""
[[ -n "$BIHVISC" ]] && BIHVISC_KWARG="biharmonic_viscosity = ${BIHVISC},"

FLUX_KWARG=""
[[ "$NCAR" == "true" ]] && FLUX_KWARG="flux_configuration = :ncar,"
[[ "$CORRECTED" == "true" ]] && FLUX_KWARG="flux_configuration = :corrected,"

SNOW_KWARG=""
[[ "$SNOW" == "true" ]] && SNOW_KWARG="with_snow = true,"

RYF_KWARG=""
[[ "$RYF" == "true" ]] && RYF_KWARG="repeat_year_forcing = true,"

if [[ -n "${JULIA:-}" ]]; then
    JULIA="${JULIA}"
elif command -v julia >/dev/null 2>&1; then
    JULIA="$(command -v julia)"
else
    JULIA="$HOME/julia-1.12.5/bin/julia"
fi

CUDA_USING="using CUDA"
if [[ "$ARCH_KIND" == "cpu" ]]; then
    CUDA_USING=""
fi

if [[ ! -x "$JULIA" ]]; then
    echo "Error: Julia executable not found or not executable: $JULIA" >&2
    echo "Set JULIA=/path/to/julia and rerun." >&2
    exit 1
fi

# Submit to PBS unless we're currently in batch payload mode or interactive
# mode is explicitly requested.
if [[ "${IN_PBS_BATCH}" != "true" && "${INTERACTIVE}" != "true" ]]; then
    NCPUS_PER_NODE=$(( THREADS * MPI_RANKS ))
    export_vars=(
        "CONFIG=${CONFIG}"
        "ARCH_KIND=${ARCH_KIND}"
        "USERDIR=${USERDIR}"
        "count=${count}"
        "max=${max}"
        "SEGMENT_MONTHS=${SEGMENT_MONTHS}"
        "THREADS=${THREADS}"
        "FORCING_DIR=${FORCING_DIR}"
        "OUTPUT_DIR=${OUTPUT_DIR}"
        "STAGING_DIR=${STAGING_DIR}"
        "BACKEND_SIZE=${BACKEND_SIZE}"
        "RYF=${RYF}"
        "NCAR=${NCAR}"
        "CORRECTED=${CORRECTED}"
        "SNOW=${SNOW}"
        "CB=${CB}"
        "KSKEW=${KSKEW}"
        "KSYMM=${KSYMM}"
        "DT=${DT}"
        "BIHARMONIC=${BIHARMONIC}"
        "BIHVISC=${BIHVISC}"
    )
    v_arg="$(IFS=,; echo "${export_vars[*]}")"

    qsub_cmd=(
        qsub
        -v "$v_arg"
        -N "OMIP_${CONFIG}"
        -l "ncpus=${NCPUS_PER_NODE}"
        -l "ngpus=${GPUS_PER_NODE}"
    )
    if [[ "${#QSUB_ARGS[@]}" -gt 0 ]]; then
        qsub_cmd+=("${QSUB_ARGS[@]}")
    fi
    qsub_cmd+=("$SCRIPT_PATH")

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "[DRY_RUN] ${qsub_cmd[*]}"
        exit 0
    fi

    "${qsub_cmd[@]}"
    exit 0
fi

JULIA_EXPR="using OMIPSimulations
using Oceananigans
using Oceananigans.Units
${CUDA_USING}
${EXTRA_USING}

sim = omip_simulation(:${CONFIG};
                      arch = ${ARCH},
                      Nz = ${NZ},
                      depth = 5500,
                      κ_skew = ${KSKEW_JULIA},
                      κ_symmetric = ${KSYMM_JULIA},
                      biharmonic_timescale = ${BIHARMONIC},
                      ${BIHVISC_KWARG}
                      ${CB_KWARG}
                      ${FLUX_KWARG}
                      ${SNOW_KWARG}
                      ${RYF_KWARG}
                      Δt = ${DT},
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      ${FILE_SPLIT}
                      output_dir = \"${OUTPUT_DIR}\",
                      filename_prefix = \"${RUN_NAME}\")

${RUN_CMD}"

STDOUT_LOG="${SCRIPT_DIR}/${RUN_NAME}_gadi_${count}.stdout"
STDERR_LOG="${SCRIPT_DIR}/${RUN_NAME}_gadi_${count}.stderr"
RUN_LOG="${SCRIPT_DIR}/${RUN_NAME}_gadi_${count}.run_log"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "[DRY_RUN] CONFIG=${CONFIG} ARCH_KIND=${ARCH_KIND} INTERACTIVE=${INTERACTIVE} count=${count} max=${max} SEGMENT_MONTHS=${SEGMENT_MONTHS} RYF=${RYF}"
    echo "[DRY_RUN] FORCING_DIR=${FORCING_DIR} OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[DRY_RUN] RUN_NAME=${RUN_NAME}"
    echo "[DRY_RUN] mpiexec --bind-to socket --map-by socket -n ${MPI_RANKS} $JULIA --project=.. --check-bounds=no -t ${THREADS} -e <JULIA_EXPR>"
else
    if [[ "${INTERACTIVE}" == "true" ]]; then
        mpiexec --bind-to socket --map-by socket -n "${MPI_RANKS}" \
            "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR" \
            2>&1 | tee "$STDOUT_LOG" "$RUN_LOG"
    else
        mpiexec --bind-to socket --map-by socket -n "${MPI_RANKS}" \
            "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR" \
            > "$STDOUT_LOG" 2> "$STDERR_LOG"
    fi
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exit 0
fi

next_count=$((count + 1))
if [[ "${CONFIG}" == "sxthdegree" && "$next_count" -le "$max" && "${IN_PBS_BATCH}" == "true" ]]; then
    echo "Resubmitting: run ${next_count} of ${max}"
    qsub -v CONFIG="${CONFIG}",ARCH_KIND="${ARCH_KIND}",USERDIR="${USERDIR}",count="${next_count}",max="${max}",SEGMENT_MONTHS="${SEGMENT_MONTHS}",THREADS="${THREADS}",FORCING_DIR="${FORCING_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",STAGING_DIR="${STAGING_DIR}",BACKEND_SIZE="${BACKEND_SIZE}",RYF="${RYF}",NCAR="${NCAR}",CORRECTED="${CORRECTED}",SNOW="${SNOW}",CB="${CB}",KSKEW="${KSKEW}",KSYMM="${KSYMM}",DT="${DT}",BIHARMONIC="${BIHARMONIC}",BIHVISC="${BIHVISC}" "$SCRIPT_PATH"
else
    echo "Completed final run: ${count} of ${max}"
fi
