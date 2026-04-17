#!/bin/bash
# Submit an OMIP simulation to SLURM.
#
# Usage:
#   ./launch.sh orca                           # ORCA with default fluxes
#   NCAR=true ./launch.sh orca                 # ORCA with NCAR bulk formulae
#   NCAR=true SNOW=true ./launch.sh orca       # ORCA + NCAR + snow
#   CB=0.1 NCAR=true ./launch.sh orca          # ORCA + NCAR + Cᵇ=0.1
#   PROFILE=true ./launch.sh orca              # nsys-profile run
#
# Credentials (e.g. ECCO_USERNAME, ECCO_WEBDAV_PASSWORD) are NOT set
# here. Export them in your shell or source a private file before
# launching, e.g.:
#
#   source ~/.ecco_credentials && ./launch.sh orca

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./launch.sh <config> [extra sbatch args...]

Configurations:
  halfdegree      Half-degree TripolarGrid
  orca            ORCA grid
  tenthdegree     1/10-degree TripolarGrid (4 GPUs)

Environment variables (physics):
  NCAR          Set to "true" for OMIP-2/NCAR bulk formulae
  CORRECTED     Set to "true" for corrected COARE 3.6 fluxes
  SNOW          Set to "true" to enable snow thermodynamics
  CB            CATKE buoyancy mixing length parameter Cᵇ (default: 0.28)

Environment variables (I/O & runtime):
  BACKEND_SIZE  Number of JRA55 time indices kept in memory (default: 240,
                i.e. 30 days of 3-hourly data ≈ 2 GB RAM for 11 variables)
  FORCING_DIR   Path to JRA55 forcing data (default: forcing_data)
  STAGING_DIR   Fast scratch directory for JRA55 staging. When set,
                files are symlinked from FORCING_DIR and progressively
                copied to STAGING_DIR ahead of each simulated year.
                Keeps ~50 GB on scratch (current + next year).
  NODE          Pin job to a specific node (default: 2904)
  PROFILE       Set to "true" for nsys profiling

Examples:
  ./launch.sh orca
  NCAR=true ./launch.sh orca
  NCAR=true SNOW=true ./launch.sh orca
  CORRECTED=true SNOW=true ./launch.sh orca
  CB=0.1 NCAR=true ./launch.sh orca
  FORCING_DIR=/data/jra55 STAGING_DIR=/scratch/jra55_staged ./launch.sh orca
  PROFILE=true ./launch.sh orca
USAGE
}

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    usage
    exit 1
fi
shift || true

case "$CONFIG" in
    halfdegree|half_degree)
        CONFIG="halfdegree"
        ;;
    orca|tenthdegree) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown configuration '$CONFIG'" >&2
        usage
        exit 1
        ;;
esac

# ── Build run name from config + options ──────────────────────────────
RUN_NAME="$CONFIG"
[[ "${CORRECTED:-false}" == "true" ]] && RUN_NAME="${RUN_NAME}_corrected"
[[ "${NCAR:-false}" == "true" ]]      && RUN_NAME="${RUN_NAME}_ncar"
[[ "${SNOW:-false}" == "true" ]]      && RUN_NAME="${RUN_NAME}_snow"
[[ -n "${CB:-}" ]]                    && RUN_NAME="${RUN_NAME}_cb${CB}"

REPORT_NAME="${REPORT_NAME:-${RUN_NAME}_report}"
JOB_NAME="${JOB_NAME:-$RUN_NAME}"
GPUS_PER_NODE=1

case "$CONFIG" in
    tenthdegree) GPUS_PER_NODE=4 ;;
esac

SBATCH_ARGS=()
NODE="${NODE:-2904}"
if [[ -n "${NODE}" ]]; then
    SBATCH_ARGS+=(-w "node${NODE}")
fi
SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}")

if [[ "${PROFILE:-false}" == "true" ]]; then
    SBATCH_ARGS+=(-o "${RUN_NAME}_profile.out")
    SBATCH_ARGS+=(-e "${RUN_NAME}_profile.err")
    SBATCH_ARGS+=(-J "${JOB_NAME}_profile")
    SBATCH_ARGS+=(--export="ALL,PROFILE=true,REPORT_NAME=${REPORT_NAME},CONFIG=${CONFIG},RUN_NAME=${RUN_NAME}")
else
    SBATCH_ARGS+=(-o "${RUN_NAME}.out")
    SBATCH_ARGS+=(-e "${RUN_NAME}.err")
    SBATCH_ARGS+=(-J "$JOB_NAME")
    SBATCH_ARGS+=(--export="ALL,CONFIG=${CONFIG},RUN_NAME=${RUN_NAME}")
fi

sbatch "${SBATCH_ARGS[@]}" "$@" <<'EOF'
#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p pi_raffaele
#SBATCH --time=120:00:00
#SBATCH --mem=150GB

source /etc/profile.d/modules.sh
module load nvhpc

JULIA="${JULIA:-$HOME/julia-1.12.5/bin/julia}"

# ── Shared environment ────────────────────────────────────────────────
FORCING_DIR="${FORCING_DIR:-forcing_data}"
STAGING_DIR="${STAGING_DIR:-}"
CB="${CB:-}"
BACKEND_SIZE="${BACKEND_SIZE:-}"
NCAR="${NCAR:-false}"
CORRECTED="${CORRECTED:-false}"
SNOW="${SNOW:-false}"

# ── Build optional kwargs strings ─────────────────────────────────────
STAGING_KWARG=""
[[ -n "$STAGING_DIR" ]] && STAGING_KWARG="staging_dir = \"${STAGING_DIR}\","

CB_KWARG=""
[[ -n "$CB" ]] && CB_KWARG="Cᵇ = ${CB},"

BACKEND_KWARG=""
[[ -n "$BACKEND_SIZE" ]] && BACKEND_KWARG="backend_size = ${BACKEND_SIZE},"

FLUX_KWARG=""
[[ "$NCAR" == "true" ]]      && FLUX_KWARG="flux_configuration = :ncar,"
[[ "$CORRECTED" == "true" ]] && FLUX_KWARG="flux_configuration = :corrected,"

SNOW_KWARG=""
[[ "$SNOW" == "true" ]] && SNOW_KWARG="with_snow = true,"

# ── Build Julia expression ────────────────────────────────────────────
case "$CONFIG" in
    halfdegree)
        JULIA_EXPR="using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA

sim = omip_simulation(:halfdegree;
                      arch = GPU(),
                      Nz = 70,
                      depth = 5500,
                      ${CB_KWARG}
                      ${FLUX_KWARG}
                      ${SNOW_KWARG}
                      Δt = 25minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      output_dir = \"${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\")

sim.stop_time = 300 * 365days
run!(sim, pickup=:latest)"
        ;;
    orca)
        JULIA_EXPR="using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA

sim = omip_simulation(:orca;
                      arch = GPU(),
                      Nz = 70,
                      depth = 5500,
                      κ_skew = 500,
                      κ_symmetric = 250,
                      biharmonic_timescale = 10days,
                      ${CB_KWARG}
                      ${FLUX_KWARG}
                      ${SNOW_KWARG}
                      Δt = 30minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      output_dir = \"${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\")

sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
    tenthdegree)
        JULIA_EXPR="using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using CUDA

sim = omip_simulation(:tenthdegree;
                      arch = Distributed(GPU(), partition=Partition(1, 4)),
                      Nz = 100,
                      depth = 5500,
                      κ_skew = nothing,
                      κ_symmetric = nothing,
                      biharmonic_timescale = nothing,
                      ${CB_KWARG}
                      ${FLUX_KWARG}
                      ${SNOW_KWARG}
                      Δt = 8minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      output_dir = \"${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\",
                      file_splitting_interval = 180days)

sim.stop_time = 91days
run!(sim)

sim.Δt = 15minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
esac

if [[ "${PROFILE:-false}" == "true" ]]; then
    echo "Profiling ${RUN_NAME} -> ${REPORT_NAME}"
    nsys profile --trace=cuda \
                 --output="$REPORT_NAME" \
                 --force-overwrite true \
                 "$JULIA" --project=.. --check-bounds=no -e "$JULIA_EXPR"
else
    "$JULIA" --project=.. --check-bounds=no -e "$JULIA_EXPR"
fi
EOF
