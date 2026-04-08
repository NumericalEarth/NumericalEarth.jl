#!/bin/bash
# Submit an OMIP simulation to SLURM.
#
# Usage:
#   ./launch.sh halfdegree             # half-degree OMIP
#   ./launch.sh eightdegree            # 1/8-degree OMIP
#   ./launch.sh orca                   # ORCA OMIP
#   PROFILE=true ./launch.sh orca      # nsys-profile run
#   NODE=2904 ./launch.sh orca         # pin to a specific node
#
# Credentials (e.g. ECCO_USERNAME, ECCO_WEBDAV_PASSWORD) are NOT set
# here. Export them in your shell or source a private file before
# launching, e.g.:
#
#   source ~/.ecco_credentials && ./launch.sh orca

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./launch.sh <halfdegree|eightdegree|orca> [extra sbatch args...]

Examples:
  ./launch.sh halfdegree
  ./launch.sh eightdegree
  ./launch.sh orca
  PROFILE=true ./launch.sh orca
  NODE=2904 ./launch.sh orca
USAGE
}

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    usage
    exit 1
fi
shift || true

case "$CONFIG" in
    halfdegree)
        CONFIG="halfdegree"
        ;;
    half_degree)
        CONFIG="halfdegree"
        ;;
    orca|eightdegree) ;;
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

REPORT_NAME="${REPORT_NAME:-${CONFIG}_report}"
JOB_NAME="${JOB_NAME:-$CONFIG}"
GPUS_PER_NODE=1

case "$CONFIG" in
    eightdegree)
        GPUS_PER_NODE=4
        ;;
esac

SBATCH_ARGS=()
if [[ -n "${NODE:-}" ]]; then
    SBATCH_ARGS+=(-w "node${NODE}")
fi
SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}")

if [[ "${PROFILE:-false}" == "true" ]]; then
    SBATCH_ARGS+=(-o "${CONFIG}_profile.out")
    SBATCH_ARGS+=(-e "${CONFIG}_profile.err")
    SBATCH_ARGS+=(-J "${JOB_NAME}_profile")
    SBATCH_ARGS+=(--export="ALL,PROFILE=true,REPORT_NAME=${REPORT_NAME},CONFIG=${CONFIG}")
else
    SBATCH_ARGS+=(-o "${CONFIG}.out")
    SBATCH_ARGS+=(-e "${CONFIG}.err")
    SBATCH_ARGS+=(-J "$JOB_NAME")
    SBATCH_ARGS+=(--export="ALL,CONFIG=${CONFIG}")
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

# Build the Julia expression from the selected config.
case "$CONFIG" in
    halfdegree)
        JULIA_EXPR='using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA

sim = omip_simulation(:halfdegree;
                      arch = CPU(),
                      Nz = 70,
                      depth = 5500,
                      Δt = 20minutes,
                      output_dir = "halfdegree_run",
                      filename_prefix = "halfdegree")

sim.stop_time = 91days
run!(sim)

sim.Δt = 30minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)'
        ;;
    orca)
        JULIA_EXPR='using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA

sim = omip_simulation(:orca;
                      arch = GPU(),
                      Nz = 70,
                      depth = 5500,
                      Δt = 30minutes,
                      output_dir = "orca_run",
                      filename_prefix = "orca")

sim.stop_time = 300 * 365days
run!(sim; pickup = true)'
        ;;
    eightdegree)
        JULIA_EXPR='using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using CUDA

# TODO: adjust this block for the 1/8-degree setup details you want.
sim = omip_simulation(:eight_degree;
                      arch = Distributed(GPU(), partition=Partition(1, 4)),
                      Nz = 100,
                      depth = 5500,
                      κ_skew = nothing,
                      κ_symmetric = nothing,
                      biharmonic_timescale = nothing,
                      Δt = 8minutes,
                      output_dir = "eightdegree_run",
                      filename_prefix = "eightdegree",
                      file_splitting_interval = 180days)

sim.stop_time = 91days
run!(sim)

sim.Δt = 15minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)'
        ;;
esac

if [[ "${PROFILE:-false}" == "true" ]]; then
    echo "Profiling ${CONFIG} configuration -> ${REPORT_NAME}"
    nsys profile --trace=cuda \
                 --output="$REPORT_NAME" \
                 --force-overwrite true \
                 "$JULIA" --project=.. --check-bounds=no -e "$JULIA_EXPR"
else
    "$JULIA" --project=.. --check-bounds=no -e 'using Pkg; Pkg.update("Oceananigans")'
    "$JULIA" --project=.. --check-bounds=no -e "$JULIA_EXPR"
fi
EOF
