#!/bin/bash
# Submit an OMIP simulation to SLURM.
#
# Usage:
#   ./launch.sh halfdegree             # half-degree OMIP
#   ./launch.sh tenthdegree            # 1/10-degree OMIP
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
Usage: ./launch.sh <config> [extra sbatch args...]

Configurations:
  halfdegree         Half-degree TripolarGrid (default fluxes)
  orca               ORCA grid (default fluxes)
  orca_corrected     ORCA grid with corrected COARE 3.6 fluxes
  orca_ncar          ORCA grid with OMIP-2/NCAR bulk formulae
  orca_corrected_snow  ORCA + corrected fluxes + snow
  orca_ncar_snow     ORCA + NCAR fluxes + snow
  tenthdegree        1/10-degree TripolarGrid (4 GPUs)

Environment variables:
  FORCING_DIR   Path to JRA55 forcing data (default: forcing_data)
  STAGING_DIR   Fast scratch directory for JRA55 staging. When set,
                files are symlinked from FORCING_DIR and progressively
                copied to STAGING_DIR ahead of each simulated year.
                Keeps ~50 GB on scratch (current + next year).
  NODE          Pin job to a specific node (default: 2904)
  PROFILE       Set to "true" for nsys profiling

Examples:
  ./launch.sh orca_ncar
  ./launch.sh orca_corrected_snow
  FORCING_DIR=/data/jra55 STAGING_DIR=/scratch/jra55_staged ./launch.sh orca_ncar
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
    halfdegree)
        CONFIG="halfdegree"
        ;;
    half_degree)
        CONFIG="halfdegree"
        ;;
    orca|tenthdegree) ;;
    orca_corrected|orca_ncar|orca_corrected_snow|orca_ncar_snow) ;;
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
    tenthdegree)
        GPUS_PER_NODE=4
        ;;
esac

SBATCH_ARGS=()
NODE="${NODE:-2904}"
if [[ -n "${NODE}" ]]; then
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

# JRA55 forcing data directories (shared across all configurations)
FORCING_DIR="${FORCING_DIR:-forcing_data}"
STAGING_DIR="${STAGING_DIR:-}"  # set to a fast scratch path to enable staging

# Build staging kwargs string
STAGING_KWARG=""
if [[ -n "$STAGING_DIR" ]]; then
    STAGING_KWARG="staging_dir = \"${STAGING_DIR}\","
fi

# Build the Julia expression from the selected config.
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
                      Δt = 25minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"halfdegree_run\",
                      filename_prefix = \"halfdegree\")

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
                      Δt = 30minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"orca_run\",
                      filename_prefix = \"orca\")

sim.stop_time = 300 * 365days
run!(sim; pickup=false)"
        ;;
    orca_corrected)
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
                      Δt = 30minutes,
                      flux_configuration = :corrected,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"orca_corrected_run\",
                      filename_prefix = \"orca_corrected\")

sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
    orca_ncar)
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
                      Δt = 30minutes,
                      flux_configuration = :ncar,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"orca_ncar_run\",
                      filename_prefix = \"orca_ncar\")

sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
    orca_corrected_snow)
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
                      Δt = 30minutes,
                      flux_configuration = :corrected,
                      with_snow = true,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"orca_corrected_snow_run\",
                      filename_prefix = \"orca_corrected_snow\")

sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
    orca_ncar_snow)
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
                      Δt = 30minutes,
                      flux_configuration = :ncar,
                      with_snow = true,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"orca_ncar_snow_run\",
                      filename_prefix = \"orca_ncar_snow\")

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
                      Δt = 8minutes,
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      output_dir = \"tenthdegree_run\",
                      filename_prefix = \"tenthdegree\",
                      file_splitting_interval = 180days)

sim.stop_time = 91days
run!(sim)

sim.Δt = 15minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
esac

if [[ "${PROFILE:-false}" == "true" ]]; then
    echo "Profiling ${CONFIG} configuration -> ${REPORT_NAME}"
    nsys profile --trace=cuda \
                 --output="$REPORT_NAME" \
                 --force-overwrite true \
                 "$JULIA" --project=.. --check-bounds=no -e "$JULIA_EXPR"
else
    "$JULIA" --project=.. --check-bounds=no -e "$JULIA_EXPR"
fi
EOF
