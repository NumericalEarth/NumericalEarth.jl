#!/bin/bash
# Submit an OMIP simulation to SLURM.
#
# Usage:
#   ./launch.sh orca                           # ORCA with default fluxes
#   NCAR=true ./launch.sh orca                 # ORCA with NCAR bulk formulae
#   NCAR=true SNOW=true ./launch.sh orca       # ORCA + NCAR + snow
#   CB=0.1 NCAR=true ./launch.sh orca          # ORCA + NCAR + Cᵇ=0.1
#   KSKEW=1000 KSYMM=500 ./launch.sh orca      # ORCA with custom eddy diffusivities
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
  KSKEW         Isopycnal skew diffusivity κ_skew (default: per-config; 0 = off)
  KSYMM         Isopycnal symmetric diffusivity κ_symmetric (default: per-config; 0 = off)
  BIHARMONIC    Biharmonic viscosity timescale (default: per-config; "nothing" = off)
  BIHVISC       Constant biharmonic viscosity ν in m^4/s (default: unset).
                When set, overrides BIHARMONIC and uses ν directly instead of
                the grid-area-scaled νhb = Az^2 / λ form.
  CB            CATKE buoyancy mixing length parameter Cᵇ (default: 0.28)
  CLOSURE       Ocean vertical closure: "catke" (default), "simple", "nori", "rbvd", or "kpp"
                ("simple" = ConvectiveAdjustment + depth-stepped background κ/ν;
                 "nori"   = NORi Richardson-number closure
                            (xkykai/NORiOceanParameterization.jl, vendored);
                 "rbvd"   = Oceananigans' built-in RiBasedVerticalDiffusivity
                            (Richardson-number-based, with built-in κ-clip and
                             time-averaging smoothing);
                 "kpp"    = K-Profile Parameterization (Large 1994 / MITgcm,
                            vendored in `KPP/`);
                 all ignore CB)
  WIND_VELOCITY Set to "true" to use absolute wind (Δu = u_atm) in the bulk
                formula instead of the OMIP-2 default relative wind
                (Δu = u_atm − u_ocean). For isolating ACC-current feedback.
  DZ_TOP        Target thickness of the top (surface) cell in meters. If set,
                the ExponentialDiscretization scale is found by bisection so
                that Δz of the surface level matches DZ_TOP within ~0.1%.
                Must satisfy 0 < DZ_TOP < depth/Nz. Default: unset (scale=1300).

Equatorial-MLD tuning knobs (closure parameters; configuration switches):
  SHEAR_GUST    Use the shear-aware Mahrt–Sun (1995) / Edson (2013) gustiness
                form (Uᴳ² = (β·w★)² + (c·|Δu|)² + Uᴳ₀², c=0.04 by default).
                Activates the :shear_aware flux configuration. Implies the
                :corrected fluxes (constant Charnock disabled, etc.).
                Useful when the equatorial mixed layer is too shallow because
                of weak convective gustiness — adds shear-driven gust at all
                wind speeds.
  MIN_SALINITY  Floor (psu) below which the freshening (salt-extracting)
                component of the air-sea freshwater flux is suppressed.
                Salt-concentrating fluxes (E > P + R) are always applied.
                Prevents NaN blow-ups in pathologically thin top cells
                under strong precip + runoff plumes. Default: 1.
  CATKE_CWUSTAR `Cᵂu★` of CATKEEquation: surface shear-driven TKE flux
                coefficient. Higher → more wind-injected TKE → deeper
                equatorial ML. Default (Oceananigans): 3.179.

Environment variables (I/O & runtime):
  BACKEND_SIZE  Number of JRA55 time indices kept in memory (default: 240,
                i.e. 30 days of 3-hourly data ≈ 2 GB RAM for 11 variables)
  FORCING_DIR   Path to JRA55 forcing data (default: ${DATA}forcing_data)
  STAGING_DIR   Base directory for JRA55 staging (default: ./staged_data).
                A per-run subdirectory (STAGING_DIR/<run_name>) is created
                with symlinks from FORCING_DIR; files are progressively
                copied ahead of each simulated year.
                Keeps ~50 GB per run (current + next year).
  NODE          Pin job to a specific node (default: 2904)
  THREADS       Number of Julia threads / CPUs per task (default: 4)
  PROFILE       Set to "true" for nsys profiling

Examples:
  ./launch.sh orca
  NCAR=true ./launch.sh orca
  NCAR=true SNOW=true ./launch.sh orca
  CORRECTED=true SNOW=true ./launch.sh orca
  CB=0.1 NCAR=true ./launch.sh orca
  KSKEW=1000 KSYMM=500 ./launch.sh orca
  KSKEW=0 ./launch.sh orca                    # disable eddy closure
  BIHARMONIC=5days ./launch.sh orca           # custom biharmonic timescale
  BIHARMONIC=nothing ./launch.sh orca         # disable biharmonic viscosity
  BIHVISC=1e12 ./launch.sh orca               # constant biharmonic viscosity ν=1e12 m^4/s
  DZ_TOP=2 ./launch.sh orca                   # 2 m top cell (scale chosen by bisection)
  SHEAR_GUST=true ./launch.sh orca            # Mahrt-Sun shear-aware gustiness
  CATKE_CWUSTAR=5.0 ./launch.sh orca          # stronger surface TKE injection in CATKE
  FORCING_DIR=/other/path/forcing_data STAGING_DIR=/scratch/staged ./launch.sh orca
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

# ── Per-config defaults ───────────────────────────────────────────────
#                     KSKEW  KSYMM  NZ   DT          BIHARMONIC  ARCH                                             GPUS  EXTRA_USING                              FILE_SPLIT  RUN_CMD
case "$CONFIG" in
    halfdegree)
        DEFAULT_KSKEW=250;  DEFAULT_KSYMM=100; NZ=70;  DEFAULT_DT="25minutes"
        DEFAULT_BIHARMONIC="40days"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 300 * 365days
run!(sim, pickup=:latest)"
        ;;
    orca)
        DEFAULT_KSKEW=500;  DEFAULT_KSYMM=250; NZ=70;  DEFAULT_DT="30minutes"
        DEFAULT_BIHARMONIC="10days"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 300 * 365days
run!(sim; pickup = :latest)"
        ;;
    tenthdegree)
        DEFAULT_KSKEW=0;    DEFAULT_KSYMM=0;   NZ=100; DEFAULT_DT="8minutes"
        DEFAULT_BIHARMONIC="nothing"; ARCH="Distributed(GPU(), partition=Partition(1, 4))"; GPUS_PER_NODE=4
        EXTRA_USING="using Oceananigans.DistributedComputations"
        FILE_SPLIT="file_splitting_interval = 180days,"
        RUN_CMD="sim.stop_time = 91days
run!(sim)

sim.Δt = 15minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
esac

# 0 means "no eddy closure" (maps to Julia `nothing`)
export KSKEW="${KSKEW:-$DEFAULT_KSKEW}"
export KSYMM="${KSYMM:-$DEFAULT_KSYMM}"
export DT="${DT:-$DEFAULT_DT}"
export BIHARMONIC="${BIHARMONIC:-$DEFAULT_BIHARMONIC}"
KSKEW_JULIA="$KSKEW"; [[ "$KSKEW" == "0" ]] && KSKEW_JULIA="nothing"
KSYMM_JULIA="$KSYMM"; [[ "$KSYMM" == "0" ]] && KSYMM_JULIA="nothing"
export KSKEW_JULIA KSYMM_JULIA
export NZ DT ARCH EXTRA_USING FILE_SPLIT RUN_CMD

# ── Build run name from config + options ──────────────────────────────
RUN_NAME="$CONFIG"
[[ "${CORRECTED:-false}" == "true" ]]  && RUN_NAME="${RUN_NAME}_corrected"
[[ "${NCAR:-false}" == "true" ]]       && RUN_NAME="${RUN_NAME}_ncar"
[[ "${SNOW:-false}" == "true" ]]       && RUN_NAME="${RUN_NAME}_snow"
[[ "${CLOSURE:-catke}" == "simple" ]]  && RUN_NAME="${RUN_NAME}_simple"
[[ "${CLOSURE:-catke}" == "nori"   ]]  && RUN_NAME="${RUN_NAME}_nori"
[[ "${CLOSURE:-catke}" == "rbvd"   ]]  && RUN_NAME="${RUN_NAME}_rbvd"
[[ "${CLOSURE:-catke}" == "kpp"    ]]  && RUN_NAME="${RUN_NAME}_kpp"
[[ "${WIND_VELOCITY:-false}" == "true" ]] && RUN_NAME="${RUN_NAME}_wind"
[[ -n "${CB:-}" ]]                     && RUN_NAME="${RUN_NAME}_cb${CB}"
[[ "$KSKEW" != "$DEFAULT_KSKEW" ]]    && RUN_NAME="${RUN_NAME}_kskew${KSKEW}"
[[ "$KSYMM" != "$DEFAULT_KSYMM" ]]    && RUN_NAME="${RUN_NAME}_ksymm${KSYMM}"
[[ "$BIHARMONIC" != "$DEFAULT_BIHARMONIC" ]] && RUN_NAME="${RUN_NAME}_bih${BIHARMONIC}"
[[ -n "${BIHVISC:-}" ]]                && RUN_NAME="${RUN_NAME}_bihvisc${BIHVISC}"
[[ -n "${DZ_TOP:-}" ]]                 && RUN_NAME="${RUN_NAME}_dz${DZ_TOP}"
[[ "${SHEAR_GUST:-false}" == "true" ]] && RUN_NAME="${RUN_NAME}_sgust"
[[ -n "${CATKE_CWUSTAR:-}" ]]          && RUN_NAME="${RUN_NAME}_cwu${CATKE_CWUSTAR}"
[[ -n "${MIN_SALINITY:-}" ]]           && RUN_NAME="${RUN_NAME}_smin${MIN_SALINITY}"

REPORT_NAME="${REPORT_NAME:-${RUN_NAME}_report}"
JOB_NAME="${JOB_NAME:-$RUN_NAME}"

SBATCH_ARGS=()
NODE="${NODE:-2904}"
if [[ -n "${NODE}" ]]; then
    SBATCH_ARGS+=(-w "node${NODE}")
fi
SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}")

export THREADS="${THREADS:-8}"
SBATCH_ARGS+=(--cpus-per-task="${THREADS}")

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
#SBATCH --time=72:00:00
#SBATCH --mem=150GB

source /etc/profile.d/modules.sh
module load nvhpc

JULIA="${JULIA:-$HOME/julia-1.12.5/bin/julia}"

# ── Shared environment ────────────────────────────────────────────────
FORCING_DIR="${FORCING_DIR:-${DATA}forcing_data}"
STAGING_DIR="${STAGING_DIR:-./staged_data}"
CB="${CB:-}"
BIHVISC="${BIHVISC:-}"
DZ_TOP="${DZ_TOP:-}"
SHEAR_GUST="${SHEAR_GUST:-false}"
CATKE_CWUSTAR="${CATKE_CWUSTAR:-}"
MIN_SALINITY="${MIN_SALINITY:-}"
BACKEND_SIZE="${BACKEND_SIZE:-}"
NCAR="${NCAR:-false}"
CORRECTED="${CORRECTED:-false}"
SNOW="${SNOW:-false}"

# ── Build optional kwargs strings ─────────────────────────────────────

# Per-run staging subdirectory to avoid conflicts between concurrent jobs
STAGING_KWARG=""
if [[ -n "$STAGING_DIR" ]]; then
    RUN_STAGING_DIR="${STAGING_DIR}/${RUN_NAME}"
    STAGING_KWARG="staging_dir = \"${RUN_STAGING_DIR}\","
fi

CB_KWARG=""
[[ -n "$CB" ]] && CB_KWARG="Cᵇ = ${CB},"

BIHVISC_KWARG=""
[[ -n "$BIHVISC" ]] && BIHVISC_KWARG="biharmonic_viscosity = ${BIHVISC},"

DZ_TOP_KWARG=""
[[ -n "$DZ_TOP" ]] && DZ_TOP_KWARG="Δz_top = ${DZ_TOP},"

CATKE_CWUSTAR_KWARG=""
[[ -n "$CATKE_CWUSTAR" ]] && CATKE_CWUSTAR_KWARG="Cᵂu★ = ${CATKE_CWUSTAR},"

MIN_SALINITY_KWARG=""
[[ -n "$MIN_SALINITY" ]] && MIN_SALINITY_KWARG="ocean_minimum_salinity = ${MIN_SALINITY},"

BACKEND_KWARG=""
[[ -n "$BACKEND_SIZE" ]] && BACKEND_KWARG="backend_size = ${BACKEND_SIZE},"

FLUX_KWARG=""
[[ "$NCAR" == "true" ]]        && FLUX_KWARG="flux_configuration = :ncar,"
[[ "$CORRECTED" == "true" ]]   && FLUX_KWARG="flux_configuration = :corrected,"
[[ "$SHEAR_GUST" == "true" ]]  && FLUX_KWARG="flux_configuration = :shear_aware,"

CLOSURE_KWARG=""
[[ "${CLOSURE:-catke}" == "simple" ]] && CLOSURE_KWARG="vertical_closure = :simple,"
[[ "${CLOSURE:-catke}" == "nori"   ]] && CLOSURE_KWARG="vertical_closure = :nori,"
[[ "${CLOSURE:-catke}" == "rbvd"   ]] && CLOSURE_KWARG="vertical_closure = :rbvd,"
[[ "${CLOSURE:-catke}" == "kpp"    ]] && CLOSURE_KWARG="vertical_closure = :kpp,"

VELOCITY_KWARG=""
[[ "${WIND_VELOCITY:-false}" == "true" ]] && VELOCITY_KWARG="velocity_formulation = :wind,"

SNOW_KWARG=""
[[ "$SNOW" == "true" ]] && SNOW_KWARG="with_snow = true,"

# ── Build and run Julia expression ────────────────────────────────────
JULIA_EXPR="using OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA
${EXTRA_USING}

sim = omip_simulation(:${CONFIG};
                      arch = ${ARCH},
                      Nz = ${NZ},
                      depth = 5500,
                      ${DZ_TOP_KWARG}
                      κ_skew = ${KSKEW_JULIA},
                      κ_symmetric = ${KSYMM_JULIA},
                      biharmonic_timescale = ${BIHARMONIC},
                      ${BIHVISC_KWARG}
                      ${CB_KWARG}
                      ${FLUX_KWARG}
                      ${CLOSURE_KWARG}
                      ${VELOCITY_KWARG}
                      ${SNOW_KWARG}
                      ${CATKE_CWUSTAR_KWARG}
                      ${MIN_SALINITY_KWARG}
                      Δt = ${DT},
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      ${FILE_SPLIT}
                      output_dir = \"${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\")

${RUN_CMD}"

THREADS="${THREADS:-8}"

if [[ "${PROFILE:-false}" == "true" ]]; then
    echo "Profiling ${RUN_NAME} -> ${REPORT_NAME}"
    nsys profile --trace=cuda \
                 --output="$REPORT_NAME" \
                 --force-overwrite true \
                 "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR"
else
    "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR"
fi
EOF
