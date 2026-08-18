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
  quarterdegree   1/4-degree TripolarGrid (2 GPUs)
  orca            ORCA grid
  twelfthdegree     1/10-degree TripolarGrid (4 GPUs)

Environment variables (physics):
  NCAR          Set to "true" for OMIP-2/NCAR bulk formulae
  CORRECTED     Set to "true" for corrected COARE 3.6 fluxes
  SNOW          Set to "true" to enable snow thermodynamics
  ICE_DYNAMICS  Set to "false" to disable sea-ice dynamics (thermo-only ice).
  ICE_LATERAL   Sea-ice lateral boundary condition: "no_slip" (default) applies the viscous
                wall stress -2 eta u / Delta on coastlines; "free_slip" leaves them stress-free.
                The old quadratic "side drag" was inert (unit mismatch) and has been removed.
  ICE_BASAL     Set to "false" to disable the Lemieux et al. (2015) landfast basal stress on
                grounded keels. Default: "true".
  ICE_DRAG      Ice-ocean drag coefficient. Default: 5.5e-3 (Hibler/McPhee, also ClimaSeaIce
                own default). The previous value 3.24e-3 is GFDL SIS2 CDW.
                Default: true.
  KSKEW         Isopycnal skew diffusivity κ_skew (default: per-config; 0 = off;
                "nemo" = NEMO's Treguier et al. 1997 coefficient, Ro² × baroclinic
                growth rate, recomputed each step and depth-uniform;
                "cesm" = CESM's Danabasoglu & Marshall 2007 coefficient,
                κ_ref · clamp(N²/N²_ref, 0.1, 1), recomputed each step with
                full vertical structure;
                "hybrid" = Treguier horizontal × Danabasoglu-Marshall vertical
                shape, with the Rossby-radius cap tightened to 20 km)
  KSYMM         Isopycnal symmetric diffusivity κ_symmetric (default: per-config; 0 = off;
                "nemo" as above, but rising to its reference value in the tropics
                with a floor of one fifth of it, per NEMO nn_aht_ijk_t = 21;
                "cesm" as above)
  BIHARMONIC    Biharmonic viscosity timescale (default: per-config; "nothing" = off)
  BIHVISC       Constant biharmonic viscosity ν in m^4/s (default: unset).
                When set, overrides BIHARMONIC and uses ν directly instead of
                the grid-area-scaled νhb = Az^2 / λ form.
  CB            CATKE buoyancy mixing length parameter Cᵇ (default: 0.28)
  CLOSURE       Ocean vertical closure: "catke" (default), "simple", "nori", "rbvd",
                "kpp", or "nemo_tke"
                ("simple" = ConvectiveAdjustment + depth-stepped background κ/ν;
                 "nori"   = NORi Richardson-number closure
                            (xkykai/NORiOceanParameterization.jl, vendored);
                 "rbvd"   = Oceananigans' built-in RiBasedVerticalDiffusivity
                            (Richardson-number-based, with built-in κ-clip and
                             time-averaging smoothing);
                 "kpp"    = K-Profile Parameterization (Large 1994 / MITgcm,
                            vendored in `KPP/`);
                 "nemo_tke" = NEMO 3.6 TKE scheme (Blanke & Delecluse 1993,
                            Madec 2017), OMIP-2 ORCAOne preset: prognostic e,
                            gradient-limited length scale, Langmuir +
                            Mellor-Blumberg wave penetration + EVD.
                            Vendored in `NEMOTKE/`;
                 all ignore CB)
  PARTIAL_CELLS Set to "true" to use partial bottom cells (PartialCellBottom) instead of
                full-cell GridFittedBottom bathymetry. Resolves sill depths and slopes
                continuously; targets the too-shallow NADW from staircased overflows.
                Adds "_pcells" to the run name.
  BBL_KAPPA     Diffusive bottom boundary layer coefficient in m² s⁻¹ (NEMO rn_ahtbbl;
                its ORCA reference value is 1000). Dense water upslope of a deeper
                neighbour is diffused along the bottom, mimicking the gravity current a
                z-coordinate model otherwise mixes away within a grid cell or two.
                Off by default; adds "_bbl<κ>" to the run name.
  BBL_GAMMA     Advective bottom boundary layer coefficient in seconds (NEMO rn_gambbl,
                reference value 10; Campin & Goosse 1999, NEMO nn_bbl_adv = 2). Opens an
                overturning circuit down the step: dense shelf water enters the bottom of
                the deep column, the water it displaces rises through that column and
                returns to the shelf. The deep cell is therefore flushed towards the shelf
                density rather than mixed towards a mean, which is the ceiling BBL_KAPPA
                cannot pass. Downslope speed is u = γ g Δρ/ρ₀, so the transport shuts off
                as the contrast is consumed. Combines with BBL_KAPPA; the two act on
                different failures. Adds "_cg<γ>" to the run name.
  OVERFLOW_RESTORE
                Diagnostic only, in DAYS. Pins T and S on the East Greenland slope
                (36-26 W, 62-66.5 N, below 1500 m) to observed Denmark Strait Overflow
                Water, Θ = 2.0 and Sᴬ = 35.065, σθ = 27.892. This is not a
                parameterisation: it forces the delivered density so the question "does
                the overflow deficit set the AMOC?" can be answered without first solving
                how to get dense water down a staircase. Both BBL schemes left the
                delivered density unchanged, so neither tested that. Adds "_dsow<days>"
                to the run name.
  ML_TAPER      Set to "true" to ramp the isopycnal-closure slopes linearly to zero
                from the mixed-layer base to the surface (Danabasoglu et al. 2008;
                NEMO ldfslp). Off by default; adds "_mltaper" to the run name.
  WIND_VELOCITY Set to "true" to use absolute wind (Δu = u_atm) in the bulk
                formula instead of the OMIP-2 default relative wind
                (Δu = u_atm − u_ocean). For isolating ACC-current feedback.
  DZ_TOP        Target thickness of the top (surface) cell in meters. If set,
                the ExponentialDiscretization scale is found by bisection so
                that Δz of the surface level matches DZ_TOP within ~0.1%.
                Must satisfy 0 < DZ_TOP < depth/Nz. Default: unset (scale=1300).

Equatorial-MLD tuning knobs (closure parameters; configuration switches):
  NORMALIZE_SALINITY "true" (default) applies the conservative, salt-conserving
                surface-salinity restoring (zero global mean). Set to "false" to use
                the raw un-normalized restoring (the old, non-conserving behavior),
                e.g. for A/B comparison. Default: true.
  NORMALIZE_FRESHWATER Removes the global mean of the atmospheric surface freshwater
                flux, holding the global ocean volume fixed (standard OMIP-2 practice;
                the sea-ice exchange is excluded and the freshwater heat content is
                corrected alongside so the carried heat flux is unchanged). Options:
                  none      no correction (default; "false" also accepted)
                  timestep  remove the instantaneous global mean; pins volume exactly
                            but also removes the seasonal land-water-storage cycle,
                            which is comparable in size ("true" also accepted)
                  annual    remove a running mean relaxed over a year; keeps the
                            seasonal cycle, spins up over the first ~2 years
  SKEW_FORMULATION How the GM skew transport is applied. Ignored when KSKEW=0.
                  diffusive  add it to the tracer flux (default)
                  advective  build the eddy-induced velocity and advect with it,
                             which also exposes the bolus transport as a model
                             field. Equivalent continuously, not discretely, so
                             the two give different answers run-to-run.
                  boundary_value  the Ferrari, Griffies, Nurser & Vallis (2010)
                             transport: each column solves
                             (c d²/dz² - N²) Y = -N² Y_GM with Y = 0 at the
                             surface and the bottom, low-passing the baroclinic
                             modes. Satisfies the boundary conditions without
                             tapering and interpolates through weakly stratified
                             layers with no floor on N². Applied advectively;
                             requires a depth-independent KSKEW (a number or
                             "nemo" -- "cesm"/"hybrid" are rejected).
                Adds "_gmadv" or "_gmbvp" to the run name.
  BVP_MODE      Mode number M in the speed c = max(BVP_CMIN, (M pi)^-1 int N dz) that
                weights the second-order operator. Larger M filters less and gives a
                larger transport; M=1 is the first baroclinic mode, whose amplitude is
                about half the truncated GM transport. Default: 2.
                Only used when SKEW_FORMULATION=boundary_value.
  BVP_CMIN      Floor c_min on that speed, in m/s. Keeps the transport bounded in
                weakly stratified columns. Default: 0.1.
  RESTORING_UNDER_ICE Set to "false" to stop the surface-salinity restoring acting under
                sea ice (weighted by the open-water fraction 1-ℵ, with the zero-mean
                correction spread over open water only, so no net salt is injected).
                WOA is poorly constrained beneath ice and the restoring there fights the
                ice-ocean salt flux. Requires NORMALIZE_SALINITY=true. Adds "_noicerest"
                to the run name. Default: true (OMIP-2 convention).
  RIVER_SPREAD  Radius in degrees over which each river/iceberg mouth's discharge is
                divided equally among the surrounding wet cells. A geographic radius
                keeps the freshwater flux per unit area resolution-independent; raise it
                if a refined grid drives coastal salinity to zero. Set to "none" for the
                historical cell-count footprint instead. Per-config default: 1.2 for the
                refined grids, "none" for orca/test (pinned to their old behaviour).
  RIVER_SPREAD_CELLS  Cells in that footprint, nearest first: a cap when RIVER_SPREAD is
                a radius, the footprint itself when it is "none". Per-config default:
                unset (uncapped) for the refined grids, 8 for orca/test.
  RIVER_MIXING  Set to "false" to disable the extra vertical diffusivity applied over
                the spread footprint. Default: true.
  RIVER_MIXING_K      That diffusivity in m^2/s (default: 0.1).
  RIVER_MIXING_DEPTH  Depth in m over which it is applied (default: 10).
  CATKE_CWUSTAR `Cᵂu★` of CATKEEquation: surface shear-driven TKE flux
                coefficient. Higher → more wind-injected TKE → deeper
                equatorial ML. Default (Oceananigans): 3.179.
  BACKGROUND_K  Interior background tracer diffusivity κ added underneath CATKE (or RBVD).
                Either a number in m^2/s (uniform), or "bryan_lewis" for the Bryan & Lewis
                (1979) depth profile, 3e-5 in the upper ocean rising across ~2500 m to 1.3e-4
                in the abyss (deep upwelling without diffusing the thermocline).
                Default: unset, i.e. the Henyey et al. (1986) latitudinal internal-wave
                scaling κ = max(2e-6, 1e-5 |sin φ|), 2e-6 at the equator to 1e-5 at the poles.
                Raising it strengthens the diapycnal upwelling that closes the AMOC lower limb
                (Bryan 1987: AMOC ~ κ^(2/3)). Adds "_bgk<value>" to the run name.
  BACKGROUND_NU Uniform interior background viscosity ν in m^2/s. Default: unset, i.e. 3e-5
                for CATKE (1e-4 for RBVD). Adds "_bgnu<value>" to the run name.
                Both are rejected by the closures that carry their own background
                (simple/nori/kpp/nemo_tke).

Environment variables (I/O & runtime):
  BACKEND_SIZE  Number of JRA55 time indices kept in memory (default: 240,
                i.e. 30 days of 3-hourly data ≈ 2 GB RAM for 11 variables)
  FORCING_DIR   Path to JRA55 forcing data (default: ${DATA}forcing_data)
  STAGING_DIR   Base directory for JRA55 staging (default: ./staged_data).
                A per-run subdirectory (STAGING_DIR/<run_name>) is created
                with symlinks from FORCING_DIR; files are progressively
                copied ahead of each simulated year.
                Keeps ~50 GB per run (current + next year).
  OUTPUT_DIR    Base directory for simulation output (default: ".").
                A per-run subdirectory (OUTPUT_DIR/<run_name>_run) is created.
  NODE          Pin job to a specific node (default: 2904)
  THREADS       Number of Julia threads / CPUs per task (default: 4)
  PROFILE       Set to "true" for nsys profiling. Also:
                  - disables the OMIP diagnostic output writers
                    (Average / JLD2 dumps / checkpoint / KE spectrum)
                    — they add per-iteration I/O and compute! overhead
                    that contaminates the trace;
                  - drops `pickup` from `run!` so the simulation always
                    starts from scratch.

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
  CATKE_CWUSTAR=5.0 ./launch.sh orca          # stronger surface TKE injection in CATKE
  BACKGROUND_K=3e-5 ./launch.sh orca          # uniform background κ = 3e-5 m^2/s (AMOC sensitivity)
  BACKGROUND_K=1e-4 ./launch.sh orca          # uniform background κ = 1e-4 m^2/s
  BACKGROUND_K=bryan_lewis ./launch.sh orca   # Bryan-Lewis depth profile, 3e-5 → 1.3e-4
  SKEW_FORMULATION=boundary_value ./launch.sh orca            # Ferrari et al. (2010) GM transport
  SKEW_FORMULATION=boundary_value KSKEW=nemo ./launch.sh orca # ... with the Treguier coefficient
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
    quarterdegree|quarter_degree)
        CONFIG="quarterdegree"
        ;;
    orca|twelfthdegree) ;;
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
        DEFAULT_KSKEW=250;  DEFAULT_KSYMM=100; NZ=70;  DEFAULT_DT="30minutes"; DEFAULT_DZ_TOP="2.0"
        DEFAULT_BIHARMONIC="40days"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 300 * 365days
run!(sim, pickup=:latest)"
        ;;
    quarterdegree)
        DEFAULT_KSKEW=0;    DEFAULT_KSYMM=0;   NZ=100; DEFAULT_DT="20minutes"; DEFAULT_DZ_TOP="1.5"
        DEFAULT_BIHARMONIC="nothing"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING="using Oceananigans.DistributedComputations"
        FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 300 * 365days
run!(sim, pickup =:latest)"
        ;;
    orca)
        DEFAULT_KSKEW=800;  DEFAULT_KSYMM=800; NZ=70;  DEFAULT_DT="30minutes"; DEFAULT_DZ_TOP="1.5"
        DEFAULT_BIHARMONIC="50days"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = 720day
run!(sim; pickup = :latest)

sim.stop_time = 300 * 365days
sim.Δt = 30minutes

run!(sim; pickup = :latest)
"
        ;;
    twelfthdegree)
        DEFAULT_KSKEW=0;    DEFAULT_KSYMM=0;   NZ=100; DEFAULT_DT="6minutes";  DEFAULT_DZ_TOP="1.5"
        DEFAULT_BIHARMONIC="nothing"; ARCH="Distributed(GPU(), partition=Partition(1, 4))"; GPUS_PER_NODE=4
        EXTRA_USING="using Oceananigans.DistributedComputations"
        FILE_SPLIT="file_splitting_interval = 180days,"
        RUN_CMD="sim.stop_time = 181days
run!(sim)

sim.Δt = 15minutes
sim.stop_time = 300 * 365days
run!(sim; pickup = true)"
        ;;
esac

# Profile mode: drop the `pickup=...` from `run!` so the simulation
# always starts from scratch. Resuming a checkpoint would skip the
# initialization phase the profiler needs to see, and the choice of
# checkpoint is irrelevant for kernel-cost measurements.
if [[ "${PROFILE:-false}" == "true" ]]; then
  RUN_CMD="sim.stop_iteration = 200; run!(sim)"
fi

# 0 means "no eddy closure" (maps to Julia `nothing`)
export KSKEW="${KSKEW:-$DEFAULT_KSKEW}"
export KSYMM="${KSYMM:-$DEFAULT_KSYMM}"
export DT="${DT:-$DEFAULT_DT}"
export DZ_TOP="${DZ_TOP:-$DEFAULT_DZ_TOP}"
export BIHARMONIC="${BIHARMONIC:-$DEFAULT_BIHARMONIC}"
KSKEW_JULIA="$KSKEW"; [[ "$KSKEW" == "0" ]] && KSKEW_JULIA="nothing"
KSYMM_JULIA="$KSYMM"; [[ "$KSYMM" == "0" ]] && KSYMM_JULIA="nothing"
[[ "$KSKEW" == "nemo" ]] && KSKEW_JULIA=":nemo"
[[ "$KSYMM" == "nemo" ]] && KSYMM_JULIA=":nemo"
[[ "$KSKEW" == "cesm" ]] && KSKEW_JULIA=":cesm"
[[ "$KSYMM" == "cesm" ]] && KSYMM_JULIA=":cesm"
[[ "$KSKEW" == "hybrid" ]] && KSKEW_JULIA=":hybrid"
[[ "$KSYMM" == "hybrid" ]] && KSYMM_JULIA=":hybrid"
export KSKEW_JULIA KSYMM_JULIA
export NZ DT ARCH EXTRA_USING FILE_SPLIT RUN_CMD

# ── Build run name from config + options ──────────────────────────────
RUN_NAME="$CONFIG"
[[ "${CORRECTED:-false}" == "true" ]]          && RUN_NAME="${RUN_NAME}_corrected"
[[ "${NCAR:-false}" == "true" ]]               && RUN_NAME="${RUN_NAME}_ncar"
[[ "${SNOW:-false}" == "true" ]]               && RUN_NAME="${RUN_NAME}_snow"
[[ "${ICE_DYNAMICS:-true}" == "false" ]]       && RUN_NAME="${RUN_NAME}_noicedyn"
[[ "${ICE_LATERAL:-no_slip}" == "no_slip" ]]   && RUN_NAME="${RUN_NAME}_noslip"
[[ "${ICE_BASAL:-true}" == "true" ]]           && RUN_NAME="${RUN_NAME}_landfast"
[[ "${ICE_DRAG:-5.5e-3}" != "3.24e-3" ]]       && RUN_NAME="${RUN_NAME}_cio${ICE_DRAG:-5.5e-3}"
[[ "${CLOSURE:-catke}" == "simple"   ]]        && RUN_NAME="${RUN_NAME}_simple"
[[ "${CLOSURE:-catke}" == "nori"     ]]        && RUN_NAME="${RUN_NAME}_nori"
[[ "${CLOSURE:-catke}" == "rbvd"     ]]        && RUN_NAME="${RUN_NAME}_rbvd"
[[ "${CLOSURE:-catke}" == "kpp"      ]]        && RUN_NAME="${RUN_NAME}_kpp"
[[ "${CLOSURE:-catke}" == "nemo_tke" ]]        && RUN_NAME="${RUN_NAME}_nemotke"
[[ "${WIND_VELOCITY:-false}" == "true" ]]      && RUN_NAME="${RUN_NAME}_wind"
[[ "${ML_TAPER:-false}" == "true" ]]           && RUN_NAME="${RUN_NAME}_mltaper"
[[ "${PARTIAL_CELLS:-false}" == "true" ]]      && RUN_NAME="${RUN_NAME}_pcells"
[[ -n "${BBL_KAPPA:-}" ]]                      && RUN_NAME="${RUN_NAME}_bbl${BBL_KAPPA}"
[[ -n "${BBL_GAMMA:-}" ]]                      && RUN_NAME="${RUN_NAME}_cg${BBL_GAMMA}"
[[ -n "${OVERFLOW_RESTORE:-}" ]]               && RUN_NAME="${RUN_NAME}_dsow${OVERFLOW_RESTORE}"
[[ "${NORMALIZE_SALINITY:-true}" == "false" ]] && RUN_NAME="${RUN_NAME}_rawsalt"
[[ "${RESTORING_UNDER_ICE:-true}" == "false" ]] && RUN_NAME="${RUN_NAME}_noicerest"
case "${NORMALIZE_FRESHWATER:-none}" in
  true|timestep) RUN_NAME="${RUN_NAME}_fwnorm" ;;
  annual)        RUN_NAME="${RUN_NAME}_fwnormann" ;;
esac
[[ "${SKEW_FORMULATION:-diffusive}" == "advective" ]]      && RUN_NAME="${RUN_NAME}_gmadv"
[[ "${SKEW_FORMULATION:-diffusive}" == "boundary_value" ]] && RUN_NAME="${RUN_NAME}_gmbvp"
[[ -n "${BVP_MODE:-}" ]]                       && RUN_NAME="${RUN_NAME}_bvpm${BVP_MODE}"
[[ -n "${BVP_CMIN:-}" ]]                       && RUN_NAME="${RUN_NAME}_bvpc${BVP_CMIN}"
[[ -n "${CB:-}" ]]                             && RUN_NAME="${RUN_NAME}_cb${CB}"
[[ "$KSKEW" != "$DEFAULT_KSKEW" ]]             && RUN_NAME="${RUN_NAME}_kskew${KSKEW}"
[[ "$KSYMM" != "$DEFAULT_KSYMM" ]]             && RUN_NAME="${RUN_NAME}_ksymm${KSYMM}"
[[ "$BIHARMONIC" != "$DEFAULT_BIHARMONIC" ]]   && RUN_NAME="${RUN_NAME}_bih${BIHARMONIC}"
[[ -n "${BIHVISC:-}" ]]                        && RUN_NAME="${RUN_NAME}_bihvisc${BIHVISC}"
[[ "$DZ_TOP" != "$DEFAULT_DZ_TOP" ]]           && RUN_NAME="${RUN_NAME}_dz${DZ_TOP}"
[[ -n "${CATKE_CWUSTAR:-}" ]]                  && RUN_NAME="${RUN_NAME}_cwu${CATKE_CWUSTAR}"
[[ -n "${BACKGROUND_K:-}" ]]                   && RUN_NAME="${RUN_NAME}_bgk${BACKGROUND_K}"
[[ -n "${BACKGROUND_NU:-}" ]]                  && RUN_NAME="${RUN_NAME}_bgnu${BACKGROUND_NU}"
[[ -n "${PVEL:-}" ]]                           && RUN_NAME="${RUN_NAME}_pvel${PVEL}"

REPORT_NAME="${REPORT_NAME:-${RUN_NAME}_report}"
JOB_NAME="${JOB_NAME:-$RUN_NAME}"

SBATCH_ARGS=()
PARTITION="${PARTITION:-pi_raffaele}"

NODE="${NODE:-2904}"
if [[ "${PARTITION}" != "default" && -n "${NODE}" ]]; then
    SBATCH_ARGS+=(-w "node${NODE}")
fi
SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}")
# Override the heredoc default `--ntasks-per-node=1` so distributed configs
# (twelfthdegree: 1×4 partition) actually launch GPUS_PER_NODE MPI ranks.
SBATCH_ARGS+=(--ntasks-per-node="${GPUS_PER_NODE}")

export THREADS="${THREADS:-8}"
SBATCH_ARGS+=(--cpus-per-task="${THREADS}")

if [[ "${PARTITION}" != "default" ]]; then
    SBATCH_ARGS+=(--partition="${PARTITION}")
fi

if [[ "${PARTITION}" == "default" ]]; then
    TIME="${TIME:-05:00:00}"
else
    TIME="${TIME:-120:00:00}"
fi
SBATCH_ARGS+=(--time="${TIME}")

MEM="${MEM:-150GB}"
SBATCH_ARGS+=(--mem="${MEM}")

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

source /etc/profile.d/modules.sh
module load nvhpc

# nvhpc only puts CUDA/NCCL/NVSHMEM on LD_LIBRARY_PATH — not HPC-X's OpenMPI
# tree. Export it ourselves so libmpi.so's dlopen of its UCX/PMIx/UCC
# neighbours resolves. Must match the libmpi path baked into MPIPreferences.
# (UCX ships under both ucx/lib and ucx/prof/lib; include both.)
HPCX_DIR="/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1/comm_libs/13.1/hpcx/hpcx-2.25.1"
export LD_LIBRARY_PATH="$HPCX_DIR/ompi/lib:$HPCX_DIR/ucx/lib:$HPCX_DIR/ucx/prof/lib:$HPCX_DIR/ucc/lib:${LD_LIBRARY_PATH:-}"

# Activate HPC-X so mpirun and friends are on PATH (hpcx_load only adjusts
# PATH on this build — relocation is handled by mpirun for its children).
source "$HPCX_DIR/hpcx-init-ompi.sh"
hpcx_load

# CUDA-aware MPI knobs (HPC-X / UCX).
export OMPI_MCA_opal_cuda_support=1
export UCX_TLS=cuda_copy,cuda_ipc,rc,sm,self
export UCX_MEMTYPE_CACHE=n          # avoids known UCX+CUDA memtype-cache bug
# Disable UCX's CUDA IPC handle cache. With JULIA_CUDA_MEMORY_POOL=none, CUDA
# recycles freed virtual addresses quickly; UCX's IPC cache then serves a
# stale handle and cuIpcOpenMemHandle() fails with "resource already mapped"
# on the importing rank a few minutes into the run (cuda_ipc_cache.c:212).
export UCX_CUDA_IPC_CACHE=n

# Disable CUDA's stream-ordered memory pool. With the pool, allocations come
# from a per-stream cache whose layout depends on prior call order, so two
# runs from the same checkpoint can land kernels on differently-aligned
# memory and pick different FMA paths → last-bit drift that compounds in
# unstable regions. `none` falls back to plain cudaMalloc/Free, which gives
# bit-reproducible kernel inputs at the cost of allocator overhead.
export JULIA_CUDA_MEMORY_POOL=none

JULIA="${JULIA:-$HOME/julia-1.12.5/bin/julia}"

# ── Shared environment ────────────────────────────────────────────────
FORCING_DIR="${FORCING_DIR:-${DATA}forcing_data}"
STAGING_DIR="${STAGING_DIR:-./staged_data}"
CB="${CB:-}"
BIHVISC="${BIHVISC:-}"
DZ_TOP="${DZ_TOP:-}"
CATKE_CWUSTAR="${CATKE_CWUSTAR:-}"
BACKGROUND_K="${BACKGROUND_K:-}"
BACKGROUND_NU="${BACKGROUND_NU:-}"
BACKEND_SIZE="${BACKEND_SIZE:-}"
NCAR="${NCAR:-false}"
CORRECTED="${CORRECTED:-false}"
SNOW="${SNOW:-false}"
ICE_DYNAMICS="${ICE_DYNAMICS:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

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

# A named profile is passed as a Julia Symbol, a number verbatim.
BACKGROUND_K_KWARG=""
case "$BACKGROUND_K" in
    "")                     ;;
    henyey|bryan_lewis)     BACKGROUND_K_KWARG="background_vertical_diffusivity = :${BACKGROUND_K}," ;;
    *)                      BACKGROUND_K_KWARG="background_vertical_diffusivity = ${BACKGROUND_K}," ;;
esac

BACKGROUND_NU_KWARG=""
[[ -n "$BACKGROUND_NU" ]] && BACKGROUND_NU_KWARG="background_vertical_viscosity = ${BACKGROUND_NU},"

# Pass the value explicitly (default true = conservative restoring) so the Julia-side default
# never silently overrides a "false" request.
NORMALIZE_SALINITY="${NORMALIZE_SALINITY:-true}"
case "$NORMALIZE_SALINITY" in
    true|false) ;;
    *) echo "NORMALIZE_SALINITY must be 'true' or 'false', got '$NORMALIZE_SALINITY'" >&2; exit 1 ;;
esac
NORMALIZE_SALINITY_KWARG="normalize_salinity = ${NORMALIZE_SALINITY},"

RESTORING_UNDER_ICE="${RESTORING_UNDER_ICE:-true}"
case "$RESTORING_UNDER_ICE" in
    true|false) ;;
    *) echo "RESTORING_UNDER_ICE must be 'true' or 'false', got '$RESTORING_UNDER_ICE'" >&2; exit 1 ;;
esac
RESTORING_UNDER_ICE_KWARG=""
[[ "$RESTORING_UNDER_ICE" == "false" ]] && RESTORING_UNDER_ICE_KWARG="restoring_under_sea_ice = false,"

NORMALIZE_FRESHWATER="${NORMALIZE_FRESHWATER:-none}"
case "$NORMALIZE_FRESHWATER" in
    none|false)    NORMALIZE_FRESHWATER_JULIA=":none" ;;
    timestep|true) NORMALIZE_FRESHWATER_JULIA=":timestep" ;;
    annual)        NORMALIZE_FRESHWATER_JULIA=":annual" ;;
    *) echo "NORMALIZE_FRESHWATER must be none|timestep|annual, got '$NORMALIZE_FRESHWATER'" >&2; exit 1 ;;
esac
NORMALIZE_FRESHWATER_KWARG="normalize_freshwater = ${NORMALIZE_FRESHWATER_JULIA},"

RIVER_KWARG=""
if [[ -n "${RIVER_SPREAD:-}" ]]; then
    RIVER_SPREAD_JULIA="$RIVER_SPREAD"
    [[ "$RIVER_SPREAD" == "none" ]] && RIVER_SPREAD_JULIA="nothing"
    RIVER_KWARG="${RIVER_KWARG}river_spread_radius = ${RIVER_SPREAD_JULIA},"
fi
[[ -n "${RIVER_SPREAD_CELLS:-}" ]]  && RIVER_KWARG="${RIVER_KWARG}river_spread_cells = ${RIVER_SPREAD_CELLS},"
[[ -n "${RIVER_MIXING_K:-}" ]]      && RIVER_KWARG="${RIVER_KWARG}river_mixing_κ = ${RIVER_MIXING_K},"
[[ -n "${RIVER_MIXING_DEPTH:-}" ]]  && RIVER_KWARG="${RIVER_KWARG}river_mixing_depth = ${RIVER_MIXING_DEPTH},"
[[ "${RIVER_MIXING:-true}" == "false" ]] && RIVER_KWARG="${RIVER_KWARG}river_mixing = false,"

SKEW_FORMULATION="${SKEW_FORMULATION:-diffusive}"
case "$SKEW_FORMULATION" in
    diffusive|advective|boundary_value) ;;
    *) echo "SKEW_FORMULATION must be diffusive|advective|boundary_value, got '$SKEW_FORMULATION'" >&2; exit 1 ;;
esac
SKEW_FORMULATION_KWARG="skew_flux_formulation = :${SKEW_FORMULATION},"

BVP_KWARG=""
[[ -n "${BVP_MODE:-}" ]] && BVP_KWARG="${BVP_KWARG}boundary_value_mode_number = ${BVP_MODE},"
[[ -n "${BVP_CMIN:-}" ]] && BVP_KWARG="${BVP_KWARG}boundary_value_minimum_speed = ${BVP_CMIN},"

BACKEND_KWARG=""
[[ -n "$BACKEND_SIZE" ]] && BACKEND_KWARG="backend_size = ${BACKEND_SIZE},"

FLUX_KWARG=""
[[ "$NCAR" == "true" ]]        && FLUX_KWARG="flux_configuration = :ncar,"
[[ "$CORRECTED" == "true" ]]   && FLUX_KWARG="flux_configuration = :corrected,"

PVELKWARG=""
[[ -n "$PVEL" ]] && PVELKWARG="piston_velocity = ${PVEL},"

CLOSURE_KWARG=""
[[ "${CLOSURE:-catke}" == "simple"   ]] && CLOSURE_KWARG="vertical_closure = :simple,"
[[ "${CLOSURE:-catke}" == "nori"     ]] && CLOSURE_KWARG="vertical_closure = :nori,"
[[ "${CLOSURE:-catke}" == "rbvd"     ]] && CLOSURE_KWARG="vertical_closure = :rbvd,"
[[ "${CLOSURE:-catke}" == "kpp"      ]] && CLOSURE_KWARG="vertical_closure = :kpp,"
[[ "${CLOSURE:-catke}" == "nemo_tke" ]] && CLOSURE_KWARG="vertical_closure = :nemo_tke,"

VELOCITY_KWARG=""
[[ "${WIND_VELOCITY:-false}" == "true" ]] && VELOCITY_KWARG="velocity_formulation = :wind,"

ML_TAPER_KWARG=""
[[ "${ML_TAPER:-false}" == "true" ]] && ML_TAPER_KWARG="mixed_layer_tapering = true,"

PARTIAL_CELLS_KWARG=""
[[ "${PARTIAL_CELLS:-false}" == "true" ]] && PARTIAL_CELLS_KWARG="partial_cell_bathymetry = true,"

BBL_KWARG=""
[[ -n "${BBL_KAPPA:-}" ]] && BBL_KWARG="bbl_diffusivity = ${BBL_KAPPA},"
[[ -n "${BBL_GAMMA:-}" ]] && BBL_KWARG="${BBL_KWARG}bbl_transport_coefficient = ${BBL_GAMMA},"
[[ -n "${OVERFLOW_RESTORE:-}" ]] && BBL_KWARG="${BBL_KWARG}overflow_restoring_timescale = ${OVERFLOW_RESTORE}days,"

SNOW_KWARG=""
[[ "$SNOW" == "true" ]] && SNOW_KWARG="with_snow = true,"

ICE_DYNAMICS_KWARG=""
[[ "$ICE_DYNAMICS" == "false" ]] && ICE_DYNAMICS_KWARG="with_ice_dynamics = false,"

ICE_LATERAL="${ICE_LATERAL:-no_slip}"
ICE_BASAL="${ICE_BASAL:-true}"
ICE_DRAG="${ICE_DRAG:-5.5e-3}"
SEA_ICE_KWARG="sea_ice_lateral_boundary_condition = :${ICE_LATERAL},"
SEA_ICE_KWARG="${SEA_ICE_KWARG}sea_ice_ocean_drag_coefficient = ${ICE_DRAG},"
[[ "$ICE_BASAL" == "false" ]] && SEA_ICE_KWARG="${SEA_ICE_KWARG}with_landfast_basal_stress = false,"

# Profile runs disable the OMIP diagnostic output writers (Average,
# JLD2 dumps, checkpoint, KE spectrum). They add per-iteration I/O and
# `compute!` overhead that pollutes nsys traces. The simulation core
# is unchanged, so kernel timings reflect pure stepping cost.
DIAGNOSTICS_KWARG=""
[[ "${PROFILE:-false}" == "true" ]] && DIAGNOSTICS_KWARG="diagnostics = false,"

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
                      ${ML_TAPER_KWARG}
                      ${PARTIAL_CELLS_KWARG}
                      ${BBL_KWARG}
                      ${SNOW_KWARG}
                      ${ICE_DYNAMICS_KWARG}
                      ${SEA_ICE_KWARG}
                      ${DIAGNOSTICS_KWARG}
                      ${PVELKWARG}
                      ${CATKE_CWUSTAR_KWARG}
                      ${BACKGROUND_K_KWARG}
                      ${BACKGROUND_NU_KWARG}
                      ${NORMALIZE_SALINITY_KWARG}
                      ${RESTORING_UNDER_ICE_KWARG}
                      ${NORMALIZE_FRESHWATER_KWARG}
                      ${RIVER_KWARG}
                      ${SKEW_FORMULATION_KWARG}
                      ${BVP_KWARG}
                      Δt = ${DT},
                      forcing_dir = \"${FORCING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      ${FILE_SPLIT}
                      output_dir = \"${OUTPUT_DIR}/${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\")

${RUN_CMD}"

THREADS="${THREADS:-8}"

# Launch via HPC-X's mpirun, NOT srun. SLURM 25.05 on this cluster has only
# pmi2/cray_shasta MPI plugins (no PMIx), and HPC-X's OpenMPI 4.1.9a1 ships
# only the pmix3x client (no native PMI-1/PMI-2 wire support). srun and HPC-X
# therefore cannot handshake. mpirun runs its own PMIx-3 server, queries SLURM
# only for the allocation (hostlist + nranks), and spawns ranks itself.
# --bind-to none avoids fighting SLURM's cgroup over CPU affinity.
NRANKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
if [[ "${PROFILE:-false}" == "true" ]]; then
    echo "Profiling ${RUN_NAME} -> ${REPORT_NAME}"
    mpirun -np "$NRANKS" --bind-to none \
                      nsys profile --trace=cuda \
                      --output="$REPORT_NAME" \
                      --force-overwrite true \
                      "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR"
else
    mpirun -np "$NRANKS" --bind-to none \
        "$JULIA" --project=.. --check-bounds=no -t "${THREADS}" -e "$JULIA_EXPR"
fi
EOF
