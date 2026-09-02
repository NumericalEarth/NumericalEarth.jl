# Running on Slurm/GPU Clusters

Hard-won operational rules for running NumericalEarth simulations on Slurm clusters with
NFS home directories and NVIDIA GPUs. Every rule below was learned from a real failure.

## Precompilation

- **Precompile serially BEFORE `mpiexec`, never inside it.** Ranks precompiling
  concurrently race the shared depot's pidfiles: duplicated work, long silent stalls.
  Batch scripts run `julia --project -e 'using Pkg; Pkg.precompile()'` as a first job
  step, then launch ranks.
- **Precompile on the partition you will run on.** Julia's cache is per CPU
  microarchitecture; a login node with a different CPU family than the compute nodes
  produces caches the job cannot reuse, and the job silently recompiles everything.
  Verify precompilation with `srun` on a compute node, not on the login node.
- **Size Julia to the allocation, not the node.** Export
  `JULIA_NUM_PRECOMPILE_TASKS=$SLURM_CPUS_PER_TASK` and
  `OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK`; the defaults assume the whole node and
  thrash the cgroup.
- **Prefer `GPUCompiler` disk caching over sysimages during development.** Set
  `disk_cache = "true"` for GPUCompiler in `LocalPreferences.toml` — it survives code
  edits. A sysimage silently pins stale package versions; if you build one, guard its
  use with a hash of `Manifest.toml` + `LocalPreferences.toml`, and expect break-even
  only after ~6 job launches per stack change.

## Judging whether a job is alive

- **Never kill a job because its log went quiet.** Slurm block-buffers stdout in ~32 KB
  chunks and NFS serves stale file attributes; a healthy job can look frozen for tens of
  minutes. Check GPU utilization (`srun --overlap --jobid=<id> nvidia-smi`), process RSS,
  and output-file mtimes before concluding anything.
- A run that dies instantly with no log output often means the script path is not
  visible from compute nodes (e.g. node-local `/tmp`) — keep scripts and outputs on the
  shared filesystem.

## Simulation scripts

- **Keep plotting packages out of simulation scripts.** Loading Makie costs minutes per
  job and buys nothing on a headless compute node; write slices to JLD2 and render in a
  separate analysis script that a CPU job (or the login node) can run after the fact.
- Log progress with wall-clock instrumentation (seconds per iteration, simulated days
  per day) so throughput regressions are visible from the log alone.

## Multi-GPU without CUDA-aware MPI

- **Grep for the technology by name before declaring it unsupported** — search the whole
  installed package including `ext/`, and check Project.toml `[weakdeps]`/`[extensions]`.
  Oceananigans ships NCCL support as the `OceananigansNCCLExt` extension; it is invisible
  to a grep that only covers `src/`.
- On a single node with NVLink, `NCCLDistributed(GPU(); partition = Partition(N))`
  (drop-in for `Distributed`, activated by `using NCCL, CUDA`) moves device halo buffers
  over NCCL while host MPI handles launch, bootstrap, and scalar reductions — a cluster
  with no CUDA-aware MPI is NOT a blocker for single-node multi-GPU.
- Validate transports with `fill_halo_regions!` on a distributed `Field` (rank-constant
  interior, assert halos equal the neighbor's constant) — a meaningful end-to-end test,
  unlike raw send/recv smoke tests.
- Only reach for CUDA-aware MPI (UCX-OpenMPI/HPC-X) when the job spans nodes.
