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

- **Check GPU topology before touching MPI** (`nvidia-smi topo -m`). If the GPUs share a
  node, NCCL over NVLink is the natural device-buffer transport and MPI only needs to
  launch ranks and carry host-side control traffic — a cluster with no CUDA-aware MPI
  (no UCX, device-pointer `MPI.Isend` segfaults) is NOT a blocker for single-node
  multi-GPU.
- Bootstrap NCCL from host MPI: rank 0 creates the NCCL unique ID and broadcasts it;
  each rank binds its CUDA device, then builds the NCCL communicator.
- NCCL has no message tags: point-to-point ops match by issue order per peer, so the
  send/recv issue order must be identical on both ranks.
- **Always wrap a halo exchange's sends and receives in `ncclGroupStart`/`ncclGroupEnd`**:
  ungrouped send/recv on a single stream deadlocks (the recv kernel blocks the stream
  head in front of the matching send).
- Enqueue NCCL ops on the same CUDA stream as compute so downstream copy-back kernels
  order after the communication with no host synchronization.
- Only reach for CUDA-aware MPI (UCX-OpenMPI/HPC-X) when the job spans nodes.
