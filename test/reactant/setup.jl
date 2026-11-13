# Shared by the test/reactant files, each of which includes it so that it also runs alone: the Reactant backend
# follows GPU_TEST.

using Reactant: Reactant
using CUDA

gpu_test = get(ENV, "GPU_TEST", "false") == "true"
Reactant.set_default_backend(gpu_test ? "gpu" : "cpu")
