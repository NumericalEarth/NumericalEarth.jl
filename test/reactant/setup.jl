# Shared by the test/reactant files, each of which includes it so that it also runs alone: the Reactant backend
# follows GPU_TEST.

using Reactant: Reactant
using CUDA

gpu_test = get(ENV, "GPU_TEST", "false") == "true"
Reactant.set_default_backend(gpu_test ? "gpu" : "cpu")

# The largest resident memory of this process so far, for the progress messages of the long compiles
peak_memory() = Base.format_bytes(Sys.maxrss())

# Runs `expr` on a task with a larger stack (16 MiB by default), as Breeze's Reactant tests do: tracing a large
# model can overflow the default stack, often only on CI machines (JuliaLang/julia#54998).
macro with_stack_size(stack_size, expr)
    return quote
        local task = Task(() -> $(esc(expr)), $(esc(stack_size)))
        schedule(task)
        fetch(task)
    end
end

macro with_stack_size(expr)
    return :(@with_stack_size 16 << 20 $(esc(expr)))
end
