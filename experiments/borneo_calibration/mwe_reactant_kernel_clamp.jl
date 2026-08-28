# Pure Reactant + KernelAbstractions reproduction: a kernel that `clamp`s a scalar Float64
# argument fails to compile on the Reactant GPU backend (dynamic `promote_rule` in
# ReactantCUDAExt), while the same kernel without the scalar argument compiles.

using Reactant, CUDA, KernelAbstractions

Reactant.set_default_backend("gpu")

@kernel function clamp_scalar!(a, x)
    i = @index(Global)
    @inbounds a[i] = clamp(x, 0.0, 1.0)
end

@kernel function clamp_array!(a, b)
    i = @index(Global)
    @inbounds a[i] = clamp(b[i], 0.0, 1.0)
end

@kernel function copy_scalar!(a, x)
    i = @index(Global)
    @inbounds a[i] = x
end

function run_kernel!(kernel!, a, arg)
    kernel!(KernelAbstractions.get_backend(a), 8)(a, arg; ndrange = 8)
    return a
end

a = Reactant.to_rarray(zeros(8))
b = Reactant.to_rarray(fill(1.7, 8))

for (label, kernel!, arg) in (("copy scalar argument", copy_scalar!, 1.7),
                              ("clamp array element", clamp_array!, b),
                              ("clamp scalar argument", clamp_scalar!, 1.7))
    try
        out = Reactant.@jit run_kernel!(kernel!, a, arg)
        @info "$label: ok, a[1] = $(Array(out)[1])"
    catch err
        @error "$label: failed" exception = (err, catch_backtrace())
    end
end
