# Pure Reactant + KernelAbstractions: a kernel argument that is a parametric struct holding a
# Float64 and an array. Inside the kernel the struct's type parameter is used as the float
# type for literals. Does the parameter survive Reactant's CUDA kernel argument conversion?

module StructParamMWE

using KernelAbstractions: @kernel, @index, get_backend
using Oceananigans: Adapt
using Reactant, CUDA

Reactant.set_default_backend("gpu")

struct Parameters{FT, A}
    Δ :: FT
    x :: A
end

Adapt.@adapt_structure Parameters
Base.eltype(::Parameters{FT}) where FT = FT

@kernel function clamp_with_param_eltype!(a, s, p)
    i = @index(Global)
    FT = eltype(p)
    @inbounds a[i] = clamp(convert(FT, s), zero(FT), one(FT)) + p.Δ
end

@kernel function clamp_with_param_scalar_only!(a, s, p)
    i = @index(Global)
    FT = typeof(p.Δ)
    @inbounds a[i] = clamp(convert(FT, s), zero(FT), one(FT)) + p.Δ
end

@kernel function report_param_type!(a, p)
    i = @index(Global)
    @inbounds a[i] = eltype(p) === Float64 ? 1.0 : 2.0
end

run!(kernel!, a, args...) = (kernel!(get_backend(a), 8)(a, args...; ndrange = 8); nothing)

a = Reactant.to_rarray(zeros(8))
p = Parameters(0.25, Reactant.to_rarray(ones(8)))

for (label, kernel!, args) in (("literals typed by the struct parameter", clamp_with_param_eltype!, (1.7, p)),
                               ("literals typed by the scalar field", clamp_with_param_scalar_only!, (1.7, p)),
                               ("struct parameter is Float64 in the kernel", report_param_type!, (p,)))
    try
        Reactant.@jit run!(kernel!, a, args...)
        @info "$label: ok, a[1] = $(Array(a)[1])"
    catch err
        io = IOBuffer(); showerror(io, err); msg = first(split(String(take!(io)), '\n'))
        @error "$label: failed — $msg"
    end
end

end # module
