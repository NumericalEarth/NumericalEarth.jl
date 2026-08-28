# Which ingredient of `_blend_tiled_land_fluxes!` breaks CUDA codegen under Reactant?
# Kernel variants on a scalar argument and on a struct-carried scalar.

using Reactant, CUDA, KernelAbstractions

Reactant.set_default_backend("gpu")

@kernel function clamp_float_bounds!(a, x)
    i = @index(Global)
    @inbounds a[i] = clamp(x, 0.0, 1.0)
end

@kernel function clamp_int_bounds!(a, x)
    i = @index(Global)
    @inbounds a[i] = clamp(x, 0, 1)
end

@kernel function convert_then_clamp!(a, x)
    i = @index(Global)
    @inbounds a[i] = clamp(convert(Float64, x), 0.0, 1.0)
end

@kernel function struct_scalar_clamp!(a, p)
    i = @index(Global)
    @inbounds a[i] = clamp(p.fraction, 0.0, 1.0)
end

@kernel function struct_eltype_clamp!(a, p)
    i = @index(Global)
    FT = typeof(p.fraction)
    @inbounds a[i] = clamp(convert(FT, p.fraction), zero(FT), one(FT))
end

@kernel function blend!(a, b, x)
    i = @index(Global)
    f = clamp(x, 0.0, 1.0)
    g = 1 - f
    @inbounds a[i] = f * b[i] + g * b[i]
end

@kernel function minmax!(a, x)
    i = @index(Global)
    @inbounds a[i] = min(max(x, 0.0), 1.0)
end

@kernel function ifelse_clamp!(a, x)
    i = @index(Global)
    f = ifelse(x > 1, one(x), ifelse(x < 0, zero(x), x))
    @inbounds a[i] = f
end

function run2!(kernel!, a, arg)
    kernel!(KernelAbstractions.get_backend(a), 8)(a, arg; ndrange = 8)
    return a
end
function run3!(kernel!, a, b, arg)
    kernel!(KernelAbstractions.get_backend(a), 8)(a, b, arg; ndrange = 8)
    return a
end

a = Reactant.to_rarray(zeros(8))
b = Reactant.to_rarray(fill(2.0, 8))

for (label, kernel!, arg) in (("clamp, Float64 bounds", clamp_float_bounds!, 1.7),
                              ("clamp, Int bounds", clamp_int_bounds!, 1.7),
                              ("convert then clamp", convert_then_clamp!, 1.7),
                              ("struct scalar clamp", struct_scalar_clamp!, (; fraction = 1.7)),
                              ("struct eltype clamp", struct_eltype_clamp!, (; fraction = 1.7)),
                              ("min/max", minmax!, 1.7),
                              ("ifelse clamp", ifelse_clamp!, 1.7))
    try
        out = Reactant.@jit run2!(kernel!, a, arg)
        @info "$label: ok, a[1] = $(Array(out)[1])"
    catch err
        io = IOBuffer(); showerror(io, err); msg = first(split(String(take!(io)), '\n'))
        @error "$label: failed — $msg"
    end
end
try
    out = Reactant.@jit run3!(blend!, a, b, 1.7)
    @info "blend: ok, a[1] = $(Array(out)[1])"
catch err
    io = IOBuffer(); showerror(io, err); msg = first(split(String(take!(io)), '\n'))
    @error "blend: failed — $msg"
end
