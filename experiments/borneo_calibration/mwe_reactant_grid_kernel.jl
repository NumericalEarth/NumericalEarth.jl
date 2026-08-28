# The blend kernel's exact shape in isolation: an Oceananigans grid argument supplies the
# float type, a scalar argument is clamped with it, and fields are blended. Run on the
# Reactant GPU backend to see whether the grid argument is what breaks CUDA codegen.

module GridKernelMWE

using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!
using Reactant, CUDA

Reactant.set_default_backend("gpu")

@kernel function clamp_with_grid_eltype!(a, x, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    @inbounds a[i, j, 1] = clamp(convert(FT, x), zero(FT), one(FT))
end

@kernel function clamp_without_grid!(a, x)
    i, j = @index(Global, NTuple)
    @inbounds a[i, j, 1] = clamp(x, 0.0, 1.0)
end

@kernel function blend_fields!(a, b, c, x, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    f = clamp(convert(FT, x), zero(FT), one(FT))
    g = one(FT) - f
    @inbounds a[i, j, 1] = f * b[i, j, 1] + g * c[i, j, 1]
end

@kernel function blend_fields_literal!(a, b, c, x, grid)
    i, j = @index(Global, NTuple)
    f = clamp(x, 0, 1)
    g = 1 - f
    @inbounds a[i, j, 1] = f * b[i, j, 1] + g * c[i, j, 1]
end

@kernel function blend_fields_field_eltype!(a, b, c, x, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(a)
    f = clamp(convert(FT, x), zero(FT), one(FT))
    g = one(FT) - f
    @inbounds a[i, j, 1] = f * b[i, j, 1] + g * c[i, j, 1]
end

grid = LatitudeLongitudeGrid(ReactantState(); size = (4, 4), latitude = (0, 1), longitude = (0, 1), topology = (Bounded, Bounded, Flat))
a = Field{Center, Center, Nothing}(grid)
b = Field{Center, Center, Nothing}(grid); parent(b) .= 2
c = Field{Center, Center, Nothing}(grid); parent(c) .= 3

run_without_grid!(a, x, grid) = (launch!(architecture(grid), grid, :xy, clamp_without_grid!, a, x); nothing)
run_with_grid!(a, x, grid) = (launch!(architecture(grid), grid, :xy, clamp_with_grid_eltype!, a, x, grid); nothing)
run_blend!(a, b, c, x, grid) = (launch!(architecture(grid), grid, :xy, blend_fields!, a, b, c, x, grid); nothing)
run_blend_literal!(a, b, c, x, grid) = (launch!(architecture(grid), grid, :xy, blend_fields_literal!, a, b, c, x, grid); nothing)
run_blend_field_eltype!(a, b, c, x, grid) = (launch!(architecture(grid), grid, :xy, blend_fields_field_eltype!, a, b, c, x, grid); nothing)

for (label, f, args) in (("clamp scalar, no grid argument", run_without_grid!, (a, 1.7, grid)),
                         ("clamp with eltype(grid)", run_with_grid!, (a, 1.7, grid)),
                         ("blend fields with eltype(grid)", run_blend!, (a, b, c, 0.7, grid)),
                         ("blend fields, literal bounds, no convert", run_blend_literal!, (a, b, c, 0.7, grid)),
                         ("blend fields with eltype(field)", run_blend_field_eltype!, (a, b, c, 0.7, grid)))
    try
        Reactant.@jit f(args...)
        @info "$label: ok, a[1,1] = $(Array(interior(a))[1, 1, 1])"
    catch err
        io = IOBuffer(); showerror(io, err); msg = first(split(String(take!(io)), '\n'))
        @error "$label: failed — $msg"
    end
end

end # module
