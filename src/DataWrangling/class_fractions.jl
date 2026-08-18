#####
##### Per-class area fractions of a categorical (class-code) map
#####

"""
    class_fraction(codes, class)

The fraction of the valid entries of `codes` that carry `class`. Averaging class codes is
meaningless, but their per-class *fractions* are continuous fields that ride the shared
bilinear regrid onto a model grid safely, and over valid cells they sum to one.

```jldoctest
julia> using NumericalEarth

julia> codes = Float32[1 1 4; 4 NaN 1];

julia> class_fraction(codes, 1), class_fraction(codes, 4)
(0.6f0, 0.4f0)
```
"""
function class_fraction(codes, class)
    valid = count(isfinite, codes)
    valid == 0 && return NaN32
    return Float32(count(code -> isfinite(code) && code == class, codes) / valid)
end

"""
    class_fractions(codes, classes, factor)

Aggregate a class map onto a lattice `factor` times coarser, as one continuous area-fraction
field per class: `Dict(class => fraction)`, each `size(codes) .÷ factor`, summing to one over
cells with any valid code. `factor` must divide both dimensions, which it does when the
coarse lattice is built by grouping whole native cells.
"""
function class_fractions(codes, classes, factor)
    Nx, Ny = size(codes)
    (mod(Nx, factor) == 0 && mod(Ny, factor) == 0) ||
        throw(ArgumentError("An aggregation factor of $factor does not divide the class " *
                            "map's $((Nx, Ny)) cells."))

    fractions = Dict(class => fill(NaN32, Nx ÷ factor, Ny ÷ factor) for class in classes)

    for j in 1:(Ny ÷ factor), i in 1:(Nx ÷ factor)
        block = view(codes, ((i - 1) * factor + 1):(i * factor),
                            ((j - 1) * factor + 1):(j * factor))
        for class in classes
            fractions[class][i, j] = class_fraction(block, class)
        end
    end

    return fractions
end
