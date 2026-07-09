#####
##### Gap-filling for cyclic (seasonal-climatology) FieldTimeSeries.
#####
##### Satellite LAI carries missing data — mainly cloud/snow-contaminated retrievals
##### removed by quality screening, heaviest in winter. Because the series is a cyclic
##### seasonal climatology and LAI varies smoothly through the year, missing periods are
##### filled per pixel by cyclic linear interpolation between that pixel's valid periods
##### (temporal interpolation, after Borak & Jasinski 2009). Pixels with no valid period
##### all year are left missing. Multi-year compositing is the complementary production
##### path (a period cloudy one year is usually clear another).
#####

using Oceananigans.Fields: interior
using Oceananigans.OutputReaders: FieldTimeSeries

"""
$(TYPEDSIGNATURES)

Cyclically fill the missing (non-finite) entries of the vector `column` in place by
linear interpolation between the nearest valid entries on either side, wrapping around
the ends. A column with no valid entry is left untouched.
"""
function fill_cyclic!(column)
    n = length(column)
    reference = copy(column)
    any(isfinite, reference) || return column
    for i in 1:n
        isfinite(reference[i]) && continue
        forward = 1
        while !isfinite(reference[mod1(i + forward, n)]); forward += 1; end
        backward = 1
        while !isfinite(reference[mod1(i - backward, n)]); backward += 1; end
        vf = reference[mod1(i + forward, n)]
        vb = reference[mod1(i - backward, n)]
        column[i] = (vb * forward + vf * backward) / (forward + backward)
    end
    return column
end

"""
$(TYPEDSIGNATURES)

Fill missing periods of a cyclic `FieldTimeSeries` in place, per pixel, by cyclic linear
interpolation in time (see [`fill_cyclic!`](@ref)). Returns the fraction of cells that
were missing before filling (reported, never silently dropped).
"""
function fill_temporal_gaps!(fts::FieldTimeSeries)
    Nx, Ny, _, Nt = size(fts)
    slices = [interior(fts[n], :, :, 1) for n in 1:Nt]
    missing_before = 0
    total = Nx * Ny * Nt
    column = zeros(eltype(fts), Nt)
    for j in 1:Ny, i in 1:Nx
        @inbounds for n in 1:Nt
            column[n] = slices[n][i, j]
            missing_before += !isfinite(column[n])
        end
        fill_cyclic!(column)
        @inbounds for n in 1:Nt
            slices[n][i, j] = column[n]
        end
    end
    return missing_before / total
end
