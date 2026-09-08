using Oceananigans.Grids: znode
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity

@inline river_mouth_κ(i, j, k, grid, clock, fields, mask) = @inbounds mask[i, j, k]

"""
    $(TYPEDSIGNATURES)

Add vertical tracer mixing below all routed river and iceberg receiving cells.
The extra diffusivity is `κ` (m²/s) at cell centers above `-mixing_depth` (m),
and zero elsewhere. The fixed mask does not depend on the current discharge.
Overlapping receiving areas retain the same diffusivity.
"""
function river_mouth_vertical_diffusivity(grid, river_routing; κ=0.1, mixing_depth=10)
    mask = CenterField(grid)
    fill!(mask, 0)
    routings = river_routing isa RiverRouting ? (river_routing,) : values(river_routing)
    for routing in routings
        n_targets = length(routing.target_i)
        n_targets == 0 && continue
        launch!(architecture(grid), grid, (n_targets, size(grid, 3)),
                _set_river_mouth_diffusivity!, mask, grid, routing.target_i, routing.target_j,
                convert(eltype(grid), κ), convert(eltype(grid), mixing_depth))
    end
    fill_halo_regions!(mask)
    return VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization();
                                    κ=river_mouth_κ, discrete_form=true,
                                    loc=(Center, Center, Center), parameters=mask)
end

@kernel function _set_river_mouth_diffusivity!(mask, grid, target_i, target_j, κ, mixing_depth)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i = target_i[n]
        j = target_j[n]
        z = znode(i, j, k, grid, Center(), Center(), Center())
        mask[i, j, k] = ifelse(z > -mixing_depth, κ, 0)
    end
end
