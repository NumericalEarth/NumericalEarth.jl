#####
##### Reference velocities for the sea ice-ocean drag
#####

# Thickness-weighted mean over the top `H` meters of the wet column, or the topmost cell where the column holds no water
@inline function surface_layer_average(i, j, k, grid, ℓx, ℓy, c, H)
    kᴺ = size(grid, 3)
    ∫c = zero(grid)
    ∫z = zero(grid)
    z  = zero(grid)
    for k′ = kᴺ:-1:1
        Δzₖ = Δz(i, j, k′, grid, ℓx, ℓy, Center()) * !peripheral_node(i, j, k′, grid, ℓx, ℓy, Center())
        δ   = min(Δzₖ, max(zero(grid), H - z))
        ∫c += δ * @inbounds c[i, j, k′]
        ∫z += δ
        z  += Δzₖ
    end
    return ifelse(∫z > 0, ∫c / ∫z, @inbounds c[i, j, kᴺ])
end

@inline surface_layer_uᶠᶜᵃ(i, j, k, grid, u, H) = surface_layer_average(i, j, k, grid, Face(), Center(), u, H)
@inline surface_layer_vᶜᶠᵃ(i, j, k, grid, v, H) = surface_layer_average(i, j, k, grid, Center(), Face(), v, H)

function EarthSystemModels.surface_layer_velocities(ocean::OceananigansModelSimulations, reference_depth)
    isnothing(reference_depth) && return ocean_surface_velocities(ocean)

    grid = ocean.model.grid
    u, v = ocean.model.velocities.u, ocean.model.velocities.v
    H = convert(eltype(grid), reference_depth)

    uˢˡ = Field(KernelFunctionOperation{Face, Center, Nothing}(surface_layer_uᶠᶜᵃ, grid, u, H))
    vˢˡ = Field(KernelFunctionOperation{Center, Face, Nothing}(surface_layer_vᶜᶠᵃ, grid, v, H))

    compute!(uˢˡ)
    compute!(vˢˡ)

    return uˢˡ, vˢˡ
end
