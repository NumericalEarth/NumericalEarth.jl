#####
##### Reference velocities for the sea ice-ocean drag
#####

# Thickness-weighted mean over the top `H` meters of the wet column, or the topmost cell where the column holds no water
@inline function surface_layer_uᶠᶜᵃ(i, j, k, grid, u, H)
    kᴺ = size(grid, 3)
    ∫u = zero(grid)
    ∫z = zero(grid)
    z  = zero(grid)
    for k′ = kᴺ:-1:1
        Δz = Δzᶠᶜᶜ(i, j, k′, grid) * !inactive_node(i, j, k′, grid, Face(), Center(), Center())
        δ  = min(Δz, max(0, H - z))
        ∫u += δ * @inbounds u[i, j, k′]
        ∫z += δ
        z  += Δz
    end
    return ifelse(∫z > 0, ∫u / ∫z, @inbounds u[i, j, kᴺ])
end

@inline function surface_layer_vᶜᶠᵃ(i, j, k, grid, v, H)
    kᴺ = size(grid, 3)
    ∫v = zero(grid)
    ∫z = zero(grid)
    z  = zero(grid)
    for k′ = kᴺ:-1:1
        Δz = Δzᶜᶠᶜ(i, j, k′, grid) * !inactive_node(i, j, k′, grid, Center(), Face(), Center())
        δ  = min(Δz, max(0, H - z))
        ∫v += δ * @inbounds v[i, j, k′]
        ∫z += δ
        z  += Δz
    end
    return ifelse(∫z > 0, ∫v / ∫z, @inbounds v[i, j, kᴺ])
end

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
