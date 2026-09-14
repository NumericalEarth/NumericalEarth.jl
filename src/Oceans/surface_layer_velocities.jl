#####
##### Reference velocities for the sea ice-ocean drag
#####

"""
$(TYPEDSIGNATURES)

Thickness-weighted mean `u` over the top `H` metres at `(i, j)`, falling back to the topmost cell where
the column carries no wet water above `H`. Dry cells contribute nothing, so a shelf column shallower
than `H` averages only the water it has.
"""
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

"""
$(TYPEDSIGNATURES)

Thickness-weighted mean `v` over the top `H` metres at `(i, j)`. See [`surface_layer_uᶠᶜᵃ`](@ref).
"""
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

# Resolves the ambiguity between the generic `(ocean, ::Nothing)` fallback and this module's
# `(::OceananigansModelSimulations, reference_depth)`: neither is more specific than the other.
EarthSystemModels.surface_layer_velocities(ocean::OceananigansModelSimulations, ::Nothing) =
    ocean_surface_velocities(ocean)

function EarthSystemModels.surface_layer_velocities(ocean::OceananigansModelSimulations, reference_depth)
    grid = ocean.model.grid
    u, v = ocean.model.velocities.u, ocean.model.velocities.v
    H = convert(eltype(grid), reference_depth)

    uˢˡ = Field(KernelFunctionOperation{Face, Center, Nothing}(surface_layer_uᶠᶜᵃ, grid, u, H))
    vˢˡ = Field(KernelFunctionOperation{Center, Face, Nothing}(surface_layer_vᶜᶠᵃ, grid, v, H))

    compute!(uˢˡ)
    compute!(vˢˡ)

    return uˢˡ, vˢˡ
end
