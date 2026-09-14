#####
##### Sea surface height for the sea ice surface-tilt term
#####

"""
$(TYPEDSIGNATURES)

The ocean's sea surface height on a `(Center, Center, Nothing)` field, which the sea-ice momentum
equation reads at its own topmost index for the surface-tilt term `- g ∇η`. A hydrostatic model
stores the displacement on the `Nz + 1` face and windows it there, so it cannot be read at that
index; this mirrors it, and [`ocean_surface_height!`](@ref) refills the mirror once per coupled step.
"""
function EarthSystemModels.ocean_surface_height(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    grid = ocean.model.grid
    isnothing(displacement(ocean.model.free_surface)) && return ZeroField(eltype(grid))
    ηˢ = Field{Center, Center, Nothing}(grid)
    EarthSystemModels.ocean_surface_height!(ηˢ, ocean)
    return ηˢ
end

"""
$(TYPEDSIGNATURES)

Refill `ηˢ` from the ocean's free surface displacement. The two carry the same interior shape — one
value per column — even when the free surface lives on the barotropic solver's halo-extended grid.
"""
function EarthSystemModels.ocean_surface_height!(ηˢ::Field, ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    η = displacement(ocean.model.free_surface)
    isnothing(η) && return nothing
    interior(ηˢ) .= interior(η)
    fill_halo_regions!(ηˢ)
    return nothing
end
