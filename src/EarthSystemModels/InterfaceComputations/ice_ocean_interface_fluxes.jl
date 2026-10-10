#####
##### Column interface solve shared by sea ice and ice shelves
#####

"""
$(TYPEDSIGNATURES)

Return the ocean–ice heat flux, melt rate, interface temperature and interface salinity at
column `(i, j)` for an ice base at height `z`. With `melt_physics = nothing` this is
[`ice_ocean_interface_heat_flux`](@ref) with melt rate `𝒬 / ℰ`.
"""
@inline ice_ocean_interface_fluxes(::Nothing, i, j, flux_formulation, ocean_state, ice_state,
                                   liquidus, z, properties, ℰ, u★) =
    default_ice_ocean_interface_fluxes(flux_formulation, ocean_state, ice_state, liquidus, z, properties, ℰ, u★)

@inline function default_ice_ocean_interface_fluxes(flux_formulation, ocean_state, ice_state, liquidus, z, properties, ℰ, u★)
    𝒬, Tᵦ, Sᵦ = ice_ocean_interface_heat_flux(flux_formulation, ocean_state, ice_state,
                                              liquidus, z, properties, ℰ, u★)
    return 𝒬, 𝒬 / ℰ, Tᵦ, Sᵦ
end

"""
$(TYPEDSIGNATURES)

Return the interface heat flux, temperature and salinity from `flux_formulation` with
the `liquidus` evaluated at the ice-base height `z`.
"""
@inline ice_ocean_interface_heat_flux(flux_formulation, ocean_state, ice_state, liquidus, z, properties, ℰ, u★) =
    compute_interface_heat_flux(flux_formulation, ocean_state, ice_state, at_depth(liquidus, z), properties, ℰ, u★)

# A liquidus without pressure dependence is the same at every depth.
@inline at_depth(liquidus, z) = liquidus
