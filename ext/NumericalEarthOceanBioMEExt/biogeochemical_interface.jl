using OceanBioME: CarbonDioxideGasExchangeBoundaryCondition,
                  OxygenGasExchangeBoundaryCondition

#####
##### gas exchange
#####
biogeochemistry_surface_exchanged_tracers(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}) =
    (biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.nutrients),
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.plankton),
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.detritus),
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.oxygen),
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.inorganic_carbon))

biogeochemistry_surface_exchanged_tracers(::Oxygen) = (:O₂, )
biogeochemistry_surface_exchanged_tracers(::AbstractInorganicCarbon{1}) = (:DIC, )
biogeochemistry_surface_exchanged_tracers(::AbstractInorganicCarbon{N}) where N = map(n->Symbol(:DIC, n), 1:N)

biogeochemical_interface(atmosphere, ocean, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}) =
    merge(
        biogeochemical_interface(atmosphere, ocean, biogeochemistry.underlying_biogeochemistry.nutrients),
        biogeochemical_interface(atmosphere, ocean, biogeochemistry.underlying_biogeochemistry.plankton),
        biogeochemical_interface(atmosphere, ocean, biogeochemistry.underlying_biogeochemistry.detritus),
        biogeochemical_interface(atmosphere, ocean, biogeochemistry.underlying_biogeochemistry.oxygen),
        biogeochemical_interface(atmosphere, ocean, biogeochemistry.underlying_biogeochemistry.inorganic_carbon)
    )

biogeochemical_interface(atmosphere, ocean, ::Oxygen) =
    (; O₂ = OxygenGasExchangeBoundaryCondition().condition)

biogeochemical_interface(atmosphere, ocean, ::AbstractInorganicCarbon) = 
    (; DIC = CarbonDioxideGasExchangeBoundaryCondition().condition) # use the same instance for every replicate

function update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, ocean, grid)
    # we might want to add more stuff like sediments or rivers here in the future
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.underlying_biogeochemistry.inorganic_carbon, ocean, grid)
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.underlying_biogeochemistry.oxygen, ocean, grid)

    return nothing
end

function update_net_ocean_biogeochemical_fluxes!(coupled_model, oxygen::Oxygen, ocean, grid)
    O₂ = ocean.model.tracers.O₂
    T  = ocean.model.tracers.T
    S  = ocean.model.tracers.S

    u = coupled_model.interfaces.exchanger.atmosphere.state.u
    v = coupled_model.interfaces.exchanger.atmosphere.state.v

    O₂_air = coupled_model.interfaces.exchanger.atmosphere.state.O₂

    

    return nothing
end

function update_net_ocean_biogeochemical_fluxes!(coupled_model, ic::AbstractInorganicCarbon{N}, ocean, grid) where N

    return nothing
end