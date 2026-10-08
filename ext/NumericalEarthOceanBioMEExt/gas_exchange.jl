using Oceananigans.Biogeochemistry: required_biogeochemical_tracers
using Oceananigans.Fields: ZeroField

using OceanBioME: CarbonDioxideGasExchangeBoundaryCondition,
                  OxygenGasExchangeBoundaryCondition

using OceanBioME.Models.GasExchangeModel: PartiallySolubleGas, OxygenSolubility, GarciaGordonOxygenSaturation,
                                          CarbonDioxideAirConcentration, MolPerKgPerAtmToMMolPerCubicMPerMicroAtm, ATM
using OceanBioME.Models.CarbonChemistryModel: CarbonChemistry, FF

import NumericalEarth.EarthSystemModels.InterfaceComputations: biogeochemical_interface
import NumericalEarth.Oceans: update_net_ocean_biogeochemical_fluxes!, biogeochemistry_surface_exchanged_tracers

biogeochemistry_surface_exchanged_tracers(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}) =
    (biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.nutrients)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.plankton)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.detritus)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.oxygen)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.inorganic_carbon)...)

biogeochemistry_surface_exchanged_tracers(::Oxygen) = (:O₂, )
biogeochemistry_surface_exchanged_tracers(::AbstractInorganicCarbon{1}) = (:DIC, )
biogeochemistry_surface_exchanged_tracers(::AbstractInorganicCarbon{N}) where N = carbon_replicate_names(Val(N))

@inline carbon_replicate_names(::Val{N}) where N = ntuple(n -> Symbol(:DIC, n), Val(N))

# The wind speed and atmospheric pressure are computed into 2D fields once per step and shared by all of
# the gas exchanges. Left as operations, each exchange would carry its own copies of the grid (one per
# operation), which takes the gas exchange kernel parameters over the GPU limit with replicate carbonate systems
surface_wind_speed(exchanger) = Field(sqrt(exchanger.atmosphere.state.u^2 + exchanger.atmosphere.state.v^2))

# prescribed atmosphere pressure is in Pa, the gas exchange takes atm
surface_atmospheric_pressure(exchanger) = Field(exchanger.atmosphere.state.p / ATM)

function biogeochemical_interface(exchanger, ocean, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}; kwargs...)
    gas_exchange_state = (wind_speed = surface_wind_speed(exchanger),
                          atmospheric_pressure = surface_atmospheric_pressure(exchanger))

    return merge(
        (; gas_exchange_state),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.nutrients; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.plankton; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.detritus; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.oxygen; gas_exchange_state, kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.inorganic_carbon; gas_exchange_state, kwargs...)
    )
end

biogeochemical_interface(exchanger, ocean, ::Oxygen; gas_exchange_state, kwargs...) =
    (; O₂ = OxygenGasExchangeBoundaryCondition(;
                wind_speed = gas_exchange_state.wind_speed,
                air_concentration = GarciaGordonOxygenSaturation(; atmospheric_pressure = gas_exchange_state.atmospheric_pressure),
                kwargs...).condition.func)

biogeochemical_interface(exchanger, ocean, ::AbstractInorganicCarbon{1}; kwargs...) =
    (; DIC = carbon_dioxide_exchange(exchanger, :DIC, :Alk; kwargs...))

function biogeochemical_interface(exchanger, ocean, ::AbstractInorganicCarbon{N}; kwargs...) where N
    names = carbon_replicate_names(Val(N))
    exchanges = ntuple(n -> carbon_dioxide_exchange(exchanger, Symbol(:DIC, n), Symbol(:Alk, n); kwargs...), Val(N))
    return NamedTuple{names}(exchanges)
end

# same solubility as the boundary condition's default, so the air and water sides share a density
function carbon_dioxide_exchange(exchanger, DIC, Alk; gas_exchange_state, carbon_chemistry = CarbonChemistry(), kwargs...)
    air_concentration = CarbonDioxideAirConcentration(; mole_fraction = exchanger.atmosphere.state.pCO₂,
                                                        atmospheric_pressure = gas_exchange_state.atmospheric_pressure,
                                                        solubility = MolPerKgPerAtmToMMolPerCubicMPerMicroAtm(FF{Float64}(),
                                                                                                              carbon_chemistry.density_function))

    return CarbonDioxideGasExchangeBoundaryCondition(;
        wind_speed = gas_exchange_state.wind_speed,
        air_concentration,
        carbon_chemistry,
        DIC, Alk,
        kwargs...
    ).condition.func
end

#####
##### Applying it
#####

function update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, ocean, grid)
    # we might want to add more stuff like sediments or rivers here in the future

    properties = coupled_model.interfaces.properties

    compute!(properties.gas_exchange_state.wind_speed)
    compute!(properties.gas_exchange_state.atmospheric_pressure)

    exchangers = gas_transfer_parametrisations(biogeochemistry, properties)

    ℵ = coupled_model.sea_ice.model.ice_concentration

    required_ocean_tracers = (:T, :S,
                              if_phosphate_available(ocean.model.tracers)...,
                              if_silicon_available(ocean.model.tracers)...,
                              required_biogeochemical_tracers(biogeochemistry.underlying_biogeochemistry.oxygen)...,
                              required_biogeochemical_tracers(biogeochemistry.underlying_biogeochemistry.inorganic_carbon)...)

    ocean_tracers = ocean.model.tracers[required_ocean_tracers]

    fluxes = tracer_fluxes(biogeochemistry, coupled_model.interfaces.net_fluxes.ocean)

    # one launch per gas, so the kernel parameters don't grow with the number of exchanged gases
    # (each carbon chemistry is a few KiB, and the GPU parameter limit is 32 KiB)
    map(fluxes, exchangers) do flux, exchanger
        launch!(architecture(grid), grid, :xy,
                compute_gas_exchange!,
                flux, grid, ocean.model.clock, ℵ, ocean_tracers, exchanger)
    end

    return nothing
end

@inline gas_transfer_parametrisations(biogeochemistry, properties) = NamedTuple()
@inline gas_transfer_parametrisations(::AbstractInorganicCarbon{1}, properties) = (; DIC = properties.DIC)
@inline gas_transfer_parametrisations(::AbstractInorganicCarbon{N}, properties) where N = properties[carbon_replicate_names(Val(N))]
@inline gas_transfer_parametrisations(::Oxygen, properties) = (; O₂ = properties.O₂)
@inline gas_transfer_parametrisations(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, properties) =
    merge(gas_transfer_parametrisations(biogeochemistry.underlying_biogeochemistry.oxygen, properties),
          gas_transfer_parametrisations(biogeochemistry.underlying_biogeochemistry.inorganic_carbon, properties))

@inline tracer_fluxes(biogeochemistry, flux) = NamedTuple()
@inline tracer_fluxes(::AbstractInorganicCarbon{1}, flux) = (; DIC = flux.DIC)
@inline tracer_fluxes(::AbstractInorganicCarbon{N}, flux) where N = flux[carbon_replicate_names(Val(N))]
@inline tracer_fluxes(::Oxygen, flux) = (; O₂ = flux.O₂)
@inline tracer_fluxes(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, flux) =
    merge(tracer_fluxes(biogeochemistry.underlying_biogeochemistry.oxygen, flux),
          tracer_fluxes(biogeochemistry.underlying_biogeochemistry.inorganic_carbon, flux))

@inline if_phosphate_available(::NamedTuple{N}) where N = :PO₄ in N ? (:PO₄, ) : tuple()
@inline if_silicon_available(::NamedTuple{N}) where N = :Si in N ? (:Si, ) : tuple()

@kernel function compute_gas_exchange!(flux, grid, clock, ice_concentration, ocean_tracers, exchanger)
    i, j = @index(Global, NTuple)

    @inbounds flux[i, j, 1] = exchanger(i, j, grid, clock, ocean_tracers) * (1 - ice_concentration[i, j, 1])
end
