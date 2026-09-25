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

@inline surface_wind_speed(exchanger) = sqrt(exchanger.atmosphere.state.u^2 + exchanger.atmosphere.state.v^2)

# prescribed atmosphere pressure is in Pa, the gas exchange takes atm
@inline surface_atmospheric_pressure(exchanger) = exchanger.atmosphere.state.p / ATM

biogeochemical_interface(exchanger, ocean, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}; kwargs...) =
    merge(
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.nutrients; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.plankton; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.detritus; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.oxygen; kwargs...),
        biogeochemical_interface(exchanger, ocean, biogeochemistry.underlying_biogeochemistry.inorganic_carbon; kwargs...)
    )

biogeochemical_interface(exchanger, ocean, ::Oxygen; kwargs...) =
    (; O₂ = OxygenGasExchangeBoundaryCondition(;
                wind_speed = surface_wind_speed(exchanger),
                air_concentration = GarciaGordonOxygenSaturation(; atmospheric_pressure = surface_atmospheric_pressure(exchanger)),
                kwargs...).condition.func)

biogeochemical_interface(exchanger, ocean, ::AbstractInorganicCarbon{1}; kwargs...) =
    (; DIC = carbon_dioxide_exchange(exchanger, :DIC, :Alk; kwargs...))

function biogeochemical_interface(exchanger, ocean, ::AbstractInorganicCarbon{N}; kwargs...) where N
    names = carbon_replicate_names(Val(N))
    exchanges = ntuple(n -> carbon_dioxide_exchange(exchanger, Symbol(:DIC, n), Symbol(:Alk, n); kwargs...), Val(N))
    return NamedTuple{names}(exchanges)
end

# same solubility as the boundary condition's default, so the air and water sides share a density
function carbon_dioxide_exchange(exchanger, DIC, Alk; carbon_chemistry = CarbonChemistry(), kwargs...)
    air_concentration = CarbonDioxideAirConcentration(; mole_fraction = exchanger.atmosphere.state.pCO₂,
                                                        atmospheric_pressure = surface_atmospheric_pressure(exchanger),
                                                        solubility = MolPerKgPerAtmToMMolPerCubicMPerMicroAtm(FF{Float64}(),
                                                                                                              carbon_chemistry.density_function))

    return CarbonDioxideGasExchangeBoundaryCondition(;
        wind_speed = surface_wind_speed(exchanger),
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

    exchangers = gas_transfer_parametrisations(biogeochemistry, coupled_model.interfaces.properties)

    ℵ = coupled_model.sea_ice.model.ice_concentration

    required_ocean_tracers = (:T, :S,
                              if_phosphate_available(ocean.model.tracers)...,
                              if_silicon_available(ocean.model.tracers)...,
                              required_biogeochemical_tracers(biogeochemistry.underlying_biogeochemistry.oxygen)...,
                              required_biogeochemical_tracers(biogeochemistry.underlying_biogeochemistry.inorganic_carbon)...)

    ocean_tracers = ocean.model.tracers[required_ocean_tracers]

    fluxes = tracer_fluxes(biogeochemistry, coupled_model.interfaces.net_fluxes.ocean)

    launch!(architecture(grid),
            grid, :xy,
            compute_all_gas_exchange!,
            grid,
            ocean.model.clock,
            fluxes,
            ℵ,
            ocean_tracers,
            biogeochemistry.underlying_biogeochemistry,
            exchangers)

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

@kernel function compute_all_gas_exchange!(grid, clock, fluxes, ice_concentration, ocean_tracers, biogeochemistry, exchangers)
    i, j = @index(Global, NTuple)

    @inbounds ℵ = ice_concentration[i, j, 1]

    compute_gas_exchange!(i, j, grid, clock, biogeochemistry.oxygen, fluxes, ℵ, ocean_tracers, exchangers)
    compute_gas_exchange!(i, j, grid, clock, biogeochemistry.inorganic_carbon, fluxes, ℵ, ocean_tracers, exchangers)
end

@inline compute_gas_exchange!(i, j, grid, clock, ::Nothing, fluxes, ℵ, ocean_tracers, exchangers) = nothing

@inline function compute_gas_exchange!(i, j, grid, clock, ::Oxygen, fluxes, ℵ, ocean_tracers, exchangers)
    @inbounds fluxes.O₂[i, j, 1] = exchangers.O₂(i, j, grid, clock, ocean_tracers) * (1 - ℵ)
    return nothing
end

@inline function compute_gas_exchange!(i, j, grid, clock, ::AbstractInorganicCarbon{1}, fluxes, ℵ, ocean_tracers, exchangers)
    @inbounds fluxes.DIC[i, j, 1] = exchangers.DIC(i, j, grid, clock, ocean_tracers) * (1 - ℵ)
    return nothing
end

# `Symbol(:DIC, n)` can not be constructed on the GPU so the replicates are unrolled here
@generated function compute_gas_exchange!(i, j, grid, clock, ::AbstractInorganicCarbon{N}, fluxes, ℵ, ocean_tracers, exchangers) where N
    exprs = map(1:N) do n
        DIC = Symbol(:DIC, n)
        :(@inbounds fluxes.$DIC[i, j, 1] = exchangers.$DIC(i, j, grid, clock, ocean_tracers) * (1 - ℵ))
    end

    return quote
        $(exprs...)

        return nothing
    end
end
