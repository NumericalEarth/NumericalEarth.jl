using Oceananigans.Biogeochemistry: required_biogeochemical_tracers
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

using OceanBioME: CarbonDioxideGasExchangeBoundaryCondition,
                  OxygenGasExchangeBoundaryCondition

using OceanBioME.Models.CarbonChemistryModel: silicate_concentration, phosphate_concentration
using OceanBioME.Models.GasExchangeModel: PartiallySolubleGas, CarbonDioxideConcentration

#####
##### gas exchange
#####
biogeochemistry_surface_exchanged_tracers(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}) = 
    (biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.nutrients)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.plankton)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.detritus)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.oxygen)...,
     biogeochemistry_surface_exchanged_tracers(biogeochemistry.underlying_biogeochemistry.inorganic_carbon)...)

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
    (; O₂ = OxygenGasExchangeBoundaryCondition().condition.func)

# for now using the same instance for every replicate, may want to change if we vary chemistry params
biogeochemical_interface(atmosphere, ocean, ::AbstractInorganicCarbon) = 
    (; DIC = CarbonDioxideGasExchangeBoundaryCondition().condition.func) 

function update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, ocean, grid)
    # we might want to add more stuff like sediments or rivers here in the future

    # gas exchange
    exchangers = gas_transfer_parametrisations(biogeochemistry, coupled_model.interfaces.properties)
    air_tracers = gas_exchange_air_tracers(biogeochemistry, coupled_model.interfaces.exchanger.atmosphere.state)

    u = coupled_model.interfaces.exchanger.atmosphere.state.u
    v = coupled_model.interfaces.exchanger.atmosphere.state.v

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
            fluxes,
            u, v, air_tracers, ℵ,
            ocean_tracers, 
            biogeochemistry.underlying_biogeochemistry, 
            exchangers)

    return nothing
end

@inline gas_transfer_parametrisations(biogeochemistry, properties) = NamedTuple()
@inline gas_transfer_parametrisations(::AbstractInorganicCarbon, properties) = (; DIC = properties.DIC)
@inline gas_transfer_parametrisations(::Oxygen, properties) = (; O₂ = properties.O₂)
@inline gas_transfer_parametrisations(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, properties) =
    merge(gas_transfer_parametrisations(biogeochemistry.underlying_biogeochemistry.oxygen, properties),
          gas_transfer_parametrisations(biogeochemistry.underlying_biogeochemistry.inorganic_carbon, properties))

@inline gas_exchange_air_tracers(biogeochemistry, state) = NamedTuple()
@inline gas_exchange_air_tracers(::AbstractInorganicCarbon, state) = (; pCO₂ = state.pCO₂)
@inline gas_exchange_air_tracers(::Oxygen, state) = (; O₂ = state.O₂)
@inline gas_exchange_air_tracers(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, state) =
    merge(gas_exchange_air_tracers(biogeochemistry.underlying_biogeochemistry.oxygen, state),
          gas_exchange_air_tracers(biogeochemistry.underlying_biogeochemistry.inorganic_carbon, state))

@inline tracer_fluxes(biogeochemistry, flux) = NamedTuple()
@inline tracer_fluxes(::AbstractInorganicCarbon{1}, flux) = (; DIC = flux.DIC)
@inline tracer_fluxes(::AbstractInorganicCarbon{N}, flux) where N = flux[tuple(map(n->Symbol(:DIC, n), 1:N)...)]
@inline tracer_fluxes(::Oxygen, flux) = (; O₂ = flux.O₂)
@inline tracer_fluxes(biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}, flux) =
    merge(tracer_fluxes(biogeochemistry.underlying_biogeochemistry.oxygen, flux),
          tracer_fluxes(biogeochemistry.underlying_biogeochemistry.inorganic_carbon, flux))

@inline if_phosphate_available(::NamedTuple{N}) where N = :PO₄ in N ? (:PO₄, ) : tuple()
@inline if_silicon_available(::NamedTuple{N}) where N = :Si in N ? (:Si, ) : tuple()

@kernel function compute_all_gas_exchange!(grid, fluxes, uᶠ, vᶠ, air_tracers, ice_thickness, ocean_tracers, biogeochemistry, exchangers)
    i, j = @index(Global, NTuple)

    u = ℑxᶜᵃᵃ(i, j, 1, grid, uᶠ)
    v = ℑyᵃᶜᵃ(i, j, 1, grid, vᶠ)

    wind = sqrt(u^2 + v^2)

    @inbounds begin
        T = ocean_tracers.T[i, j, grid.Nz]
        S = ocean_tracers.S[i, j, grid.Nz]
        ℵ = ice_thickness[i, j, 1]
    end
    
    compute_gas_exchange!(i, j, grid, biogeochemistry.oxygen, fluxes, wind, T, S, air_tracers, ℵ, ocean_tracers, exchangers)
    compute_gas_exchange!(i, j, grid, biogeochemistry.inorganic_carbon, fluxes, wind, T, S, air_tracers, ℵ, ocean_tracers, exchangers)
end

@inline compute_gas_exchange!(i, j, grid, ::Nothing, fluxes, wind, T, S, air_tracers, ℵ, ocean_tracers, exchangers) = nothing

@inline function compute_gas_exchange!(i, j, grid, ::Oxygen, fluxes, wind, T, S, air_tracers, ℵ, ocean_tracers, exchangers)
    exchange = exchangers.O₂
    flux = fluxes.O₂

    k = exchange.transfer_velocity
    k₀ = k.base_transfer_velocity.parametrisation(wind)
    Sc = k.schmidt_number(T)
    ζ  = k.solubility(T, S) # nb: this isn't really solubility it also converts units e.g. ppmv to mmol/m³ etc (which depend on density)

    ocean = ocean_tracers.O₂[i, j, grid.Nz]
    air = air_gas_concentration(i, j, 1, grid, air_tracers.O₂, ocean_tracers, exchange.air_concentration)

    @inbounds flux[i, j, 1] = k₀ * sqrt(Sc / convert(typeof(T), 660)) * ζ * (ocean - air) * (1 - ℵ)
end

@inline function compute_gas_exchange!(i, j, grid, ::AbstractInorganicCarbon{1}, fluxes, wind, T, S, air_tracers, ℵ, ocean_tracers, exchangers)
    exchange = exchangers.DIC

    k = exchange.transfer_velocity
    k₀ = k.base_transfer_velocity.parametrisation(wind)
    Sc = k.schmidt_number(T)
    Sc′ = sqrt(Sc / convert(typeof(T), 660))
    ζ  = k.solubility(T, S) # nb: this isn't really solubility it also converts units e.g. ppmv to mmol/m³ etc (which depend on density)
    air = air_gas_concentration(i, j, 1, grid, air_tracers.pCO₂, ocean_tracers, exchange.air_concentration)

    silicate = silicate_concentration(grid, i, j, grid.Nz, ocean_tracers)
    phosphate = phosphate_concentration(grid, i, j, grid.Nz, ocean_tracers)

    @inbounds begin
        ocean = exchange.water_concentration.carbon_chemistry(;
                    DIC = ocean_tracers.DIC[i, j, grid.Nz], 
                    Alk = ocean_tracers.Alk[i, j, grid.Nz], 
                    T, S, silicate, phosphate, 
                    output = Val(:pCO₂)
                )

        fluxes.DIC[i, j, 1] = k₀ * Sc′ * ζ * (ocean - air) * (1 - ℵ)
    end
end

@generated function compute_gas_exchange!(i, j, grid, 
                                                  ::AbstractInorganicCarbon{N}, 
                                                  fluxes, 
                                                  wind, 
                                                  T, S, 
                                                  air_tracers,
                                                  ℵ, 
                                                  ocean_tracers, 
                                                  exchangers) where N
    exprs = map(1:N) do n
        DIC = Symbol(:DIC, n)
        Alk = Symbol(:Alk, n)

        quote
            @inbounds begin
                ocean = exchange.water_concentration.carbon_chemistry(;
                            DIC = ocean_tracers.$DIC[i, j, grid.Nz],
                            Alk = ocean_tracers.$Alk[i, j, grid.Nz],
                            T, S, silicate, phosphate,
                            output = Val(:pCO₂)
                        )

                fluxes.$DIC[i, j, 1] = k₀ * Sc′ * ζ * (ocean - air) * (1 - ℵ)
            end
        end
    end

    return quote
        @inline begin
            exchange = exchangers.DIC

            k  = exchange.transfer_velocity
            k₀ = k.base_transfer_velocity.parametrisation(wind)
            Sc = k.schmidt_number(T)
            Sc′ = sqrt(Sc / convert(typeof(T), 660))
            ζ  = k.solubility(T, S)
            air = air_gas_concentration(i, j, 1, grid, air_tracers.pCO₂, ocean_tracers, exchange.air_concentration)

            silicate = silicate_concentration(grid, i, j, grid.Nz, ocean_tracers)
            phosphate = phosphate_concentration(grid, i, j, grid.Nz, ocean_tracers)

            $(exprs...)
        end
        return nothing
    end
end

@inline air_gas_concentration(i, j, k, grid, air_tracer, ocean_tracers, conc) = @inbounds air_tracer[i, j, k]
@inline air_gas_concentration(i, j, k, ::AbstractGrid{FT}, air_tracer, ocean_tracers, conc::PartiallySolubleGas) where FT =
    @inbounds conc.solubility(ocean_tracers.T[i, j, k] + convert(FT, 273.15), ocean_tracers.S[i, j, k]) * air_tracer[i, j, k]