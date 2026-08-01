using ClimaSeaIce.SeaIceThermodynamics: melting_temperature, LinearLiquidus
using Oceananigans.Operators: Δzᶜᶜᶜ

using ..EarthSystemModels: EarthSystemModels, EarthSystemModel, NoSeaIceInterface
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations

#####
##### A workaround when you don't have a sea ice model
#####

struct FreezingLimitedOceanTemperature{L, F}
    liquidus    :: L
    frazil_heat :: F
end

"""
    FreezingLimitedOceanTemperature(FT=Float64; liquidus=LinearLiquidus(FT))

The minimal possible sea ice representation, clipping the temperature below to the freezing point.
Not really a "model" per se, however, it is the most simple way to make sure that temperature
does not dip below freezing.

The melting temperature is a function of salinity and is controlled by the `liquidus`.
"""
FreezingLimitedOceanTemperature(FT::DataType=Oceananigans.defaults.FloatType; liquidus=LinearLiquidus(FT)) =
    FreezingLimitedOceanTemperature(liquidus, nothing)

const FreezingLimitedEarthSystemModel = EarthSystemModel{R, A, L, <:FreezingLimitedOceanTemperature, O, <:NoSeaIceInterface} where {R, A, L, O}

function EarthSystemModels.materialize_sea_ice!(sea_ice::FreezingLimitedOceanTemperature, ocean)
    frazil_heat = Field{Center, Center, Nothing}(ocean.model.grid)
    return FreezingLimitedOceanTemperature(sea_ice.liquidus, frazil_heat)
end

EarthSystemModels.materialize_sea_ice!(sea_ice::FreezingLimitedOceanTemperature, ::Nothing) = sea_ice

# Extend interface methods to work with a `FreezingLimitedOceanTemperature`
EarthSystemModels.sea_ice_concentration(::FreezingLimitedOceanTemperature) = ZeroField()
EarthSystemModels.sea_ice_thickness(::FreezingLimitedOceanTemperature) = ZeroField()
EarthSystemModels.intercepted_snowfall(::FreezingLimitedOceanTemperature) = ZeroField()

# does not matter
EarthSystemModels.reference_density(::FreezingLimitedOceanTemperature) = 0
EarthSystemModels.heat_capacity(::FreezingLimitedOceanTemperature) = 0
Oceananigans.TimeSteppers.time_step!(::FreezingLimitedOceanTemperature, Δt) = nothing

# FreezingLimitedOceanTemperature handles temperature limiting in compute_sea_ice_ocean_fluxes!
EarthSystemModels.above_freezing_ocean_temperature!(ocean, grid, ::FreezingLimitedOceanTemperature) = nothing

# No atmosphere-sea ice or sea ice-ocean interface for FreezingLimitedOceanTemperature
InterfaceComputations.default_ai_temperature(::FreezingLimitedOceanTemperature) = nothing
InterfaceComputations.ThreeEquationHeatFlux(::FreezingLimitedOceanTemperature) = nothing
InterfaceComputations.atmosphere_sea_ice_interface(grid, atmos, ::FreezingLimitedOceanTemperature, args...) = nothing
InterfaceComputations.atmosphere_sea_ice_interface(grid, ::Nothing, ::FreezingLimitedOceanTemperature, args...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::FreezingLimitedOceanTemperature, ocean, flux_formulation; kwargs...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::FreezingLimitedOceanTemperature, ::Nothing, flux_formulation; kwargs...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::FreezingLimitedOceanTemperature, ocean, ::ThreeEquationHeatFlux; kwargs...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::FreezingLimitedOceanTemperature, ::Nothing, ::ThreeEquationHeatFlux; kwargs...) = nothing

InterfaceComputations.net_fluxes(::FreezingLimitedOceanTemperature) = nothing

const OnlyOceanwithFreezingLimited      = EarthSystemModel{<:Any, <:Nothing, <:Any, <:FreezingLimitedOceanTemperature, <:Any}
const OnlyAtmospherewithFreezingLimited = EarthSystemModel{<:Any, <:Any,     <:Any, <:FreezingLimitedOceanTemperature, <:Nothing}
const SingleComponentPlusFreezingLimited = Union{OnlyAtmospherewithFreezingLimited, OnlyOceanwithFreezingLimited}

# Also for the ocean nothing really happens here
EarthSystemModels.update_net_fluxes!(::SingleComponentPlusFreezingLimited, ocean::Simulation{<:HydrostaticFreeSurfaceModel}) = nothing

# No need to compute fluxes for this "sea ice model"
InterfaceComputations.compute_atmosphere_sea_ice_fluxes!(cm::FreezingLimitedEarthSystemModel) = nothing

# Same for the sea_ice ocean fluxes
function InterfaceComputations.compute_sea_ice_ocean_fluxes!(cm::FreezingLimitedEarthSystemModel)
    ocean   = cm.ocean
    sea_ice = cm.sea_ice
    liquidus = sea_ice.liquidus
    𝒬ᶠʳᶻ = sea_ice.frazil_heat
    grid = ocean.model.grid
    arch = architecture(grid)
    Sᵒᶜ = ocean.model.tracers.S
    Tᵒᶜ = ocean.model.tracers.T
    Δt = ocean.Δt
    ocean_properties = cm.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity

    # Guard for ocean.model.clock.iteration == 0
    Δt_frazil = ifelse(ocean.model.clock.iteration == 0, convert(typeof(Δt), Inf), Δt)

    launch!(arch, grid, :xy, _freeze_ocean_temperature!, 𝒬ᶠʳᶻ, Tᵒᶜ, Sᵒᶜ, liquidus, grid, ρᵒᶜ, cᵒᶜ, Δt_frazil)

    return nothing
end

@kernel function _freeze_ocean_temperature!(𝒬ᶠʳᶻ, Tᵒᶜ, Sᵒᶜ, liquidus, grid, ρᵒᶜ, cᵒᶜ, Δt)
    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    δ𝒬ᶠʳᶻ = zero(grid)

    for k = Nz:-1:1
        @inbounds begin
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            Tᵏ = Tᵒᶜ[i, j, k]
            Sᵏ = Sᵒᶜ[i, j, k]
        end

        Tₘ = melting_temperature(liquidus, Sᵏ)
        freezing = Tᵏ < Tₘ
        δE = freezing * ρᵒᶜ * cᵒᶜ * (Tₘ - Tᵏ)

        @inbounds Tᵒᶜ[i, j, k] = ifelse(freezing, Tₘ, Tᵏ)

        δ𝒬ᶠʳᶻ -= δE * Δz / Δt
    end

    @inbounds 𝒬ᶠʳᶻ[i, j, 1] = δ𝒬ᶠʳᶻ
end

Base.summary(::FreezingLimitedOceanTemperature) = "FreezingLimitedOceanTemperature"

function Base.show(io::IO, sea_ice::FreezingLimitedOceanTemperature)
    print(io, summary(sea_ice), "\n")
    print(io, "├── liquidus: ", summary(sea_ice.liquidus), "\n")
    print(io, "└── frazil_heat: ", summary(sea_ice.frazil_heat))
end

#####
##### Checkpointing (not needed for FreezingLimitedOceanTemperature)
#####

Oceananigans.prognostic_state(::FreezingLimitedOceanTemperature) = nothing
Oceananigans.restore_prognostic_state!(flt::FreezingLimitedOceanTemperature, state) = flt
Oceananigans.restore_prognostic_state!(flt::FreezingLimitedOceanTemperature, ::Nothing) = flt
