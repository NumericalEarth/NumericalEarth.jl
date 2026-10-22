#####
##### Surface radiation uses positive-upward fluxes (W m⁻²).
##### The net surface flux is ε σ Tₛ⁴ + ε ℐꜜˡʷ + (1 - α) ℐꜜˢʷ.
#####

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, inactive_node
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant,
                                 emitted_longwave_radiation, absorbed_longwave_radiation,
                                 transmitted_shortwave_radiation

const BreezeRTM = Breeze.RadiativeTransferModel
const GrayBreezeRTM = Breeze.RadiativeTransferModel{<:Number, <:Any, <:Any, Nothing}

# Bind the interfaces' diagnostic skin temperature — what the atmosphere actually sees;
# equal to land.temperature only for bulk formulations — into an RTM constructed without
# one. Explicit construction wins; with no land interface, Breeze errors at first solve.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_temperature(rtm::BreezeRTM, interfaces)
    isnothing(rtm.surface_radiation.surface_temperature) || return rtm
    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_radiation.surface_temperature = Tˢ
end

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(rtm::BreezeRTM, exchange_grid; kw...)
    state = (; ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid))

    return ComponentExchanger(state, nothing)
end

# Downwelling fluxes are negative in both Breeze and the interface state.
@kernel function _interpolate_breeze_radiation_state!(state, ℐꜜˢʷ, ℐꜜˡʷ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.ℐꜜˢʷ[i, j, 1] = ℐꜜˢʷ[i, j, 1]
        state.ℐꜜˡʷ[i, j, 1] = ℐꜜˡʷ[i, j, 1]
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, rtm::BreezeRTM, coupled_model)
    state = exchanger.state

    launch!(architecture(exchange_grid), exchange_grid, :xy,
            _interpolate_breeze_radiation_state!,
            state,
            rtm.downwelling_shortwave_flux,
            rtm.downwelling_longwave_flux)

    return nothing
end

# σ is NumericalEarth's default: Breeze's `stefan_bolzmann_constant` is not reachable from the
# model, so land emission pairs with atmospheric absorption only while Breeze keeps that default.
function NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(rtm::BreezeRTM)
    FT = eltype(rtm.downwelling_shortwave_flux)
    ε = rtm.surface_radiation.surface_emissivity
    α = blue_sky_albedo(rtm)
    return (σ = convert(FT, default_stefan_boltzmann_constant),
            surface_properties = (; land = SurfaceRadiationProperties(α, ε)))
end

# Gray optics transmits the direct beam only.
blue_sky_albedo(rtm::GrayBreezeRTM) = rtm.surface_radiation.direct_surface_albedo

# Two-stream optics reflect the direct and diffuse beams with their own albedos;
# their blend is the ratio of upwelling to downwelling shortwave at the surface face k = 1.
blue_sky_albedo(rtm::BreezeRTM) =
    KernelFunctionOperation{Center, Center, Nothing}(reflected_shortwave_fraction,
                                                     rtm.downwelling_shortwave_flux.grid,
                                                     rtm.upwelling_shortwave_flux,
                                                     rtm.downwelling_shortwave_flux)

@inline reflected_shortwave_fraction(i, j, k, grid, ℐꜛˢʷ, ℐꜜˢʷ) =
    @inbounds ifelse(ℐꜜˢʷ[i, j, 1] < 0, - ℐꜛˢʷ[i, j, 1] / ℐꜜˢʷ[i, j, 1], zero(grid))

@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, grid, Tˢ, ε, σ, ℐꜜˡʷ, ℐꜜˢʷ, α)
    i, j = @index(Global, NTuple)

    inactive = inactive_node(i, j, 1, grid, Center(), Center(), Center())

    @inbounds begin
        ℐꜛˡʷ = emitted_longwave_radiation(Tˢ[i, j, 1], σ, ε[i, j, 1])
        ℐₐˡʷ = absorbed_longwave_radiation(ε[i, j, 1], ℐꜜˡʷ[i, j, 1])
        ℐₜˢʷ = transmitted_shortwave_radiation(α[i, j, 1], ℐꜜˢʷ[i, j, 1])
        Es[i, j, 1] += !inactive * (ℐꜛˡʷ + ℐₐˡʷ + ℐₜˢʷ)
    end
end

# The land emits from the current Tˢ; the atmosphere absorbs emission from Tˢ at the last RTM solve.
function NumericalEarth.EarthSystemModels.apply_air_land_radiative_fluxes!(
        coupled_model :: NumericalEarth.EarthSystemModels.EarthSystemModel{<:BreezeRTM})

    land = coupled_model.land
    isnothing(land) && return nothing

    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    fluxes = land.fluxes
    hasproperty(fluxes, :surface_energy_flux) || return nothing
    Es = fluxes.surface_energy_flux

    rtm = coupled_model.radiation
    grid = coupled_model.interfaces.exchanger.grid
    arch = architecture(grid)
    rk = NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(rtm)
    surface_properties = rk.surface_properties.land
    Tˢ = al_interface.temperature   # land skin temperature (K)

    state = coupled_model.interfaces.exchanger.radiation.state

    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
            grid,
            Tˢ,
            surface_properties.emissivity,
            rk.σ,
            state.ℐꜜˡʷ,
            state.ℐꜜˢʷ,
            surface_properties.albedo)
    return nothing
end

# The air–sea analog: dispatch peels off no-ocean and prescribed-SST (no net fluxes) cases.
# A responsive ocean under a Breeze RTM raises a MethodError until its radiative heating
# is implemented.
NumericalEarth.EarthSystemModels.apply_air_sea_radiative_fluxes!(
        coupled_model :: NumericalEarth.EarthSystemModels.EarthSystemModel{<:BreezeRTM}) =
    apply_breeze_air_sea_radiative_fluxes!(coupled_model, coupled_model.ocean)

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ::Nothing) = nothing

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean) =
    apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean,
        NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(ocean))

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean, ::Nothing) = nothing
