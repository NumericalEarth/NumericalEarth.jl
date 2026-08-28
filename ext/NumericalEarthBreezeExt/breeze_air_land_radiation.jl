#####
##### Surface energy balance coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### Each coupled step adds the net upward surface radiative flux, ℐˡʷꜛ + ℐˡʷꜜ + (1 - α) ℐˢʷꜜ,
##### to the slab's `surface_energy_flux` (positive = upward; downwelling stored negative).
##### Longwave up is rebuilt from the live surface state, ℐˡʷꜛ = ε σ Tₛ⁴ - (1 - ε) ℐˡʷꜜ, since
##### the RTM's own upwelling longwave is stale between scheduled solves. Shortwave keeps only
##### the absorbed fraction (1 - α): Breeze stores gross SW↓ with no upwelling field to read
##### back. Exact for coincident direct/diffuse albedos — the coupled configuration.
##### TODO: distinct direct/diffuse albedos need Breeze to expose the direct/diffuse SW↓ split.
#####

using Oceananigans.Fields: ConstantField
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics

const BreezeRTM = Breeze.RadiativeTransferModel

function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_properties(rtm::BreezeRTM, interfaces)
    al_interface = interfaces.atmosphere_land_interface
    temperature = isnothing(al_interface) ? nothing : al_interface.temperature

    # A canopy owns its surface optics, overriding configured properties: σ Tᵉᶠᶠ⁴ is the
    # column's total upwelling longwave — emission plus reflected downwelling — so a blackbody
    # at Tᵉᶠᶠ (ε = 1) reproduces it exactly, one broadband albedo in both shortwave slots.
    if temperature isa CanopyAirSpaceDiagnostics
        rtm = @set rtm.surface_properties.surface_temperature = temperature.effective
        rtm = @set rtm.surface_properties.surface_emissivity = ConstantField(one(eltype(temperature.effective)))
        rtm = @set rtm.surface_properties.direct_surface_albedo = temperature.effective_albedo
        return @set rtm.surface_properties.diffuse_surface_albedo = temperature.effective_albedo
    end

    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# RRTMGP copies scalar surface optics into its solver boundary conditions at construction
# only, so field-valued emissivity and albedo are republished each coupled step.
@kernel function _update_rrtmgp_surface_optics!(sfc_emis, sfc_alb_direct, sfc_alb_diffuse,
                                                emissivity, direct_albedo, diffuse_albedo, Nx)
    i, j = @index(Global, NTuple)
    c = i + (j - 1) * Nx
    @inbounds begin
        for band in axes(sfc_emis, 1)
            sfc_emis[band, c] = emissivity[i, j, 1]
        end
        for band in axes(sfc_alb_direct, 1)
            sfc_alb_direct[band, c]  = direct_albedo[i, j, 1]
            sfc_alb_diffuse[band, c] = diffuse_albedo[i, j, 1]
        end
    end
end

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, rtm::BreezeRTM)
    # The all-sky and clear-sky solvers re-read the surface properties at every solve themselves.
    hasproperty(rtm.longwave_solver, :bcs) || return nothing
    grid = coupled_model.interfaces.exchanger.grid
    properties = rtm.surface_properties
    launch!(architecture(grid), grid, :xy,
            _update_rrtmgp_surface_optics!,
            rtm.longwave_solver.bcs.sfc_emis,
            rtm.shortwave_solver.bcs.sfc_alb_direct,
            rtm.shortwave_solver.bcs.sfc_alb_diffuse,
            properties.surface_emissivity,
            properties.direct_surface_albedo,
            properties.diffuse_surface_albedo,
            grid.Nx)
    return nothing
end

# A Breeze RTM needs no exchange state; without this method the generic constructor
# would pass the RTM's solver internals into the flux kernel, which cannot compile on GPU.
NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(::BreezeRTM, exchange_grid; kw...) = nothing

# Empty `surface_properties` keeps radiation out of the turbulent-flux kernel:
# with a Breeze RTM the radiative term enters via `apply_air_land_radiative_fluxes!` below.
NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(::BreezeRTM) =
    (surface_properties = NamedTuple(),)

@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, Tˢ, ε, σ, ℐˡʷꜜ, ℐˢʷꜜ, α)
    i, j = @index(Global, NTuple)
    @inbounds begin
        εᵢⱼ = ε[i, j, 1]
        ℐˡʷꜛ = εᵢⱼ * σ * Tˢ[i, j, 1]^4 - (1 - εᵢⱼ) * ℐˡʷꜜ[i, j, 1]
        Es[i, j, 1] += ℐˡʷꜛ + ℐˡʷꜜ[i, j, 1] + (1 - α[i, j, 1]) * ℐˢʷꜜ[i, j, 1]
    end
end

# The generic method reads `PrescribedRadiation`-style `interface_fluxes`;
# a Breeze RTM carries its surface flux fields directly on the model.
function NumericalEarth.EarthSystemModels.apply_air_land_radiative_fluxes!(
        coupled_model :: NumericalEarth.EarthSystemModels.EarthSystemModel{<:BreezeRTM})

    land = coupled_model.land
    isnothing(land) && return nothing

    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    # A canopy (single or tiled) absorbs radiation inside its own solve; nothing is added here.
    al_interface.temperature isa CanopyAirSpaceDiagnostics && return nothing

    fluxes = land.fluxes
    hasproperty(fluxes, :surface_energy_flux) || return nothing
    Es = fluxes.surface_energy_flux

    rtm = coupled_model.radiation
    grid = land.grid
    arch = architecture(grid)
    σ = convert(eltype(grid), NumericalEarth.Radiations.default_stefan_boltzmann_constant)
    Tˢ = rtm.surface_properties.surface_temperature
    ε = rtm.surface_properties.surface_emissivity

    # Equals `diffuse_surface_albedo` in the coupled configuration; always indexable.
    α = rtm.surface_properties.direct_surface_albedo

    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
            Tˢ,
            ε,
            σ,
            rtm.downwelling_longwave_flux,
            rtm.downwelling_shortwave_flux,
            α)
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
