#####
##### Surface energy balance coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### The RTM lives at `coupled_model.radiation`. Its surface-level flux fields
##### already bake in albedo and emissivity at the surface (RRTMGP handles
##### both reflection of downwelling shortwave and ε σ Tₛ⁴ for upwelling
##### longwave internally), so we simply add the *net radiative flux* to the
##### slab's `surface_energy_flux` accumulator. Both use the "positive flux =
##### upward" sign convention (downwelling components are negative), so the
##### net upward radiative flux at the surface face `k = 1` is
#####
#####    ℐˡʷꜛ + ℐˡʷꜜ + ℐˢʷꜜ
#####
##### This runs in `update_state!` after the turbulent (sensible + latent)
##### flux has been written to `surface_energy_flux`, so the kernel adds the
##### radiative term on top.
#####

const BreezeRTM = Breeze.RadiativeTransferModel

# Bind the interfaces' diagnostic skin temperature — what the atmosphere actually sees;
# equal to land.temperature only for bulk formulations — into an RTM constructed without
# one. Explicit construction wins; with no land interface, Breeze errors at first solve.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_temperature(rtm::BreezeRTM, interfaces)
    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# The turbulent-flux kernel asks the radiation for "kernel properties" used to
# augment its interface energy balance. With a Breeze RTM the radiative
# contribution to the surface energy balance is handled separately by
# `apply_air_land_radiative_fluxes!` below, so we return an empty
# `surface_properties` here — `air_land_interface_radiation_state` already
# handles the "no land surface_properties" path by returning a zero radiation
# state.
NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(::BreezeRTM) =
    (surface_properties = NamedTuple(),)

@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, ℐˡʷꜛ, ℐˡʷꜜ, ℐˢʷꜜ)
    i, j = @index(Global, NTuple)
    @inbounds Es[i, j, 1] += ℐˡʷꜛ[i, j, 1] + ℐˡʷꜜ[i, j, 1] + ℐˢʷꜜ[i, j, 1]
end

# Dispatch on `EarthSystemModel{<:BreezeRTM}`: the existing generic
# `apply_air_land_radiative_fluxes!` only handles `PrescribedRadiation`-style
# radiation (which carries `interface_fluxes.land` etc.); the Breeze RTM
# carries the surface flux fields directly on the model.
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
    grid = land.grid
    arch = architecture(grid)
    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
            rtm.upwelling_longwave_flux,
            rtm.downwelling_longwave_flux,
            rtm.downwelling_shortwave_flux)
    return nothing
end
