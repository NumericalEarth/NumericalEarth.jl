using Oceananigans.Grids: inactive_node

using ..EarthSystemModels: EarthSystemModel
using ..EarthSystemModels.InterfaceComputations: kernel_radiation_properties, skin_conductance, CanopyAirSpaceDiagnostics

"""
    apply_air_land_radiative_fluxes!(coupled_model)

Add the radiative contribution to the land `surface_energy_flux` (positive
upward) and, for a radiation that keeps them, write the diagnostic radiative
fluxes into `coupled_model.radiation.interface_fluxes.land`. A skin that closes
its own energy balance absorbs the radiation itself, so only the diagnostics are
written, at the temperature and albedo it radiates with.

When `coupled_model.radiation === nothing`, this is a no-op.
"""
EarthSystemModels.apply_air_land_radiative_fluxes!(::EarthSystemModel{<:Nothing}) = nothing

EarthSystemModels.apply_air_land_radiative_fluxes!(coupled_model::EarthSystemModel) =
    apply_air_land_radiative_fluxes!(coupled_model, coupled_model.land)

# No land: nothing to do.
apply_air_land_radiative_fluxes!(coupled_model, ::Nothing) = nothing

# Diagnostic radiative fluxes at the land surface; a radiation that keeps none gets empty slots.
land_interface_radiation_flux(radiation) = InterfaceRadiationFlux(nothing, nothing, nothing)
land_interface_radiation_flux(radiation::PrescribedRadiation) = radiation.interface_fluxes.land

function apply_air_land_radiative_fluxes!(coupled_model, land)
    # No atmosphere--land interface (no atmosphere): nothing to do.
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    # Skip when the radiation was not configured with a land surface.
    radiation = coupled_model.radiation
    rk = kernel_radiation_properties(radiation, coupled_model.interfaces.exchanger.radiation)
    land_surface_props = get(rk.surface_properties, :land, nothing)
    isnothing(land_surface_props) && return nothing

    grid  = coupled_model.interfaces.exchanger.grid
    arch  = architecture(grid)
    clock = coupled_model.clock
    radiation_state = coupled_model.interfaces.exchanger.radiation.state

    # A skin that closes its own energy balance absorbs the radiation and drives the slab by
    # conduction, so nothing is added to the slab's flux.
    land_energy_flux = isnothing(skin_conductance(al_interface)) ? land_surface_energy_flux(land) : nothing

    # `nothing` for components that diagnose their own upwelling radiation.
    land_shortwave_upwelling_flux = land_surface_shortwave_up(land)
    land_longwave_upwelling_flux = land_surface_longwave_up(land)

    launch!(arch, grid, :xy,
            _apply_air_land_radiative_fluxes!,
            land_energy_flux,
            land_interface_radiation_flux(radiation),
            grid,
            clock,
            rk,
            radiation_state,
            al_interface.temperature,
            land_shortwave_upwelling_flux,
            land_longwave_upwelling_flux)

    return nothing
end

# Opting out with `nothing` is resolved at compile time, so it costs nothing in the kernel.
@inline maybe_set_radiative_flux!(::Nothing, i, j, value) = nothing
@inline maybe_set_radiative_flux!(field, i, j, value) = @inbounds field[i, j, 1] = value
@inline maybe_add_radiative_flux!(::Nothing, i, j, value) = nothing
@inline maybe_add_radiative_flux!(field, i, j, value) = @inbounds field[i, j, 1] += value

# Temperature, albedo, and emissivity the land radiates with: a single skin uses the
# radiation's surface properties; a canopy column radiates as a black body at its effective
# temperature, σ T⁴ = LWꜛ above the canopy, with the column's albedo.
@inline radiating_surface(Tₛ, i, j, rs) = @inbounds (Tₛ[i, j, 1], rs.α, rs.ϵ)
@inline radiating_surface(Ts::CanopyAirSpaceDiagnostics, i, j, rs) =
    @inbounds (Ts.effective[i, j, 1], Ts.effective_albedo[i, j, 1], one(rs.ϵ))

@kernel function _apply_air_land_radiative_fluxes!(land_energy_flux,
                                                  interface_radiation_flux,
                                                  grid,
                                                  clock,
                                                  rk,
                                                  radiation_state,
                                                  land_surface_temperature,
                                                  land_shortwave_upwelling_flux,
                                                  land_longwave_upwelling_flux)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)

    rs = air_land_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)
    Tₛ, α, ϵ = radiating_surface(land_surface_temperature, i, j, rs)

    ℐꜛˡʷ = emitted_longwave_radiation(Tₛ, rs.σ, ϵ)
    ℐₐˡʷ = absorbed_longwave_radiation(ϵ, rs.ℐꜜˡʷ)
    ℐₜˢʷ = transmitted_shortwave_radiation(α, rs.ℐꜜˢʷ)

    # Reflected shortwave, positive upward (`ΣQ` uses the absorbed part instead).
    ℐꜛˢʷ = - α * rs.ℐꜜˢʷ

    # Net radiative flux, positive upward like `surface_energy_flux`: emitted longwave minus
    # absorbed longwave and shortwave (`ℐₐˡʷ` and `ℐₜˢʷ` are negative).
    ΣQ = ℐꜛˡʷ + ℐₐˡʷ + ℐₜˢʷ

    inactive = inactive_node(i, j, 1, grid, Center(), Center(), Center())

    maybe_add_radiative_flux!(land_energy_flux, i, j, ifelse(inactive, zero(grid), ΣQ))

    # Same `Tₛ`, `α`, `ϵ` as `ΣQ`, so the diagnostics cannot disagree with the assembled budget.
    maybe_set_radiative_flux!(interface_radiation_flux.upwelling_longwave,    i, j, ℐꜛˡʷ)
    maybe_set_radiative_flux!(interface_radiation_flux.downwelling_longwave,  i, j, ℐₐˡʷ)
    maybe_set_radiative_flux!(interface_radiation_flux.downwelling_shortwave, i, j, ℐₜˢʷ)
    maybe_set_radiative_flux!(land_shortwave_upwelling_flux, i, j, ℐꜛˢʷ)
    maybe_set_radiative_flux!(land_longwave_upwelling_flux, i, j, ℐꜛˡʷ)
end
