using NumericalEarth.Oceans

import NumericalEarth.EarthSystemModels.InterfaceComputations:
    net_fluxes,
    initialize!,
    ComponentExchanger,
    default_exchange_grid,
    compute_sea_ice_ocean_fluxes!

import NumericalEarth.EarthSystemModels:
    interpolate_state!,
    update_net_fluxes!

import NumericalEarth.Oceans: get_radiative_forcing

using Oceananigans.Fields: interior

using MITgcm: MITgcmOceanSimulation,
              set_fu!, set_fv!, set_qnet!, set_empmr!, set_qsw!, set_saltflux!

#####
##### ComponentExchanger
#####

function ComponentExchanger(ocean::MITgcmOceanSimulation, grid)
    # MITgcm uses Arakawa C-grid: u at (Face, Center), v at (Center, Face)
    state = (; u = Field{Face,   Center, Nothing}(grid),
               v = Field{Center, Face,   Nothing}(grid),
               T = Field{Center, Center, Nothing}(grid),
               S = Field{Center, Center, Nothing}(grid))

    return ComponentExchanger(state, nothing)
end

default_exchange_grid(atmosphere, ocean::MITgcmOceanSimulation, sea_ice) = surface_grid(ocean)

#####
##### Net fluxes container
#####

@inline function net_fluxes(ocean::MITgcmOceanSimulation)
    grid = surface_grid(ocean)
    u = Field{Face,   Center, Nothing}(grid)
    v = Field{Center, Face,   Nothing}(grid)
    T = Field{Center, Center, Nothing}(grid)
    S = Field{Center, Center, Nothing}(grid)

    return (; u, v, T, S)
end

# ============================================================
# State interpolation
# ============================================================

function interpolate_state!(exchanger, exchange_grid, ocean::MITgcmOceanSimulation, coupled_model)
    u = exchanger.state.u
    v = exchanger.state.v
    T = exchanger.state.T
    S = exchanger.state.S

    u_surf, v_surf = ocean_surface_velocities(ocean)
    T_surf = ocean_surface_temperature(ocean)
    S_surf = ocean_surface_salinity(ocean)

    Nx = size(ocean.xc, 1)
    Ny = size(ocean.xc, 2)

    # u at (Face, Center): with periodic longitude, interior is (Nx, Ny) — direct match
    u_interior = interior(u, :, :, 1)
    u_interior .= u_surf

    # v at (Center, Face): with bounded latitude, interior is (Nx, Ny+1).
    # MITgcm vVel(i,j) is at the southern face of cell (i,j), so maps to face indices 1:Ny.
    # The Ny+1-th face (northern boundary) is set to zero.
    v_interior = interior(v, :, :, 1)
    v_interior[:, 1:Ny] .= v_surf
    v_interior[:, Ny+1] .= 0

    # T, S at (Center, Center): interior is (Nx, Ny) — direct match
    interior(T, :, :, 1) .= T_surf
    interior(S, :, :, 1) .= S_surf

    return nothing
end

initialize!(exchanger::ComponentExchanger, grid, ::MITgcmOceanSimulation) = nothing

get_radiative_forcing(ocean::MITgcmOceanSimulation) = nothing

# MITgcm handles its own freezing/temperature limiting internally,
# so no-op the FreezingLimitedOceanTemperature sea-ice-ocean flux computation
using NumericalEarth.EarthSystemModels: EarthSystemModel, NoSeaIceInterface
using NumericalEarth.SeaIces: FreezingLimitedOceanTemperature

const MITgcmFreezingLimited = EarthSystemModel{<:FreezingLimitedOceanTemperature, <:Any, <:MITgcmOceanSimulation, <:NoSeaIceInterface}
compute_sea_ice_ocean_fluxes!(::MITgcmFreezingLimited) = nothing

# ============================================================
# Flux update: coupled model → MITgcm surface forcing
# ============================================================

function update_net_fluxes!(coupled_model, ocean::MITgcmOceanSimulation)

    # Use the generic ocean flux assembler to compute net fluxes
    # on the exchange grid (momentum, heat, freshwater)
    exchange_grid = coupled_model.interfaces.exchanger.grid
    Oceans.update_net_ocean_fluxes!(coupled_model, ocean, exchange_grid)

    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean

    Nx = size(ocean.xc, 1)
    Ny = size(ocean.xc, 2)

    ρ₀ = ocean.reference_density
    cₚ = ocean.heat_capacity

    # Extract assembled fluxes.
    # NumericalEarth computes:
    #   τx, τy = kinematic stress (m²/s² = N/m² / ρ₀)
    #   Jᵀ = temperature tendency (K·m/s = W/m² / (ρ₀·cₚ))
    #   Jˢ = salinity tendency (PSU·m/s)
    # MITgcm expects:
    #   fu, fv = wind stress (N/m²) → multiply by ρ₀
    #   Qnet = net heat flux (W/m², positive = ocean cooling) → multiply by ρ₀·cₚ
    #   EmPmR = freshwater flux (kg/m²/s) → derive from Jˢ (or set to 0 if not available)
    #   Qsw = shortwave (W/m²) → set to 0 (already included in Qnet)
    #   saltFlux = salt flux (g/m²/s) → derive from Jˢ

    τx = interior(net_ocean_fluxes.u, :, :, 1)
    τy = interior(net_ocean_fluxes.v, :, :, 1)
    JT = interior(net_ocean_fluxes.T, :, :, 1)
    JS = interior(net_ocean_fluxes.S, :, :, 1)

    for j in 1:Ny, i in 1:Nx
        ocean.fu[i, j]       = - τx[i, j] * ρ₀
        ocean.fv[i, j]       = - τy[i, j] * ρ₀
        ocean.qnet[i, j]     = JT[i, j] * ρ₀ * cₚ
        ocean.saltflux[i, j] = JS[i, j] * ρ₀
    end

    # These are not needed
    fill!(ocean.empmr, 0)
    fill!(ocean.qsw,   0)

    # Push to MITgcm
    set_fu!(ocean.library,       ocean.fu)
    set_fv!(ocean.library,       ocean.fv)
    set_qnet!(ocean.library,     ocean.qnet)
    set_empmr!(ocean.library,    ocean.empmr)
    set_qsw!(ocean.library,      ocean.qsw)
    set_saltflux!(ocean.library, ocean.saltflux)

    return nothing
end
