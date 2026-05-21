using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(ocean::VerosOceanSimulation, grid)
    state = (; u = Field{Face, Center, Nothing}(grid),
               v = Field{Center, Face, Nothing}(grid),
               T = Field{Center, Center, Nothing}(grid),
               S = Field{Center, Center, Nothing}(grid))

    return ComponentExchanger(state, nothing)
end

NumericalEarth.EarthSystemModels.exchange_grid(atmosphere, ocean::VerosOceanSimulation, sea_ice) = surface_grid(ocean)

@inline function NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(ocean::VerosOceanSimulation)
    grid = surface_grid(ocean)
    u = Field{Face,   Center, Nothing}(grid)
    v = Field{Center, Face,   Nothing}(grid)
    T = Field{Center, Center, Nothing}(grid)
    S = Field{Center, Center, Nothing}(grid)

    return (; u, v, T, S)
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, ocean::VerosOceanSimulation, coupled_model)
    u = exchanger.state.u
    v = exchanger.state.v
    T = exchanger.state.T
    S = exchanger.state.S

    set!(u, ocean.setup.state.variables.u)
    set!(v, ocean.setup.state.variables.v)
    set!(T, ocean.setup.state.variables.temp)
    set!(S, ocean.setup.state.variables.salt)

    return nothing
end

Oceananigans.initialize!(exchanger::ComponentExchanger, grid, ::VerosOceanSimulation) = nothing

NumericalEarth.Oceans.get_radiative_forcing(ocean::VerosOceanSimulation) = nothing

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, ocean::VerosOceanSimulation)

    # Update the flux containers
    NumericalEarth.Oceans.update_net_ocean_fluxes!(coupled_model, ocean, coupled_model.interfaces.exchanger.grid)
    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean

    # Pass the flux values to the python ocean
    nx = pyconvert(Int, ocean.setup.state.settings.nx) + 4
    ny = pyconvert(Int, ocean.setup.state.settings.ny) + 4

    ρᵒᶜ = pyconvert(eltype(ocean), ocean.setup.state.settings.rho_0)
    taux = view(parent(net_ocean_fluxes.u), 1:nx, 1:ny, 1) .* ρᵒᶜ
    tauy = view(parent(net_ocean_fluxes.v), 1:nx, 1:ny, 1) .* ρᵒᶜ

    set!(ocean, "surface_taux", taux; path=:variables)
    set!(ocean, "surface_tauy", tauy; path=:variables)

    temp_flux = view(parent(net_ocean_fluxes.T), 1:nx, 1:ny, 1)
    salt_flux = view(parent(net_ocean_fluxes.S), 1:nx, 1:ny, 1)

    set!(ocean, "forc_temp_surface", temp_flux; path=:variables)
    set!(ocean, "forc_salt_surface", salt_flux; path=:variables)

    return nothing
end
