NumericalEarth.EarthSystemModels.surface_state(ocean::VerosOceanSimulation) = ocean.surface_state

NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(ocean::VerosOceanSimulation) = ocean.net_fluxes

# The index i of u (j of v) is the east (north) face of cell i (j), and tracers carry 2 halos in x and y
function NumericalEarth.EarthSystemModels.update_surface_state!(state, ocean::VerosOceanSimulation)
    veros_u = current_time_level(ocean, :u)
    veros_v = current_time_level(ocean, :v)
    veros_temperature = current_time_level(ocean, :temp)
    veros_salinity = current_time_level(ocean, :salt)

    Nx, Ny, Nz = size(veros_temperature)
    TX = topology(Grids.grid(ocean), 1)
    i_faces = TX == Periodic ? (2:Nx-3) : (2:Nx-2)

    set!(state.u, view(veros_u, i_faces, 3:Ny-2, Nz))
    set!(state.v, view(veros_v, 3:Nx-2, 2:Ny-2, Nz))
    set!(state.T, view(veros_temperature, 3:Nx-2, 3:Ny-2, Nz))
    set!(state.S, view(veros_salinity, 3:Nx-2, 3:Ny-2, Nz))

    fill_halo_regions!((state.u, state.v))
    fill_halo_regions!(state.T)
    fill_halo_regions!(state.S)

    return nothing
end

NumericalEarth.Oceans.get_radiative_forcing(ocean::VerosOceanSimulation) = nothing

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, ocean::VerosOceanSimulation)
    fluxes = ocean.net_fluxes
    fill_halo_regions!((fluxes.u, fluxes.v))
    fill_halo_regions!((fluxes.u_coefficient, fluxes.v_coefficient))

    nx, ny = size(Grids.grid(ocean), 1) + 4, size(Grids.grid(ocean), 2) + 4
    Hx, Hy, _ = halo_size(Grids.grid(ocean))
    ρᵒᶜ = NumericalEarth.EarthSystemModels.reference_density(ocean)

    # Index p of a Veros tracer array holds the center of cell p - 2
    i_centers = (1:nx) .+ Hx .- 2
    j_centers = (1:ny) .+ Hy .- 2

    # Index p of Veros's u (v) holds the east (north) face of cell p - 2, which is the west (south) face of cell p - 1
    i_faces = (1:nx-1) .+ Hx .- 1
    j_faces = (1:ny-1) .+ Hy .- 1

    # The ice-ocean drag is Fₑ + λ uᵒ, with both terms explicit in Veros
    (; u, v) = ocean.surface_state
    τˣ = view(parent(fluxes.u), i_faces, j_centers, 1) .+ view(parent(fluxes.u_coefficient), i_faces, j_centers, 1) .* view(parent(u), i_faces, j_centers, 1)
    τʸ = view(parent(fluxes.v), i_centers, j_faces, 1) .+ view(parent(fluxes.v_coefficient), i_centers, j_faces, 1) .* view(parent(v), i_centers, j_faces, 1)

    taux = zeros(nx, ny)
    tauy = zeros(nx, ny)
    view(taux, 1:nx-1, 1:ny) .= - ρᵒᶜ .* τˣ
    view(tauy, 1:nx, 1:ny-1) .= - ρᵒᶜ .* τʸ

    # Veros has no surface volume flux, so the freshwater flux Jʷ acts on the surface salinity as a virtual salt flux
    Jᵀ = view(parent(fluxes.T), i_centers, j_centers, 1)
    Jˢ = view(parent(fluxes.S), i_centers, j_centers, 1)
    Jʷ = view(parent(fluxes.η), i_centers, j_centers, 1)
    S = current_time_level(ocean, :salt)
    Sᴺ = view(S, :, :, size(S, 3))

    # Veros takes surface fluxes positive into the ocean
    set!(ocean, "surface_taux", taux)
    set!(ocean, "surface_tauy", tauy)
    set!(ocean, "forc_temp_surface", - Jᵀ)
    set!(ocean, "forc_salt_surface", - Jˢ .- Sᴺ .* Jʷ)

    update_surface_temperature!(ocean)

    return nothing
end

# Frazil formation adjusts the surface temperature in place at the sea ice-ocean interface.
function update_surface_temperature!(ocean::VerosOceanSimulation)
    T = pyimport("numpy").asarray(Array(view(interior(ocean.surface_state.T), :, :, 1)))
    pyexec("""
    from veros.core.operators import update, at
    vs = setup.state.variables
    with vs.unlock():
        vs.temp = update(vs.temp, at[2:-2, 2:-2, -1, vs.tau], T)
    """, Main, (; setup=ocean.setup, T))
    return nothing
end

function NumericalEarth.EarthSystemModels.above_freezing_ocean_temperature!(ocean::VerosOceanSimulation, grid, sea_ice::Oceananigans.Simulation)
    NumericalEarth.EarthSystemModels.update_surface_state!(ocean.surface_state, ocean)
    invoke(NumericalEarth.EarthSystemModels.above_freezing_ocean_temperature!, Tuple{Any, Any, Any}, ocean, grid, sea_ice)
    update_surface_temperature!(ocean)
    return nothing
end
