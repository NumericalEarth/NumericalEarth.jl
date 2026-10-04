using DocStringExtensions: TYPEDSIGNATURES

#####
##### Melt-rate computation at the ice shelf base
#####

"""
$(TYPEDSIGNATURES)

Compute heat and salt fluxes and the melt rate at the ice shelf-ocean
interface, storing them in `interface.fluxes` (and the interface temperature,
salinity, and friction velocity in the corresponding `interface` fields).

`ocean` may be a `Simulation` or the ocean model itself. For each cavity
column the three-equation system is solved with the ocean state
boundary-layer-averaged over a fixed physical thickness beneath the topmost
wet cell `k_draft(i, j)` (see [`boundary_layer_average`](@ref)), using the
liquidus evaluated at the depth of the ice base (the top face of that cell),
see [`ice_ocean_interface_heat_flux`](@ref).

Call this every time step, for example with
`add_callback!(simulation, sim -> compute_ice_shelf_fluxes!(interface, sim), IterationInterval(1))`,
so that the fluxes applied by [`ice_shelf_tracer_forcing`](@ref) stay in sync
with the ocean state.
"""
compute_ice_shelf_fluxes!(interface::IceShelfOceanInterface, ocean::Simulation) =
    compute_ice_shelf_fluxes!(interface, ocean.model)

function compute_ice_shelf_fluxes!(interface::IceShelfOceanInterface, model)
    grid = model.grid
    arch = architecture(grid)

    Tᵒᶜ = model.tracers.T
    Sᵒᶜ = model.tracers.S
    uᵒᶜ = model.velocities.u
    vᵒᶜ = model.velocities.v

    launch!(arch, grid, :xy, _compute_ice_shelf_fluxes!,
            interface.fluxes, interface.temperature, interface.salinity,
            interface.friction_velocity, interface.k_draft, interface.ice_concentration, grid,
            interface.flux_formulation, ice_shelf_liquidus(interface, model),
            Tᵒᶜ, Sᵒᶜ, uᵒᶜ, vᵒᶜ, interface.properties)

    return nothing
end

@kernel function _compute_ice_shelf_fluxes!(fluxes, T★, S★, u★_field, k_draft, ℵ, grid,
                                            flux_formulation, liquidus,
                                            Tᵒᶜ, Sᵒᶜ, uᵒᶜ, vᵒᶜ, properties)
    i, j = @index(Global, NTuple)

    FT = eltype(grid)
    kd = @inbounds k_draft[i, j, 1]

    𝒬  = zero(FT)
    q  = zero(FT)
    Jᵀ = zero(FT)
    Jˢ = zero(FT)
    Tᵦ = zero(FT)
    Sᵦ = zero(FT)
    u★ = zero(FT)

    if kd > 0
        # The ice base sits at the top face of the topmost wet cell
        zᵈ = znode(i, j, kd + 1, grid, Center(), Center(), Face())

        H_TBL = properties.boundary_layer_thickness
        Tᵈ = boundary_layer_average(i, j, kd, grid, H_TBL, k -> @inbounds(Tᵒᶜ[i, j, k]))
        Sᵈ = boundary_layer_average(i, j, kd, grid, H_TBL, k -> @inbounds(Sᵒᶜ[i, j, k]))

        ρᵒᶜ = properties.reference_density
        cᵒᶜ = properties.heat_capacity
        ℰ   = properties.latent_heat
        Sˢⁱ = properties.ice_salinity

        u★ = ice_shelf_friction_velocity(flux_formulation.friction_velocity, i, j, kd, grid, uᵒᶜ, vᵒᶜ, H_TBL)

        ocean_state = (; T = Tᵈ, S = Sᵈ)
        ice_state = (; S = Sˢⁱ, h = zero(FT), hc = zero(FT), ℵ = @inbounds(ℵ[i, j, 1]), T = zero(FT))
        𝒬, q, Tᵦ, Sᵦ = ice_ocean_interface_fluxes(nothing, i, j, flux_formulation, ocean_state, ice_state,
                                                 liquidus, zᵈ, properties, ℰ, u★)

        # Kinematic fluxes, positive out of the ocean.
        Jᵀ = 𝒬 / (ρᵒᶜ * cᵒᶜ)
        Jˢ = q / ρᵒᶜ * (Sᵈ - Sˢⁱ)
    end

    @inbounds begin
        fluxes.interface_heat[i, j, 1] = 𝒬
        fluxes.temperature[i, j, 1]    = Jᵀ
        fluxes.salt[i, j, 1]           = Jˢ
        fluxes.melt_rate[i, j, 1]      = q
        T★[i, j, 1] = Tᵦ
        S★[i, j, 1] = Sᵦ
        u★_field[i, j, 1] = u★
    end
end
