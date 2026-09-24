using Oceananigans.Grids: inactive_node

#####
##### Atmosphere-Land interface constructor

atmosphere_land_interface(grid, ::Nothing,    land;     kw...) = nothing
atmosphere_land_interface(grid, atmosphere, ::Nothing; kw...) = nothing
atmosphere_land_interface(grid, ::Nothing,  ::Nothing; kw...) = nothing

"""
    atmosphere_land_interface(grid, atmosphere, land;
                              fluxes               = default_atmosphere_land_fluxes(land, eltype(grid)),
                              temperature          = BulkTemperature(),
                              velocity_difference  = RelativeVelocity(),
                              specific_humidity    = default_al_specific_humidity(land))

Build the atmosphere--land interface on `grid` from `atmosphere` and `land` with
the given turbulent-flux closure, interface-temperature model, atmosphere-relative
velocity model, and specific-humidity formulation. Pass the result as
`atmosphere_land_interface = ...` to `ComponentInterfaces` /
`AtmosphereLandModel` to override the default.

The flux closure's roughness lengths and zero-plane displacement may be per-cell
`Field`s at `(Center, Center, Nothing)` on `grid` — for example from
`urban_roughness` or a canopy roughness closure — localized to each cell before the
Monin--Obukhov solve.
"""
function atmosphere_land_interface(grid, atmosphere, land;
                                   fluxes              = default_atmosphere_land_fluxes(land, eltype(grid)),
                                   temperature         = BulkTemperature(),
                                   velocity_difference = RelativeVelocity(),
                                   specific_humidity   = default_al_specific_humidity(land))
    validate_flux_formulation(fluxes, grid)

    al_fluxes = AtmosphereSurfaceFluxes(grid)
    al_properties = InterfaceProperties(specific_humidity, temperature, velocity_difference)
    interface_temperature = Field{Center, Center, Nothing}(grid)
    interface_specific_humidity = Field{Center, Center, Nothing}(grid)
    return AtmosphereInterface(al_fluxes, fluxes, interface_temperature,
                               interface_specific_humidity, al_properties)
end

#####
##### Flux compute driver
#####

compute_atmosphere_land_fluxes!(coupled_model) =
    compute_atmosphere_land_fluxes!(coupled_model, coupled_model.interfaces.atmosphere_land_interface)

compute_atmosphere_land_fluxes!(coupled_model, ::Nothing) = nothing

function compute_atmosphere_land_fluxes!(coupled_model, atmosphere_land_interface)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    atmosphere_fields = exchanger.atmosphere.state

    # See compute_atmosphere_ocean_fluxes! for rationale.
    atmosphere_data = merge(atmosphere_fields,
                            (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

    flux_formulation = atmosphere_land_interface.flux_formulation
    interface_fluxes = atmosphere_land_interface.fluxes
    interface_temperature = atmosphere_land_interface.temperature
    interface_specific_humidity = atmosphere_land_interface.specific_humidity
    interface_properties = atmosphere_land_interface.properties
    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    land_state = exchanger.land.state

    radiation = coupled_model.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation)
    radiation_exchanger    = exchanger.radiation
    radiation_state        = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state

    # Interior cells only: halo cells of the atmosphere exchanger state are uninitialized when the
    # atmosphere grid is a regional cutout matching the exchange-grid interior exactly.
    launch!(arch, grid, :xy,
            _compute_atmosphere_land_interface_state!,
            interface_fluxes,
            interface_temperature,
            interface_specific_humidity,
            grid,
            clock,
            flux_formulation,
            land_state,
            atmosphere_data,
            interface_properties,
            atmosphere_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

#####
##### Land state read by the interface solver
#####

"""
$(TYPEDSIGNATURES)

Names of the land-state quantities that the atmosphere--land interface described by
`properties` reads at each cell. The land component publishes exactly these, on the
exchange grid, through its `ComponentExchanger`; the flux kernel reads them pointwise
as the land state `Ψˡᵃ`. The bulk land temperature `T` is always included: it is the
initial guess for the interface temperature.
"""
land_state_names(properties::InterfaceProperties) = land_state_names(properties.specific_humidity_formulation)

land_state_names(specific_humidity_formulation) = (:T, :𝒮)
land_state_names(::SkinHumidity) = (:T,)
land_state_names(::FractionalHumidity{<:Number}) = (:T,)

"""
    land_state_field(land, ::Val{name})

Return the land quantity `name` on the land grid, e.g. the bulk land temperature for
`Val(:T)` or the surface saturation for `Val(:𝒮)`. Land components extend this for
every name in `land_state_names` they support.
"""
function land_state_field end

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes,
                                                           interface_temperature,
                                                           interface_specific_humidity,
                                                           grid,
                                                           clock,
                                                           turbulent_flux_formulation,
                                                           land_state,
                                                           atmosphere_state,
                                                           interface_properties,
                                                           atmosphere_properties,
                                                           radiation_kernel_props,
                                                           radiation_exchanger_state)

    i, j = @index(Global, NTuple)
    time = Time(clock.time)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Ψᵃᵗ = local_atmosphere_state(i, j, atmosphere_state, atmosphere_properties)
    Ψˡᵃ = state2dindex(land_state, i, j, grid, time)
    FT  = typeof(Ψˡᵃ.T)

    # Collapse Field-valued roughness lengths and displacement to this cell's values.
    local_turbulent_flux_formulation = local_flux_formulation(turbulent_flux_formulation, i, j)

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    # The bulk land temperature and its saturation humidity are the initial guess;
    # the solver recomputes both via the interface formulations.
    q_formulation = interface_properties.specific_humidity_formulation
    u★  = convert(FT, 1e-4)
    qⁱⁿ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Ψˡᵃ.T, Ψᵃᵗ.p, q_formulation.phase))
    Ψⁱⁿ⁰ = AirLandInterfaceState(u★, u★, u★, zero(FT), zero(FT), Ψˡᵃ.T, qⁱⁿ)

    Ψⁱⁿ = compute_interface_state(local_turbulent_flux_formulation,
                                  Ψⁱⁿ⁰,
                                  Ψᵃᵗ,
                                  Ψˡᵃ,
                                  radiation_state,
                                  interface_properties,
                                  atmosphere_properties,
                                  (;))

    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Ψᵃᵗ.T)

    store_interface_fluxes!(interface_fluxes, interface_temperature, interface_specific_humidity, i, j,
                            Ψⁱⁿ, Ψᵃᵗ, ℂᵃᵗ, ℒˡ, Ψⁱⁿ.temperature, interface_properties)
end
