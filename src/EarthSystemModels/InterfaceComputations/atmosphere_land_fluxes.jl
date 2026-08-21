using Oceananigans.Grids: inactive_node

#####
##### Atmosphere-Land interface constructor
#####
##### The atmosphere–land turbulent fluxes share their container type
##### with atmosphere–ocean ([`AtmosphereSurfaceFluxes`](@ref)); only
##### the compute kernel differs.
#####

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
"""
function atmosphere_land_interface(grid, atmosphere, land;
                                   fluxes              = default_atmosphere_land_fluxes(land, eltype(grid)),
                                   temperature         = BulkTemperature(),
                                   velocity_difference = RelativeVelocity(),
                                   specific_humidity   = default_al_specific_humidity(land))
    if requires_retention_curve(specific_humidity) && isnothing(surface_retention_curve(land))
        throw(ArgumentError("$(summary(specific_humidity)) measures plant-available water on " *
                            "the soil's retention curve, but this land's hydrology carries none. " *
                            "Use a hydrology that owns one (VariablySaturatedHydrology), or a " *
                            "moisture stress defined on the hydrology's own saturation " *
                            "(CriticalSaturation)."))
    end
    al_fluxes = AtmosphereSurfaceFluxes(grid)
    al_properties = InterfaceProperties(specific_humidity, temperature, velocity_difference)
    interface_temperature = build_interface_temperature(temperature, grid)
    return AtmosphereInterface(al_fluxes, fluxes, interface_temperature, al_properties)
end

# The atmosphere-facing interface temperature. A single field for the ordinary
# closures; a `CanopyAirSpace` additionally needs the two diagnostic skins, the
# skin→slab ground heat flux, and the two-source (leaf/ground) sensible and latent
# shares, so it carries a [`CanopyAirSpaceDiagnostics`](@ref) — the type is the signal
# downstream (`slab_land.jl`, `apply_air_land_radiative_fluxes.jl`) that the radiation
# is internalized and the slab is driven by conduction.
@inline build_interface_temperature(temperature_formulation, grid) = Field{Center, Center, Nothing}(grid)
@inline build_interface_temperature(::CanopyAirSpace, grid) = CanopyAirSpaceDiagnostics(grid)

# Store the diagnostic surface temperature(s) from the converged interface state.
# Ordinary closures write the single skin temperature; a `CanopyAirSpace` re-runs its
# (cheap, converged) solve to recover the leaf/soil-skin temperatures, the ground heat
# flux, and the two-source (leaf/ground) sensible and latent shares — the atmosphere-facing
# `interface.fluxes` carry only their sums.
@inline store_interface_temperature!(Ts, i, j, formulation, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ) =
    (@inbounds Ts[i, j, 1] = Ψₛ.temperature; nothing)

@inline function store_interface_temperature!(Ts, i, j, cas::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    sol = canopy_air_space_solve(cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    @inbounds begin
        Ts.interface[i, j, 1]              = sol.Tᵃᶜ
        Ts.canopy[i, j, 1]                 = sol.Tᵛ
        Ts.soil_skin[i, j, 1]              = sol.Tᵍ
        Ts.effective[i, j, 1]              = sol.Teff
        Ts.ground_heat_flux[i, j, 1]        = sol.Gᶜ
        Ts.canopy_latent_heat[i, j, 1]     = sol.LEᵛ
        Ts.soil_latent_heat[i, j, 1]       = sol.LEᵍ
        Ts.canopy_sensible_heat[i, j, 1]   = sol.Hᵛ
        Ts.soil_sensible_heat[i, j, 1]     = sol.Hᵍ
        Ts.canopy_evaporation[i, j, 1]     = sol.Eʷ
        Ts.canopy_wet_latent_heat[i, j, 1] = sol.LEʷ
    end
    return nothing
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
    interface_properties = atmosphere_land_interface.properties
    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    # Land surface state from the exchanger. `interface_energy_state` /
    # `interface_hydrology_state` read these per cell to build the land
    # interface state; the surface models derive `β`, the reservoir
    # temperature, etc. from them.
    land_exchanger_state = exchanger.land.state
    land_state = (T = land_exchanger_state.T,
                  saturation = land_exchanger_state.saturation,
                  canopy_water_storage = land_exchanger_state.canopy_water_storage,
                  canopy_water_capacity = land_exchanger_state.canopy_water_capacity,
                  retention_curve = land_exchanger_state.retention_curve)

    land_properties = atmosphere_land_surface_properties(land_exchanger_state)

    # Prescribed leaf area index off the canopy formulation (or `nothing`),
    # reduced to a kernel-friendly value plus its host-side time interpolator.
    leaf_area_index = canopy_leaf_area_index(interface_properties.specific_humidity_formulation)
    vegetation, leaf_area_index_time_interpolator = kernel_surface_field(leaf_area_index, arch, clock.time)

    radiation = coupled_model.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation)
    radiation_exchanger    = exchanger.radiation
    radiation_state        = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state

    # Land turbulent fluxes are evaluated only over interior cells; the
    # downstream SlabLand step uses `:xy` (interior-only), and halo
    # cells of the atmosphere exchanger state may not be initialized
    # when the atmosphere grid is a regional cutout matching the
    # exchange-grid interior exactly (`interface_kernel_parameters`
    # iterates 0:Nx+1 for the ocean's benefit; we do not need that
    # here).
    launch!(arch, grid, :xy,
            _compute_atmosphere_land_interface_state!,
            interface_fluxes,
            interface_temperature,
            grid,
            clock,
            flux_formulation,
            land_state,
            vegetation,
            leaf_area_index_time_interpolator,
            atmosphere_data,
            interface_properties,
            atmosphere_properties,
            land_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

# Roughness and zero-plane displacement live on the flux closure
# (`atmosphere_land_fluxes`), not the land state — `SlabLand` carries neither.
# A land model that provides per-cell values (`momentum_roughness_length`,
# `scalar_roughness_length`, `zero_plane_displacement`) overrides these for its
# own land state type.
@inline atmosphere_land_surface_properties(land_state) = (;)
@inline local_atmosphere_land_surface_properties(land_properties, i, j) = (;)

#####
##### Prescribed, possibly time-varying surface inputs.
##### `surface_field_value` reads the per-cell value from a `Number`, a static
##### `Field`, or a `FieldTimeSeries` interpolated to the model clock.
#####

@inline surface_field_value(x, i, j, time_interpolator) = state2dindex(x, i, j)

# `i, j` name an exact cell, so the value is interpolated in time only. Passing them to
# `interpolate` as spatial fractional indices would also read the neighbor at
# `(i + 1, j + 1)`, which does not exist in a `Flat` direction or at the last cell.
#
# The indices are converted because `time_interpolator` reaches the kernel as an argument,
# where its scalar fields can arrive as `CuTracedRNumber` rather than `Int`
# (CliMA/Oceananigans.jl#4230).
#
# TODO: drop this blend once Oceananigans accepts a `TimeInterpolator` in `getindex`
# (CliMA/Oceananigans.jl#5886), leaving `x[i, j, 1, time_interpolator]`. The flux kernels take
# an interpolator precomputed on the host, as the prescribed-atmosphere path does, because
# `fts[i, j, k, Time(t)]` recomputes the time indices in every thread; today Oceananigans
# accepts one only through `interpolate`, which interpolates in space as well.
@inline function surface_field_value(x::FlavorOfFTS, i, j, time_interpolator)
    ñ  = time_interpolator.fractional_index
    n₁ = convert(Int, time_interpolator.first_index)
    n₂ = convert(Int, time_interpolator.second_index)

    @inbounds ψ₁ = x[i, j, 1, n₁]
    @inbounds ψ₂ = x[i, j, 1, n₂]

    return ifelse(n₁ == n₂, ψ₁, ψ₂ * ñ + ψ₁ * (1 - ñ))
end

# Host-side: pair a prescribed-surface spec (LAI, vegetation fraction, …) with the time
# index used to interpolate it. Constants and static fields pass through untouched
# (`nothing` interpolator); a `FieldTimeSeries` keeps its time index precomputed on the
# host, so the kernel does not recompute it.
@inline kernel_surface_field(surface_field, arch, time) = (surface_field, nothing)
@inline function kernel_surface_field(surface_field::FieldTimeSeries, arch, time)
    time_interpolator = cpu_interpolating_time_indices(arch, surface_field.times,
                                                       surface_field.time_indexing, time)
    return surface_field, time_interpolator
end

# The LAI spec lives on the canopy humidity formulation; other formulations carry
# none. The canopy / composite methods are defined alongside those formulations.
@inline canopy_leaf_area_index(q_formulation) = nothing

#####
##### Land surface state materialized into the interface state.
#####
##### The surface model (`interface_model`, here the specific-humidity
##### formulation) dispatches these helpers to pull *exactly* the per-cell land
##### state it consumes — saturation for the moisture-availability models, the
##### bulk temperature for the reservoir model — and nothing otherwise. The
##### model then derives `β`, the reservoir temperature, etc. from what it pulled.
#####

@inline land_saturation(i, j, grid, land_state) =
    (saturation = state2dindex(land_state.saturation, i, j),)

# Hydrology state, per humidity formulation.
@inline interface_hydrology_state(i, j, grid, ::BulkHumidity, land_state) = land_saturation(i, j, grid, land_state)
@inline interface_hydrology_state(i, j, grid, q::FractionalHumidity, land_state) =
    interface_hydrology_state(i, j, grid, q.efficiency, land_state)
@inline requires_retention_curve(q::FractionalHumidity) = requires_retention_curve(q.efficiency)
@inline interface_hydrology_state(i, j, grid, ::CriticalSaturation, land_state) = land_saturation(i, j, grid, land_state)
# The stress endpoints live on the *land's* retention curve, whose parameters may vary
# per cell; evaluate them here, once per cell, so the flux solve reads plain scalars.
@inline function interface_hydrology_state(i, j, grid, p::PlantAvailableWaterStress, land_state)
    𝒮 = state2dindex(land_state.saturation, i, j)
    FT = typeof(𝒮)
    r  = land_state.retention_curve
    return (saturation = 𝒮,
            field_capacity_saturation = effective_saturation(i, j, grid, r, convert(FT, p.field_capacity_head)),
            wilting_saturation        = effective_saturation(i, j, grid, r, convert(FT, p.wilting_point_head)))
end
@inline interface_hydrology_state(i, j, grid, ::DryLayerHumidity, land_state) =
    land_saturation(i, j, grid, land_state)
@inline interface_hydrology_state(i, j, grid, interface_model, land_state) = (;) # default: pulls nothing

# Energy state: humidity formulations that need the bulk land temperature
# (the SkinHumidity reservoir and the DryLayerHumidity dry-layer model)
# pull it from the materialized land state.
@inline interface_energy_state(i, j, grid, ::SkinHumidity, land_state) =
    (temperature = state2dindex(land_state.T, i, j),)
@inline interface_energy_state(i, j, grid, ::DryLayerHumidity, land_state) =
    (temperature = state2dindex(land_state.T, i, j),)
@inline interface_energy_state(i, j, grid, interface_model, land_state) = (;) # default: pulls nothing

# Vegetation state, per humidity formulation. Only the canopy formulations
# (defined in their own files) pull a leaf area index; everything else is empty.
@inline interface_vegetation_state(i, j, grid, interface_model, vegetation, time_interpolator) = (;)

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes,
                                                           interface_temperature,
                                                           grid,
                                                           clock,
                                                           turbulent_flux_formulation,
                                                           land_state,
                                                           vegetation,
                                                           leaf_area_index_time_interpolator,
                                                           atmosphere_state,
                                                           interface_properties,
                                                           atmosphere_properties,
                                                           land_properties,
                                                           radiation_kernel_props,
                                                           radiation_exchanger_state)

    i, j = @index(Global, NTuple)
    time = Time(clock.time)

    @inbounds begin
        uᵃᵗ = atmosphere_state.u[i, j, 1]
        vᵃᵗ = atmosphere_state.v[i, j, 1]
        Tᵃᵗ = atmosphere_state.T[i, j, 1]
        pᵃᵗ = atmosphere_state.p[i, j, 1]
        qᵃᵗ = atmosphere_state.q[i, j, 1]
    end

    q_formulation = interface_properties.specific_humidity_formulation

    # Bulk land temperature serves as the initial skin-temperature guess.
    Tₛ = state2dindex(land_state.T, i, j)
    FT = typeof(Tₛ)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    zᵃᵗ = state2dindex(atmosphere_properties.surface_layer_height, i, j)

    local_atmosphere_state = (z = zᵃᵗ,
                              u = uᵃᵗ,
                              v = vᵃᵗ,
                              T = Tᵃᵗ,
                              p = pᵃᵗ,
                              q = qᵃᵗ,
                              h_bℓ = state2dindex(atmosphere_state.h_bℓ, i, j))

    # Surface velocities are zero for land.
    uₛ = zero(FT)
    vₛ = zero(FT)

    local_interior_state = (u = uₛ, v = vₛ, T = Tₛ)
    local_land_properties = local_atmosphere_land_surface_properties(land_properties, i, j)

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    # Estimate initial interface state. Use the saturated value as the initial
    # surface humidity guess (the solver recomputes it via the formulation).
    u★ = convert(FT, 1e-4)
    qₛ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, interface_phase(q_formulation)))
    initial_interface_state = AirLandInterfaceState(i, j, grid,
                                                    InterfaceFluxScales(u★, u★, u★),
                                                    InterfaceVelocities(uₛ, vₛ),
                                                    q_formulation, land_state,
                                                    vegetation, leaf_area_index_time_interpolator,
                                                    Tₛ, qₛ)

    interface_state = compute_interface_state(turbulent_flux_formulation,
                                              initial_interface_state,
                                              local_atmosphere_state,
                                              local_interior_state,
                                              radiation_state,
                                              interface_properties,
                                              atmosphere_properties,
                                              local_land_properties)

    u★ = interface_state.fluxes.u★
    θ★ = interface_state.fluxes.θ★
    q★ = interface_state.fluxes.q★

    Ψₛ = interface_state
    Ψₐ = local_atmosphere_state
    Δu, Δv = velocity_difference(interface_properties.velocity_formulation, Ψₐ, Ψₛ)
    ΔU = sqrt(Δu^2 + Δv^2)

    τˣ = ifelse(ΔU == 0, zero(grid), - u★^2 * Δu / ΔU)
    τʸ = ifelse(ΔU == 0, zero(grid), - u★^2 * Δv / ΔU)

    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)

    𝒬ᵛ  = interface_fluxes.latent_heat
    𝒬ᵀ  = interface_fluxes.sensible_heat
    Jᵛ  = interface_fluxes.water_vapor
    ρτˣ = interface_fluxes.x_momentum
    ρτʸ = interface_fluxes.y_momentum

    @inbounds begin
        𝒬ᵛ[i, j, 1]  = - ρᵃᵗ * ℒˡ * u★ * q★
        𝒬ᵀ[i, j, 1]  = - ρᵃᵗ * cᵖᵐ * u★ * θ★
        Jᵛ[i, j, 1]  = - ρᵃᵗ * u★ * q★
        ρτˣ[i, j, 1] = + ρᵃᵗ * τˣ
        ρτʸ[i, j, 1] = + ρᵃᵗ * τʸ

        interface_fluxes.friction_velocity[i, j, 1] = u★
        interface_fluxes.temperature_scale[i, j, 1] = θ★
        interface_fluxes.water_vapor_scale[i, j, 1] = q★
    end

    store_interface_temperature!(interface_temperature, i, j,
                                 interface_properties.temperature_formulation,
                                 interface_state, local_atmosphere_state, local_interior_state,
                                 radiation_state, atmosphere_properties)
end
