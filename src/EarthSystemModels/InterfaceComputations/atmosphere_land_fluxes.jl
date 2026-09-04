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

    if requires_retention_curve(specific_humidity) && isnothing(surface_retention_curve(land))
        throw(ArgumentError("$(summary(specific_humidity)) needs a soil retention curve, " *
                            "which $(summary(land)) does not carry"))
    end

    al_fluxes = AtmosphereSurfaceFluxes(grid)
    al_properties = InterfaceProperties(specific_humidity, temperature, velocity_difference)
    interface_temperature = build_interface_temperature(temperature, grid)
    return AtmosphereInterface(al_fluxes, fluxes, interface_temperature, al_properties)
end

# The atmosphere-facing interface temperature: a single field, or a
# `CanopyAirSpaceDiagnostics` carrying the two skins, the ground heat flux, and the
# per-source sensible and latent shares.
@inline build_interface_temperature(temperature_formulation, grid) = Field{Center, Center, Nothing}(grid)
@inline build_interface_temperature(cas::CanopyAirSpace, grid) = CanopyAirSpaceDiagnostics(grid, cas.storage)

# Store the diagnostic surface temperature(s) from the converged interface state.
@inline store_interface_temperature!(Ts, i, j, formulation, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ) =
    (@inbounds Ts[i, j, 1] = Ψₛ.temperature; nothing)

@inline function store_interface_temperature!(Ts, i, j, cas::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    sol = canopy_air_space_solve(cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    @inbounds begin
        Ts.interface[i, j, 1]              = sol.Tᵃᶜ
        Ts.canopy[i, j, 1]                 = sol.Tˡᵉᵃᶠ
        Ts.soil_skin[i, j, 1]              = sol.Tᵍ
        Ts.effective[i, j, 1]              = sol.effective_temperature
        Ts.ground_heat_flux[i, j, 1]        = sol.𝒬ᵍ
        Ts.canopy_latent_heat[i, j, 1]     = sol.LEˡᵉᵃᶠ
        Ts.soil_latent_heat[i, j, 1]       = sol.LEᵍ
        Ts.canopy_sensible_heat[i, j, 1]   = sol.Hˡᵉᵃᶠ
        Ts.soil_sensible_heat[i, j, 1]     = sol.Hᵍ
        Ts.canopy_evaporation[i, j, 1]     = sol.Eʷᵉᵗ
        Ts.canopy_wet_latent_heat[i, j, 1] = sol.LEʷᵉᵗ
    end
    return nothing
end

# Initial interface values for the fixed point. Diagnostic formulations cold-start from
# the bulk land temperature and its saturation humidity; a prognostic `CanopyAirSpace`
# reads the stored node back once the clock has taken a step.
@inline clock_has_stepped(clock) =
    (clock.iteration > 0) & isfinite(clock.last_Δt) & (clock.last_Δt > 0)

@inline initial_interface_values(formulation, Ts, i, j, T₀, q₀, clock) = (T₀, q₀)

@inline function initial_interface_values(::PrognosticCanopyAirSpace, Ts::CanopyAirSpaceDiagnostics,
                                          i, j, T₀, q₀, clock)
    stepped = clock_has_stepped(clock)
    @inbounds T = ifelse(stepped, Ts.state.temperature[i, j, 1], T₀)
    @inbounds q = ifelse(stepped, Ts.state.specific_humidity[i, j, 1], q₀)
    return T, q
end

# Finalize the interface state after the fixed point: store diagnostics and, for
# prognostic storage, advance the stored state and return the state whose flux
# scales the kernel exports. The default (all diagnostic formulations) is exactly
# the previous behavior.
@inline function advance_interface_state!(Ts, i, j, formulation, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, clock)
    store_interface_temperature!(Ts, i, j, formulation, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return Ψₛ
end

# Prognostic canopy air: the node was frozen through the fixed point, so the skins
# equilibrate against it once here; the node then advances by the exponential
# relaxation toward the conductance-weighted equilibrium, and the exported scales
# are re-evaluated at the step-mean node state, closing the step ledger exactly:
# flux to the atmosphere = Kirchhoff supply − storage tendency. The stored
# diagnostics (shares, node) are the step-mean-consistent ones the ledger uses.
@inline function advance_interface_state!(Ts, i, j, cas::PrognosticCanopyAirSpace,
                                          Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, clock)
    sol = canopy_air_space_solve(cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)   # node frozen inside

    FT  = eltype(Ψₛ)
    # Δt = 0 before any time step (including the first-time-step preparation call):
    # the advance then lands on the equilibrium — the diagnostic initialization.
    Δt  = ifelse(clock_has_stepped(clock), convert(FT, clock.last_Δt), zero(FT))
    h_c = convert(FT, state2dindex(cas.storage.layer_depth, i, j))
    Cᵀ = sol.ρᵃᵗ * sol.cᵖ * h_c
    Cᵛ = sol.ρᵃᵗ * h_c

    T⁻, q⁻ = Ψₛ.temperature, Ψₛ.specific_humidity
    T⁺ = advance_canopy_air(T⁻, sol.T_eq, sol.Σgᵀ, Cᵀ, Δt)
    q⁺ = advance_canopy_air(q⁻, sol.q_eq, sol.Σgᵛ, Cᵛ, Δt)
    T̄  = step_mean_canopy_air(T⁻, sol.T_eq, sol.Σgᵀ, Cᵀ, Δt)
    q̄  = step_mean_canopy_air(q⁻, sol.q_eq, sol.Σgᵛ, Cᵛ, Δt)

    @inbounds begin
        Ts.state.temperature[i, j, 1]       = T⁺
        Ts.state.specific_humidity[i, j, 1] = q⁺
        Ts.interface[i, j, 1]              = T⁺
        Ts.canopy[i, j, 1]                 = sol.Tˡᵉᵃᶠ
        Ts.soil_skin[i, j, 1]              = sol.Tᵍ
        Ts.effective[i, j, 1]              = sol.effective_temperature
        Ts.ground_heat_flux[i, j, 1]        = sol.𝒬ᵍ
        Ts.canopy_latent_heat[i, j, 1]     = sol.ℒ * sol.gˡᵉᵃᶠᵛ * (sol.qˡᵉᵃᶠ - q̄)
        Ts.soil_latent_heat[i, j, 1]       = sol.ℒ * sol.Gᵉ * (sol.qᵉ - q̄)
        Ts.canopy_sensible_heat[i, j, 1]   = sol.gˡᵉᵃᶠᵀ * (sol.Tˡᵉᵃᶠ - T̄)
        Ts.soil_sensible_heat[i, j, 1]     = sol.gᵍᵀ * (sol.Tᵍ - T̄)
        Ts.canopy_evaporation[i, j, 1]     = sol.Eʷᵉᵗ
        Ts.canopy_wet_latent_heat[i, j, 1] = sol.LEʷᵉᵗ
    end

    # Exported scales at the step-mean node (the same floored transfer coefficients
    # the node balance uses), so −ρ cᵖ u★ θ★ = gᵃᵀ (T̄ − θᵃᵗ) and the vapor analog.
    # `InterfaceFluxScales` fields share one type: convert, since the thermodynamic
    # constants (and hence θᵃᵗ) may carry a wider float type than the state.
    θᵃᵗ = convert(FT, surface_atmosphere_temperature(Ψₐ, ℙₐ))
    χθ⁺ = max(zero(FT), Ψₛ.fluxes.χθ)
    χq⁺ = max(zero(FT), Ψₛ.fluxes.χq)
    fluxes = InterfaceFluxScales(Ψₛ.fluxes.u★,
                                 convert(FT, χθ⁺ * (θᵃᵗ - T̄)),
                                 convert(FT, χq⁺ * (Ψₐ.q - q̄)),
                                 Ψₛ.fluxes.χθ, Ψₛ.fluxes.χq)
    return rebuild_interface_state(Ψₛ, fluxes, convert(FT, T⁺), convert(FT, q⁺))
end

# A prognostic energy-balance skin reads its stored temperature back from the
# interface-temperature field; before any time step it starts from the bulk guess,
# and the advance then initializes the field at the equilibrium root.
@inline function initial_interface_values(::PrognosticEnergyBalanceTemperature, Ts,
                                          i, j, T₀, q₀, clock)
    stepped = clock_has_stepped(clock)
    @inbounds T = ifelse(stepped, Ts[i, j, 1], T₀)
    return T, q₀
end

# Prognostic energy-balance skin: frozen through the fixed point, advanced once per
# step by a backward-Euler update of C dTₛ/dt = Rₙ + G − H − LE, solved by a fixed
# three-iteration Newton on R(T) = C (T − Tₛ)/Δt − F(T) (re-linearizing the radiative
# and vapor curvature each iterate, so violent adjustment steps stay energy-consistent).
# The imbalance the massless solve has to dissipate instantly lands in the storage
# tendency instead. Exported scales are re-evaluated at the end-of-step skin.
@inline function advance_interface_state!(Ts, i, j, t::PrognosticEnergyBalanceTemperature,
                                          Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, clock)
    FT = eltype(Ψₛ)
    Tₛ = Ψₛ.temperature
    C  = convert(FT, state2dindex(t.storage.heat_capacity, i, j))
    stepped = clock_has_stepped(clock)
    # Δt⁻¹ = 0 before any time step (including the first-time-step preparation call):
    # the Newton then lands on the equilibrium root — the diagnostic initialization,
    # as the prognostic canopy-air node does.
    Δt⁻¹ = ifelse(stepped, 1 / convert(FT, clock.last_Δt), zero(FT))

    T⁺ = Tₛ
    for _ in 1:3
        F, Σλ = skin_energy_imbalance(T⁺, t, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ)
        R  = C * (T⁺ - Tₛ) * Δt⁻¹ - F
        dR = C * Δt⁻¹ + Σλ
        # dR = 0 only before the first step (Δt⁻¹ = 0) on a skin with no restoring
        # conductance at all — no conduction, no radiative feedback, no turbulent
        # exchange. F is then independent of T and there is no root to step toward.
        T⁺ = ifelse(dR > 0, T⁺ - R / dR, T⁺)
    end

    @inbounds Ts[i, j, 1] = T⁺

    u★  = Ψₛ.fluxes.u★
    χθ⁺ = max(zero(FT), Ψₛ.fluxes.χθ)
    χq⁺ = max(zero(FT), Ψₛ.fluxes.χq)
    Tᵃᵗ = convert(FT, surface_atmosphere_temperature(Ψₐ, ℙₐ))
    q⁺  = compute_interface_humidity(ℙₛ.specific_humidity_formulation, T⁺, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    fluxes = InterfaceFluxScales(u★, convert(FT, χθ⁺ * (Tᵃᵗ - T⁺)), convert(FT, χq⁺ * (Ψₐ.q - q⁺)),
                                 Ψₛ.fluxes.χθ, Ψₛ.fluxes.χq)
    return rebuild_interface_state(Ψₛ, fluxes, convert(FT, T⁺), convert(FT, q⁺))
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
            radiation_kernel_props,
            radiation_state)

    return nothing
end

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

    # `CanopyAirSpace` optics slots may be per-cell `Field`s; collapse them to this cell's
    # values before the index-free solve.
    temperature_formulation = local_interface_formulation(interface_properties.temperature_formulation, i, j)
    q_formulation           = local_interface_formulation(interface_properties.specific_humidity_formulation, i, j)

    local_interface_properties = InterfaceProperties(q_formulation, temperature_formulation,
                                                     interface_properties.velocity_formulation)

    # Bulk land temperature serves as the initial skin-temperature guess.
    Tₛ = state2dindex(land_state.T, i, j)
    FT = typeof(Tₛ)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    zᵃᵗ = state2dindex(atmosphere_properties.surface_layer_height, i, j)

    # Collapse Field-valued roughness lengths and displacement to this cell's values.
    local_turbulent_flux_formulation = local_flux_formulation(turbulent_flux_formulation, i, j)

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

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    # Estimate initial interface state. Use the saturated value as the initial
    # surface humidity guess (the solver recomputes it via the formulation);
    # prognostic formulations read their stored state back instead.
    u★ = convert(FT, 1e-4)
    qₛ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, interface_phase(q_formulation)))
    Tₛ, qₛ = initial_interface_values(temperature_formulation,
                                      interface_temperature, i, j, Tₛ, qₛ, clock)
    initial_interface_state = AirLandInterfaceState(i, j, grid,
                                                    InterfaceFluxScales(u★, u★, u★),
                                                    InterfaceVelocities(uₛ, vₛ),
                                                    q_formulation, land_state,
                                                    vegetation, leaf_area_index_time_interpolator,
                                                    Tₛ, qₛ)

    interface_state = compute_interface_state(local_turbulent_flux_formulation,
                                              initial_interface_state,
                                              local_atmosphere_state,
                                              local_interior_state,
                                              radiation_state,
                                              local_interface_properties,
                                              atmosphere_properties,
                                              (;))

    # Store diagnostics; prognostic formulations also advance their stored state and
    # hand back the state whose scales the flux exports below use (step-mean values,
    # so the step energy/vapor ledger closes against the storage tendency).
    interface_state = advance_interface_state!(interface_temperature, i, j,
                                               temperature_formulation,
                                               interface_state, local_atmosphere_state,
                                               local_interior_state, radiation_state,
                                               local_interface_properties, atmosphere_properties,
                                               clock)

    u★ = interface_state.fluxes.u★
    θ★ = interface_state.fluxes.θ★
    q★ = interface_state.fluxes.q★

    Ψₛ = interface_state
    Ψₐ = local_atmosphere_state
    Δu, Δv = velocity_difference(local_interface_properties.velocity_formulation, Ψₐ, Ψₛ)
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
end
