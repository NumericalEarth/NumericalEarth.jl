# Idealized 0D forcing, parameters, the bare slab of `examples/era5_forced_slab_land.jl`,
# and the reset/loss/gradient functions shared by the compile-scaling diagnostic and the
# NaN probe.

include(joinpath(@__DIR__, "borneo_model.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using Printf

FT = Float64
Δt = 10minutes
times = range(0, 30 * 3600, step = 3600)
backend = get(ENV, "ARCH", "cpu")
Reactant.set_default_backend(backend)

air_temperature(t)       = 299 - 3 * cos(2π * t / day)
downwelling_shortwave(t) = max(0, 800 * cos(2π * (t - day/2) / day))
rain_rate(t)             = ifelse(6hours ≤ t ≤ 8hours, 2e-3, 0.0)
idealized_forcing = (; times = collect(times),
    u = [3.0 for t in times], v = [0.0 for t in times],
    T = [air_temperature(t) for t in times], q = [0.016 for t in times], p = [100000.0 for t in times],
    rain = [rain_rate(t) for t in times],
    sw = [downwelling_shortwave(t) for t in times], lw = [400.0 for t in times])

parameters = (; porosity = 0.45, residual_liquid_fraction = 0.06, inverse_air_entry_head = 2.0,
                pore_size_uniformity = 1.4, matching_point_conductivity = 5e-6, pore_connectivity_exponent = 0.5,
                leaf_area_index = 4.0, tile_lai = 4.5, vegetation_fraction = 0.9, canopy_height = 25.0,
                vegetated_roughness_length = 1.0, bare_roughness_length = 0.05,
                vegetated_scalar_roughness_length = 0.1, bare_scalar_roughness_length = 0.005,
                albedo = 0.13, emissivity = 0.97, dry_heat_capacity = 840 * 0.15 * 1200.0,
                deep_temperature = 298.0, infiltration_capacity = 2e-3, scalar_porosity = 0.45)
θ₀, T₀, q₀, θ_target = 0.30, 298.0, 0.016, 0.33

# The bare slab of the ERA5 example: same soil and energy closures, a single dry-layer
# humidity interface, no canopy, no interception or pond.
function bare_slab_model(grid, s; similarity_iterations = 8)
    atmosphere, radiation = forcing_on(grid, idealized_forcing; land_surface = SurfaceRadiationProperties(s.albedo, s.emissivity),
                                       surface_layer_height = 10, boundary_layer_height = 800)
    soil = VariablySaturatedHydrology(FT; slab_depth = surface_field(grid), storage_height = 1000,
        porosity = s.porosity, residual_liquid_fraction = s.residual_liquid_fraction,
        retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = s.inverse_air_entry_head, pore_size_uniformity = s.pore_size_uniformity),
        hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = s.matching_point_conductivity,
            pore_size_uniformity = s.pore_size_uniformity, pore_connectivity_exponent = s.pore_connectivity_exponent),
        deep_liquid_flux = FreeDrainageFlux(), runoff = InfiltrationCapacityRunoff(FT; infiltration_capacity = s.infiltration_capacity))
    energy = WaterCoupledEnergy(FT; dry_heat_capacity = s.dry_heat_capacity, liquid_heat_capacity = 4186,
                                deep_temperature = s.deep_temperature, deep_time_scale = 1day)
    land = SlabLand(grid; energy, hydrology = soil)
    humidity = DryLayerHumidity(FT; dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 1.0, dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3, molecular_diffusivity = 2.4e-5, tortuosity = PowerLawTortuosity()),
        thermal_exchange_depth = 0.05, porosity = s.scalar_porosity)
    fluxes = SimilarityTheoryFluxes(FT; momentum_roughness_length = s.vegetated_roughness_length,
        temperature_roughness_length = s.vegetated_scalar_roughness_length, water_vapor_roughness_length = s.vegetated_scalar_roughness_length,
        stability_functions = atmosphere_land_stability_functions(FT), solver_stop_criteria = FixedIterations(similarity_iterations))
    interface = atmosphere_land_interface(grid, atmosphere, land; specific_humidity = humidity, fluxes)
    return AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = interface, clock = Clock(grid))
end

soil_hydrology(model) = model.land.hydrology isa VariablySaturatedHydrology ? model.land.hydrology : model.land.hydrology.soil.soil

function reset!(model, h)
    hydrology = soil_hydrology(model)
    parent(hydrology.slab_depth) .= parent(h)
    parent(model.land.water_storage) .= 1000 .* θ₀ .* parent(h)
    parent(model.land.temperature) .= T₀
    parent(model.land.saturation) .= clamp((θ₀ - hydrology.residual_liquid_fraction) / (hydrology.porosity - hydrology.residual_liquid_fraction), 0, 1)
    for field in values(model.land.prognostic)
        parent(field) .= 0
    end
    interface = model.interfaces.atmosphere_land_interface
    if interface isa TiledLandInterface
        for tile in (interface.vegetated, interface.bare)
            parent(tile.temperature.state.temperature) .= T₀
            parent(tile.temperature.state.specific_humidity) .= q₀
        end
    end
    update_state!(model)   # fluxes consistent with the reset state
    return nothing
end

soil_water(model, h) = parent(model.land.water_storage) ./ (1000 .* parent(h))

forward_step!(model, Δt) = (time_step!(model, Δt); nothing)

function loss(model, h, Δt, nsteps)
    reset!(model, h)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return sum((soil_water(model, h) .- θ_target).^2)
end

function grad(model, dmodel, h, dh, Δt, nsteps)
    parent(dh) .= 0
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(h, dh), Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dh, L
end

