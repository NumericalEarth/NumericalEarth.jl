# Model builders shared by the column, map and GPU scripts: the vegetated slab land of
# `experiments/conus_slab_canopy_v2` (variably saturated soil under a surface pond and an
# interception store, water-coupled energy, a two-tile canopy-air-space interface) forced
# by an in-memory ERA5 atmosphere and radiation. Every per-cell parameter arrives either
# as a `Number` (column) or as a `Field` on the model grid (map); the same builders serve
# CPU, GPU and `ReactantState` grids because fields are filled through their `parent`.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: Clock, update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations: atmosphere_land_stability_functions, FixedIterations

surface_field(grid) = Field{Center, Center, Nothing}(grid)

# A per-cell parameter on `grid`: a number stays a number; an array of cell values (or a
# CPU field's parent) is copied into a fresh field through its parent, which works on
# every architecture including `ReactantState`.
surface_property(grid, value::Number) = value
function surface_property(grid, values::AbstractArray)
    field = surface_field(grid)
    if size(values) == size(parent(field))
        parent(field) .= values
    else
        cpu = surface_field(Oceananigans.Architectures.on_architecture(CPU(), grid))
        set!(cpu, values)
        parent(field) .= parent(cpu)
    end
    return field
end

# In-memory ERA5 forcing on `grid` from the cached hourly slices (`borneo_setup.jl`); each
# slice is a number (column) or a parent-layout array (map).
function forcing_on(grid, forcing; land_surface, surface_layer_height, boundary_layer_height)
    times = forcing.times
    atmosphere = PrescribedAtmosphere(grid, times; surface_layer_height, boundary_layer_height)
    radiation  = PrescribedRadiation(grid, times; land_surface, ocean_surface = nothing, sea_ice_surface = nothing)
    for n in eachindex(times)
        parent(atmosphere.velocities.u[n])            .= forcing.u[n]
        parent(atmosphere.velocities.v[n])            .= forcing.v[n]
        parent(atmosphere.temperature[n])             .= forcing.T[n]
        parent(atmosphere.specific_humidity[n])       .= forcing.q[n]
        parent(atmosphere.pressure[n])                .= forcing.p[n]
        parent(atmosphere.precipitation_flux.rain[n]) .= forcing.rain[n]
        parent(radiation.downwelling_shortwave[n])    .= forcing.sw[n]
        parent(radiation.downwelling_longwave[n])     .= forcing.lw[n]
    end
    update_state!(atmosphere)
    update_state!(radiation)
    return atmosphere, radiation
end

# The land: `s` holds the per-cell surface parameters (numbers or fields on `grid`).
function borneo_land(grid, FT, s; slab_depth, deep_liquid_flux = FreeDrainageFlux(), deep_pressure_head = 0,
                     infiltration_capacity = s.infiltration_capacity)
    soil = VariablySaturatedHydrology(FT;
        slab_depth,
        storage_height = 1000,
        porosity = s.porosity,
        residual_liquid_fraction = s.residual_liquid_fraction,
        retention_curve = VanGenuchtenRetention(FT;
            inverse_air_entry_head = s.inverse_air_entry_head,
            pore_size_uniformity   = s.pore_size_uniformity),
        hydraulic_conductivity = VanGenuchtenConductivity(FT;
            matching_point_conductivity = s.matching_point_conductivity,
            pore_size_uniformity        = s.pore_size_uniformity,
            pore_connectivity_exponent  = s.pore_connectivity_exponent),
        deep_liquid_flux, deep_pressure_head,
        runoff = InfiltrationCapacityRunoff(FT; infiltration_capacity))

    hydrology = InterceptingHydrology(FT;
        soil = SurfaceWaterStore(FT; soil, drainage_timescale = 1hour),
        leaf_area_index = s.leaf_area_index,
        capacity_per_leaf_area = 0.2,
        drainage_smoothing_width = 0.05)

    energy = WaterCoupledEnergy(FT;
        dry_heat_capacity = s.dry_heat_capacity,
        liquid_heat_capacity = 4186,
        deep_temperature = s.deep_temperature,
        deep_time_scale = 1day)

    return SlabLand(grid; energy, hydrology)
end

# The two-tile canopy interface. Fixed iteration counts keep the Monin–Obukhov solve a
# static graph for Reactant (and every GPU thread on the same path).
function borneo_interface(grid, FT, atmosphere, land, s; inner_iterations = 16, similarity_iterations = 8)
    dry_layer_soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.05,
                                                    dry_layer_onset_saturation = 1.0,
                                                    dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                     molecular_diffusivity = 2.4e-5,
                                                     tortuosity = PowerLawTortuosity()),
        thermal_exchange_depth = 0.05,
        porosity = s.scalar_porosity)

    canopy = CanopyConductanceHumidity(FT;
        leaf_area_index = s.tile_lai,
        conductance = JarvisConductance(FT),
        moisture_stress = PlantAvailableWaterStress(FT),
        absorbed_par = InteractiveAbsorbedPAR(FT))

    vegetated = CanopyAirSpace(FT;
        soil = dry_layer_soil,
        canopy,
        soil_skin_flux = SoilConductiveFlux(FT(1.5), FT(0.05)),
        undercanopy_conductance = FrictionVelocityUndercanopyConductance(FT),
        inner_iterations,
        interception = CanopyInterception(),
        storage = PrognosticCanopyAir(layer_depth = s.canopy_height),
        leaf_albedo = s.albedo,
        ground_albedo = s.albedo,
        ground_emissivity = s.emissivity)

    land_fluxes(ℓᵐ, ℓˢ) = SimilarityTheoryFluxes(FT;
        momentum_roughness_length    = ℓᵐ,
        temperature_roughness_length = ℓˢ,
        water_vapor_roughness_length = ℓˢ,
        stability_functions          = atmosphere_land_stability_functions(FT),
        solver_stop_criteria         = FixedIterations(similarity_iterations))

    return TiledLandInterface(grid, atmosphere, land;
        vegetated,
        fraction = s.vegetation_fraction,
        vegetated_fluxes = land_fluxes(s.vegetated_roughness_length, s.vegetated_scalar_roughness_length),
        bare_fluxes      = land_fluxes(s.bare_roughness_length, s.bare_scalar_roughness_length))
end

function borneo_coupled_model(grid, FT, forcing, s; slab_depth, exchanger_correction = nothing,
                              surface_layer_height = 10, boundary_layer_height = 800,
                              inner_iterations = 16, similarity_iterations = 8,
                              deep_liquid_flux = FreeDrainageFlux(), deep_pressure_head = 0,
                              infiltration_capacity = s.infiltration_capacity)
    land_surface = SurfaceRadiationProperties(s.albedo, s.emissivity)
    atmosphere, radiation = forcing_on(grid, forcing; land_surface, surface_layer_height, boundary_layer_height)
    land = borneo_land(grid, FT, s; slab_depth, deep_liquid_flux, deep_pressure_head, infiltration_capacity)
    interface = borneo_interface(grid, FT, atmosphere, land, s; inner_iterations, similarity_iterations)
    return AtmosphereLandModel(atmosphere, land; radiation,
                               atmosphere_land_interface = interface,
                               exchanger_correction,
                               clock = Clock(grid))
end

# The names in `static_r*.jld2` the model reads per cell, plus the derived scalar
# roughness lengths.
surface_parameter_names = (:porosity, :residual_liquid_fraction, :inverse_air_entry_head,
                           :pore_size_uniformity, :matching_point_conductivity, :pore_connectivity_exponent,
                           :leaf_area_index, :tile_lai, :vegetation_fraction, :canopy_height,
                           :vegetated_roughness_length, :bare_roughness_length,
                           :albedo, :emissivity, :dry_heat_capacity, :deep_temperature)

# Per-cell parameters as numbers (one column, cell `(i, j)`) or as fields on `grid`.
function surface_parameters(static, grid, FT, cell = nothing)
    pick(a) = isnothing(cell) ? surface_property(grid, FT.(a)) : FT(a[cell...])
    values = NamedTuple{surface_parameter_names}(map(name -> pick(static[name]), surface_parameter_names))
    return merge(values, (; vegetated_scalar_roughness_length = pick(static.vegetated_roughness_length ./ 10),
                            bare_scalar_roughness_length      = pick(static.bare_roughness_length ./ 10),
                            infiltration_capacity = FT(static.infiltration_capacity),
                            scalar_porosity       = FT(static.scalar_porosity)))
end

# Forcing slices of one column of the (CPU) grid the slices were stored on.
column_forcing(forcing, grid, cell) = (; times = forcing.times,
    (name => [slice[cell[1] + grid.Hx, cell[2] + grid.Hy, 1] for slice in forcing[name]]
     for name in (:u, :v, :T, :q, :p, :rain, :sw, :lw))...)
