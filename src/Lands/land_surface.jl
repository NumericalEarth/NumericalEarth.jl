"""
$(TYPEDSIGNATURES)

Land-owned surface physics and thermal/moisture storage. `temperature` specifies the
surface balance and `specific_humidity` specifies evaporation and transpiration.

```jldoctest
using NumericalEarth

LandSurface()

# output
LandSurface(BulkTemperature)
```
"""
struct LandSurface{T, Q, P, N}
    temperature :: T
    specific_humidity :: Q
    prognostic :: P
    next_state :: N
end

function LandSurface(temperature = BulkTemperature();
                     specific_humidity = temperature isa CanopyAirSpace ? temperature : BulkHumidity())
    return LandSurface(temperature, specific_humidity, nothing, nothing)
end

"""
$(TYPEDSIGNATURES)

Vegetated and bare surface physics sharing a soil column, with vegetated area `fraction`.
"""
struct TiledLandSurface{V, B, F}
    vegetated :: V
    bare :: B
    fraction :: F
end

function TiledLandSurface(vegetated::LandSurface; fraction,
                          bare = LandSurface(EnergyBalanceTemperature(vegetated.temperature.ground_heat_flux;
                                                                      storage = DiagnosticSkin());
                                             specific_humidity = vegetated.temperature.soil))
    return TiledLandSurface(vegetated, bare, fraction)
end

function materialize_land_surface(surface::LandSurface, grid)
    temperature = surface.temperature
    names = if temperature isa PrognosticCanopyAirSpace
        (:temperature, :specific_humidity)
    elseif temperature isa PrognosticEnergyBalanceTemperature
        (:temperature,)
    else
        ()
    end
    prognostic = NamedTuple{names}(map(_ -> Field{Center, Center, Nothing}(grid), names))
    next_state = NamedTuple{names}(map(_ -> Field{Center, Center, Nothing}(grid), names))
    return LandSurface(temperature, surface.specific_humidity, prognostic, next_state)
end

materialize_land_surface(surface::TiledLandSurface, grid) =
    TiledLandSurface(materialize_land_surface(surface.vegetated, grid),
                     materialize_land_surface(surface.bare, grid), surface.fraction)

function time_step!(surface::LandSurface)
    foreach(values(surface.prognostic), values(surface.next_state)) do state, next
        copyto!(parent(state), parent(next))
        fill_halo_regions!(state)
    end
    return nothing
end

function time_step!(surface::TiledLandSurface)
    time_step!(surface.vegetated)
    time_step!(surface.bare)
    return nothing
end

Oceananigans.prognostic_state(surface::LandSurface) = map(prognostic_state, surface.prognostic)
Oceananigans.prognostic_state(surface::TiledLandSurface) =
    (; vegetated=prognostic_state(surface.vegetated), bare=prognostic_state(surface.bare))

function Oceananigans.restore_prognostic_state!(surface::LandSurface, state)
    for name in keys(surface.prognostic)
        restore_prognostic_state!(surface.prognostic[name], state[name])
    end
    return surface
end

function Oceananigans.restore_prognostic_state!(surface::TiledLandSurface, state)
    restore_prognostic_state!(surface.vegetated, state.vegetated)
    restore_prognostic_state!(surface.bare, state.bare)
    return surface
end

Base.summary(surface::LandSurface) = string("LandSurface(", summary(surface.temperature), ")")
Base.show(io::IO, surface::LandSurface) = print(io, summary(surface))
Base.summary(surface::TiledLandSurface) = string("TiledLandSurface(fraction=", prettysummary(surface.fraction), ")")
Base.show(io::IO, surface::TiledLandSurface) = print(io, summary(surface))

InterfaceComputations.default_al_temperature(land::SlabLand) =
    land.surface isa LandSurface ? land.surface.temperature : land.surface
InterfaceComputations.default_al_specific_humidity(temperature, land::SlabLand) =
    land.surface isa LandSurface ? land.surface.specific_humidity : nothing
InterfaceComputations.default_al_specific_humidity(temperature::CanopyAirSpace, land::SlabLand) = temperature

function InterfaceComputations.land_surface_state(land::SlabLand, temperature)
    surface = land.surface
    if surface isa TiledLandSurface
        surface = temperature isa CanopyAirSpace ? surface.vegetated : surface.bare
    end
    return (; state=surface.prognostic, next=surface.next_state)
end

EarthSystemModels.initialize_land_surface!(land::SlabLand) = time_step!(land.surface)

function InterfaceComputations.atmosphere_land_interface(grid, atmosphere, land, surface::TiledLandSurface;
                                                        fluxes, velocity_difference, specific_humidity)
    return TiledLandInterface(grid, atmosphere, land;
                              vegetated_fluxes = fluxes, bare_fluxes = fluxes, velocity_difference)
end

materialize_surface_hydrology(hydrology, surface) = hydrology

function materialize_surface_hydrology(hydrology::InterceptingHydrology, surface)
    tiled = surface isa TiledLandSurface
    canopy = tiled ? surface.vegetated : surface
    leaf_area_index = canopy_leaf_area_index(canopy.specific_humidity)
    fraction = tiled ? surface.fraction : 1
    if isnothing(leaf_area_index)
        leaf_area_index = hydrology.leaf_area_index
        fraction = isnothing(hydrology.cover_fraction) ? fraction : hydrology.cover_fraction
    else
        isnothing(hydrology.leaf_area_index) || isequal(hydrology.leaf_area_index, leaf_area_index) ||
            throw(ArgumentError("InterceptingHydrology and the land surface must share leaf_area_index."))
        isnothing(hydrology.cover_fraction) || isequal(hydrology.cover_fraction, fraction) ||
            throw(ArgumentError("InterceptingHydrology and the land surface must share cover_fraction."))
    end
    isnothing(leaf_area_index) && throw(ArgumentError("InterceptingHydrology requires a canopy surface or leaf_area_index."))
    return InterceptingHydrology(typeof(hydrology.capacity_per_leaf_area);
                                 soil = hydrology.soil, leaf_area_index, cover_fraction = fraction,
                                 capacity_per_leaf_area = hydrology.capacity_per_leaf_area,
                                 extinction = hydrology.extinction, clumping = hydrology.clumping)
end

soil_hydrology(hydrology) = hydrology
soil_hydrology(hydrology::InterceptingHydrology) = hydrology.soil

function InterfaceComputations.surface_hydrology(land::SlabLand, stress::PlantAvailableWaterStress, grid)
    hydrology = soil_hydrology(land.hydrology)
    return plant_available_water(hydrology, stress, grid)
end

function plant_available_water(hydrology::VariablySaturatedHydrology, stress, grid)
    curve = hydrology.retention_curve
    FT = eltype(grid)
    field_capacity_saturation = KernelFunctionOperation{Center, Center, Nothing}(retention_saturation,
        grid, curve, convert(FT, stress.field_capacity_head))
    wilting_saturation = KernelFunctionOperation{Center, Center, Nothing}(retention_saturation,
        grid, curve, convert(FT, stress.wilting_point_head))
    return (; field_capacity_saturation, wilting_saturation)
end

plant_available_water(hydrology, stress, grid) =
    throw(ArgumentError("PlantAvailableWaterStress requires soil water retention physics."))

@inline retention_saturation(i, j, k, grid, curve, head) = effective_saturation(i, j, grid, curve, head)

function InterfaceComputations.surface_hydrology(land::SlabLand, humidity::DryLayerHumidity, grid)
    vapor = humidity.vapor_exchange
    diffusivity = soil_vapor_diffusivity(soil_hydrology(land.hydrology), vapor.tortuosity,
                                       vapor, land.saturation, grid)
    return (; vapor_diffusivity=diffusivity)
end

soil_vapor_diffusivity(hydrology, ::ConstantTortuosity, vapor, saturation, grid) = vapor.molecular_diffusivity

function soil_vapor_diffusivity(hydrology::VariablySaturatedHydrology, ::PowerLawTortuosity, vapor, saturation, grid)
    return KernelFunctionOperation{Center, Center, Nothing}(surface_vapor_diffusivity, grid,
        saturation, layer(hydrology.porosity, 1), layer(hydrology.residual_liquid_fraction, 1), vapor)
end

@inline function surface_vapor_diffusivity(i, j, k, grid, saturation, porosity, residual, vapor)
    FT = eltype(grid)
    ν = convert(FT, stateindex(porosity, i, j, 1))
    θʳ = convert(FT, stateindex(residual, i, j, 1))
    θ = θʳ + saturation[i, j, 1] * (ν - θʳ)
    return effective_vapor_diffusivity(vapor, ν, θ)
end

InterfaceComputations.surface_hydrology(land::SlabLand, humidity::CanopyConductanceHumidity, grid) =
    surface_hydrology(land, humidity.moisture_stress, grid)
InterfaceComputations.surface_hydrology(land::SlabLand, humidity::FractionalHumidity, grid) =
    surface_hydrology(land, humidity.efficiency, grid)
InterfaceComputations.surface_hydrology(land::SlabLand, canopy::CanopyAirSpace, grid) =
    merge(surface_hydrology(land, canopy.soil, grid), surface_hydrology(land, canopy.canopy, grid))
