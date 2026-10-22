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
                     specific_humidity = temperature isa CanopyGroundSurface ? temperature : BulkHumidity())
    return LandSurface(temperature, specific_humidity, nothing, nothing)
end

"""
$(TYPEDSIGNATURES)

Vegetated and bare surface physics sharing a soil column, with vegetated `cover_fraction`.
"""
struct TiledLandSurface{V, B, F}
    vegetated :: V
    bare :: B
    cover_fraction :: F
end

function TiledLandSurface(vegetated::LandSurface; cover_fraction,
                          bare = LandSurface(EnergyBalanceTemperature(vegetated.temperature.ground_heat_flux;
                                                                      storage = DiagnosticSkin());
                                             specific_humidity = vegetated.temperature.soil_humidity))
    return TiledLandSurface(vegetated, bare, cover_fraction)
end

function materialize_land_surface(surface::LandSurface, grid)
    temperature = surface.temperature
    names = if temperature isa PrognosticCanopyGroundSurface
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
                     materialize_land_surface(surface.bare, grid), surface.cover_fraction)

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
Base.summary(surface::TiledLandSurface) = string("TiledLandSurface(cover_fraction=", prettysummary(surface.cover_fraction), ")")
Base.show(io::IO, surface::TiledLandSurface) = print(io, summary(surface))

# The tile whose canopy sets the land's defaults.
vegetated_tile(surface::LandSurface) = surface
vegetated_tile(surface::TiledLandSurface) = surface.vegetated

InterfaceComputations.land_surface(land::SlabLand) = land.surface
InterfaceComputations.land_surface_state(surface::LandSurface) = (; state=surface.prognostic, next=surface.next_state)
InterfaceComputations.default_al_temperature(land::SlabLand) = vegetated_tile(land.surface).temperature
InterfaceComputations.default_al_specific_humidity(temperature, land::SlabLand) = vegetated_tile(land.surface).specific_humidity

EarthSystemModels.initialize_land_surface!(land::SlabLand) = time_step!(land.surface)

InterfaceComputations.atmosphere_land_interface(grid, atmosphere, land, surface::TiledLandSurface;
                                                fluxes, velocity_difference, kw...) =
    TiledLandInterface(grid, atmosphere, land; vegetated_fluxes = fluxes, bare_fluxes = fluxes, velocity_difference)

materialize_surface_hydrology(hydrology, surface) = hydrology

# The store's leaf area index and cover fraction default to the canopy's and the vegetated fraction.
function materialize_surface_hydrology(hydrology::InterceptingHydrology, surface)
    leaf_area_index = something(hydrology.leaf_area_index,
                                canopy_leaf_area_index(vegetated_tile(surface).specific_humidity), Some(nothing))
    isnothing(leaf_area_index) && throw(ArgumentError("InterceptingHydrology requires a canopy surface or leaf_area_index."))
    cover_fraction = something(hydrology.cover_fraction, surface isa TiledLandSurface ? surface.cover_fraction : 1)
    return InterceptingHydrology(typeof(hydrology.capacity_per_leaf_area);
                                 soil = hydrology.soil, leaf_area_index, cover_fraction,
                                 capacity_per_leaf_area = hydrology.capacity_per_leaf_area,
                                 extinction = hydrology.extinction, clumping = hydrology.clumping)
end

soil_hydrology(hydrology) = hydrology
soil_hydrology(hydrology::InterceptingHydrology) = hydrology.soil

function InterfaceComputations.surface_hydrology(land::SlabLand, stress::PlantAvailableWaterStress, grid)
    hydrology = soil_hydrology(land.hydrology)
    return plant_available_water(hydrology, stress, grid)
end

# The stress endpoints are fixed by the retention curve, so they are evaluated once.
function plant_available_water(hydrology::VariablySaturatedHydrology, stress, grid)
    curve = hydrology.retention_curve
    FT = eltype(grid)
    endpoint(head) = compute!(Field(KernelFunctionOperation{Center, Center, Nothing}(effective_saturation,
                                                                                     grid, curve, convert(FT, head))))
    return (; field_capacity_saturation = endpoint(stress.field_capacity_head),
              wilting_point_saturation = endpoint(stress.wilting_point_head))
end

plant_available_water(hydrology, stress, grid) =
    throw(ArgumentError("PlantAvailableWaterStress requires soil water retention physics."))

function InterfaceComputations.surface_hydrology(land::SlabLand, humidity::DryLayerHumidity, grid)
    vapor = humidity.vapor_exchange
    diffusivity = soil_vapor_diffusivity(soil_hydrology(land.hydrology), vapor.tortuosity,
                                       vapor, land.saturation, grid)
    return (; effective_vapor_diffusivity=diffusivity)
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
    θˡ = θʳ + saturation[i, j, 1] * (ν - θʳ)
    return effective_vapor_diffusivity(vapor, ν, θˡ)
end

InterfaceComputations.surface_hydrology(land::SlabLand, humidity::CanopyConductanceHumidity, grid) =
    surface_hydrology(land, humidity.moisture_stress, grid)
InterfaceComputations.surface_hydrology(land::SlabLand, humidity::FractionalHumidity, grid) =
    surface_hydrology(land, humidity.efficiency, grid)
InterfaceComputations.surface_hydrology(land::SlabLand, canopy::CanopyGroundSurface, grid) =
    merge(surface_hydrology(land, canopy.soil_humidity, grid), surface_hydrology(land, canopy.canopy_humidity, grid))
