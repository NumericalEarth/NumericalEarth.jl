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
InterfaceComputations.default_al_specific_humidity(temperature, land::SlabLand) =
    temperature isa CanopyGroundSurface ? temperature : vegetated_tile(land.surface).specific_humidity

EarthSystemModels.initialize_land_surface!(land::SlabLand) = time_step!(land.surface)

InterfaceComputations.atmosphere_land_interface(grid, atmosphere, land, surface::TiledLandSurface;
                                                fluxes, velocity_difference, kw...) =
    TiledLandInterface(grid, atmosphere, land; vegetated_fluxes = fluxes, bare_fluxes = fluxes, velocity_difference)

materialize_surface_hydrology(hydrology, surface) = hydrology

function materialize_surface_hydrology(hydrology::InterceptingHydrology, surface)
    leaf_area_index = something(hydrology.leaf_area_index,
                                canopy_leaf_area_index(vegetated_tile(surface).specific_humidity), Some(nothing))
    isnothing(leaf_area_index) && throw(ArgumentError("InterceptingHydrology requires a canopy surface or leaf_area_index."))
    cover_fraction = hydrology.cover_fraction
    if surface isa TiledLandSurface
        surface_fraction = convert_if_number(typeof(hydrology.capacity_per_leaf_area), surface.cover_fraction)
        matching_fraction = cover_fraction isa Number && surface_fraction isa Number ?
                            cover_fraction == surface_fraction : cover_fraction === surface_fraction
        isnothing(cover_fraction) || matching_fraction ||
            throw(ArgumentError("InterceptingHydrology cover_fraction must match the tiled surface's value or share its field or time series."))
        cover_fraction = surface_fraction
    else
        cover_fraction = something(cover_fraction, 1)
    end
    return InterceptingHydrology(typeof(hydrology.capacity_per_leaf_area);
                                 soil = hydrology.soil, leaf_area_index, cover_fraction,
                                 capacity_per_leaf_area = hydrology.capacity_per_leaf_area,
                                 extinction = hydrology.extinction, clumping = hydrology.clumping)
end

soil_hydrology(hydrology) = hydrology
soil_hydrology(hydrology::InterceptingHydrology) = hydrology.soil

# The plant-stress endpoints are fixed by the soil's retention curve, so they are evaluated once.
(s::InterfaceComputations.SaturationAtSuction)(land::SlabLand) =
    saturation_at_suction(soil_hydrology(land.hydrology), s.suction_head, land.grid)

function saturation_at_suction(hydrology::VariablySaturatedHydrology, suction_head, grid)
    saturation = KernelFunctionOperation{Center, Center, Nothing}(effective_saturation, grid, hydrology.retention_curve,
                                                                  convert(eltype(grid), suction_head))
    return compute!(Field(saturation))
end

saturation_at_suction(hydrology, suction_head, grid) =
    throw(ArgumentError("PlantAvailableWaterStress requires soil water retention physics."))

function (d::InterfaceComputations.SoilVaporDiffusivity)(land::SlabLand)
    vapor = d.vapor_exchange
    return soil_vapor_diffusivity(soil_hydrology(land.hydrology), vapor.tortuosity, vapor, land.saturation, land.grid)
end

soil_vapor_diffusivity(hydrology, ::ConstantTortuosity, vapor, saturation, grid) = vapor.molecular_diffusivity

function soil_vapor_diffusivity(hydrology::VariablySaturatedHydrology, ::PowerLawTortuosity, vapor, saturation, grid)
    return KernelFunctionOperation{Center, Center, Nothing}(surface_vapor_diffusivity, grid,
        saturation, hydrology.porosity, hydrology.residual_liquid_fraction, vapor)
end

@inline function surface_vapor_diffusivity(i, j, k, grid, saturation, porosity, residual, vapor)
    FT = eltype(grid)
    ν = convert(FT, stateindex(porosity, i, j, 1))
    θʳ = convert(FT, stateindex(residual, i, j, 1))
    θˡ = θʳ + saturation[i, j, 1] * (ν - θʳ)
    return effective_vapor_diffusivity(vapor, ν, θˡ)
end
