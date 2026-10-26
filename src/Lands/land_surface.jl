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

function time_step!(surface::LandSurface)
    foreach(values(surface.prognostic), values(surface.next_state)) do state, next
        copyto!(parent(state), parent(next))
        fill_halo_regions!(state)
    end
    return nothing
end

Oceananigans.prognostic_state(surface::LandSurface) = map(prognostic_state, surface.prognostic)

function Oceananigans.restore_prognostic_state!(surface::LandSurface, state)
    for name in keys(surface.prognostic)
        restore_prognostic_state!(surface.prognostic[name], state[name])
    end
    return surface
end

Base.summary(surface::LandSurface) = string("LandSurface(", summary(surface.temperature), ")")
Base.show(io::IO, surface::LandSurface) = print(io, summary(surface))

InterfaceComputations.land_surface(land::SlabLand) = land.surface
InterfaceComputations.land_surface_state(surface::LandSurface) = (; state=surface.prognostic, next=surface.next_state)
InterfaceComputations.default_al_temperature(land::SlabLand) = land.surface.temperature
InterfaceComputations.default_al_specific_humidity(temperature, land::SlabLand) =
    temperature isa CanopyGroundSurface ? temperature : land.surface.specific_humidity

EarthSystemModels.initialize_land_surface!(land::SlabLand) = time_step!(land.surface)

materialize_surface_hydrology(hydrology, surface) = hydrology

# The store's leaf area index defaults to the canopy's.
function materialize_surface_hydrology(hydrology::InterceptingHydrology, surface)
    leaf_area_index = something(hydrology.leaf_area_index,
                                canopy_leaf_area_index(surface.specific_humidity), Some(nothing))
    isnothing(leaf_area_index) && throw(ArgumentError("InterceptingHydrology requires a canopy surface or leaf_area_index."))
    return InterceptingHydrology(typeof(hydrology.capacity_per_leaf_area);
                                 soil = hydrology.soil, leaf_area_index,
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
