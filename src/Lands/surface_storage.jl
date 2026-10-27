# TODO: Make the atmosphere–land interface prognostic and move canopy-air storage
# ownership from SlabLand into the interface.

"""
$(TYPEDSIGNATURES)

Canopy-air heat and moisture storage for `SlabLand(surface_storage=...)`.
`layer_depth` is the air-layer depth in meters, supplied as a number or a horizontal field.
Temperature and specific humidity are initialized by the coupled interface solve.

```jldoctest
using NumericalEarth

PrognosticCanopyAir(layer_depth=12)

# output
PrognosticCanopyAir(hᵃᶜᵃⁿ=12.0)
```
"""
struct PrognosticCanopyAir{H, P, N}
    layer_depth :: H
    prognostic :: P
    next_state :: N
end

PrognosticCanopyAir(FT::Type = Oceananigans.defaults.FloatType; layer_depth=10) =
    PrognosticCanopyAir(convert_if_number(FT, layer_depth), nothing, nothing)

"""
$(TYPEDSIGNATURES)

Ground-skin heat storage for `SlabLand(surface_storage=...)`.
`heat_capacity` is the areal heat capacity in J m⁻² K⁻¹, supplied as a number or a
horizontal field. The bulk land heat capacity describes the soil below this skin.
Temperature is initialized by the coupled interface solve.

```jldoctest
using NumericalEarth

PrognosticSkin()

# output
PrognosticSkin(C=100000.0)
```
"""
struct PrognosticSkin{C, P, N}
    heat_capacity :: C
    prognostic :: P
    next_state :: N
end

PrognosticSkin(FT::Type = Oceananigans.defaults.FloatType; heat_capacity=1e5) =
    PrognosticSkin(convert_if_number(FT, heat_capacity), nothing, nothing)

const SurfaceStorage = Union{PrognosticCanopyAir, PrognosticSkin}

storage_variables(::PrognosticCanopyAir) = (:temperature, :specific_humidity)
storage_variables(::PrognosticSkin) = (:temperature,)

materialize_surface_storage(::Nothing, grid) = nothing
function materialize_surface_storage(storage::SurfaceStorage, grid)
    names = storage_variables(storage)
    prognostic = NamedTuple{names}(map(_ -> Field{Center, Center, Nothing}(grid), names))
    next_state = NamedTuple{names}(map(_ -> Field{Center, Center, Nothing}(grid), names))
    return materialize_surface_storage(storage, prognostic, next_state)
end

materialize_surface_storage(s::PrognosticCanopyAir, prognostic, next_state) =
    PrognosticCanopyAir(s.layer_depth, prognostic, next_state)
materialize_surface_storage(s::PrognosticSkin, prognostic, next_state) =
    PrognosticSkin(s.heat_capacity, prognostic, next_state)

step_surface_storage!(::Nothing) = nothing
function step_surface_storage!(storage::SurfaceStorage)
    foreach(values(storage.prognostic), values(storage.next_state)) do state, next
        copyto!(parent(state), parent(next))
        fill_halo_regions!(state)
    end
    return nothing
end

Oceananigans.prognostic_state(storage::SurfaceStorage) = map(prognostic_state, storage.prognostic)

function Oceananigans.restore_prognostic_state!(storage::SurfaceStorage, state)
    for name in keys(storage.prognostic)
        restore_prognostic_state!(storage.prognostic[name], state[name])
    end
    return storage
end

Base.summary(s::PrognosticCanopyAir) = string("PrognosticCanopyAir(hᵃᶜᵃⁿ=", prettysummary(s.layer_depth), ")")
Base.summary(s::PrognosticSkin) = string("PrognosticSkin(C=", prettysummary(s.heat_capacity), ")")
Base.show(io::IO, s::SurfaceStorage) = print(io, summary(s))

Adapt.@adapt_structure PrognosticCanopyAir
Adapt.@adapt_structure PrognosticSkin

InterfaceComputations.land_surface_state(land::SlabLand) = surface_storage_state(land.surface_storage)

surface_storage_state(::Nothing) = (; state=nothing, next=(;))
surface_storage_state(s::PrognosticCanopyAir) =
    (; state=merge(s.prognostic, (; s.layer_depth)), next=s.next_state)
surface_storage_state(s::PrognosticSkin) =
    (; state=merge(s.prognostic, (; s.heat_capacity)), next=s.next_state)

EarthSystemModels.initialize_land_surface!(land::SlabLand) = step_surface_storage!(land.surface_storage)

function InterfaceComputations.validate_surface_storage(land::SlabLand, state)
    storage = land.surface_storage
    compatible = isnothing(storage) ||
                 (storage isa PrognosticCanopyAir && state isa CanopyInterface) ||
                 (storage isa PrognosticSkin && state isa NamedTuple && state.temperature isa EnergyBalanceTemperature)
    compatible || throw(ArgumentError("$(summary(storage)) is incompatible with the interface state formulation."))
    return nothing
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
