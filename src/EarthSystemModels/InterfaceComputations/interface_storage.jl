using Oceananigans.Utils: prettysummary

#####
##### Prognostic interface storage: canopy-air or skin state owned by the atmosphere–land interface
#####

"""
$(TYPEDSIGNATURES)

Canopy-air heat and moisture storage for an atmosphere–land interface with a [`CanopyInterface`](@ref).
`layer_depth` is the air-layer depth in meters, supplied as a number or a horizontal field.
Temperature and specific humidity are initialized by the coupled interface solve.

```jldoctest
using NumericalEarth

PrognosticCanopyAir(layer_depth=12)

# output
PrognosticCanopyAir(hᵃᶜᵃⁿ=12.0)
```
"""
struct PrognosticCanopyAir{H, P}
    layer_depth :: H
    prognostic :: P
end

PrognosticCanopyAir(FT::Type = Oceananigans.defaults.FloatType; layer_depth=10) =
    PrognosticCanopyAir(convert_if_number(FT, layer_depth), nothing)

"""
$(TYPEDSIGNATURES)

Ground heat storage for an atmosphere–land interface with an [`EnergyBalanceTemperature`](@ref).
`heat_capacity` is the areal heat capacity in J m⁻² K⁻¹, supplied as a number or a
horizontal field. The bulk land heat capacity describes the soil below the ground surface.
Temperature is initialized by the coupled interface solve.

```jldoctest
using NumericalEarth

PrognosticSkin()

# output
PrognosticSkin(C=100000.0)
```
"""
struct PrognosticSkin{C, P}
    heat_capacity :: C
    prognostic :: P
end

PrognosticSkin(FT::Type = Oceananigans.defaults.FloatType; heat_capacity=1e5) =
    PrognosticSkin(convert_if_number(FT, heat_capacity), nothing)

const InterfaceStorage = Union{PrognosticCanopyAir, PrognosticSkin}

storage_variables(::PrognosticCanopyAir) = (:temperature, :specific_humidity)
storage_variables(::PrognosticSkin) = (:temperature,)

materialize_interface_storage(::Nothing, grid) = nothing

function materialize_interface_storage(storage::InterfaceStorage, grid)
    names = storage_variables(storage)
    prognostic = NamedTuple{names}(map(_ -> Field{Center, Center, Nothing}(grid), names))
    return with_prognostic(storage, prognostic)
end

with_prognostic(s::PrognosticCanopyAir, prognostic) = PrognosticCanopyAir(s.layer_depth, prognostic)
with_prognostic(s::PrognosticSkin, prognostic) = PrognosticSkin(s.heat_capacity, prognostic)

validate_interface_storage(::Nothing, state) = nothing
validate_interface_storage(::PrognosticCanopyAir, ::CanopyInterface) = nothing
validate_interface_storage(::PrognosticSkin, state::NamedTuple) =
    state.temperature isa EnergyBalanceTemperature ? nothing : incompatible_storage(PrognosticSkin, state)
validate_interface_storage(storage, state) = incompatible_storage(typeof(storage).name.wrapper, state)

incompatible_storage(S, state) =
    throw(ArgumentError("$S is incompatible with the interface state formulation $(summary(state))."))

# The storage the flux kernel reads at each cell: the prognostic state with its capacity parameter.
storage_state(::Nothing) = nothing
storage_state(s::PrognosticCanopyAir) = merge(s.prognostic, (; s.layer_depth))
storage_state(s::PrognosticSkin) = merge(s.prognostic, (; s.heat_capacity))

# Initialization solves for a massless canopy air layer, whose state is the equilibrium of its sources.
initialization_storage_state(storage) = storage_state(storage)
initialization_storage_state(s::PrognosticCanopyAir) = merge(s.prognostic, (; layer_depth = zero(eltype(s.prognostic.temperature))))

storage_prognostic(::Nothing) = (;)
storage_prognostic(storage) = storage.prognostic

Oceananigans.prognostic_state(storage::InterfaceStorage) = map(prognostic_state, storage.prognostic)

function Oceananigans.restore_prognostic_state!(storage::InterfaceStorage, state)
    for name in keys(storage.prognostic)
        restore_prognostic_state!(storage.prognostic[name], state[name])
    end
    return storage
end

Base.summary(s::PrognosticCanopyAir) = string("PrognosticCanopyAir(hᵃᶜᵃⁿ=", prettysummary(s.layer_depth), ")")
Base.summary(s::PrognosticSkin) = string("PrognosticSkin(C=", prettysummary(s.heat_capacity), ")")
Base.show(io::IO, s::InterfaceStorage) = print(io, summary(s))

Adapt.@adapt_structure PrognosticCanopyAir
Adapt.@adapt_structure PrognosticSkin
