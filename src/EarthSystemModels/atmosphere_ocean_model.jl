using Oceananigans.TimeSteppers: Clock

"""
    SlabOcean(sea_surface_temperature;
              mixed_layer_depth = 50,
              density = 1025,
              heat_capacity = 4000)

A simple slab ocean model that evolves sea surface temperature (SST) in response
to surface heat fluxes using a single well-mixed layer of fixed depth.

The SST tendency equation is:

    ρ cₚ H ∂T/∂t = -Q_net

where `ρ` is the seawater density, `cₚ` is the specific heat capacity,
`H` is the mixed layer depth, and `Q_net` is the net downward surface heat flux.

Arguments
=========

- `sea_surface_temperature`: An Oceananigans `Field` representing the SST.

Keyword Arguments
=================

- `mixed_layer_depth`: Depth of the slab ocean mixed layer in meters. Default: 50.
- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
"""
struct SlabOcean{T, H, ρ, C}
    sea_surface_temperature :: T
    mixed_layer_depth :: H
    density :: ρ
    heat_capacity :: C
end

function SlabOcean(sea_surface_temperature;
                   mixed_layer_depth = 50,
                   density = 1025,
                   heat_capacity = 4000)

    return SlabOcean(sea_surface_temperature,
                     mixed_layer_depth,
                     density,
                     heat_capacity)
end

Base.summary(ocean::SlabOcean) = "SlabOcean(H=$(ocean.mixed_layer_depth) m)"

"""
    AtmosphereOceanModel(; architecture, clock, atmosphere, ocean, ocean_surface_fluxes)

A coupled atmosphere-ocean model that combines an atmospheric model with a slab ocean.
After each atmosphere timestep, the net surface heat flux is used to update the SST
via forward Euler time stepping.

The coupling algorithm is:
1. Step the atmosphere model (which uses the current SST in its boundary conditions)
2. Extract net surface heat fluxes from the atmosphere
3. Update SST: `T -= Δt * Q_net / (ρ cₚ H)`
4. Advance the clock

This type is defined generically (no Breeze dependency). Breeze-specific constructor
and methods are provided by the `NumericalEarthBreezeExt` extension.
"""
mutable struct AtmosphereOceanModel{A, O, Q, C, Arch} <: AbstractModel{Nothing, Arch}
    architecture :: Arch
    clock :: C
    atmosphere :: A
    ocean :: O
    ocean_surface_fluxes :: Q
end

const AOM = AtmosphereOceanModel

function Base.summary(model::AOM)
    A = nameof(typeof(architecture(model)))
    return string("AtmosphereOceanModel{$A}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::AOM)
    print(io, summary(model), "\n")
    print(io, "├── atmosphere: ", summary(model.atmosphere), "\n")
    print(io, "└── ocean: ", summary(model.ocean))
    return nothing
end

architecture(model::AOM) = model.architecture
timestepper(::AOM)       = nothing
time(model::AOM)         = model.clock.time
iteration(model::AOM)    = model.clock.iteration
prettytime(model::AOM)   = prettytime(model.clock.time)

default_included_properties(::AOM) = tuple()
prognostic_fields(::AOM)           = nothing
fields(::AOM)                      = NamedTuple()
update_state!(::AOM)               = nothing
initialization_update_state!(::AOM) = nothing
