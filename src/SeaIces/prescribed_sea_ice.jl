using ClimaSeaIce: ConductiveFlux
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: set!
using Oceananigans.Grids: Grids, LatitudeLongitudeGrid
using Oceananigans.OutputReaders: update_field_time_series!, FieldTimeSeries, cpu_interpolating_time_indices
using Oceananigans.TimeSteppers: tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: prettysummary

using ..EarthSystemModels: AbstractPrescribedComponent, DegreesKelvin
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters
using ..Oceans: PrescribedOcean, interpolate_prescribed_state!

#####
##### A fixed freezing point, in Kelvin
#####

"""
    FixedFreezingPoint(freshwater_melting_temperature, seawater_freezing_temperature)

A liquidus for [`PrescribedSeaIce`](@ref), in Kelvin: the ice surface melts at
`freshwater_melting_temperature` and the ocean under the ice sits at `seawater_freezing_temperature`
whatever its salinity, which is how the prescribed sea surface temperature is built.
"""
struct FixedFreezingPoint{FT}
    freshwater_melting_temperature :: FT
    seawater_freezing_temperature :: FT
end

@inline ClimaSeaIce.SeaIceThermodynamics.melting_temperature(liquidus::FixedFreezingPoint, salinity, z=0) =
    liquidus.seawater_freezing_temperature

Base.summary(liquidus::FixedFreezingPoint) =
    string("FixedFreezingPoint(freshwater_melting_temperature = ", prettysummary(liquidus.freshwater_melting_temperature),
           ", seawater_freezing_temperature = ", prettysummary(liquidus.seawater_freezing_temperature), ")")

#####
##### Prescribed sea ice
#####

struct PrescribedSeaIce{G, C, P, A, H, T, K, L} <: AbstractPrescribedComponent
    grid :: G
    clock :: C
    concentration :: P              # prescribed FieldTimeSeries, which may overshoot [0, 1]
    ice_concentration :: A          # the series at the clock time, clipped to [0, 1]
    ice_thickness :: H              # (m)
    top_surface_temperature :: T    # diagnosed skin temperature (K)
    initial_surface_temperature :: T # skin temperature at the previous time step, from which the skin solve starts (K)
    internal_heat_flux :: K
    liquidus :: L
end

# CICE's prescribed mode: 2 m of ice in the Northern Hemisphere, 1 m in the Southern.
default_prescribed_ice_thickness(grid) = 2
default_prescribed_ice_thickness(::LatitudeLongitudeGrid) = (λ, φ) -> ifelse(φ > 0, 2, 1)

"""
$(SIGNATURES)

A sea ice component for `EarthSystemModel` with a prescribed concentration and a diagnosed surface
temperature, for atmosphere-only runs over a [`PrescribedOcean`](@ref) on the same `grid`
(sea surface temperature and sea ice concentration prescribed, as in AMIP). `times` are the time
instances of the default (ice-free) `concentration`.

Concentration
=============

`concentration` is a `FieldTimeSeries` of the ice area fraction. It is read at the sea ice clock by
linear interpolation between the two bracketing snapshots and *then* clipped to `[0, 1]`. The order
matters for boundary-condition datasets such as `PCMDIAMIPBoundaryConditions`' `siconcbcs`, whose
mid-month values overshoot the physical bounds by design so that the monthly mean of the interpolated
and clipped series reproduces the observed monthly mean (Taylor, Williamson and Zwiers, 2000, PCMDI
Report 60). `sea_ice_concentration(sea_ice)` is the clipped field, which splits the ocean fraction of the
surface partition into open water `ℵᵒᶜ (1 - ℵˢⁱ)` and ice `ℵᵒᶜ ℵˢⁱ`.

Surface temperature
===================

The ice is a slab of prescribed `thickness` `hˢⁱ` and `conductivity`, without snow and without heat
capacity, over an ocean at `freezing_temperature`. Its surface temperature `Tⁱⁿ` is diagnosed from the
surface energy balance,

    𝒬ᵀ(Tⁱⁿ) + 𝒬ᵛ(Tⁱⁿ) + ϵ σ (Tⁱⁿ)⁴ + ϵ ℐꜜˡʷ + (1 - α) ℐꜜˢʷ = conductivity (freezing_temperature - Tⁱⁿ) / hˢⁱ,

and capped at `melting_temperature`. All fluxes are positive upward (so the downwelling `ℐꜜˡʷ` and
`ℐꜜˢʷ` are negative); `𝒬ᵀ` and `𝒬ᵛ` are the turbulent sensible and latent heat fluxes over ice, and the
radiative terms, the sea ice albedo `α` and the emissivity `ϵ` are those of the model's `radiation` (no
radiative terms without one). At the cap, the residual of the balance is the energy that would melt ice;
it is discarded, since concentration and thickness are prescribed. The balance is solved together with
the turbulent fluxes by the `SkinTemperature` iteration that a prognostic ClimaSeaIce model uses, started
from the surface temperature of the previous time step; both temperatures are checkpointed. Its
limits are `Tⁱⁿ → freezing_temperature` for thin ice and the conduction-free radiative-turbulent
equilibrium for thick ice.

This is the zero-layer balance of Semtner (1976) with prescribed thickness, the usual treatment of sea
ice in atmosphere-only runs: CICE's prescribed mode for CAM (2 m of ice in the Northern Hemisphere and
1 m in the Southern, the default `thickness` on a `LatitudeLongitudeGrid`) and ECHAM (Roeckner et al.,
2003, MPI Report 349) both conduct heat through ice of fixed thickness to a base at the freezing point.

Sea surface temperature
=======================

The sea surface temperature that a `PrescribedOcean` passes to the interfaces is clipped from below at
`freezing_temperature`, after its own interpolation in time (the default 271.38 K is the PCMDI
recommendation for the AMIP boundary conditions).

Keyword Arguments
=================

- `clock`: Clock for tracking sea ice time. Default: `Clock{FT}(time=0)`.
- `concentration`: `FieldTimeSeries` of the ice area fraction. Default: ice-free.
- `thickness`: Ice thickness in m, anything `set!` accepts. Default: 2 (Northern Hemisphere) and 1
  (Southern Hemisphere) on a `LatitudeLongitudeGrid`, 2 elsewhere.
- `conductivity`: Ice thermal conductivity in W m⁻¹ K⁻¹. Default: 2.
- `freezing_temperature`: Temperature of the ocean under and around the ice, in K. Default: 271.38.
- `melting_temperature`: Upper bound of the ice surface temperature, in K. Default: 273.15.

```jldoctest
using NumericalEarth, Oceananigans

grid = LatitudeLongitudeGrid(size=(36, 18), longitude=(0, 360), latitude=(-90, 90), topology=(Periodic, Bounded, Flat))
sea_ice = PrescribedSeaIce(grid)

# output

PrescribedSeaIce{Float64, CPU, LatitudeLongitudeGrid}(Nt = 1)
├── grid: 36×18×1 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Flat} on CPU with 3×3×0 halo
├── ice_thickness: 36×18×1 Field{Center, Center, Nothing} reduced over dims = (3,) on LatitudeLongitudeGrid on CPU
├── conductivity: 2.0
└── liquidus: FixedFreezingPoint(freshwater_melting_temperature = 273.15, seawater_freezing_temperature = 271.38)
```
"""
function PrescribedSeaIce(grid, times=[zero(grid)];
                          clock = Clock{eltype(grid)}(time=0),
                          concentration = FieldTimeSeries{Center, Center, Nothing}(grid, times),
                          thickness = default_prescribed_ice_thickness(grid),
                          conductivity = 2,
                          freezing_temperature = 271.38,
                          melting_temperature = 273.15)

    FT = eltype(grid)
    liquidus = FixedFreezingPoint(convert(FT, melting_temperature), convert(FT, freezing_temperature))
    internal_heat_flux = ConductiveFlux(FT; conductivity)

    ice_concentration = Field{Center, Center, Nothing}(grid)
    ice_thickness = Field{Center, Center, Nothing}(grid)
    top_surface_temperature = Field{Center, Center, Nothing}(grid)
    initial_surface_temperature = Field{Center, Center, Nothing}(grid)

    # The skin temperature iteration starts from the surface temperature of the previous time step
    set!(top_surface_temperature, liquidus.seawater_freezing_temperature)
    fill_halo_regions!(top_surface_temperature)
    parent(initial_surface_temperature) .= parent(top_surface_temperature)

    sea_ice = PrescribedSeaIce(grid, clock, concentration, ice_concentration, ice_thickness,
                               top_surface_temperature, initial_surface_temperature, internal_heat_flux, liquidus)

    set!(sea_ice; h=thickness)
    interpolate_prescribed_concentration!(sea_ice)

    return sea_ice
end

Grids.grid(sea_ice::PrescribedSeaIce) = sea_ice.grid

function Oceananigans.set!(sea_ice::PrescribedSeaIce; ℵ=nothing, h=nothing)
    !isnothing(ℵ) && set!(sea_ice.concentration, ℵ)
    !isnothing(h) && set!(sea_ice.ice_thickness, h)

    # The interface kernels read the halo columns too (see `interpolate_prescribed_state!`)
    fill_halo_regions!(sea_ice.concentration)
    fill_halo_regions!(sea_ice.ice_thickness)

    return nothing
end

function Base.summary(sea_ice::PrescribedSeaIce)
    FT = eltype(sea_ice.grid)
    A = nameof(typeof(architecture(sea_ice.grid)))
    G = nameof(typeof(sea_ice.grid))
    return string("PrescribedSeaIce{$FT, $A, $G}(Nt = ", length(sea_ice.concentration.times), ")")
end

function Base.show(io::IO, sea_ice::PrescribedSeaIce)
    print(io, summary(sea_ice), "\n",
          "├── grid: ", summary(sea_ice.grid), "\n",
          "├── ice_thickness: ", summary(sea_ice.ice_thickness), "\n",
          "├── conductivity: ", prettysummary(sea_ice.internal_heat_flux.conductivity), "\n",
          "└── liquidus: ", summary(sea_ice.liquidus))
end

#####
##### EarthSystemModels interface
#####

EarthSystemModels.is_ocean_component(::PrescribedSeaIce) = false
EarthSystemModels.adopt_clock(sea_ice::PrescribedSeaIce, clock) = EarthSystemModels.reclock(sea_ice, clock)

EarthSystemModels.sea_ice_concentration(sea_ice::PrescribedSeaIce) = sea_ice.ice_concentration
EarthSystemModels.sea_ice_thickness(sea_ice::PrescribedSeaIce) = sea_ice.ice_thickness
EarthSystemModels.temperature_units(::PrescribedSeaIce) = DegreesKelvin()

# Prescribed ice has no heat budget
EarthSystemModels.reference_density(::PrescribedSeaIce) = 0
EarthSystemModels.heat_capacity(::PrescribedSeaIce) = 0

# The prescribed sea surface temperature is clipped where the interfaces read it
EarthSystemModels.above_freezing_ocean_temperature!(ocean, grid, ::PrescribedSeaIce) = nothing

#####
##### InterfaceComputations interface
#####

InterfaceComputations.sea_ice_liquidus(sea_ice::PrescribedSeaIce) = sea_ice.liquidus
InterfaceComputations.sea_ice_surface_temperature(sea_ice::PrescribedSeaIce) = sea_ice.top_surface_temperature
InterfaceComputations.default_ai_temperature(sea_ice::PrescribedSeaIce) = SkinTemperature(sea_ice.internal_heat_flux)
InterfaceComputations.net_fluxes(::PrescribedSeaIce) = nothing

# The ocean under prescribed ice does not respond: no sea ice-ocean interface
InterfaceComputations.ThreeEquationHeatFlux(::PrescribedSeaIce) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::PrescribedSeaIce, ocean, flux_formulation; kwargs...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::PrescribedSeaIce, ::Nothing, flux_formulation; kwargs...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::PrescribedSeaIce, ocean, ::ThreeEquationHeatFlux; kwargs...) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::PrescribedSeaIce, ::Nothing, ::ThreeEquationHeatFlux; kwargs...) = nothing

function InterfaceComputations.ComponentExchanger(sea_ice::PrescribedSeaIce, exchange_grid)
    sea_ice.grid == exchange_grid ||
        throw(ArgumentError("PrescribedSeaIce must be built on the exchange grid, $(summary(exchange_grid))"))

    FT = eltype(exchange_grid)
    no_field = ZeroField(FT) # no ice velocity, no snow, and ice of any thickness is consolidated

    state = (u = no_field, v = no_field,
             hi = sea_ice.ice_thickness,
             hc = no_field,
             ℵ  = sea_ice.ice_concentration,
             hs = no_field)

    return ComponentExchanger(state, nothing)
end

# Once the model is stepping, the skin solve iterates from the surface temperature of the previous time step,
# whatever the number of `update_state!`s since: its solution depends on where it starts at the level of the
# solver tolerance, and a restored model must find the solution of the model it was checkpointed from.
# Before the first step, successive `update_state!`s refine the skin from one to the next.
function EarthSystemModels.interpolate_state!(exchanger, grid, sea_ice::PrescribedSeaIce, coupled_model)
    if sea_ice.clock.iteration > 0
        parent(sea_ice.top_surface_temperature) .= parent(sea_ice.initial_surface_temperature)
    end
    interpolate_prescribed_concentration!(sea_ice)
    return nothing
end

# Read the series at the sea ice clock by linear interpolation between the two snapshots that bracket it,
# then clip: the boundary-condition values overshoot [0, 1]. Covers the halo columns, as
# `interpolate_prescribed_state!` does.
function interpolate_prescribed_concentration!(sea_ice::PrescribedSeaIce)
    concentration = sea_ice.concentration
    grid = sea_ice.grid
    arch = architecture(grid)
    time_index = cpu_interpolating_time_indices(arch, concentration.times, concentration.time_indexing, sea_ice.clock.time)

    launch!(arch, grid, interface_kernel_parameters(grid), _interpolate_prescribed_concentration!,
            sea_ice.ice_concentration, concentration, time_index)

    return nothing
end

@kernel function _interpolate_prescribed_concentration!(ice_concentration, concentration, time_index)
    i, j = @index(Global, NTuple)
    @inbounds ice_concentration[i, j, 1] = clamp(concentration[i, j, 1, time_index], 0, 1)
end

const PrescribedSeaIceModel = EarthSystemModel{<:Any, <:Any, <:Any, <:PrescribedSeaIce}

function EarthSystemModels.interpolate_state!(exchanger, grid, ocean::PrescribedOcean, coupled_model::PrescribedSeaIceModel)
    interpolate_prescribed_state!(exchanger.state, ocean)
    freezing_temperature = coupled_model.sea_ice.liquidus.seawater_freezing_temperature
    launch!(architecture(grid), grid, interface_kernel_parameters(grid), _clip_sea_surface_temperature!,
            exchanger.state.T, freezing_temperature)
    return nothing
end

@kernel function _clip_sea_surface_temperature!(T, freezing_temperature)
    i, j = @index(Global, NTuple)
    @inbounds T[i, j, 1] = max(T[i, j, 1], freezing_temperature)
end

#####
##### Time stepping: tick the clock, update the FieldTimeSeries backend
#####

function Oceananigans.TimeSteppers.time_step!(sea_ice::PrescribedSeaIce, Δt)
    parent(sea_ice.initial_surface_temperature) .= parent(sea_ice.top_surface_temperature)
    tick!(sea_ice.clock, Δt)
    update_state!(sea_ice)
    return nothing
end

function Oceananigans.TimeSteppers.update_state!(sea_ice::PrescribedSeaIce)
    update_field_time_series!(sea_ice.concentration, Time(sea_ice.clock.time))
    return nothing
end

#####
##### Checkpointing
#####

# The skin temperature is diagnosed, but by an iteration that starts from `initial_surface_temperature`
Oceananigans.prognostic_state(sea_ice::PrescribedSeaIce) =
    (; clock = Oceananigans.prognostic_state(sea_ice.clock),
       top_surface_temperature = Oceananigans.prognostic_state(sea_ice.top_surface_temperature),
       initial_surface_temperature = Oceananigans.prognostic_state(sea_ice.initial_surface_temperature))

function Oceananigans.restore_prognostic_state!(sea_ice::PrescribedSeaIce, state)
    Oceananigans.restore_prognostic_state!(sea_ice.clock, state.clock)

    # Checkpoints written before the skin temperature was saved have no such entries
    for name in (:top_surface_temperature, :initial_surface_temperature)
        hasproperty(state, name) && Oceananigans.restore_prognostic_state!(getproperty(sea_ice, name), getproperty(state, name))
    end

    update_state!(sea_ice)
    return sea_ice
end

Oceananigans.restore_prognostic_state!(sea_ice::PrescribedSeaIce, ::Nothing) = sea_ice
