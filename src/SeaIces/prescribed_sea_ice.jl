using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, cpu_interpolating_time_indices
using Oceananigans.TimeSteppers: tick!, update_state!
using Oceananigans.Units: Time
using ClimaSeaIce: PhaseTransitions, ConductiveFlux

using ..EarthSystemModels: AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: SkinTemperature, interface_kernel_parameters

"""
    PrescribedSeaIce(grid, times=[zero(grid)];
                     concentration = 0,
                     ice_thickness = 1,
                     conductivity = 2,
                     surface_temperature = SkinTemperature(ConductiveFlux(eltype(grid); conductivity)),
                     albedo = 0.7,
                     emissivity = 1,
                     phase_transitions = PhaseTransitions(eltype(grid)),
                     clock = Clock{eltype(grid)}(time=0))

A sea ice component for `EarthSystemModel` whose concentration `ℵ` follows prescribed data
rather than evolving. Each exchange-grid cell is split into open water `(1 - ℵ)`, which
exchanges fluxes with the ocean surface, and ice `ℵ`, which exchanges fluxes with the
atmosphere through the atmosphere–sea ice interface.

The default ice surface temperature closes the surface energy balance
`𝒬ᵀ + 𝒬ᵛ + ℐꜛˡʷ - (1 - α) ℐꜜˢʷ - ϵ ℐꜜˡʷ = k (Tᵇ - Tˢ) / h`,
with conduction through a slab of fixed `ice_thickness` `h` and `conductivity` `k`
from a base held at the freezing temperature `Tᵇ` of the underlying ocean,
capped at the freshwater melting temperature. Pass `surface_temperature = BulkTemperature()`
to hold the ice surface at the ocean surface temperature instead.

Arguments
=========

- `grid`: the two-dimensional surface grid, which must be the exchange grid of the coupled model.
- `times`: times of the prescribed concentration. Default: `[zero(grid)]` (constant).

Keyword Arguments
=================

- `concentration`: a `FieldTimeSeries` of ice concentration, or anything `FieldTimeSeries(concentration, grid)`
  accepts, such as a `Metadata` for `:sea_ice_concentration`. Missing values count as open water.
- `ice_thickness`: thickness of the conducting ice slab in m. Default: 1.
- `conductivity`: ice thermal conductivity in W m⁻¹ K⁻¹. Default: 2.
- `surface_temperature`: atmosphere–sea ice interface temperature formulation.
- `albedo`: ice surface albedo. Default: 0.7.
- `emissivity`: ice surface emissivity. Default: 1.
- `phase_transitions`: ice density, heat capacity, and liquidus.
- `clock`: clock tracking the sea ice time.

```jldoctest
using Oceananigans
using NumericalEarth

grid = LatitudeLongitudeGrid(size=(4, 4), longitude=(0, 360), latitude=(60, 90), topology=(Periodic, Bounded, Flat))
sea_ice = PrescribedSeaIce(grid, [0, 3600])

# output
PrescribedSeaIce{Float64}(Nt = 2)
├── ice_thickness: 1.0
├── albedo: 0.7
└── emissivity: 1.0
```
"""
struct PrescribedSeaIce{G, C, N, F, H, T, A, E, P} <: AbstractPrescribedComponent
    grid :: G
    clock :: C
    concentration_time_series :: N
    concentration :: F
    ice_thickness :: H
    surface_temperature_formulation :: T
    surface_temperature :: F
    albedo :: A
    emissivity :: E
    phase_transitions :: P
end

prescribed_concentration(concentration::FieldTimeSeries, grid, times) = concentration
prescribed_concentration(concentration, grid, times) = FieldTimeSeries(concentration, grid)

function prescribed_concentration(concentration::Number, grid, times)
    fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    parent(fts) .= concentration
    return fts
end

function PrescribedSeaIce(grid, times=[zero(grid)];
                          concentration = 0,
                          ice_thickness = 1,
                          conductivity = 2,
                          surface_temperature = SkinTemperature(ConductiveFlux(eltype(grid); conductivity)),
                          albedo = 0.7,
                          emissivity = 1,
                          phase_transitions = PhaseTransitions(eltype(grid)),
                          clock = Clock{eltype(grid)}(time=0))

    FT = eltype(grid)
    concentration_time_series = prescribed_concentration(concentration, grid, times)
    fill_halo_regions!(concentration_time_series)

    ℵ = Field{Center, Center, Nothing}(grid)
    Tˢ = Field{Center, Center, Nothing}(grid)
    Oceananigans.set!(Tˢ, phase_transitions.liquidus.freshwater_melting_temperature)

    sea_ice = PrescribedSeaIce(grid, clock, concentration_time_series, ℵ,
                               ConstantField(convert(FT, ice_thickness)),
                               surface_temperature, Tˢ,
                               convert(FT, albedo), convert(FT, emissivity),
                               phase_transitions)

    interpolate_concentration!(sea_ice)

    return sea_ice
end

function interpolate_concentration!(sea_ice::PrescribedSeaIce)
    grid = sea_ice.grid
    arch = architecture(grid)
    ℵᵗ = sea_ice.concentration_time_series
    τ = cpu_interpolating_time_indices(arch, ℵᵗ.times, ℵᵗ.time_indexing, sea_ice.clock.time)
    launch!(arch, grid, interface_kernel_parameters(grid), _interpolate_concentration!, sea_ice.concentration, ℵᵗ, τ)
    return nothing
end

@kernel function _interpolate_concentration!(ℵ, ℵᵗ, τ)
    i, j = @index(Global, NTuple)
    @inbounds ℵᵢ = ℵᵗ[i, j, 1, τ]
    # ifelse, not ?:, so the kernel stays branch-free on GPU; NaN marks land in reanalysis products
    @inbounds ℵ[i, j, 1] = ifelse(isnan(ℵᵢ), zero(ℵᵢ), ℵᵢ)
end

Base.summary(sea_ice::PrescribedSeaIce) =
    string("PrescribedSeaIce{", eltype(sea_ice.grid), "}(Nt = ", length(sea_ice.concentration_time_series.times), ")")

function Base.show(io::IO, sea_ice::PrescribedSeaIce)
    print(io, summary(sea_ice), "\n",
          "├── ice_thickness: ", sea_ice.ice_thickness.constant, "\n",
          "├── albedo: ", sea_ice.albedo, "\n",
          "└── emissivity: ", sea_ice.emissivity)
end

#####
##### EarthSystemModels interface
#####

EarthSystemModels.adopt_clock(sea_ice::PrescribedSeaIce, clock) = EarthSystemModels.reclock(sea_ice, clock)
EarthSystemModels.sea_ice_concentration(sea_ice::PrescribedSeaIce) = sea_ice.concentration
EarthSystemModels.sea_ice_thickness(sea_ice::PrescribedSeaIce) = sea_ice.ice_thickness
EarthSystemModels.intercepted_snowfall(::PrescribedSeaIce) = ZeroField()
EarthSystemModels.reference_density(sea_ice::PrescribedSeaIce) = sea_ice.phase_transitions.density
EarthSystemModels.heat_capacity(sea_ice::PrescribedSeaIce) = sea_ice.phase_transitions.heat_capacity
EarthSystemModels.above_freezing_ocean_temperature!(ocean, grid, ::PrescribedSeaIce) = nothing
EarthSystemModels.update_net_fluxes!(coupled_model, ::PrescribedSeaIce) = nothing

EarthSystemModels.interpolate_state!(exchanger, grid, sea_ice::PrescribedSeaIce, coupled_model) =
    interpolate_concentration!(sea_ice)

function EarthSystemModels.materialize_sea_ice!(sea_ice::PrescribedSeaIce, ocean)
    sea_ice.grid == Oceananigans.Grids.grid(ocean) ||
        throw(ArgumentError("PrescribedSeaIce must share the grid of the ocean it covers"))
    return sea_ice
end

EarthSystemModels.materialize_sea_ice!(::PrescribedSeaIce, ocean::Simulation) =
    throw(ArgumentError("PrescribedSeaIce couples only to a PrescribedOcean"))

#####
##### InterfaceComputations interface
#####

InterfaceComputations.net_fluxes(::PrescribedSeaIce) = nothing
InterfaceComputations.default_ai_temperature(sea_ice::PrescribedSeaIce) = sea_ice.surface_temperature_formulation
InterfaceComputations.sea_ice_liquidus(sea_ice::PrescribedSeaIce) = sea_ice.phase_transitions.liquidus
InterfaceComputations.sea_ice_surface_temperature(sea_ice::PrescribedSeaIce) = sea_ice.surface_temperature
InterfaceComputations.ThreeEquationHeatFlux(::PrescribedSeaIce) = nothing
InterfaceComputations.sea_ice_ocean_interface(grid, ::PrescribedSeaIce, ocean, ::Nothing) = nothing

const PrescribedSeaIceModel = EarthSystemModel{<:Any, <:Any, <:Any, <:PrescribedSeaIce}
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::PrescribedSeaIceModel) = nothing

InterfaceComputations.ComponentExchanger(sea_ice::PrescribedSeaIce, grid) =
    ComponentExchanger((; ℵ = sea_ice.concentration, hi = sea_ice.ice_thickness, hs = ZeroField(), hc = ZeroField()), nothing)

InterfaceComputations.sea_ice_radiation_properties(::PrescribedSeaIce, ::Nothing) = nothing

function InterfaceComputations.sea_ice_radiation_properties(sea_ice::PrescribedSeaIce, radiation_properties)
    sea_ice_surface = (albedo = sea_ice.albedo, emissivity = sea_ice.emissivity)
    surface_properties = merge(radiation_properties.surface_properties, (; sea_ice = sea_ice_surface))
    return merge(radiation_properties, (; surface_properties))
end

#####
##### Time stepping: tick the clock and load the concentration data
#####

function Oceananigans.TimeSteppers.time_step!(sea_ice::PrescribedSeaIce, Δt)
    tick!(sea_ice.clock, Δt)
    update_state!(sea_ice)
    return nothing
end

function Oceananigans.TimeSteppers.update_state!(sea_ice::PrescribedSeaIce)
    update_field_time_series!(sea_ice.concentration_time_series, Time(sea_ice.clock.time))
    interpolate_concentration!(sea_ice)
    return nothing
end

Oceananigans.prognostic_state(sea_ice::PrescribedSeaIce) = (; clock = Oceananigans.prognostic_state(sea_ice.clock))

function Oceananigans.restore_prognostic_state!(sea_ice::PrescribedSeaIce, state)
    Oceananigans.restore_prognostic_state!(sea_ice.clock, state.clock)
    update_state!(sea_ice)
    return sea_ice
end
