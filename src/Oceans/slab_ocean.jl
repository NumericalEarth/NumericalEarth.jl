using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Fields: ConstantField, ZeroField

"""
    SlabOcean(sea_surface_temperature;
              sea_surface_salinity = nothing,
              mixed_layer_depth = 50,
              density = 1025,
              heat_capacity = 4000,
              clock = Clock{FT}(time=0))

A slab ocean component for `EarthSystemModel`. Represents the ocean as a single
well-mixed layer of fixed depth, with sea surface temperature (SST) evolving
in response to surface heat fluxes.

The SST tendency equation is:

    ∂T/∂t = -Jᵀ / H

where `H` is the mixed layer depth and `Jᵀ` is the temperature flux
(in units of K m/s) assembled by the `EarthSystemModel` coupling.

Arguments
=========

- `sea_surface_temperature`: An Oceananigans `Field` representing the SST (in Kelvin or Celsius
  depending on the coupled model configuration).

Keyword Arguments
=================

- `sea_surface_salinity`: An Oceananigans `Field` representing sea surface salinity, or `nothing`
  to use a constant default of 35 psu. Default: `nothing`.
- `mixed_layer_depth`: Depth of the slab in meters. Default: 50.
- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
- `clock`: Clock for tracking slab ocean time. Default: `Clock{FT}(time=0)` where `FT` is the
  float type of the grid.
"""
struct SlabOcean{FT, G, Clk, T, S, F, H, ρ, C}
    grid :: G
    clock :: Clk
    sea_surface_temperature :: T
    sea_surface_salinity :: S
    temperature_flux :: F
    mixed_layer_depth :: H
    density :: ρ
    heat_capacity :: C
end

function SlabOcean(sea_surface_temperature;
                   FT = eltype(sea_surface_temperature.grid),
                   sea_surface_salinity = nothing,
                   mixed_layer_depth = 50,
                   density = 1025,
                   heat_capacity = 4000,
                   clock = Clock{FT}(time=0))

    grid = sea_surface_temperature.grid
    temperature_flux = CenterField(grid)

    return SlabOcean{FT}(grid, clock, sea_surface_temperature, sea_surface_salinity,
                         temperature_flux, mixed_layer_depth, density, heat_capacity)
end

# Inner constructor that captures FT
SlabOcean{FT}(grid, clock, sst, sss, tf, mld, ρ, c) where FT =
    SlabOcean{FT, typeof(grid), typeof(clock), typeof(sst), typeof(sss),
              typeof(tf), typeof(mld), typeof(ρ), typeof(c)}(grid, clock, sst, sss, tf, mld, ρ, c)

Base.summary(ocean::SlabOcean) = "SlabOcean(H=$(ocean.mixed_layer_depth) m)"
Base.show(io::IO, ocean::SlabOcean) = print(io, summary(ocean))
Base.eltype(::SlabOcean{FT}) where FT = FT

#####
##### EarthSystemModels interface
#####

reference_density(ocean::SlabOcean) = ocean.density
heat_capacity(ocean::SlabOcean) = ocean.heat_capacity
ocean_temperature(ocean::SlabOcean) = ocean.sea_surface_temperature
ocean_salinity(ocean::SlabOcean{FT, G, Clk, T, Nothing}) where {FT, G, Clk, T} = ConstantField(convert(FT, 35))
ocean_salinity(ocean::SlabOcean) = ocean.sea_surface_salinity
ocean_surface_temperature(ocean::SlabOcean) = ocean.sea_surface_temperature
ocean_surface_salinity(ocean::SlabOcean{FT, G, Clk, T, Nothing}) where {FT, G, Clk, T} = ConstantField(convert(FT, 35))
ocean_surface_salinity(ocean::SlabOcean) = ocean.sea_surface_salinity
ocean_surface_velocities(::SlabOcean{FT}) where FT = ZeroField(FT), ZeroField(FT)

#####
##### InterfaceComputations interface
#####

function ComponentExchanger(ocean::SlabOcean, exchange_grid)
    T = ocean.sea_surface_temperature
    S = ocean_surface_salinity(ocean)
    u, v = ocean_surface_velocities(ocean)
    return ComponentExchanger((; u, v, T, S), nothing)
end

function net_fluxes(ocean::SlabOcean)
    grid = ocean.grid
    Jˢ = CenterField(grid)
    τx = CenterField(grid)
    τy = CenterField(grid)
    return (T=ocean.temperature_flux, S=Jˢ, u=τx, v=τy)
end

# No interpolation needed: the slab ocean IS on the exchange grid
interpolate_state!(exchanger, grid, ::SlabOcean, coupled_model) = nothing

# Assemble net ocean fluxes from interface computations
update_net_fluxes!(coupled_model, ocean::SlabOcean) =
    Oceans.update_net_ocean_fluxes!(coupled_model, ocean, ocean.grid)

#####
##### Time stepping
#####

function Oceananigans.TimeSteppers.time_step!(ocean::SlabOcean, Δt)
    tick!(ocean.clock, Δt)
    T = ocean.sea_surface_temperature
    Jᵀ = ocean.temperature_flux
    H = ocean.mixed_layer_depth
    parent(T) .-= parent(Jᵀ) .* Δt ./ H
    return nothing
end

#####
##### Checkpointing
#####

Oceananigans.prognostic_state(ocean::SlabOcean) = (; T = Array(interior(ocean.sea_surface_temperature)))

function Oceananigans.restore_prognostic_state!(ocean::SlabOcean, state)
    interior(ocean.sea_surface_temperature) .= state.T
    return ocean
end

Oceananigans.restore_prognostic_state!(ocean::SlabOcean, ::Nothing) = ocean
