using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Fields: ConstantField, ZeroField

"""
    SlabOcean(grid;
              depth = 50,
              density = 1025,
              heat_capacity = 4000,
              clock = Clock{FT}(time=0))

A slab ocean component for `EarthSystemModel`. Represents the ocean as a single
well-mixed layer of fixed depth, with temperature evolving in response to
net surface heat fluxes:

    ∂T/∂t = Q / (ρ cₚ H)

where `Q` is the net downward surface heat flux (W/m²), `ρ` is the seawater density,
`cₚ` is the heat capacity, and `H` is the slab depth.

Internally, the `EarthSystemModel` coupling assembles a temperature flux `Jᵀ` (in K m/s),
so the tendency is computed as `∂T/∂t = -Jᵀ / H`.

Use `set!(ocean, T=value)` to initialize the slab temperature after construction.

Arguments
=========

- `grid`: An Oceananigans grid for the slab ocean domain.

Keyword Arguments
=================

- `depth`: Depth of the slab in meters. Default: 50.
- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
- `clock`: Clock for tracking slab ocean time. Default: `Clock{FT}(time=0)` where `FT` is the
  float type of the grid.
"""
struct SlabOcean{FT, G, Clk, T, F, H, ρ, C}
    grid :: G
    clock :: Clk
    temperature :: T
    temperature_flux :: F
    depth :: H
    density :: ρ
    heat_capacity :: C
end

function SlabOcean(grid;
                   FT = eltype(grid),
                   depth = 50,
                   density = 1025,
                   heat_capacity = 4000,
                   clock = Clock{FT}(time=0))

    temperature = CenterField(grid)
    temperature_flux = CenterField(grid)

    return SlabOcean{FT}(grid, clock, temperature,
                         temperature_flux, depth, density, heat_capacity)
end

# Inner constructor that captures FT
SlabOcean{FT}(grid, clock, T, tf, d, ρ, c) where FT =
    SlabOcean{FT, typeof(grid), typeof(clock), typeof(T),
              typeof(tf), typeof(d), typeof(ρ), typeof(c)}(grid, clock, T, tf, d, ρ, c)

function Oceananigans.set!(ocean::SlabOcean; T=nothing)
    !isnothing(T) && set!(ocean.temperature, T)
    return nothing
end

Base.summary(ocean::SlabOcean) = "SlabOcean(H=$(ocean.depth) m)"
Base.show(io::IO, ocean::SlabOcean) = print(io, summary(ocean))
Base.eltype(::SlabOcean{FT}) where FT = FT

#####
##### EarthSystemModels interface
#####

reference_density(ocean::SlabOcean) = ocean.density
heat_capacity(ocean::SlabOcean) = ocean.heat_capacity
ocean_temperature(ocean::SlabOcean) = ocean.temperature
ocean_salinity(ocean::SlabOcean{FT}) where FT = ConstantField(convert(FT, 35))
ocean_surface_temperature(ocean::SlabOcean) = ocean.temperature
ocean_surface_salinity(ocean::SlabOcean{FT}) where FT = ConstantField(convert(FT, 35))
ocean_surface_velocities(::SlabOcean{FT}) where FT = ZeroField(FT), ZeroField(FT)

#####
##### InterfaceComputations interface
#####

function ComponentExchanger(ocean::SlabOcean, exchange_grid)
    T = ocean.temperature
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
    T = ocean.temperature
    Jᵀ = ocean.temperature_flux
    H = ocean.depth
    parent(T) .-= parent(Jᵀ) .* Δt ./ H
    return nothing
end

#####
##### Checkpointing
#####

Oceananigans.prognostic_state(ocean::SlabOcean) = (; T = Array(interior(ocean.temperature)))

function Oceananigans.restore_prognostic_state!(ocean::SlabOcean, state)
    interior(ocean.temperature) .= state.T
    return ocean
end

Oceananigans.restore_prognostic_state!(ocean::SlabOcean, ::Nothing) = ocean
