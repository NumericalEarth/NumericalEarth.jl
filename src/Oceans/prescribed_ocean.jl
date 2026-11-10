using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: update_field_time_series!, FieldTimeSeries, cpu_interpolating_time_indices
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.MultiRegion: MultiRegionGrids
using Oceananigans.Utils: @apply_regionally, construct_regionally, launch!, prettysummary, prettytime
using KernelAbstractions: @kernel, @index
using NumericalEarth.EarthSystemModels: AbstractPrescribedComponent

"""
    PrescribedOcean(grid, times=[zero(grid)];
                    density = 1025,
                    heat_capacity = 4000,
                    clock = Clock{FT}(time=0),
                    sea_surface_temperature = default_prescribed_sst(grid, times),
                    sea_surface_salinity = default_prescribed_sss(grid, times),
                    velocities = default_prescribed_velocities(grid, times))

A prescribed ocean component for `EarthSystemModel` with sea surface
temperature, salinity, and velocities prescribed as `FieldTimeSeries`.

The ocean state does not evolve in response to surface fluxes — it follows
the prescribed data. Surface fluxes are still computed so the atmosphere
feels the ocean.

Arguments
=========

- `grid`: An Oceananigans grid for the ocean domain.
- `times`: Time instances for the prescribed data. Default: `[zero(grid)]` (constant).

Keyword Arguments
=================

- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
- `clock`: Clock for tracking ocean time. Default: `Clock{FT}(time=0)`.
- `sea_surface_temperature`: `FieldTimeSeries` for SST.
- `sea_surface_salinity`: `FieldTimeSeries` for SSS.
- `velocities`: `NamedTuple` of `FieldTimeSeries` for `(u, v)`.
"""
mutable struct PrescribedOcean{FT, G, C, T, S, U, TI, R, HC} <: AbstractPrescribedComponent
    grid :: G
    clock :: C
    sea_surface_temperature :: T
    sea_surface_salinity :: S
    velocities :: U
    times :: TI
    density :: R
    heat_capacity :: HC

    function PrescribedOcean{FT}(grid::G, clock::C, sst::T, sss::S, u::U, times::TI, ρ::R, cp::HC) where {FT, G, C, T, S, U, TI, R, HC}
        return new{FT, G, C, T, S, U, TI, R, HC}(grid, clock, sst, sss, u, times, ρ, cp)
    end
end

# On a multi-region grid, a `MultiRegionObject` holding one series per region
prescribed_field_time_series(grid, times) =
    construct_regionally(FieldTimeSeries{Center, Center, Nothing}, grid, times)

default_prescribed_sst(grid, times) = prescribed_field_time_series(grid, times)

function default_prescribed_sss(grid, times)
    sss = prescribed_field_time_series(grid, times)
    @apply_regionally fill!(parent(sss), 35)
    return sss
end

function default_prescribed_velocities(grid, times)
    u = prescribed_field_time_series(grid, times)
    v = prescribed_field_time_series(grid, times)
    return (; u, v)
end

function PrescribedOcean(grid, times=[zero(grid)];
                         FT = eltype(grid),
                         density = 1025,
                         heat_capacity = 4000,
                         clock = Clock{FT}(time=0),
                         sea_surface_temperature = default_prescribed_sst(grid, times),
                         sea_surface_salinity = default_prescribed_sss(grid, times),
                         velocities = default_prescribed_velocities(grid, times))

    ocean = PrescribedOcean{FT}(grid,
                                clock,
                                sea_surface_temperature,
                                sea_surface_salinity,
                                velocities,
                                times,
                                convert(FT, density),
                                convert(FT, heat_capacity))

    fill_prescribed_halo_regions!(ocean)

    return ocean
end

Grids.grid(ocean::PrescribedOcean) = ocean.grid

function Oceananigans.set!(ocean::PrescribedOcean; T=nothing, S=nothing, u=nothing, v=nothing)
    !isnothing(T) && @apply_regionally set!(ocean.sea_surface_temperature, T)
    !isnothing(S) && @apply_regionally set!(ocean.sea_surface_salinity, S)
    !isnothing(u) && @apply_regionally set!(ocean.velocities.u, u)
    !isnothing(v) && @apply_regionally set!(ocean.velocities.v, v)
    fill_prescribed_halo_regions!(ocean)
    return nothing
end

# The exchanger reads the halo columns straight out of the time series (see
# `interpolate_prescribed_state!`), so their halos must be valid. Fill them whenever the data is
# set; a partly-in-memory series fills its own halos each time it loads a new window.
# On a multi-region grid the exchanger fills the halos of the interpolated state instead.
fill_prescribed_halo_regions!(::PrescribedOcean{<:Any, <:MultiRegionGrids}) = nothing

function fill_prescribed_halo_regions!(ocean::PrescribedOcean)
    fill_halo_regions!(ocean.sea_surface_temperature)
    fill_halo_regions!(ocean.sea_surface_salinity)
    fill_halo_regions!(ocean.velocities.u)
    fill_halo_regions!(ocean.velocities.v)
    return nothing
end

function Base.summary(ocean::PrescribedOcean{FT}) where FT
    A = nameof(typeof(architecture(ocean.grid)))
    G = nameof(typeof(ocean.grid))
    Nt = length(ocean.times)
    return string("PrescribedOcean{$FT, $A, $G}",
                  "(Nt = ", Nt, ")")
end

function Base.show(io::IO, ocean::PrescribedOcean)
    print(io, summary(ocean), "\n",
          "├── grid: ", summary(ocean.grid), "\n",
          "├── times: ", prettysummary(ocean.times), "\n",
          "├── density: ", prettysummary(ocean.density), "\n",
          "└── heat_capacity: ", prettysummary(ocean.heat_capacity))
end

Base.eltype(::PrescribedOcean{FT}) where FT = FT

#####
##### EarthSystemModels interface
#####

EarthSystemModels.is_sea_ice_component(::PrescribedOcean) = false

function EarthSystemModels.adopt_clock(ocean::PrescribedOcean{FT}, clock) where FT
    new_clock = EarthSystemModels.matching_clock(ocean.clock, clock)
    isnothing(new_clock) && return ocean
    EarthSystemModels.warn_clock_coercion(ocean, new_clock)
    return PrescribedOcean{FT}(ocean.grid,
                               new_clock,
                               ocean.sea_surface_temperature,
                               ocean.sea_surface_salinity,
                               ocean.velocities,
                               ocean.times,
                               ocean.density,
                               ocean.heat_capacity)
end

EarthSystemModels.reference_density(ocean::PrescribedOcean) = ocean.density
EarthSystemModels.heat_capacity(ocean::PrescribedOcean) = ocean.heat_capacity
EarthSystemModels.temperature_units(::PrescribedOcean) = DegreesKelvin()

EarthSystemModels.ocean_temperature(ocean::PrescribedOcean) = ocean.sea_surface_temperature
EarthSystemModels.ocean_salinity(ocean::PrescribedOcean) = ocean.sea_surface_salinity
EarthSystemModels.ocean_surface_temperature(ocean::PrescribedOcean) = ocean.sea_surface_temperature
EarthSystemModels.ocean_surface_salinity(ocean::PrescribedOcean) = ocean.sea_surface_salinity
EarthSystemModels.ocean_surface_velocities(ocean::PrescribedOcean) = ocean.velocities.u, ocean.velocities.v

#####
##### InterfaceComputations interface
#####

function EarthSystemModels.InterfaceComputations.ComponentExchanger(ocean::PrescribedOcean, exchange_grid)
    grid = ocean.grid
    T = CenterField(grid)
    S = CenterField(grid)
    u = CenterField(grid)
    v = CenterField(grid)

    state = (; u, v, T, S)
    interpolate_prescribed_state!(state, ocean)

    return ComponentExchanger(state, nothing)
end

EarthSystemModels.InterfaceComputations.net_fluxes(ocean::PrescribedOcean) = nothing
EarthSystemModels.InterfaceComputations.biogeochemical_interface(exchanger, ::PrescribedOcean; kwargs...) = NamedTuple()

EarthSystemModels.interpolate_state!(exchanger, grid, ocean::PrescribedOcean, coupled_model) =
    interpolate_prescribed_state!(exchanger.state, ocean)

# Read each series at the ocean clock, linearly interpolating between the two snapshots that bracket it.
# The interface flux kernels run over the halo columns too (`interface_kernel_parameters` spans 0:N+1), and
# the atmosphere's face-centered momentum fluxes average them into the first interior face, so the
# interpolation covers those columns rather than leaving them at zero (#754).
function interpolate_prescribed_state!(state, ocean::PrescribedOcean)
    uᵒ, vᵒ = ocean.velocities
    @apply_regionally interpolate_prescribed_state!(state, ocean.grid, ocean.clock.time,
                                                    ocean.sea_surface_temperature, ocean.sea_surface_salinity, uᵒ, vᵒ)
    fill_multi_region_halos!(state, ocean.grid)
    return nothing
end

function interpolate_prescribed_state!(state, grid, t, Tᵒ, Sᵒ, uᵒ, vᵒ)
    arch = architecture(grid)
    interpolator(fts) = cpu_interpolating_time_indices(arch, fts.times, fts.time_indexing, t)

    launch!(arch, grid, interface_kernel_parameters(grid), _interpolate_prescribed_ocean_state!,
            state.T, state.S, state.u, state.v,
            Tᵒ, Sᵒ, uᵒ, vᵒ,
            interpolator(Tᵒ), interpolator(Sᵒ), interpolator(uᵒ), interpolator(vᵒ))

    return nothing
end

# The halo columns of a single region series cannot hold the neighboring regions' data, so on a
# multi-region grid the halos of the interpolated state are filled from it instead.
fill_multi_region_halos!(state, grid) = nothing

function fill_multi_region_halos!(state, ::MultiRegionGrids)
    fill_halo_regions!(state.T)
    fill_halo_regions!(state.S)
    fill_halo_regions!((state.u, state.v))
    return nothing
end

@kernel function _interpolate_prescribed_ocean_state!(T, S, u, v, Tᵒ, Sᵒ, uᵒ, vᵒ, τT, τS, τu, τv)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T[i, j, 1] = Tᵒ[i, j, 1, τT]
        S[i, j, 1] = Sᵒ[i, j, 1, τS]
        u[i, j, 1] = uᵒ[i, j, 1, τu]
        v[i, j, 1] = vᵒ[i, j, 1, τv]
    end
end

# A prescribed ocean follows its data rather than responding to fluxes, so net flux assembly is not needed.
EarthSystemModels.update_net_fluxes!(coupled_model, ocean::PrescribedOcean) = nothing

#####
##### Time stepping — update FieldTimeSeries backends, tick the clock
#####

function Oceananigans.TimeSteppers.time_step!(ocean::PrescribedOcean, Δt)
    tick!(ocean.clock, Δt)
    update_state!(ocean)
    return nothing
end

function Oceananigans.TimeSteppers.update_state!(ocean::PrescribedOcean)
    time = Time(ocean.clock.time)
    @apply_regionally update_field_time_series!(ocean.sea_surface_temperature, time)
    @apply_regionally update_field_time_series!(ocean.sea_surface_salinity, time)
    @apply_regionally update_field_time_series!(ocean.velocities.u, time)
    @apply_regionally update_field_time_series!(ocean.velocities.v, time)
    return nothing
end

#####
##### Checkpointing
#####

Oceananigans.prognostic_state(ocean::PrescribedOcean) = (; clock = Oceananigans.prognostic_state(ocean.clock))

function Oceananigans.restore_prognostic_state!(ocean::PrescribedOcean, state)
    Oceananigans.restore_prognostic_state!(ocean.clock, state.clock)
    update_state!(ocean)
    return ocean
end

Oceananigans.restore_prognostic_state!(ocean::PrescribedOcean, ::Nothing) = ocean
