using Oceananigans: LatitudeLongitudeGrid, Periodic, Bounded, Flat, CPU, Field,
                    Face, Center

import Oceananigans.TimeSteppers: time_step!, initialize!
import Oceananigans.Architectures: architecture
import Oceananigans.Fields: set!

using NumericalEarth.EarthSystemModels: EarthSystemModel
import NumericalEarth.EarthSystemModels: default_nan_checker,
                                         reference_density,
                                         heat_capacity,
                                         ocean_temperature,
                                         ocean_salinity,
                                         ocean_surface_salinity,
                                         ocean_surface_temperature,
                                         ocean_surface_velocities

import Base: eltype

using MITgcm: MITgcmOceanSimulation, MITgcmLibrary,
              set_timestep!, get_timestep, step!, refresh_state!,
              set_fu!, set_fv!, set_qnet!, set_empmr!, set_qsw!, set_saltflux!

default_nan_checker(model::EarthSystemModel{<:Any, <:Any, <:MITgcmOceanSimulation}) = nothing
initialize!(::MITgcmOceanSimulation) = nothing

architecture(model::EarthSystemModel{<:Any, <:Any, <:MITgcmOceanSimulation}) = CPU()
eltype(model::EarthSystemModel{<:Any, <:Any, <:MITgcmOceanSimulation}) = Float64

##### 
##### Time stepping
##### 

"""
    time_step!(ocean::MITgcmOceanSimulation, Δt)

Advance the MITgcm ocean by `Δt` seconds. Sets all MITgcm timestep parameters
(deltaT, deltaTMom, deltaTClock, deltaTFreeSurf, dTtracerLev) to `Δt`, then
takes a single forward step.
"""
function time_step!(ocean::MITgcmOceanSimulation, Δt)
    set_timestep!(ocean.library, Float64(Δt))
    step!(ocean.library)
    refresh_state!(ocean)
    return nothing
end

reference_density(ocean::MITgcmOceanSimulation) = ocean.reference_density
heat_capacity(ocean::MITgcmOceanSimulation) = ocean.heat_capacity

ocean_temperature(ocean::MITgcmOceanSimulation) = ocean.theta
ocean_salinity(ocean::MITgcmOceanSimulation) = ocean.salt

ocean_surface_temperature(ocean::MITgcmOceanSimulation) = view(ocean.theta, :, :, 1)
ocean_surface_salinity(ocean::MITgcmOceanSimulation) = view(ocean.salt, :, :, 1)

function ocean_surface_velocities(ocean::MITgcmOceanSimulation)
    u_surface = view(ocean.uvel, :, :, 1)
    v_surface = view(ocean.vvel, :, :, 1)
    return u_surface, v_surface
end

"""
    surface_grid(ocean::MITgcmOceanSimulation)

Construct an Oceananigans `LatitudeLongitudeGrid` from MITgcm's grid arrays.
Assumes the MITgcm grid is a regular lat-lon grid.
"""
function surface_grid(ocean::MITgcmOceanSimulation)
    xc = ocean.xc
    yc = ocean.yc
    Nx = size(xc, 1)
    Ny = size(xc, 2)

    # Extract 1D coordinate vectors (MITgcm stores 2D, but for lat-lon they are separable)
    λc = xc[:, 1]     # longitude centers
    φc = yc[1, :]     # latitude centers

    # Compute face (edge) coordinates from centers
    # Assume uniform spacing (can be extended for stretched grids)
    Δλ = length(λc) > 1 ? λc[2] - λc[1] : 1.0
    Δφ = length(φc) > 1 ? φc[2] - φc[1] : 1.0

    λf = vcat(λc[1] - Δλ/2, [λc[i] + Δλ/2 for i in 1:Nx])
    φf = vcat(φc[1] - Δφ/2, [φc[j] + Δφ/2 for j in 1:Ny])

    # Determine topology: periodic in longitude if grid spans ~360 degrees
    lon_span = λf[end] - λf[1]
    TX = lon_span >= 359.0 ? Periodic : Bounded

    return LatitudeLongitudeGrid(size = (Nx, Ny),
                                  longitude = λf,
                                  latitude = φf,
                                  topology = (TX, Bounded, Flat),
                                  halo = (2, 2))
end
