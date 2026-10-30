module NumericalEarthBreezeExt

using Breeze: Breeze
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: launch!
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels: sea_ice_concentration
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, computed_fluxes

include("coupled_radiation.jl")
include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")
include("breeze_prognostic_state.jl")
include("breeze_state_exchanger.jl")
include("breeze_nested_atmosphere.jl")
include("breeze_air_land_radiation.jl")
include("breeze_surface_albedo.jl")

function __init__()
    EXPLICIT_LID_W_SPONGE[] = lowercase(get(ENV, "NE_EXPLICIT_LID_W_SPONGE", "false")) != "false"
    EXPLICIT_LID_W_SPONGE[] && @info "Explicit ρw lid sponge on (environment, pre-T21 path)"
    rate = get(ENV, "NE_LATERAL_W_RELAXATION_RATE", "")
    LATERAL_W_RELAXATION_RATE[] = isempty(rate) ? nothing : parse(Float64, rate)
    isempty(rate) || @info "Explicit ρw relaxation rate $rate 1/s (environment)"
end

end # module NumericalEarthBreezeExt
