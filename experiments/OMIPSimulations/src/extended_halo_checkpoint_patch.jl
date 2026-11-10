#####
##### Restore checkpoints written by Oceananigans ss/for-omip before 1831002
#####
#
# Those checkpoints store the barotropic η, U, V and every time average of η with the y-halo extended for the
# substeps (1456×1343 on eORCA025 against 1456×1067 today), and the model state carries no `coriolis` entry.
# Trim the arrays to the current halo and restore a missing `coriolis` as `nothing`.
# Delete this file once no such checkpoint needs restoring.

using Oceananigans.Fields: Field
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

function trim_extended_halo(restored, from)
    Δy = size(from, 2) - size(restored, 2)
    widened_in_y = ndims(from) >= 2 && ndims(from) == ndims(restored) && size(from, 1) == size(restored, 1) &&
                   size(from)[3:end] == size(restored)[3:end] && Δy > 0 && iseven(Δy)
    widened_in_y || return from
    j₀ = Δy ÷ 2
    return from[:, j₀+1:j₀+size(restored, 2), ntuple(_ -> Colon(), ndims(from) - 2)...]
end

function Oceananigans.restore_prognostic_state!(restored::DenseArray, from::Array)
    copyto!(restored, trim_extended_halo(restored, from))
    return restored
end

function Oceananigans.restore_prognostic_state!(restored::Field, from::NamedTuple)
    Oceananigans.restore_prognostic_state!(parent(restored), from.data)
    haskey(from, :boundary_conditions) && Oceananigans.restore_prognostic_state!(restored.boundary_conditions, from.boundary_conditions)
    return restored
end

function Oceananigans.restore_prognostic_state!(restored::HydrostaticFreeSurfaceModel, from::NamedTuple)
    state = haskey(from, :coriolis) ? from : merge(from, (; coriolis = nothing))
    return invoke(Oceananigans.restore_prognostic_state!, Tuple{HydrostaticFreeSurfaceModel, Any}, restored, state)
end
