module NumericalEarthVelocityTransportExt

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: AbstractField, Field, XFaceField, YFaceField,
                           compute_at!, interior
using Oceananigans.Grids: Center, Face, LatitudeLongitudeGrid, RightFaceFolded,
                          new_data, ξnode, ηnode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Operators: Δx, Δy, extrinsic_vector
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

using NumericalEarth: Diagnostics

using GeoInterface: GeoInterface
using LibGEOS: LibGEOS
using LinearAlgebra: Diagonal, mul!
using SparseArrays: sparse

include("line_intersection_regridder.jl")
include("transport_operations.jl")

end # module NumericalEarthVelocityTransportExt
