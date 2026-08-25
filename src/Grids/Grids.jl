module Grids

export PressureLevelVerticalDiscretization, PressureLevelGrid, surface_elevation, is_three_dimensional

include("pressure_level_vertical_discretization.jl")

"""
    is_three_dimensional(grid)

Whether a prescribed component over `grid` carries volumetric (z-`Center`) fields rather than surface
(z-`Nothing`) ones. `Nz == 1` ⇒ surface forcing (ocean / sea-ice coupling); `Nz > 1` ⇒ a volumetric
state, such as the parent of a nest. A single-level grid — even one with a `Bounded` z, like an ocean
coupling grid — is treated as surface.
"""
@inline is_three_dimensional(grid) = size(grid, 3) > 1

end # module
