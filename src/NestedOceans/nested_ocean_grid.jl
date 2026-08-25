#####
##### nested_ocean_grid: a child grid whose bathymetry matches the parent's at the open boundaries
#####
#
# The child's lateral boundaries carry a transport integrated over the child's own column, so where the
# child's seafloor disagrees with the parent's, the prescribed transport does not fit the geometry it is
# imposed on. Blending the child bathymetry toward the parent's over an outer frame removes that
# mismatch at the walls while leaving the interior at the child's own resolution.
#
# This is a grid-level constructor rather than a `nested_ocean_model` keyword because the bathymetry is
# fixed when the `ImmersedBoundaryGrid` is built — `compute_numerical_bottom_height!` snaps the bottom to
# cell interfaces there — so by the time a model receives its grid the blend is already too late.

"""
$(TYPEDSIGNATURES)

Build the `ImmersedBoundaryGrid` of a nested ocean child over `underlying_grid`: bathymetry regridded
from `bathymetry_dataset`, blended toward `parent_dataset`'s own seafloor over the outermost
`blend_width` cells so the geometry at the open boundaries matches the state that will drive them.

The parent's seafloor is derived from where `parent_dataset` has missing values at `date` — the only
record of it, since the parent series are inpainted (see [`bathymetry_from_missing_values`](@ref)). Pass
`blend_width = 0` to keep the child's own bathymetry everywhere.

Additional keyword arguments flow to `regrid_bathymetry`.
"""
function nested_ocean_grid(underlying_grid, parent_dataset;
                           date,
                           bathymetry_dataset = ETOPO2022(),
                           blend_width = 5,
                           parent_padding = default_horizontal_padding(parent_dataset),
                           dir = default_download_directory(parent_dataset),
                           kw...)

    bottom_height = regrid_bathymetry(underlying_grid; dataset = bathymetry_dataset, kw...)

    if blend_width > 0
        region = BoundingBox(underlying_grid; padding = parent_padding)
        metadatum = Metadatum(:temperature; dataset = parent_dataset, date, region, dir)
        parent_bottom_height = bathymetry_from_missing_values(underlying_grid, metadatum)
        blend_parent_terrain!(bottom_height, parent_bottom_height; width = blend_width)
    end

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
end
