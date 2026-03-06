using Oceananigans.BoundaryConditions: fill_halo_regions!, FPivotZipperBoundaryCondition,
    NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!
using Oceananigans.Grids: RightFaceFolded, generate_coordinate
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!

using ..DataWrangling: dataset_variable_name
using ..DataWrangling.ORCA: ORCA1, default_south_rows_to_remove

"""
    read_2d_nemo_variable(ds, name)

Read a 2D variable from a NEMO NetCDF dataset, handling varying
dimension layouts: `(x, y)`, `(x, y, z)`, or `(x, y, z, t)`.
"""
function read_2d_nemo_variable(ds, name)
    var = ds[name]
    nd = ndims(var)
    if nd == 2
        return Array(var[:, :])
    elseif nd == 3
        return Array(var[:, :, 1])
    else
        return Array(var[:, :, 1, 1])
    end
end

# Detect periodic overlap columns in NEMO data.
# The eORCA grid has `n` trailing columns that are copies of the first `n` columns
# (e.g., columns 361:362 repeat columns 1:2 for eORCA1).
function periodic_overlap_index(λCC)
    Nx = size(λCC, 1)
    for n in min(div(Nx, 4), 10):-1:1
        if all(isapprox.(λCC[Nx-n+1:Nx, :], λCC[1:n, :]; atol=1e-4))
            return n
        end
    end
    return 0
end

# Shift Face-x data by -1 index while preserving the periodic overlap structure.
# NEMO U[i] is the eastern face of T[i], but Oceananigans Face[i] is the western
# face of Center[i], so Face[i] should get U[i-1].  A naive circshift breaks the
# overlap columns; instead we re-slice:
#   shifted = data[[Nx_unique; 1:Nx-1], :]
# which gives shifted[i] = data[i-1] with correct overlap at the trailing end.
function shift_face_x(data, overlap)
    Nx = size(data, 1)
    No = Nx - overlap
    return data[vcat(No, 1:Nx-1), :]
end

# Helper: copy NEMO data into a Field, fill halos, extract as OffsetArray.
#
# Stagger offsets (NEMO → Oceananigans):
#   Face-x:  shifted by -1 in x via shift_face_x (preserves overlap columns)
#   Face-y:  shifted by +1 in y (row 1 left empty, filled by continue_south!)
function halo_filled_data(data, helper_grid, bcs, LX, LY, overlap)
    TX, TY, _ = topology(helper_grid)
    Nx, Ny, _ = size(helper_grid)
    Ni = Base.length(LX(), TX(), Nx)
    Nj = size(data, 2)

    # Shift Face-x data to account for NEMO vs Oceananigans stagger convention
    shifted_data = LX === Face ? shift_face_x(data, overlap) : data

    field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    if LY === Center  # Center-y: no y-shift
        field.data[1:Ni, 1:Nj, 1] .= shifted_data[1:Ni, 1:Nj]
    else              # Face-y: shift +1 in y
        field.data[1:Ni, 2:Nj+1, 1] .= shifted_data[1:Ni, 1:Nj]
    end
    fill_halo_regions!(field)
    
    return deepcopy(dropdims(field.data, dims = 3))
end

"""
    ORCAGrid(arch = CPU(), FT::DataType = Float64;
             dataset,
             halo = (4, 4, 4),
             z = (-6000, 0),
             Nz = 50,
             radius = Oceananigans.defaults.planet_radius,
             with_bathymetry = true,
             active_cells_map = true,
             south_rows_to_remove = default_south_rows_to_remove(dataset))

Construct an `OrthogonalSphericalShellGrid` with `(Periodic, RightFaceFolded, Bounded)`
topology using coordinate and metric data from a NEMO eORCA `mesh_mask` file.

The `dataset` keyword argument specifies which ORCA configuration to use (e.g., `ORCA1()`).
The mesh mask and bathymetry files are downloaded automatically via the
`DataWrangling.ORCA` metadata interface.

The horizontal grid (including coordinates, scale factors, and areas) is loaded
directly from the `mesh_mask` NetCDF file, which contains data at all four staggered
locations (T, U, V, F points). The user provides the vertical discretization via the `z`
keyword argument.

When `with_bathymetry = true` (the default), the bathymetry is also downloaded
and the grid is returned as an `ImmersedBoundaryGrid` with a `GridFittedBottom`.

Positional Arguments
====================

- `arch`: The architecture (e.g., `CPU()` or `GPU()`). Default: `CPU()`.
- `FT`: Floating point type. Default: `Float64`.

Keyword Arguments
=================

- `dataset`: The ORCA dataset to use. Default: `ORCA1()` (from Zenodo; <https://doi.org.10.5281/zenodo.4436658>).
- `halo`: Halo size tuple `(Hx, Hy, Hz)`. Default: `(4, 4, 4)`.
- `z`: Vertical coordinate specification. Can be a 2-tuple `(z_bottom, z_top)`, an array of z-interfaces,
       or, e.g., an `ExponentialDiscretization`. Default: `(-6000, 0)`.
- `Nz`: Number of vertical levels (only used when `z` is a 2-tuple). Default: `50`.
- `radius`: Planet radius. Default: `Oceananigans.defaults.planet_radius`.
- `with_bathymetry`: If `true`, download the bathymetry and return an `ImmersedBoundaryGrid` with
                     `GridFittedBottom`. Default: `true`.
- `active_cells_map`: If `true` and `with_bathymetry = true`, build an active cells map
                      for efficient kernel execution over wet cells only. Default: `true`.
- `south_rows_to_remove`: Number of southern rows to remove from the eORCA grid.  The "extended" eORCA grid
                          contains degenerate padding rows near Antarctica that are entirely land.
                          Removing them reduces memory usage and computation.
"""
function ORCAGrid(arch = CPU(), FT::DataType = Float64;
                  dataset = ORCA1(),
                  halo = (4, 4, 4),
                  z = (-6000, 0),
                  Nz = 50,
                  radius = Oceananigans.defaults.planet_radius,
                  with_bathymetry = true,
                  active_cells_map = true,
                  south_rows_to_remove = default_south_rows_to_remove(dataset))

    # Validate z specification against Nz (mirrors Oceananigans' input_validation.jl)
    if z isa AbstractVector
        Nξ = length(z)
        if Nξ < Nz + 1
            throw(ArgumentError("length(z) = $Nξ has too few interfaces for the dimension size $Nz!"))
        elseif Nξ > Nz + 1
            throw(ArgumentError("length(z) = $Nξ has too many interfaces for the dimension size $Nz!"))
        end
    end

    # Download mesh_mask via the metadata interface
    mesh_meta = Metadatum(:mesh_mask; dataset)
    mesh_mask_path = download_dataset(mesh_meta)

    ds = Dataset(mesh_mask_path)

    # Read 2D coordinate arrays
    # NEMO stagger: T → (Center, Center), U → (Face, Center),
    #               V → (Center, Face),   F → (Face, Face)
    λCC = read_2d_nemo_variable(ds, "glamt")
    λFC = read_2d_nemo_variable(ds, "glamu")
    λCF = read_2d_nemo_variable(ds, "glamv")
    λFF = read_2d_nemo_variable(ds, "glamf")

    φCC = read_2d_nemo_variable(ds, "gphit")
    φFC = read_2d_nemo_variable(ds, "gphiu")
    φCF = read_2d_nemo_variable(ds, "gphiv")
    φFF = read_2d_nemo_variable(ds, "gphif")

    # Read scale factors (cell widths in meters)
    e1t = read_2d_nemo_variable(ds, "e1t")
    e1u = read_2d_nemo_variable(ds, "e1u")
    e1v = read_2d_nemo_variable(ds, "e1v")
    e1f = read_2d_nemo_variable(ds, "e1f")

    e2t = read_2d_nemo_variable(ds, "e2t")
    e2u = read_2d_nemo_variable(ds, "e2u")
    e2v = read_2d_nemo_variable(ds, "e2v")
    e2f = read_2d_nemo_variable(ds, "e2f")

    # Read pre-computed areas if available, otherwise compute from scale factors
    varnames = keys(ds)

    if "e1e2t" in varnames
        AzCC = read_2d_nemo_variable(ds, "e1e2t")
        AzFC = read_2d_nemo_variable(ds, "e1e2u")
        AzCF = read_2d_nemo_variable(ds, "e1e2v")
        AzFF = read_2d_nemo_variable(ds, "e1e2f")
    else
        AzCC = e1t .* e2t
        AzFC = e1u .* e2u
        AzCF = e1v .* e2v
        AzFF = e1f .* e2f
    end

    close(ds)

    # Extract tripolar pole parameters from F-point coordinates.
    # The two singularities sit at the F-points with maximum latitude
    # in the last row.
    last_row_φ = φFF[:, end]
    pole_idx   = argmax(last_row_φ)
    north_poles_latitude  = Float64(last_row_φ[pole_idx])
    first_pole_longitude  = Float64(λFF[pole_idx, end])

    Nx_nemo, Ny_nemo = size(λCC)
    Nx = Nx_nemo

    # Detect periodic overlap columns (e.g., eORCA1 has 2 trailing overlap columns)
    overlap = periodic_overlap_index(λCC)

    # The "extended" eORCA grid (eORCA) has extra rows near Antarctica
    # that are entirely land with degenerate metrics (scale factors ~ 4 m).
    # Removing these rows reduces cost.
    jr = south_rows_to_remove
    if jr > 0
        chop_south(data) = data[:, jr+1:end]
        λCC  = chop_south(λCC);  λFC  = chop_south(λFC)
        λCF  = chop_south(λCF);  λFF  = chop_south(λFF)
        φCC  = chop_south(φCC);  φFC  = chop_south(φFC)
        φCF  = chop_south(φCF);  φFF  = chop_south(φFF)
        e1t  = chop_south(e1t);  e1u  = chop_south(e1u)
        e1v  = chop_south(e1v);  e1f  = chop_south(e1f)
        e2t  = chop_south(e2t);  e2u  = chop_south(e2u)
        e2v  = chop_south(e2v);  e2f  = chop_south(e2f)
        AzCC = chop_south(AzCC); AzFC = chop_south(AzFC)
        AzCF = chop_south(AzCF); AzFF = chop_south(AzFF)

        Ny_nemo = size(λCC, 2)
    end

    southernmost_latitude = Float64(minimum(φCC))

    # NEMO stores all variables with size (Nx, Ny_nemo).  NEMO V[j] is the
    # northern face of T-cell j, but Oceananigans Face[j] is the southern face
    # of Center-cell j.  With Ny = Ny_nemo + 1 and RightFaceFolded:
    #   - Center-y interior has Ny - 1 = Ny_nemo points  ← matches NEMO T data
    #   - Face-y   interior has Ny     = Ny_nemo+1 points ← NEMO V data shifted +1
    # Face-y row 1 (southernmost) has no NEMO data and is filled by continue_south!.
    Ny = Ny_nemo + 1
    Hx, Hy, Hz = halo

    # Set up vertical coordinate
    topology = (Periodic, RightFaceFolded, Bounded)
    Lz, z_coord = generate_coordinate(FT, topology, (Nx, Ny, Nz), halo, z, :z, 3, CPU())

    # Helper RectilinearGrid for filling halo regions
    # Matches the TripolarGrid pattern in Oceananigans
    helper_grid = RectilinearGrid(; size = (Nx, Ny),
                                    halo = (Hx, Hy),
                                    x = (0, 1), y = (0, 1),
                                    topology = (Periodic, RightFaceFolded, Flat))

    bcs = FieldBoundaryConditions(north  = FPivotZipperBoundaryCondition(),
                                  south  = NoFluxBoundaryCondition(),
                                  west   = Oceananigans.PeriodicBoundaryCondition(),
                                  east   = Oceananigans.PeriodicBoundaryCondition(),
                                  top    = nothing,
                                  bottom = nothing)

    # Fill halo regions for coordinates
    λᶜᶜᵃ = halo_filled_data(λCC, helper_grid, bcs, Center, Center, overlap)
    λᶠᶜᵃ = halo_filled_data(λFC, helper_grid, bcs, Face,   Center, overlap)
    λᶜᶠᵃ = halo_filled_data(λCF, helper_grid, bcs, Center, Face,   overlap)
    λᶠᶠᵃ = halo_filled_data(λFF, helper_grid, bcs, Face,   Face,   overlap)

    φᶜᶜᵃ = halo_filled_data(φCC, helper_grid, bcs, Center, Center, overlap)
    φᶠᶜᵃ = halo_filled_data(φFC, helper_grid, bcs, Face,   Center, overlap)
    φᶜᶠᵃ = halo_filled_data(φCF, helper_grid, bcs, Center, Face,   overlap)
    φᶠᶠᵃ = halo_filled_data(φFF, helper_grid, bcs, Face,   Face,   overlap)

    # Fill halo regions for scale factors
    Δxᶜᶜᵃ = halo_filled_data(e1t, helper_grid, bcs, Center, Center, overlap)
    Δxᶠᶜᵃ = halo_filled_data(e1u, helper_grid, bcs, Face,   Center, overlap)
    Δxᶜᶠᵃ = halo_filled_data(e1v, helper_grid, bcs, Center, Face,   overlap)
    Δxᶠᶠᵃ = halo_filled_data(e1f, helper_grid, bcs, Face,   Face,   overlap)

    Δyᶜᶜᵃ = halo_filled_data(e2t, helper_grid, bcs, Center, Center, overlap)
    Δyᶠᶜᵃ = halo_filled_data(e2u, helper_grid, bcs, Face,   Center, overlap)
    Δyᶜᶠᵃ = halo_filled_data(e2v, helper_grid, bcs, Center, Face,   overlap)
    Δyᶠᶠᵃ = halo_filled_data(e2f, helper_grid, bcs, Face,   Face,   overlap)

    # Fill halo regions for areas
    Azᶜᶜᵃ = halo_filled_data(AzCC, helper_grid, bcs, Center, Center, overlap)
    Azᶠᶜᵃ = halo_filled_data(AzFC, helper_grid, bcs, Face,   Center, overlap)
    Azᶜᶠᵃ = halo_filled_data(AzCF, helper_grid, bcs, Center, Face,   overlap)
    Azᶠᶠᵃ = halo_filled_data(AzFF, helper_grid, bcs, Face,   Face,   overlap)

    # Continue metrics to the south using a reference LatitudeLongitudeGrid.
    # The eORCA grid has degenerate padding cells near the southern boundary
    # and the south halo rows contain zeros after fill_halo_regions!.
    # Following the TripolarGrid pattern, we overwrite south halo metrics
    # with values from a regular LatitudeLongitudeGrid.
    latitude  = (southernmost_latitude, 90)
    longitude = (-180, 180)

    latitude_longitude_grid = LatitudeLongitudeGrid(; size = (Nx, Ny, Nz),
                                                      latitude,
                                                      longitude,
                                                      halo,
                                                      z = (0, 1),
                                                      radius)

    continue_south!(Δxᶠᶠᵃ, latitude_longitude_grid.Δxᶠᶠᵃ)
    continue_south!(Δxᶠᶜᵃ, latitude_longitude_grid.Δxᶠᶜᵃ)
    continue_south!(Δxᶜᶠᵃ, latitude_longitude_grid.Δxᶜᶠᵃ)
    continue_south!(Δxᶜᶜᵃ, latitude_longitude_grid.Δxᶜᶜᵃ)

    continue_south!(Δyᶠᶠᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶠᶜᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶜᶠᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)
    continue_south!(Δyᶜᶜᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)

    continue_south!(Azᶠᶠᵃ, latitude_longitude_grid.Azᶠᶠᵃ)
    continue_south!(Azᶠᶜᵃ, latitude_longitude_grid.Azᶠᶜᵃ)
    continue_south!(Azᶜᶠᵃ, latitude_longitude_grid.Azᶜᶠᵃ)
    continue_south!(Azᶜᶜᵃ, latitude_longitude_grid.Azᶜᶜᵃ)

    underlying_grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
        arch,
        Nx, Ny, Nz,
        Hx, Hy, Hz,
        convert(FT, Lz),
        on_architecture(arch, map(FT, λᶜᶜᵃ)),
        on_architecture(arch, map(FT, λᶠᶜᵃ)),
        on_architecture(arch, map(FT, λᶜᶠᵃ)),
        on_architecture(arch, map(FT, λᶠᶠᵃ)),
        on_architecture(arch, map(FT, φᶜᶜᵃ)),
        on_architecture(arch, map(FT, φᶠᶜᵃ)),
        on_architecture(arch, map(FT, φᶜᶠᵃ)),
        on_architecture(arch, map(FT, φᶠᶠᵃ)),
        on_architecture(arch, z_coord),
        on_architecture(arch, map(FT, Δxᶜᶜᵃ)),
        on_architecture(arch, map(FT, Δxᶠᶜᵃ)),
        on_architecture(arch, map(FT, Δxᶜᶠᵃ)),
        on_architecture(arch, map(FT, Δxᶠᶠᵃ)),
        on_architecture(arch, map(FT, Δyᶜᶜᵃ)),
        on_architecture(arch, map(FT, Δyᶠᶜᵃ)),
        on_architecture(arch, map(FT, Δyᶜᶠᵃ)),
        on_architecture(arch, map(FT, Δyᶠᶠᵃ)),
        on_architecture(arch, map(FT, Azᶜᶜᵃ)),
        on_architecture(arch, map(FT, Azᶠᶜᵃ)),
        on_architecture(arch, map(FT, Azᶜᶠᵃ)),
        on_architecture(arch, map(FT, Azᶠᶠᵃ)),
        convert(FT, radius),
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude))

    if !with_bathymetry
        return underlying_grid
    end

    # Load bathymetry via the metadata interface
    bathy_meta = Metadatum(:bottom_height; dataset)
    bathymetry_path = download_dataset(bathy_meta)

    bathy_varname = dataset_variable_name(bathy_meta)
    bathy_ds = Dataset(bathymetry_path)
    bathy_data = Array(bathy_ds[bathy_varname][:, :])
    close(bathy_ds)

    # Chop off the same southern rows from bathymetry
    if jr > 0
        bathy_data = chop_south(bathy_data)
    end

    # NEMO stores bathymetry as positive depth; convert to negative bottom height
    # (Oceananigans convention: z < 0 below sea level).
    # In NEMO, bathymetry == 0 means land. We map these to bottom_height = 100
    # (above sea level) so that GridFittedBottom correctly masks them as land.
    bottom_height = convert.(FT, bathy_data)
    bottom_height .= ifelse.(bottom_height .> 0, .-bottom_height, FT(100))
    bottom_height = on_architecture(arch, bottom_height)

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
end
