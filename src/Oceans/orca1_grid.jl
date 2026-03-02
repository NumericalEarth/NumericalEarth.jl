using Downloads
using NCDatasets
using Scratch
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!, FPivotZipperBoundaryCondition,
    NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!
using Oceananigans.Grids: RightFaceFolded, generate_coordinate
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!

# Zenodo record 4436658: eORCA1 mesh_mask and bathymetry
const ORCA1_mesh_mask_url      = "https://zenodo.org/records/4436658/files/eORCA1.2_mesh_mask.nc"
const ORCA1_mesh_mask_file     = "eORCA1.2_mesh_mask.nc"
const ORCA1_bathymetry_url     = "https://zenodo.org/records/4436658/files/eORCA_R1_bathy_meter_v2.2.nc"
const ORCA1_bathymetry_file    = "eORCA_R1_bathy_meter_v2.2.nc"

orca1_cache_dir::String = ""
function init_orca1_cache!()
    global orca1_cache_dir
    if isempty(orca1_cache_dir)
        orca1_cache_dir = @get_scratch!("ORCA1_grid")
    end
    return orca1_cache_dir
end

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

const default_south_rows_to_remove_ORCA1 = 35

"""
    ORCA1Grid(arch = CPU(), FT::DataType = Float64;
              halo = (4, 4, 4),
              z = (-6000, 0),
              Nz = 50,
              radius = Oceananigans.defaults.planet_radius,
              with_bathymetry = true,
              active_cells_map = true,
              south_rows_to_remove = $default_south_rows_to_remove_ORCA1)

Construct an `OrthogonalSphericalShellGrid` with `(Periodic, RightFaceFolded, Bounded)`
topology using coordinate and metric data from the NEMO eORCA1 `mesh_mask` file
(Zenodo; doi:10.5281/zenodo.4436658).

The horizontal grid (including coordinates, scale factors, and areas) is loaded
directly from the `mesh_mask` NetCDF file, which contains data at all four staggered
locations (T, U, V, F points). The user provides the vertical discretization via the `z`
keyword argument.

When `with_bathymetry = true` (the default), the ORCA1 bathymetry is also downloaded
from Zenodo and the grid is returned as an `ImmersedBoundaryGrid` with a `GridFittedBottom`.

Positional Arguments
====================

- `arch`: The architecture (e.g., `CPU()` or `GPU()`). Default: `CPU()`.
- `FT`: Floating point type. Default: `Float64`.

Keyword Arguments
=================

- `halo`: Halo size tuple `(Hx, Hy, Hz)`. Default: `(4, 4, 4)`.
- `z`: Vertical coordinate specification. Can be a 2-tuple `(z_bottom, z_top)`, an array of z-interfaces,
       or, e.g., an `ExponentialDiscretization`. Default: `(-6000, 0)`.
- `Nz`: Number of vertical levels (only used when `z` is a 2-tuple). Default: `50`.
- `radius`: Planet radius. Default: `Oceananigans.defaults.planet_radius`.
- `with_bathymetry`: If `true`, download the eORCA1 bathymetry and return an `ImmersedBoundaryGrid` with
                     `GridFittedBottom`. Default: `true`.
- `active_cells_map`: If `true` and `with_bathymetry = true`, build an active cells map
                      for efficient kernel execution over wet cells only. Default: `true`.
- `south_rows_to_remove`: Number of southern rows to remove from the eORCA grid.  The "extended" eORCA1 grid
                          contains degenerate padding rows near Antarctica that are entirely land.
                          Removing them reduces memory usage and computation. Default: `35`.
"""
function ORCA1Grid(arch = CPU(), FT::DataType = Float64;
                   halo = (4, 4, 4),
                   z = (-6000, 0),
                   Nz = 50,
                   radius = Oceananigans.defaults.planet_radius,
                   with_bathymetry = true,
                   active_cells_map = true,
                   south_rows_to_remove = default_south_rows_to_remove_ORCA1)

    # Download mesh_mask if not already cached
    cache_dir = init_orca1_cache!()
    mesh_mask_path = joinpath(cache_dir, ORCA1_mesh_mask_file)

    if !isfile(mesh_mask_path)
        @info "Downloading eORCA1 mesh_mask to $cache_dir..."
        Downloads.download(ORCA1_mesh_mask_url, mesh_mask_path)
    end

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

    # The "extended" eORCA1 grid (eORCA) has extra rows near Antarctica
    # that are entirely land with degenerate metrics (scale factors ~ 4 m).
    # Removing these rows reduces cost.
    jr = south_rows_to_remove
    if jr > 0
        chop_south(data) = data[:, jr+1:end]

        λCC = chop_south(λCC);  λFC = chop_south(λFC)
        λCF = chop_south(λCF);  λFF = chop_south(λFF)
        φCC = chop_south(φCC);  φFC = chop_south(φFC)
        φCF = chop_south(φCF);  φFF = chop_south(φFF)
        e1t = chop_south(e1t);  e1u = chop_south(e1u)
        e1v = chop_south(e1v);  e1f = chop_south(e1f)
        e2t = chop_south(e2t);  e2u = chop_south(e2u)
        e2v = chop_south(e2v);  e2f = chop_south(e2f)
        AzCC = chop_south(AzCC); AzFC = chop_south(AzFC)
        AzCF = chop_south(AzCF); AzFF = chop_south(AzFF)

        Ny_nemo = size(λCC, 2)
    end

    southernmost_latitude = Float64(minimum(φCC))

    # NEMO stores all variables with size (Nx, Ny_nemo).
    # With RightFaceFolded topology and Ny = Ny_nemo:
    #   - Center-y fields have Ny - 1 interior points (the fold row is handled by BC)
    #   - Face-y fields have Ny interior points
    # So we trim Center-y data (T, U points) to Ny-1 rows and keep
    # Face-y data (V, F points) as-is with all Ny rows.
    Ny = Ny_nemo
    Hx, Hy, Hz = halo

    # Trim helper: for Center-y data, drop the last (fold) row
    trim_center_y(data) = data[:, 1:Ny_nemo-1]

    # Set up vertical coordinate
    topology = (Periodic, RightFaceFolded, Bounded)
    Nz_val = z isa Tuple ? Nz : length(z) - 1
    Lz, z_coord = generate_coordinate(FT, topology, (Nx, Ny, Nz_val), halo, z, :z, 3, CPU())

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

    # Helper: set data into a Field, fill halos, extract as OffsetArray
    function halo_filled_data(data, LX, LY)
        field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
        set!(field, data)
        fill_halo_regions!(field)
        return deepcopy(dropdims(field.data, dims = 3))
    end

    # Fill halo regions for coordinates
    # Center-y (T, U) data must be trimmed; Face-y (V, F) data used as-is
    λᶜᶜᵃ = halo_filled_data(trim_center_y(λCC), Center, Center)
    λᶠᶜᵃ = halo_filled_data(trim_center_y(λFC), Face,   Center)
    λᶜᶠᵃ = halo_filled_data(λCF,                Center, Face)
    λᶠᶠᵃ = halo_filled_data(λFF,                Face,   Face)

    φᶜᶜᵃ = halo_filled_data(trim_center_y(φCC), Center, Center)
    φᶠᶜᵃ = halo_filled_data(trim_center_y(φFC), Face,   Center)
    φᶜᶠᵃ = halo_filled_data(φCF,                Center, Face)
    φᶠᶠᵃ = halo_filled_data(φFF,                Face,   Face)

    # Fill halo regions for scale factors
    Δxᶜᶜᵃ = halo_filled_data(trim_center_y(e1t), Center, Center)
    Δxᶠᶜᵃ = halo_filled_data(trim_center_y(e1u), Face,   Center)
    Δxᶜᶠᵃ = halo_filled_data(e1v,                Center, Face)
    Δxᶠᶠᵃ = halo_filled_data(e1f,                Face,   Face)

    Δyᶜᶜᵃ = halo_filled_data(trim_center_y(e2t), Center, Center)
    Δyᶠᶜᵃ = halo_filled_data(trim_center_y(e2u), Face,   Center)
    Δyᶜᶠᵃ = halo_filled_data(e2v,                Center, Face)
    Δyᶠᶠᵃ = halo_filled_data(e2f,                Face,   Face)

    # Fill halo regions for areas
    Azᶜᶜᵃ = halo_filled_data(trim_center_y(AzCC), Center, Center)
    Azᶠᶜᵃ = halo_filled_data(trim_center_y(AzFC), Face,   Center)
    Azᶜᶠᵃ = halo_filled_data(AzCF,                Center, Face)
    Azᶠᶠᵃ = halo_filled_data(AzFF,                Face,   Face)

    # Continue metrics to the south using a reference LatitudeLongitudeGrid.
    # The eORCA grid has degenerate padding cells near the southern boundary
    # and the south halo rows contain zeros after fill_halo_regions!.
    # Following the TripolarGrid pattern, we overwrite south halo metrics
    # with values from a regular LatitudeLongitudeGrid.
    latitude  = (southernmost_latitude, 90)
    longitude = (-180, 180)
    latitude_longitude_grid = LatitudeLongitudeGrid(; size = (Nx, Ny, Nz_val),
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
        Nx, Ny, Nz_val,
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

    # Load ORCA1 bathymetry
    bathymetry_path = joinpath(cache_dir, ORCA1_bathymetry_file)

    if !isfile(bathymetry_path)
        @info "Downloading eORCA1 bathymetry to $cache_dir..."
        Downloads.download(ORCA1_bathymetry_url, bathymetry_path)
    end

    bathy_ds = Dataset(bathymetry_path)
    bathy_data = Array(bathy_ds["Bathymetry"][:, :])
    close(bathy_ds)

    # Chop off the same southern rows from bathymetry
    if jr > 0
        bathy_data = chop_south(bathy_data)
    end

    # Bathymetry is T-point (Center, Center) data: trim last row for RightFaceFolded
    bathy_data = trim_center_y(bathy_data)

    # NEMO stores bathymetry as positive depth; convert to negative bottom height
    # (Oceananigans convention: z < 0 below sea level)
    bottom_height = -FT.(bathy_data)

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
end
