using Downloads
using NCDatasets
using Scratch
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!, FPivotZipperBoundaryCondition,
    NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!
using Oceananigans.Grids: RightFaceFolded, generate_coordinate
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar

# Zenodo record 4436658: eORCA1 mesh_mask and bathymetry
const ORCA1_mesh_mask_url  = "https://zenodo.org/records/4436658/files/eORCA1.2_mesh_mask.nc"
const ORCA1_mesh_mask_file = "eORCA1.2_mesh_mask.nc"

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

"""
    ORCA1Grid(arch = CPU(), FT::DataType = Float64;
              halo = (4, 4, 4),
              z = (-6000, 0),
              Nz = 50,
              radius = Oceananigans.defaults.planet_radius)

Construct an `OrthogonalSphericalShellGrid` with `(Periodic, RightFaceFolded, Bounded)`
topology using coordinate and metric data from the NEMO eORCA1 mesh_mask file
(Zenodo record 4436658).

The horizontal grid (coordinates, scale factors, and areas) is loaded directly from
the mesh_mask NetCDF file, which contains data at all four stagger locations
(T, U, V, F points). The user provides the vertical discretization via the `z`
keyword argument.

Positional Arguments
====================

- `arch`: The architecture (e.g., `CPU()` or `GPU()`). Default: `CPU()`.
- `FT`: Floating point type. Default: `Float64`.

Keyword Arguments
=================

- `halo`: Halo size tuple `(Hx, Hy, Hz)`. Default: `(4, 4, 4)`.
- `z`: Vertical coordinate specification. Can be a 2-tuple `(z_bottom, z_top)`,
       an array of z-interfaces, or an `ExponentialDiscretization`. Default: `(-6000, 0)`.
- `Nz`: Number of vertical levels (only used when `z` is a 2-tuple). Default: `50`.
- `radius`: Planet radius. Default: `Oceananigans.defaults.planet_radius`.
"""
function ORCA1Grid(arch = CPU(), FT::DataType = Float64;
                   halo = (4, 4, 4),
                   z = (-6000, 0),
                   Nz = 50,
                   radius = Oceananigans.defaults.planet_radius)

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

    # Extract tripolar pole parameters from F-point coordinates.
    # The two singularities sit at the F-points with maximum latitude
    # in the last row.
    last_row_φ = φFF[:, end]
    pole_idx   = argmax(last_row_φ)
    north_poles_latitude  = Float64(last_row_φ[pole_idx])
    first_pole_longitude  = Float64(λFF[pole_idx, end])
    southernmost_latitude = Float64(minimum(φCC))

    close(ds)

    # NEMO stores all variables with size (Nx_nemo, Ny_nemo).
    # With RightFaceFolded topology and Ny = Ny_nemo:
    #   - Center-y fields have Ny - 1 interior points (the fold row is handled by BC)
    #   - Face-y fields have Ny interior points
    # So we trim Center-y data (T, U points) to Ny-1 rows and keep
    # Face-y data (V, F points) as-is with all Ny rows.
    Nx, Ny_nemo = size(λCC)
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

    grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
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

    return grid
end
