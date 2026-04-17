using Oceananigans.BoundaryConditions: fill_halo_regions!, FPivotZipperBoundaryCondition,
    NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!
using Oceananigans.Grids: RightFaceFolded, generate_coordinate
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!

using ..DataWrangling: dataset_variable_name, default_download_directory
using ..DataWrangling.ORCA: ORCA1, ORCA12, default_south_rows_to_remove

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

has_all_variables(ds, names) = all(name -> name in keys(ds), names)

function orient_xy(data, Nx, Ny; name = "variable")
    sx, sy = size(data)
    if (sx, sy) == (Nx, Ny)
        return data
    elseif (sx, sy) == (Ny, Nx)
        return permutedims(data, (2, 1))
    else
        throw(ArgumentError("Cannot orient $name with size $(size(data)) to (Nx, Ny)=($Nx, $Ny)."))
    end
end

@inline wrap_longitude(λ) = mod(λ + 180, 360) - 180

@inline function midpoint_longitude(λ₁, λ₂)
    Δλ = λ₂ - λ₁
    Δλ = ifelse(Δλ > 180, Δλ - 360, Δλ)
    Δλ = ifelse(Δλ < -180, Δλ + 360, Δλ)
    return wrap_longitude(λ₁ + Δλ / 2)
end

function average_x_periodic(data)
    Nx, Ny = size(data)
    avg = similar(data)
    @inbounds for j in 1:Ny, i in 1:Nx
        avg[i, j] = (data[i, j] + data[mod1(i+1, Nx), j]) / 2
    end
    return avg
end

function average_y_with_north_copy(data)
    Nx, Ny = size(data)
    avg = similar(data)
    @inbounds for j in 1:Ny-1, i in 1:Nx
        avg[i, j] = (data[i, j] + data[i, j+1]) / 2
    end
    @inbounds for i in 1:Nx
        avg[i, Ny] = data[i, Ny]
    end
    return avg
end

"""
    read_orca_staggered_mesh(ds)

Read ORCA horizontal coordinates and metrics.

Supports both:
- full NEMO staggered mesh files (`glamt/gphit/e1u/...`), and
- reduced files with only `longitude`, `latitude`, `e1t`, and `e2t`
  by reconstructing U/V/F staggered fields.
"""
function read_orca_staggered_mesh(ds)
    full_stagger_vars = ("glamt", "glamu", "glamv", "glamf",
                         "gphit", "gphiu", "gphiv", "gphif",
                         "e1t", "e1u", "e1v", "e1f",
                         "e2t", "e2u", "e2v", "e2f")

    if has_all_variables(ds, full_stagger_vars)
        read_2d = read_2d_nemo_variable
        λCC, λFC, λCF, λFF = read_2d(ds, "glamt"), read_2d(ds, "glamu"), read_2d(ds, "glamv"), read_2d(ds, "glamf")
        φCC, φFC, φCF, φFF = read_2d(ds, "gphit"), read_2d(ds, "gphiu"), read_2d(ds, "gphiv"), read_2d(ds, "gphif")
        e1t, e1u, e1v, e1f = read_2d(ds, "e1t"),   read_2d(ds, "e1u"),   read_2d(ds, "e1v"),   read_2d(ds, "e1f")
        e2t, e2u, e2v, e2f = read_2d(ds, "e2t"),   read_2d(ds, "e2u"),   read_2d(ds, "e2v"),   read_2d(ds, "e2f")

        if "e1e2t" in keys(ds)
            AzCC, AzFC = read_2d(ds, "e1e2t"), read_2d(ds, "e1e2u")
            AzCF, AzFF = read_2d(ds, "e1e2v"), read_2d(ds, "e1e2f")
        else
            AzCC, AzFC, AzCF, AzFF = e1t .* e2t, e1u .* e2u, e1v .* e2v, e1f .* e2f
        end

        return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
                  e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
                  AzCC, AzFC, AzCF, AzFF)
    end

    reduced_vars = ("longitude", "latitude", "e1t", "e2t")
    has_all_variables(ds, reduced_vars) || throw(ArgumentError("Unsupported ORCA mesh format. Missing required coordinate variables."))

    λ = collect(ds["longitude"][:])
    φ = collect(ds["latitude"][:])
    Nx, Ny = length(λ), length(φ)

    e1t = orient_xy(read_2d_nemo_variable(ds, "e1t"), Nx, Ny; name = "e1t")
    e2t = orient_xy(read_2d_nemo_variable(ds, "e2t"), Nx, Ny; name = "e2t")

    λFC₁ = similar(λ)
    @inbounds for i in 1:Nx
        λFC₁[i] = midpoint_longitude(λ[i], λ[mod1(i+1, Nx)])
    end

    φCF₁ = similar(φ)
    @inbounds for j in 1:Ny-1
        φCF₁[j] = (φ[j] + φ[j+1]) / 2
    end
    φCF₁[Ny] = φ[Ny]

    λCC = repeat(reshape(λ, Nx, 1), 1, Ny)
    λFC = repeat(reshape(λFC₁, Nx, 1), 1, Ny)
    λCF = copy(λCC)
    λFF = copy(λFC)

    φCC = repeat(reshape(φ, 1, Ny), Nx, 1)
    φFC = copy(φCC)
    φCF = repeat(reshape(φCF₁, 1, Ny), Nx, 1)
    φFF = copy(φCF)

    e1u = average_x_periodic(e1t)
    e2u = average_x_periodic(e2t)
    e1v = average_y_with_north_copy(e1t)
    e2v = average_y_with_north_copy(e2t)
    e1f = average_x_periodic(e1v)
    e2f = average_x_periodic(e2v)

    AzCC, AzFC, AzCF, AzFF = e1t .* e2t, e1u .* e2u, e1v .* e2v, e1f .* e2f

    return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
              e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
              AzCC, AzFC, AzCF, AzFF)
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
# face of Center[i], so Face[i] should get U[i-1].
function shift_face_x(data, overlap)
    Nx = size(data, 1)
    No = Nx - overlap
    return data[vcat(No, 1:Nx-1), :]
end

# Copy NEMO data into a Field on `helper_grid`, fill halos, return as OffsetArray.
#
# Stagger offsets (NEMO → Oceananigans):
#   Face-x: shifted by -1 in x via shift_face_x
#   Face-y: shifted by +1 in y (row 1 left empty, filled by continue_south!)
#
# With RightFaceFolded topology, Face-y fields have Ny+1 interior points.
# NEMO data (Ny_nemo rows) fills rows 2:Ny+1, covering the fold row at Ny+1.
function halo_filled_data(data, helper_grid, bcs, LX, LY, overlap)
    TX, TY, _ = topology(helper_grid)
    Nx, Ny, _ = size(helper_grid)
    Ni = Base.length(LX(), TX(), Nx)
    Nj = size(data, 2)

    shifted_data = LX === Face ? shift_face_x(data, overlap) : data

    field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    if LY === Center
        field.data[1:Ni, 1:Nj, 1] .= shifted_data[1:Ni, 1:Nj]
    else
        field.data[1:Ni, 2:Nj+1, 1] .= shifted_data[1:Ni, 1:Nj]
    end
    fill_halo_regions!(field)

    return deepcopy(dropdims(field.data, dims = 3))
end

# Fill halos for all four stagger locations (CC, FC, CF, FF) at once.
function halo_fill_stagger(CC, FC, CF, FF, helper_grid, bcs, overlap)
    return (
        halo_filled_data(CC, helper_grid, bcs, Center, Center, overlap),
        halo_filled_data(FC, helper_grid, bcs, Face,   Center, overlap),
        halo_filled_data(CF, helper_grid, bcs, Center, Face,   overlap),
        halo_filled_data(FF, helper_grid, bcs, Face,   Face,   overlap),
    )
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
             south_rows_to_remove = default_south_rows_to_remove(dataset),
             dir = default_download_directory(dataset))

Construct an `OrthogonalSphericalShellGrid` with `(Periodic, RightFaceFolded, Bounded)`
topology using coordinate and metric data from a NEMO eORCA `mesh_mask` file.

The `dataset` keyword argument specifies which ORCA configuration to use (e.g., `ORCA1() or ORCA12()`).
The mesh mask and bathymetry files are downloaded automatically via the
`DataWrangling.ORCA` metadata interface.

The horizontal grid (including coordinates, scale factors, and areas) is loaded
directly from the `mesh_mask` NetCDF file. If all staggered NEMO fields are present
(`T`, `U`, `V`, `F` points), they are used directly. Otherwise, reduced-coordinate
files (longitude/latitude plus `e1t` and `e2t`) are supported via reconstructed
staggered fields.

When `with_bathymetry = true` (the default), the bathymetry is also downloaded
and the grid is returned as an `ImmersedBoundaryGrid` with a `GridFittedBottom`.

Positional Arguments
====================

- `arch`: The architecture (e.g., `CPU()` or `GPU()`). Default: `CPU()`.
- `FT`: Floating point type. Default: `Float64`.

Keyword Arguments
=================

- `dataset`: The ORCA dataset to use. Default: `ORCA1()`. `ORCA12()` is also supported (ORCA1 data from Zenodo; <https://doi.org/10.5281/zenodo.4436658>).
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
- `dir`: Directory to store and look up ORCA files (`mesh_mask` and bathymetry).
         Defaults to the dataset scratch cache via `default_download_directory(dataset)`.
"""
function ORCAGrid(arch = CPU(), FT::DataType = Float64;
                  dataset = ORCA1(),
                  halo = (4, 4, 4),
                  z = (-6000, 0),
                  Nz = 50,
                  radius = Oceananigans.defaults.planet_radius,
                  with_bathymetry = true,
                  active_cells_map = true,
                  south_rows_to_remove = default_south_rows_to_remove(dataset),
                  dir = default_download_directory(dataset))

    # Download mesh_mask via the metadata interface
    mesh_meta = Metadatum(:mesh_mask; dataset, dir)
    mesh_mask_path = download_dataset(mesh_meta)

    ds = Dataset(mesh_mask_path)
    mesh = read_orca_staggered_mesh(ds)
    close(ds)

    λCC, λFC, λCF, λFF = mesh.λCC, mesh.λFC, mesh.λCF, mesh.λFF
    φCC, φFC, φCF, φFF = mesh.φCC, mesh.φFC, mesh.φCF, mesh.φFF
    e1t, e1u, e1v, e1f = mesh.e1t, mesh.e1u, mesh.e1v, mesh.e1f
    e2t, e2u, e2v, e2f = mesh.e2t, mesh.e2u, mesh.e2v, mesh.e2f
    AzCC, AzFC, AzCF, AzFF = mesh.AzCC, mesh.AzFC, mesh.AzCF, mesh.AzFF

    # Extract tripolar pole parameters from F-point coordinates
    last_row_φ = φFF[:, end]
    pole_idx   = argmax(last_row_φ)
    north_poles_latitude = min(Float64(last_row_φ[pole_idx]), 89.999)
    first_pole_longitude = Float64(λFF[pole_idx, end])

    Nx, Ny = size(λCC)
    Ny_full = Ny

    # Detect periodic overlap columns (e.g., eORCA1 has 2 trailing overlap columns)
    overlap = periodic_overlap_index(λCC)

    # Remove degenerate southern rows from the extended eORCA grid
    jr = south_rows_to_remove
    if jr > 0
        chop(data) = data[:, jr+1:end]

        λCC, λFC, λCF, λFF = chop(λCC), chop(λFC), chop(λCF), chop(λFF)
        φCC, φFC, φCF, φFF = chop(φCC), chop(φFC), chop(φCF), chop(φFF)
        e1t, e1u, e1v, e1f = chop(e1t), chop(e1u), chop(e1v), chop(e1f)
        e2t, e2u, e2v, e2f = chop(e2t), chop(e2u), chop(e2v), chop(e2f)
        AzCC, AzFC, AzCF, AzFF = chop(AzCC), chop(AzFC), chop(AzCF), chop(AzFF)

        Ny = size(λCC, 2)
    end

    southernmost_latitude = Float64(minimum(φCC))

    # With RightFaceFolded (Bounded-like) topology:
    #   Center-y has Ny interior points        ← matches NEMO data
    #   Face-y   has Ny + 1 interior points    ← NEMO V/F data shifted +1, fold at Ny+1
    Hx, Hy, Hz = halo

    # Vertical coordinate
    topo = (Periodic, RightFaceFolded, Bounded)
    Lz, z_coord = generate_coordinate(FT, topo, (Nx, Ny, Nz), halo, z, :z, 3, CPU())

    # Helper grid and boundary conditions for halo filling
    helper_grid = RectilinearGrid(; size = (Nx, Ny), halo = (Hx, Hy),
                                    x = (0, 1), y = (0, 1),
                                    topology = (Periodic, RightFaceFolded, Flat))

    bcs = FieldBoundaryConditions(north  = FPivotZipperBoundaryCondition(),
                                  south  = NoFluxBoundaryCondition(),
                                  west   = Oceananigans.PeriodicBoundaryCondition(),
                                  east   = Oceananigans.PeriodicBoundaryCondition(),
                                  top    = nothing,
                                  bottom = nothing)

    # Fill halos for all stagger locations
    λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ    = halo_fill_stagger(λCC, λFC, λCF, λFF, helper_grid, bcs, overlap)
    φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ    = halo_fill_stagger(φCC, φFC, φCF, φFF, helper_grid, bcs, overlap)
    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ = halo_fill_stagger(e1t, e1u, e1v, e1f, helper_grid, bcs, overlap)
    Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ = halo_fill_stagger(e2t, e2u, e2v, e2f, helper_grid, bcs, overlap)
    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = halo_fill_stagger(AzCC, AzFC, AzCF, AzFF, helper_grid, bcs, overlap)

    # Fill south halo metrics from a reference LatitudeLongitudeGrid
    # (the eORCA south halo has degenerate/zero values after fill_halo_regions!)
    ref_grid = LatitudeLongitudeGrid(; size = (Nx, Ny, Nz),
                                       latitude = (southernmost_latitude, 90),
                                       longitude = (-180, 180),
                                       halo, z = (0, 1), radius)

    for (field, ref_name) in ((Δxᶜᶜᵃ, :Δxᶜᶜᵃ), (Δxᶠᶜᵃ, :Δxᶠᶜᵃ), (Δxᶜᶠᵃ, :Δxᶜᶠᵃ), (Δxᶠᶠᵃ, :Δxᶠᶠᵃ),
                              (Δyᶜᶜᵃ, :Δyᶜᶠᵃ), (Δyᶠᶜᵃ, :Δyᶠᶜᵃ), (Δyᶜᶠᵃ, :Δyᶜᶠᵃ), (Δyᶠᶠᵃ, :Δyᶠᶜᵃ),
                              (Azᶜᶜᵃ, :Azᶜᶜᵃ), (Azᶠᶜᵃ, :Azᶠᶜᵃ), (Azᶜᶠᵃ, :Azᶜᶠᵃ), (Azᶠᶠᵃ, :Azᶠᶠᵃ))
        continue_south!(field, getproperty(ref_grid, ref_name))
    end

    # Build the grid
    to_arch(data) = on_architecture(arch, map(FT, data))

    underlying_grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
        arch,
        Nx, Ny, Nz,
        Hx, Hy, Hz,
        convert(FT, Lz),
        to_arch(λᶜᶜᵃ), to_arch(λᶠᶜᵃ), to_arch(λᶜᶠᵃ), to_arch(λᶠᶠᵃ),
        to_arch(φᶜᶜᵃ), to_arch(φᶠᶜᵃ), to_arch(φᶜᶠᵃ), to_arch(φᶠᶠᵃ),
        on_architecture(arch, z_coord),
        to_arch(Δxᶜᶜᵃ), to_arch(Δxᶠᶜᵃ), to_arch(Δxᶜᶠᵃ), to_arch(Δxᶠᶠᵃ),
        to_arch(Δyᶜᶜᵃ), to_arch(Δyᶠᶜᵃ), to_arch(Δyᶜᶠᵃ), to_arch(Δyᶠᶠᵃ),
        to_arch(Azᶜᶜᵃ), to_arch(Azᶠᶜᵃ), to_arch(Azᶜᶠᵃ), to_arch(Azᶠᶠᵃ),
        convert(FT, radius),
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude))

    with_bathymetry || return underlying_grid

    # Load bathymetry
    bathy_meta = Metadatum(:bottom_height; dataset, dir)
    bathymetry_path = download_dataset(bathy_meta)

    bathy_ds = Dataset(bathymetry_path)
    bathy_name = dataset_variable_name(bathy_meta)
    bathy_data = read_2d_nemo_variable(bathy_ds, bathy_name)
    close(bathy_ds)

    bathy_data = orient_xy(bathy_data, Nx, Ny_full; name = string(bathy_name))

    if jr > 0
        bathy_data = chop(bathy_data)
    end

    # NEMO bathymetry is positive depth; convert to negative bottom height.
    # Land (bathymetry == 0) gets mapped to +100 so GridFittedBottom masks it.
    bottom_height = FT.(coalesce.(bathy_data, FT(0)))
    bottom_height .= ifelse.(isfinite.(bottom_height) .& (bottom_height .> 0), .-bottom_height, FT(100))
    bottom_height = on_architecture(arch, bottom_height)

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
end
