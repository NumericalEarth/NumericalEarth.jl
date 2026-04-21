using Oceananigans.BoundaryConditions: fill_halo_regions!, FPivotZipperBoundaryCondition,
    NoFluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Fields: set!, convert_to_0_360
using Oceananigans.Grids: RightFaceFolded, generate_coordinate
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!
using CubedSphere.SphericalGeometry: lat_lon_to_cartesian, cartesian_to_lat_lon,
    spherical_area_quadrilateral
using Distances: haversine

using ..DataWrangling: dataset_variable_name, default_download_directory
using ..DataWrangling.ORCA: ORCA1, ORCA12, default_south_rows_to_remove

"""
    read_2d_nemo_variable(ds, name)

Read a 2D variable from a NEMO NetCDF dataset, handling varying
dimension layouts: `(x, y)`, `(x, y, z)`, or `(x, y, z, t)`.
"""
function read_2d_nemo_variable(ds, name)
    var = ds[name]
    # NOTE: `var[:]` does linear indexing and flattens to 1D.
    # We need the full shaped array here.
    data = Array(var)

    if ndims(data) < 2
        throw(ArgumentError("Variable $name could not be reduced to 2D. Size after slicing: $(size(data))"))
    end

    if ndims(data) > 2
        # Keep the two largest dimensions as horizontal (x, y), and pick
        # the first index for all other dimensions (for example t=1, z=1).
        # This handles layouts like (t, x, y), (x, y, t), (t, z, y, x), etc.
        sizes = collect(size(data))
        keep = sort(sortperm(sizes; rev = true)[1:2])
        indices = ntuple(d -> (d in keep ? Colon() : 1), ndims(data))
        data = @view data[indices...]
    end

    if ndims(data) != 2
        throw(ArgumentError("Variable $name could not be reduced to 2D. Size after slicing: $(size(data))"))
    end

    return Array(data)
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

function orient_orca_xy(data; name = "variable")
    sx, sy = size(data)
    if sx >= sy
        return data
    else
        # ORCA global grids should have Nx > Ny. If not, we assume (y, x) layout.
        return permutedims(data, (2, 1))
    end
end

@inline wrap_longitude(λ) = convert_to_0_360(λ + 180) - 180

@inline function midpoint_longitude(λ₁, λ₂)
    Δλ = λ₂ - λ₁
    Δλ = ifelse(Δλ > 180, Δλ - 360, Δλ)
    Δλ = ifelse(Δλ < -180, Δλ + 360, Δλ)
    return wrap_longitude(λ₁ + Δλ / 2)
end

@inline function spherical_midpoint(λ₁, φ₁, λ₂, φ₂)
    x₁, y₁, z₁ = lat_lon_to_cartesian(φ₁, λ₁; radius = 1, check_latitude_bounds = false)
    x₂, y₂, z₂ = lat_lon_to_cartesian(φ₂, λ₂; radius = 1, check_latitude_bounds = false)
    x = x₁ + x₂
    y = y₁ + y₂
    z = z₁ + z₂
    n = sqrt(x^2 + y^2 + z^2)

    if n < 1e-12
        λm = midpoint_longitude(λ₁, λ₂)
        φm = (φ₁ + φ₂) / 2
        return λm, φm
    end

    x /= n
    y /= n
    z /= n

    φm, λm = cartesian_to_lat_lon(x, y, z)
    λm = wrap_longitude(λm)
    return λm, φm
end

@inline function spherical_quadrilateral_area_unit(λ₁, φ₁, λ₂, φ₂, λ₃, φ₃, λ₄, φ₄)
    a = lat_lon_to_cartesian(φ₁, λ₁; radius = 1, check_latitude_bounds = false)
    b = lat_lon_to_cartesian(φ₂, λ₂; radius = 1, check_latitude_bounds = false)
    c = lat_lon_to_cartesian(φ₃, λ₃; radius = 1, check_latitude_bounds = false)
    d = lat_lon_to_cartesian(φ₄, λ₄; radius = 1, check_latitude_bounds = false)
    return spherical_area_quadrilateral(a, b, c, d; radius = 1)
end

function reconstruct_orca_staggered_mesh_from_t_f_points(λCC, φCC, λFF, φFF; radius)
    size(λCC) == size(φCC) || throw(ArgumentError("glamt and gphit size mismatch: $(size(λCC)) vs $(size(φCC))."))
    size(λFF) == size(φFF) || throw(ArgumentError("glamf and gphif size mismatch: $(size(λFF)) vs $(size(φFF))."))
    size(λCC) == size(λFF) || throw(ArgumentError("T-point and F-point grids must have matching size, got $(size(λCC)) and $(size(λFF))."))

    Nx, Ny = size(λCC)
    AFT = promote_type(eltype(λCC), eltype(φCC), eltype(λFF), eltype(φFF), typeof(radius))

    λFC = similar(λCC, AFT)
    φFC = similar(φCC, AFT)
    λCF = similar(λCC, AFT)
    φCF = similar(φCC, AFT)

    @inbounds for j in 1:Ny, i in 1:Nx
        iE = mod1(i + 1, Nx)
        λm, φm = spherical_midpoint(λCC[i, j], φCC[i, j], λCC[iE, j], φCC[iE, j])
        λFC[i, j] = λm
        φFC[i, j] = φm
    end

    if Ny > 1
        @inbounds for j in 1:Ny-1, i in 1:Nx
            λm, φm = spherical_midpoint(λCC[i, j], φCC[i, j], λCC[i, j+1], φCC[i, j+1])
            λCF[i, j] = λm
            φCF[i, j] = φm
        end
    end

    # Northern V-points are inferred from the northern F-point edge.
    @inbounds for i in 1:Nx
        iE = mod1(i + 1, Nx)
        λm, φm = spherical_midpoint(λFF[i, Ny], φFF[i, Ny], λFF[iE, Ny], φFF[iE, Ny])
        λCF[i, Ny] = λm
        φCF[i, Ny] = φm
    end

    e1u = similar(λCC, AFT)
    e2u = similar(λCC, AFT)
    e1v = similar(λCC, AFT)
    e2v = similar(λCC, AFT)
    e1f = similar(λCC, AFT)
    e2f = similar(λCC, AFT)
    e1t = similar(λCC, AFT)
    e2t = similar(λCC, AFT)

    @inbounds for j in 1:Ny, i in 1:Nx
        iE = mod1(i + 1, Nx)
        e1u[i, j] = haversine((λCC[i, j], φCC[i, j]), (λCC[iE, j], φCC[iE, j]), radius)
        e1v[i, j] = haversine((λFF[i, j], φFF[i, j]), (λFF[iE, j], φFF[iE, j]), radius)
        e1f[i, j] = haversine((λCF[i, j], φCF[i, j]), (λCF[iE, j], φCF[iE, j]), radius)
    end

    @inbounds for j in 1:Ny-1, i in 1:Nx
        e2u[i, j] = haversine((λFC[i, j], φFC[i, j]), (λFC[i, j+1], φFC[i, j+1]), radius)
        e2v[i, j] = haversine((λCC[i, j], φCC[i, j]), (λCC[i, j+1], φCC[i, j+1]), radius)
        e2f[i, j] = haversine((λFC[i, j], φFC[i, j]), (λFC[i, j+1], φFC[i, j+1]), radius)
    end

    if Ny > 1
        @inbounds for i in 1:Nx
            e2u[i, Ny] = e2u[i, Ny-1]
            e2v[i, Ny] = e2v[i, Ny-1]
            e2f[i, Ny] = e2f[i, Ny-1]
        end
    else
        @inbounds for i in 1:Nx
            e2u[i, 1] = e1u[i, 1]
            e2v[i, 1] = e1v[i, 1]
            e2f[i, 1] = e1f[i, 1]
        end
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        iW = mod1(i - 1, Nx)
        e1t[i, j] = haversine((λFC[iW, j], φFC[iW, j]), (λFC[i, j], φFC[i, j]), radius)
    end

    if Ny > 1
        @inbounds for i in 1:Nx
            e2t[i, 1] = e2v[i, 1]
            for j in 2:Ny
                e2t[i, j] = (e2v[i, j-1] + e2v[i, j]) / 2
            end
        end
    else
        @inbounds for i in 1:Nx
            e2t[i, 1] = e2v[i, 1]
        end
    end

    AzCC = similar(λCC, AFT)
    AzFC = e1u .* e2u
    AzCF = e1v .* e2v
    AzFF = similar(λCC, AFT)

    if Ny > 1
        @inbounds for j in 1:Ny-1, i in 1:Nx
            iE = mod1(i + 1, Nx)
            A = spherical_quadrilateral_area_unit(λFF[i, j],   φFF[i, j],
                                                  λFF[iE, j],  φFF[iE, j],
                                                  λFF[iE, j+1], φFF[iE, j+1],
                                                  λFF[i, j+1],  φFF[i, j+1])
            AzCC[i, j] = A * radius^2
        end
        @inbounds for i in 1:Nx
            AzCC[i, Ny] = AzCC[i, Ny-1]
        end

        @inbounds for j in 2:Ny, i in 1:Nx
            iW = mod1(i - 1, Nx)
            A = spherical_quadrilateral_area_unit(λCC[iW, j-1], φCC[iW, j-1],
                                                  λCC[i, j-1],  φCC[i, j-1],
                                                  λCC[i, j],    φCC[i, j],
                                                  λCC[iW, j],   φCC[iW, j])
            AzFF[i, j] = A * radius^2
        end
        @inbounds for i in 1:Nx
            AzFF[i, 1] = AzFF[i, 2]
        end
    else
        AzCC .= e1t .* e2t
        AzFF .= AzCC
    end

    return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
              e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
              AzCC, AzFC, AzCF, AzFF)
end

"""
    read_orca_staggered_mesh(ds)

Read ORCA horizontal coordinates and metrics.

Supports:
- full NEMO staggered mesh variables (`glamt/gphit/e1u/...`), and
- approximate reconstruction from T/F coordinates only (`glamt/gphit/glamf/gphif`)
  using Tripolar-style spherical metric assumptions.
"""
function read_orca_staggered_mesh(ds; radius = Oceananigans.defaults.planet_radius)
    full_stagger_vars = ("glamt", "glamu", "glamv", "glamf",
                         "gphit", "gphiu", "gphiv", "gphif",
                         "e1t", "e1u", "e1v", "e1f",
                         "e2t", "e2u", "e2v", "e2f")

    if has_all_variables(ds, full_stagger_vars)
        read_2d = read_2d_nemo_variable
        λCC = orient_orca_xy(read_2d(ds, "glamt"); name = "glamt")
        Nx, Ny = size(λCC)

        orient(data, name) = orient_xy(data, Nx, Ny; name)

        λFC, λCF, λFF = orient(read_2d(ds, "glamu"), "glamu"), orient(read_2d(ds, "glamv"), "glamv"), orient(read_2d(ds, "glamf"), "glamf")
        φCC, φFC, φCF, φFF = orient(read_2d(ds, "gphit"), "gphit"), orient(read_2d(ds, "gphiu"), "gphiu"), orient(read_2d(ds, "gphiv"), "gphiv"), orient(read_2d(ds, "gphif"), "gphif")
        e1t, e1u, e1v, e1f = orient(read_2d(ds, "e1t"), "e1t"), orient(read_2d(ds, "e1u"), "e1u"), orient(read_2d(ds, "e1v"), "e1v"), orient(read_2d(ds, "e1f"), "e1f")
        e2t, e2u, e2v, e2f = orient(read_2d(ds, "e2t"), "e2t"), orient(read_2d(ds, "e2u"), "e2u"), orient(read_2d(ds, "e2v"), "e2v"), orient(read_2d(ds, "e2f"), "e2f")

        if "e1e2t" in keys(ds)
            AzCC, AzFC = orient(read_2d(ds, "e1e2t"), "e1e2t"), orient(read_2d(ds, "e1e2u"), "e1e2u")
            AzCF, AzFF = orient(read_2d(ds, "e1e2v"), "e1e2v"), orient(read_2d(ds, "e1e2f"), "e1e2f")
        else
            AzCC, AzFC, AzCF, AzFF = e1t .* e2t, e1u .* e2u, e1v .* e2v, e1f .* e2f
        end

        return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
                  e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
                  AzCC, AzFC, AzCF, AzFF)
    end

    tf_vars = ("glamt", "gphit", "glamf", "gphif")
    if has_all_variables(ds, tf_vars)
        read_2d = read_2d_nemo_variable
        λCC = orient_orca_xy(read_2d(ds, "glamt"); name = "glamt")
        Nx, Ny = size(λCC)
        λFF = orient_xy(read_2d(ds, "glamf"), Nx, Ny; name = "glamf")
        φCC = orient_xy(read_2d(ds, "gphit"), Nx, Ny; name = "gphit")
        φFF = orient_xy(read_2d(ds, "gphif"), Nx, Ny; name = "gphif")
        return reconstruct_orca_staggered_mesh_from_t_f_points(λCC, φCC, λFF, φFF; radius)
    end

    throw(ArgumentError("Unsupported ORCA mesh format. Missing either full staggered variables $(full_stagger_vars) or T/F variables $(tf_vars)."))
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
(`T`, `U`, `V`, `F` points), they are used directly. If only `T` and `F`
coordinates are available (`glamt/gphit/glamf/gphif`), staggered coordinates and
metrics are reconstructed approximately using Tripolar-style spherical assumptions.

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
    mesh = read_orca_staggered_mesh(ds; radius)
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
