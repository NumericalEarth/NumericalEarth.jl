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

@inline east_idx(i, Nx) = ifelse(i == Nx, 1, i + 1)
@inline west_idx(i, Nx) = ifelse(i == 1, Nx, i - 1)

@kernel function _reconstruct_λFC_φFC_λCF_φCF!(λFC, φFC, λCF, φCF, λCC, φCC, λFF, φFF, Nx, Ny)
    i, j = @index(Global, NTuple)
    iE = east_idx(i, Nx)
    iW = west_idx(i, Nx)
    λm₁, φm₁ = spherical_midpoint(λCC[iW, j], φCC[iW, j], λCC[i, j], φCC[i, j])
    λFC[i, j] = λm₁
    φFC[i, j] = φm₁
    λm₂, φm₂ = spherical_midpoint(λFF[i, j], φFF[i, j], λFF[iE, j], φFF[iE, j])
    λCF[i, j] = λm₂
    φCF[i, j] = φm₂
end

@kernel function _reconstruct_e1_e2_metrics!(e1u, e1v, e1f, e1t, e2u, e2v, e2f, e2t, λCC, φCC, λFF, φFF, λFC, φFC, λCF, φCF, radius, Nx, Ny)
    i, j = @index(Global, NTuple)
    iE = east_idx(i, Nx)
    iW = west_idx(i, Nx)
    e1u[i, j] = haversine((λCC[iW, j], φCC[iW, j]), (λCC[i, j], φCC[i, j]), radius)
    e1v[i, j] = haversine((λFF[i, j], φFF[i, j]), (λFF[iE, j], φFF[iE, j]), radius)
    e1f[i, j] = haversine((λCF[i, j], φCF[i, j]), (λCF[iE, j], φCF[iE, j]), radius)
    if Ny == 1
        e2u[i, j] = e1u[i, j]
        e2v[i, j] = e1v[i, j]
        e2f[i, j] = e1f[i, j]
    else
        if j < Ny
            e2u[i, j] = haversine((λFC[i, j], φFC[i, j]), (λFC[i, j+1], φFC[i, j+1]), radius)
        else
            e2u[i, Ny] = e2u[i, Ny-1]
        end

        if j > 1
            e2v[i, j] = haversine((λCC[i, j-1], φCC[i, j-1]), (λCC[i, j], φCC[i, j]), radius)
            e2f[i, j] = haversine((λFC[i, j-1], φFC[i, j-1]), (λFC[i, j], φFC[i, j]), radius)
        else
            e2v[i, 1] = e1v[i, 1]
            e2f[i, 1] = e1f[i, 1]
        end
    end

    e1t[i, j] = haversine((λFC[iW, j], φFC[iW, j]), (λFC[i, j], φFC[i, j]), radius)
    if Ny == 1
        e2t[i, j] = e2v[i, j]
    elseif j < Ny
        e2t[i, j] = (e2v[i, j] + e2v[i, j+1]) / 2
    else
        e2t[i, Ny] = e2v[i, Ny]
    end
end

@kernel function _reconstruct_Az_interior!(AzCC, AzFF, λCC, φCC, λFF, φFF, radius, Nx, Ny)
    i, j = @index(Global, NTuple)
    iE = east_idx(i, Nx)
    iW = west_idx(i, Nx)
    if j < Ny
        A = spherical_quadrilateral_area_unit(λFF[i, j],    φFF[i, j],
                                              λFF[iE, j],   φFF[iE, j],
                                              λFF[iE, j+1], φFF[iE, j+1],
                                              λFF[i, j+1],  φFF[i, j+1])
        AzCC[i, j] = A * radius^2
    end
    if j > 1
        A = spherical_quadrilateral_area_unit(λCC[iW, j-1], φCC[iW, j-1],
                                              λCC[i, j-1],  φCC[i, j-1],
                                              λCC[i, j],    φCC[i, j],
                                              λCC[iW, j],   φCC[iW, j])
        AzFF[i, j] = A * radius^2
    end
end

@kernel function _fill_AzCC_boundaries!(AzCC, AzFF, Ny)
    i = @index(Global, Linear)
    AzCC[i, Ny] = AzCC[i, Ny-1]
    AzFF[i, 1] = AzFF[i, 2]
end

function reconstruct_orca_mesh_from_CC_FF_points(λCC, φCC, λFF, φFF; radius)
    size(λCC) == size(φCC) || throw(ArgumentError("glamt and gphit size mismatch: $(size(λCC)) vs $(size(φCC))."))
    size(λFF) == size(φFF) || throw(ArgumentError("glamf and gphif size mismatch: $(size(λFF)) vs $(size(φFF))."))
    size(λCC) == size(λFF) || throw(ArgumentError("T-point and F-point grids must have matching size, got $(size(λCC)) and $(size(λFF))."))

    Nx, Ny = size(λCC)
    overlap = periodic_overlap_index(λCC)
    AFT = promote_type(eltype(λCC), eltype(φCC), eltype(λFF), eltype(φFF), typeof(radius))

    λFFₒ = shift_face_y(shift_face_x(λFF, overlap))
    φFFₒ = shift_face_y(shift_face_x(φFF, overlap))

    λFC  = similar(λCC, AFT)
    φFC  = similar(φCC, AFT)
    λCF  = similar(λCC, AFT)
    φCF  = similar(φCC, AFT)
    dev  = Oceananigans.Architectures.device(architecture(λFC))

    launch_xy = KernelParameters(1:Nx, 1:Ny)

    _reconstruct_λFC_φFC_λCF_φCF!(dev, (Nx, Ny), (16, 16))(λFC, φFC, λCF, φCF, λCC, φCC, λFFₒ, φFFₒ, Nx, Ny)

    e1u = similar(λCC, AFT)
    e2u = similar(λCC, AFT)
    e1v = similar(λCC, AFT)
    e2v = similar(λCC, AFT)
    e1f = similar(λCC, AFT)
    e2f = similar(λCC, AFT)
    e1t = similar(λCC, AFT)
    e2t = similar(λCC, AFT)

    _reconstruct_e1_e2_metrics!(dev, (Nx, Ny), (16, 16))(e1u, e1v, e1f, e1t, e2u, e2v, e2f, e2t, λCC, φCC, λFFₒ, φFFₒ, λFC, φFC, λCF, φCF, radius, Nx, Ny)

    AzCC = similar(λCC, AFT)
    AzFC = e1u .* e2u
    AzCF = e1v .* e2v
    AzFF = similar(λCC, AFT)

    if Ny > 1
        _reconstruct_Az_interior!(dev, (Nx, Ny), (16, 16))(AzCC, AzFF, λCC, φCC, λFFₒ, φFFₒ, radius, Nx, Ny)
        _fill_AzCC_boundaries!(dev, Nx, 16)(AzCC, AzFF, Ny)
    else
        AzCC .= e1t .* e2t
        AzFF .= AzCC
    end

    return (; λCC, λFC, λCF, λFF = λFFₒ, φCC, φFC, φCF, φFF = φFFₒ,
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
    metrics = ("glamt", "glamu", "glamv", "glamf",
               "gphit", "gphiu", "gphiv", "gphif",
               "e1t", "e1u", "e1v", "e1f",
               "e2t", "e2u", "e2v", "e2f")

    # Assume ORCA horizontal variables are stored as (Nx, Ny).
    λCC = read_2d_nemo_variable(ds, "glamt")
    Nx, Ny = size(λCC)
    overlap = periodic_overlap_index(λCC)

    orcaread(data, name) = orient_xy(read_2d_nemo_variable(data, name), Nx, Ny; name)
    shift_x(data) = shift_face_x(data, overlap)
    shift_y(data) = shift_face_y(data)
    shift_xy(data) = shift_y(shift_x(data))

    if has_all_variables(ds, metrics)
        λCC, λFC, λCF, λFF = orcaread(ds, "glamt"), shift_x(orcaread(ds, "glamu")), shift_y(orcaread(ds, "glamv")), shift_xy(orcaread(ds, "glamf"))
        φCC, φFC, φCF, φFF = orcaread(ds, "gphit"), shift_x(orcaread(ds, "gphiu")), shift_y(orcaread(ds, "gphiv")), shift_xy(orcaread(ds, "gphif"))
        e1t, e1u, e1v, e1f = orcaread(ds, "e1t"),   shift_x(orcaread(ds, "e1u")),   shift_y(orcaread(ds, "e1v")),   shift_xy(orcaread(ds, "e1f"))
        e2t, e2u, e2v, e2f = orcaread(ds, "e2t"),   shift_x(orcaread(ds, "e2u")),   shift_y(orcaread(ds, "e2v")),   shift_xy(orcaread(ds, "e2f"))

        if "e1e2t" in keys(ds)
            AzCC, AzFC = orcaread(ds, "e1e2t"), shift_x(orcaread(ds, "e1e2u"))
            AzCF, AzFF = shift_y(orcaread(ds, "e1e2v")), shift_xy(orcaread(ds, "e1e2f"))
        else
            AzCC, AzFC, AzCF, AzFF = e1t .* e2t, e1u .* e2u, e1v .* e2v, e1f .* e2f
        end

        return (; λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF,
                  e1t, e1u, e1v, e1f, e2t, e2u, e2v, e2f,
                  AzCC, AzFC, AzCF, AzFF)
    end

    coords = ("glamt", "gphit", "glamf", "gphif")
    if has_all_variables(ds, coords)
        λCC = orcaread(ds, "glamt")
        λFF = orcaread(ds, "glamf")
        φCC = orcaread(ds, "gphit")
        φFF = orcaread(ds, "gphif")
        return reconstruct_orca_mesh_from_CC_FF_points(λCC, φCC, λFF, φFF; radius)
    end

    throw(ArgumentError("Unsupported ORCA mesh format. Missing either full staggered variables $(metrics) or T/F variables $(coords)."))
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

# Reindex x-face fields from NEMO to Oceananigans.
# In NEMO, U[i, j] sits on the east face of T[i, j].
# In Oceananigans, Face-x[i, j] is the west face of Center[i, j].
# So Oceananigans Face-x[i, j] should receive NEMO U[i-1, j] (with periodic wrap).
# `overlap` is the duplicated periodic tail-column count; `No = Nx - overlap`
# maps i=1 to the last unique column instead of a duplicated overlap column.
function shift_face_x(data, overlap)
    Nx = size(data, 1)
    No = Nx - overlap
    return data[vcat(No, 1:Nx-1), :]
end

# Reindex y-face fields from NEMO to Oceananigans.
# NEMO V/F are indexed as faces north of T-row j, while Oceananigans Face-y[j]
# is treated as the south face of Center-row j. This is a -1 row shift:
#   out[:, j] <- in[:, j-1] for j >= 2.
# Row 1 has no southern source row in the input, so we keep in[:, 1] and let
# halo filling / boundary-condition handling manage the exterior face.
function shift_face_y(data)
    Nx, Ny = size(data)
    shifted = similar(data)
    shifted[:, 1] .= data[:, 1]
    if Ny > 1
        shifted[:, 2:Ny] .= data[:, 1:Ny-1]
    end
    return shifted
end

# Copy NEMO data into a Field on `helper_grid`, fill halos, return as OffsetArray.
#
# Data is expected to already be in Oceananigans indexing when this is called.
function halo_filled_data(data, helper_grid, bcs, LX, LY)
    TX, TY, _ = topology(helper_grid)
    Nx, Ny, _ = size(helper_grid)
    Ni = Base.length(LX(), TX(), Nx)
    Nj = size(data, 2)

    field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    field.data[1:Ni, 1:Nj, 1] .= data[1:Ni, 1:Nj]
    fill_halo_regions!(field)

    return deepcopy(dropdims(field.data, dims = 3))
end

# Fill halos for all four stagger locations (CC, FC, CF, FF) at once.
function halo_fill_stagger(CC, FC, CF, FF, helper_grid, bcs)
    return (
        halo_filled_data(CC, helper_grid, bcs, Center, Center),
        halo_filled_data(FC, helper_grid, bcs, Face,   Center),
        halo_filled_data(CF, helper_grid, bcs, Center, Face),
        halo_filled_data(FF, helper_grid, bcs, Face,   Face),
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

    λCC,  λFC,  λCF,  λFF  = mesh.λCC,  mesh.λFC,  mesh.λCF,  mesh.λFF
    φCC,  φFC,  φCF,  φFF  = mesh.φCC,  mesh.φFC,  mesh.φCF,  mesh.φFF
    e1t,  e1u,  e1v,  e1f  = mesh.e1t,  mesh.e1u,  mesh.e1v,  mesh.e1f
    e2t,  e2u,  e2v,  e2f  = mesh.e2t,  mesh.e2u,  mesh.e2v,  mesh.e2f
    AzCC, AzFC, AzCF, AzFF = mesh.AzCC, mesh.AzFC, mesh.AzCF, mesh.AzFF

    # Extract tripolar pole parameters from F-point coordinates
    pole_idx = argmin(φFF[:, end])
    north_poles_latitude = φFF[pole_idx]
    first_pole_longitude = Float64(λFF[pole_idx])

    Nx, Ny = size(λCC)

    # Remove degenerate southern rows from the extended eORCA grid
    jr = south_rows_to_remove
    if jr > 0
        chop(data) = data[:, jr+1:end]

        λCC, λFC, λCF, λFF     = chop(λCC),  chop(λFC),  chop(λCF),  chop(λFF)
        φCC, φFC, φCF, φFF     = chop(φCC),  chop(φFC),  chop(φCF),  chop(φFF)
        e1t, e1u, e1v, e1f     = chop(e1t),  chop(e1u),  chop(e1v),  chop(e1f)
        e2t, e2u, e2v, e2f     = chop(e2t),  chop(e2u),  chop(e2v),  chop(e2f)
        AzCC, AzFC, AzCF, AzFF = chop(AzCC), chop(AzFC), chop(AzCF), chop(AzFF)

        Ny = size(λCC, 2)
    end

    southernmost_latitude = Float64(minimum(φCC))

    # With RightFaceFolded (Bounded-like) topology:
    #   Center-y has Ny interior points        ← matches NEMO data
    #   Face-y   has Ny + 1 interior points
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
    λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ     = halo_fill_stagger(λCC,  λFC,  λCF,  λFF,  helper_grid, bcs)
    φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ     = halo_fill_stagger(φCC,  φFC,  φCF,  φFF,  helper_grid, bcs)
    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ = halo_fill_stagger(e1t,  e1u,  e1v,  e1f,  helper_grid, bcs)
    Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ = halo_fill_stagger(e2t,  e2u,  e2v,  e2f,  helper_grid, bcs)
    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = halo_fill_stagger(AzCC, AzFC, AzCF, AzFF, helper_grid, bcs)

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
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude)
    )

    with_bathymetry || return underlying_grid

    # Load bathymetry
    bathy_meta = Metadatum(:bottom_height; dataset, dir)
    bathymetry_path = download_dataset(bathy_meta)

    bathy_ds   = Dataset(bathymetry_path)
    bathy_name = dataset_variable_name(bathy_meta)
    bathy_data = read_2d_nemo_variable(bathy_ds, bathy_name)
    close(bathy_ds)

    bathy_data = orient_xy(bathy_data, size(bathy_data)...; name = string(bathy_name))

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
