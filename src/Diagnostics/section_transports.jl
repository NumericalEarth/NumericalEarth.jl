using ..EarthSystemModels: reference_density, heat_capacity

"""
$(TYPEDEF)

The u and v faces crossed by a hydrographic section, with sign +1 where positive velocity
flows to the right of the section (walking from its first waypoint to its last), -1 where it
flows to the left, and 0 on faces the section does not cross. `u_signs[i, j, 1]` is the sign of
the u face between cells `(i-1, j)` and `(i, j)`, and `v_signs[i, j, 1]` that of the v face
between `(i, j-1)` and `(i, j)`.
"""
struct TransportSection{U, V}
    u_signs :: U
    v_signs :: V
end

@inline unit_vector(λ, φ) = (cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))
@inline cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
@inline dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline cell_unit_vector(i, j, grid) = unit_vector(λnode(i, j, 1, grid, Center(), Center(), Center()),
                                                   φnode(i, j, 1, grid, Center(), Center(), Center()))

# Cells right of the great circle through a and b (normal n = a × b), between a and b, and not on the domain's edge
@inline function right_of_segment(i, j, grid, a, b, n)
    x = cell_unit_vector(i, j, grid)
    between = (dot3(cross3(a, x), cross3(x, b)) ≥ 0) & (dot3(x, a .+ b) > 0)
    interior = (1 < i < grid.Nx) & (1 < j < grid.Ny)
    return interior & between & (dot3(x, n) < 0)
end

# A face crosses the segment if one of its cells is right of the segment and the other left of its great circle
@inline function crossing_sign(i₁, j₁, i₂, j₂, grid, a, b, n)
    s = right_of_segment(i₂, j₂, grid, a, b, n) - right_of_segment(i₁, j₁, grid, a, b, n)
    x = ifelse(s > 0, cell_unit_vector(i₁, j₁, grid), cell_unit_vector(i₂, j₂, grid))
    return ifelse(dot3(x, n) ≥ 0, s, 0)
end

@kernel function _add_segment_crossings!(u_signs, v_signs, grid, a, b, n)
    i, j = @index(Global, NTuple)
    @inbounds u_signs[i, j, 1] += crossing_sign(i-1, j, i, j, grid, a, b, n)
    @inbounds v_signs[i, j, 1] += crossing_sign(i, j-1, i, j, grid, a, b, n)
end

"""
$(TYPEDSIGNATURES)

Return the `TransportSection` through the waypoints `longitudes`, `latitudes` (in degrees),
joined by great circles. Positive transport flows to the right of the section, walking from the
first waypoint to the last. Both ends should be on land so that the section spans the whole strait,
and the section must not cross the domain's edges.

```jldoctest
using NumericalEarth
using Oceananigans

grid = LatitudeLongitudeGrid(size = (20, 20, 2), longitude = (-10, 10), latitude = (60, 80), z = (-100, 0))
section = TransportSection(grid, [5, -5], [70, 70])

# output
TransportSection on 20×20×2 LatitudeLongitudeGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×2 halo
└── crosses 0 u faces and 10 v faces
```
"""
function TransportSection(grid, longitudes, latitudes)
    u_signs = Field{Face, Center, Nothing}(grid)
    v_signs = Field{Center, Face, Nothing}(grid)
    Nx, Ny = size(grid, 1), size(grid, 2)

    for n in 1:length(longitudes)-1
        a = unit_vector(longitudes[n],   latitudes[n])
        b = unit_vector(longitudes[n+1], latitudes[n+1])
        launch!(architecture(grid), grid, (Nx+1, Ny+1), _add_segment_crossings!,
                u_signs, v_signs, grid, a, b, cross3(a, b))
    end

    return TransportSection(u_signs, v_signs)
end

count_faces(signs) = count(!=(0), on_architecture(CPU(), interior(signs)))

function Base.show(io::IO, section::TransportSection)
    print(io, "TransportSection on ", summary(section.u_signs.grid), '\n',
          "└── crosses ", count_faces(section.u_signs), " u faces and ", count_faces(section.v_signs), " v faces")
end

# Faces on or below the bathymetry are closed; open boundaries are not
@inline open_face(i, j, k, grid, ℓx, ℓy, ℓz) =
    !immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) & !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline function u_flux(i, j, k, grid, u_signs, u, c, α, β)
    F = @inbounds u_signs[i, j, 1] * u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid) * (α + β * ℑxᶠᵃᵃ(i, j, k, grid, c))
    return ifelse(open_face(i, j, k, grid, Face(), Center(), Center()), F, zero(F))
end

@inline function v_flux(i, j, k, grid, v_signs, v, c, α, β)
    F = @inbounds v_signs[i, j, 1] * v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid) * (α + β * ℑyᵃᶠᵃ(i, j, k, grid, c))
    return ifelse(open_face(i, j, k, grid, Center(), Face(), Center()), F, zero(F))
end

# West and south faces of cell (i, j, k), plus east and north faces in the last column and row
@inline function section_flux(i, j, k, grid, u_signs, v_signs, u, v, c, α, β)
    F = u_flux(i, j, k, grid, u_signs, u, c, α, β) + v_flux(i, j, k, grid, v_signs, v, c, α, β)
    F += ifelse(i == grid.Nx, u_flux(i+1, j, k, grid, u_signs, u, c, α, β), zero(F))
    F += ifelse(j == grid.Ny, v_flux(i, j+1, k, grid, v_signs, v, c, α, β), zero(F))
    return F
end

# Transport of velocity × (α + β c) through the section, summed over the box of cells around its faces
function section_transport(ocean, section, c, α, β)
    grid = ocean.model.grid
    Nx, Ny = size(grid, 1), size(grid, 2)
    FT = eltype(grid)
    u, v = ocean.model.velocities.u, ocean.model.velocities.v

    faces = vcat(findall(!=(0), on_architecture(CPU(), interior(section.u_signs))),
                 findall(!=(0), on_architecture(CPU(), interior(section.v_signs))))
    i₁, i₂ = extrema(min(I[1], Nx) for I in faces)
    j₁, j₂ = extrema(min(I[2], Ny) for I in faces)

    flux = KernelFunctionOperation{Center, Center, Center}(section_flux, grid, section.u_signs, section.v_signs,
                                                           u, v, c, convert(FT, α), convert(FT, β))
    flux = Field(flux; indices = (i₁:i₂, j₁:j₂, :))
    return Field(Reduction(sum!, flux, dims = (1, 2, 3)))
end

"""
$(TYPEDSIGNATURES)

Return a scalar `Field` of the volume transport (m³ s⁻¹) of `ocean` through `section`,

    Σ  s u Ax + s v Ay

summed over the open faces of the section, with `s` their signs and `Ax`, `Ay` their areas.
"""
volume_transport(ocean, section::TransportSection) = section_transport(ocean, section, ZeroField(), 1, 0)

"""
$(TYPEDSIGNATURES)

Return a scalar `Field` of the heat transport (W) of `ocean` through `section`, relative to
`reference_temperature` `Tᵣ` (°C): `ρᵒᶜ cᵒᶜ (T - Tᵣ)` times the volume flux, with `ρᵒᶜ` and `cᵒᶜ`
the ocean's reference density and heat capacity and `T` interpolated to the faces.
"""
function heat_transport(ocean, section::TransportSection; reference_temperature = 0)
    ρc = reference_density(ocean) * heat_capacity(ocean)
    return section_transport(ocean, section, ocean.model.tracers.T, -ρc * reference_temperature, ρc)
end

"""
$(TYPEDSIGNATURES)

Return a scalar `Field` of the salt transport (kg s⁻¹) of `ocean` through `section`:
`ρᵒᶜ S / 1000` times the volume flux, with `S` in g kg⁻¹ interpolated to the faces.
"""
salt_transport(ocean, section::TransportSection) =
    section_transport(ocean, section, ocean.model.tracers.S, 0, reference_density(ocean) / 1000)

"""
$(TYPEDSIGNATURES)

Return a scalar `Field` of the freshwater transport (m³ s⁻¹) of `ocean` through `section`, relative
to `reference_salinity` `Sᵣ` (g kg⁻¹): `(Sᵣ - S) / Sᵣ` times the volume flux, with `S` interpolated
to the faces.
"""
freshwater_transport(ocean, section::TransportSection; reference_salinity = 34.8) =
    section_transport(ocean, section, ocean.model.tracers.S, 1, -1 / reference_salinity)
