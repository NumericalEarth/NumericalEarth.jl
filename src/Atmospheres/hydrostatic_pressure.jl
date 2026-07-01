#####
##### hydrostatic_pressure_from_surface: balanced pressure from a surface-pressure anchor
#####

using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Fields: CenterField, interior
using Oceananigans.Grids: znode
using Oceananigans.BoundaryConditions: fill_halo_regions!

"""
$(TYPEDSIGNATURES)

Hydrostatically-balanced pressure on the grid of `temperature`, integrating
`d(ln p)/dz = −g/(Rᵐ T)` upward from `surface_pressure` (the surface pressure at the `orography`
height) with the moist mixture gas constant `Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ`, `qᵈ = 1 − qᵛ − qᶜ − qⁱ`.

Each column is anchored at the grid's bottom face — the terrain surface on a terrain-following
grid, the domain bottom on a regular height grid — and `surface_pressure` is reduced
hydrostatically from `orography` to that face. Anchoring on the surface pressure (rather than
interpolating pressure to the node heights) avoids the spurious near-surface density a
sub-surface-clamped pressure interpolation produces over high terrain.

`surface_pressure` and `orography` are `(Nx, Ny)` arrays. The moisture fields `qᵛ`, `qᶜ`, `qⁱ`
are optional `Field`s; omitting them gives the dry result (`Rᵐ = Rᵈ`). The calculation launches
one kernel thread per horizontal column, integrating upward through `k`.

```jldoctest
using Oceananigans
using NumericalEarth

grid = RectilinearGrid(size = (1, 1, 4), x = (0, 1), y = (0, 1), z = (0, 1000),
                       topology = (Periodic, Periodic, Bounded))
T = CenterField(grid)
set!(T, 250)

p = hydrostatic_pressure_from_surface(T, fill(1.0e5, 1, 1), fill(0.0, 1, 1);
                                      dry_gas_constant = 287.0,
                                      vapor_gas_constant = 461.0,
                                      gravitational_acceleration = 9.81)
round(Int, interior(p)[1, 1, 1] / 100)  # surface pressure (hPa) at the lowest level

# output
983
```
"""
@kernel function _hydrostatic_pressure_from_surface!(p, grid, temperature, qᵛ, qᶜ, qⁱ,
                                                     surface_pressure, orography, Rᵈ, Rᵛ, g)
    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)

    T⁻ = @inbounds temperature[i, j, 1]
    qᵛ⁻ = isnothing(qᵛ) ? zero(T⁻) : @inbounds qᵛ[i, j, 1]
    qᶜ⁻ = isnothing(qᶜ) ? zero(T⁻) : @inbounds qᶜ[i, j, 1]
    qⁱ⁻ = isnothing(qⁱ) ? zero(T⁻) : @inbounds qⁱ[i, j, 1]

    z⁻   = znode(i, j, 1, grid, Center(), Center(), Face())
    RᵐT⁻ = ((1 - qᵛ⁻ - qᶜ⁻ - qⁱ⁻) * Rᵈ + qᵛ⁻ * Rᵛ) * T⁻
    p⁻   = @inbounds surface_pressure[i, j] * exp(-g * (z⁻ - orography[i, j]) / RᵐT⁻)

    for k in 1:Nz
        zᵏ = znode(i, j, k, grid, Center(), Center(), Center())
        Tᵏ = @inbounds temperature[i, j, k]
        qᵛᵏ = isnothing(qᵛ) ? zero(Tᵏ) : @inbounds qᵛ[i, j, k]
        qᶜᵏ = isnothing(qᶜ) ? zero(Tᵏ) : @inbounds qᶜ[i, j, k]
        qⁱᵏ = isnothing(qⁱ) ? zero(Tᵏ) : @inbounds qⁱ[i, j, k]
        RᵐTᵏ = ((1 - qᵛᵏ - qᶜᵏ - qⁱᵏ) * Rᵈ + qᵛᵏ * Rᵛ) * Tᵏ

        @inbounds p[i, j, k] = p⁻ * exp(-g * (zᵏ - z⁻) / ((RᵐT⁻ + RᵐTᵏ) / 2))

        z⁻ = zᵏ
        RᵐT⁻ = RᵐTᵏ
        @inbounds p⁻ = p[i, j, k]
    end
end

function hydrostatic_pressure_from_surface(temperature, surface_pressure, orography;
                                           qᵛ = nothing, qᶜ = nothing, qⁱ = nothing,
                                           dry_gas_constant,
                                           vapor_gas_constant,
                                           gravitational_acceleration)
    grid = temperature.grid
    arch = architecture(grid)

    surface_pressure = on_architecture(arch, surface_pressure)
    orography = on_architecture(arch, orography)

    Rᵈ = dry_gas_constant
    Rᵛ = vapor_gas_constant
    g  = gravitational_acceleration

    pressure = CenterField(grid)

    launch!(arch, grid, :xy, _hydrostatic_pressure_from_surface!,
            pressure, grid, temperature, qᵛ, qᶜ, qⁱ, surface_pressure, orography, Rᵈ, Rᵛ, g)

    fill_halo_regions!(pressure)
    return pressure
end
