#####
##### hydrostatic_pressure_from_surface: balanced pressure from a surface-pressure anchor
#####

using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: CenterField, interior, set!
using Oceananigans.Grids: znode
using Oceananigans.BoundaryConditions: fill_halo_regions!

# Host interior arrays of `temperature` and the optional moisture fields, with omitted moisture
# zero-filled (the dry limit). Shared by `hydrostatic_pressure_from_surface` and `density_from_pressure`.
function interior_temperature_and_moisture(temperature, qᵛ, qᶜ, qⁱ)
    Tᵃ = Array(interior(temperature))
    no_moisture = zero(Tᵃ)
    qᵛᵃ = isnothing(qᵛ) ? no_moisture : Array(interior(qᵛ))
    qᶜᵃ = isnothing(qᶜ) ? no_moisture : Array(interior(qᶜ))
    qⁱᵃ = isnothing(qⁱ) ? no_moisture : Array(interior(qⁱ))
    return Tᵃ, qᵛᵃ, qᶜᵃ, qⁱᵃ
end

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
are optional `Field`s; omitting them gives the dry result (`Rᵐ = Rᵈ`). One-time host computation
(not a kernel) — it loops over columns on the CPU.

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
function hydrostatic_pressure_from_surface(temperature, surface_pressure, orography;
                                           qᵛ = nothing, qᶜ = nothing, qⁱ = nothing,
                                           dry_gas_constant,
                                           vapor_gas_constant,
                                           gravitational_acceleration)
    grid = temperature.grid
    Nx, Ny, Nz = size(grid)
    cpu_grid = on_architecture(CPU(), grid)

    Tᵃ, qᵛᵃ, qᶜᵃ, qⁱᵃ = interior_temperature_and_moisture(temperature, qᵛ, qᶜ, qⁱ)

    Rᵈ = dry_gas_constant
    Rᵛ = vapor_gas_constant
    g  = gravitational_acceleration

    moist_RT(i, j, k) = ((1 - qᵛᵃ[i, j, k] - qᶜᵃ[i, j, k] - qⁱᵃ[i, j, k]) * Rᵈ +
                         qᵛᵃ[i, j, k] * Rᵛ) * Tᵃ[i, j, k]

    p = similar(Tᵃ)
    for j in 1:Ny, i in 1:Nx
        z⁻   = znode(i, j, 1, cpu_grid, Center(), Center(), Face())    # grid bottom face
        RᵐT⁻ = moist_RT(i, j, 1)
        p⁻   = surface_pressure[i, j] * exp(-g * (z⁻ - orography[i, j]) / RᵐT⁻)
        for k in 1:Nz
            zᵏ   = znode(i, j, k, cpu_grid, Center(), Center(), Center())
            RᵐTᵏ = moist_RT(i, j, k)
            p[i, j, k] = p⁻ * exp(-g * (zᵏ - z⁻) / ((RᵐT⁻ + RᵐTᵏ) / 2))
            z⁻ = zᵏ; RᵐT⁻ = RᵐTᵏ; p⁻ = p[i, j, k]
        end
    end

    pressure = CenterField(grid)
    set!(pressure, p)
    fill_halo_regions!(pressure)
    return pressure
end

"""
$(TYPEDSIGNATURES)

Moist-air density `ρ = p / (Rᵐ T)` on the grid of `temperature`, with the moist mixture gas constant
`Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ`, `qᵈ = 1 − qᵛ − qᶜ − qⁱ` — the same EOS as [`hydrostatic_pressure_from_surface`](@ref),
so a density built from that pressure is mutually consistent. `temperature` and `pressure` are `Field`s
on the same grid; the moisture `qᵛ`, `qᶜ`, `qⁱ` are optional `Field`s (omitting them gives the dry
result `ρ = p / (Rᵈ T)`). Returns a `CenterField`. Useful to initialize a compressible model's density
from an analysis temperature + (hydrostatic) pressure, e.g. `set!(model; ρ = density_from_pressure(T, p; …))`.
"""
function density_from_pressure(temperature, pressure;
                               qᵛ = nothing, qᶜ = nothing, qⁱ = nothing,
                               dry_gas_constant,
                               vapor_gas_constant)
    grid = temperature.grid

    Tᵃ, qᵛᵃ, qᶜᵃ, qⁱᵃ = interior_temperature_and_moisture(temperature, qᵛ, qᶜ, qⁱ)
    pᵃ = Array(interior(pressure))

    Rᵈ = dry_gas_constant
    Rᵛ = vapor_gas_constant
    Rᵐ = @. (1 - qᵛᵃ - qᶜᵃ - qⁱᵃ) * Rᵈ + qᵛᵃ * Rᵛ
    ρ  = @. pᵃ / (Rᵐ * Tᵃ)

    density = CenterField(grid)
    set!(density, ρ)
    fill_halo_regions!(density)
    return density
end
