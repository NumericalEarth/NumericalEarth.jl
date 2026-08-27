# Isobaric and surface diagnostics for a Breeze child on a terrain-following grid.
#
# The model carries no pressure levels: pressure is a 3-D diagnostic that varies column by column,
# so a fixed-pressure surface cuts across model levels. For each column we find the model interval
# that brackets the target pressure and interpolate linearly in log p (pressure is exponential in
# height, so log p is the variable that interpolates without bias over a model layer). Columns whose
# surface pressure is below the target — a target under the terrain, e.g. 1000 hPa over the Rockies —
# are filled with NaN rather than extrapolated, so the plots show a hole instead of invented air.
#
# The fields are filled by a callback and saved as plain `Field`s, so the writer serializes the grid
# with them and they read back as `FieldTimeSeries`. Callbacks run before writers within a step, so
# sharing one schedule keeps the saved slice current.

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: compute!, location
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Utils: launch!
using Breeze.AtmosphereModels: dynamics_pressure
using KernelAbstractions: @kernel, @index

# Mean Earth radius (m). Geopotential height is Z = Φ/g₀ with Φ = ∫₀^z g dz′ and g = g₀ (a/(a+z))²,
# which integrates to Z = a z / (a + z) — the standard conversion from geometric to geopotential
# height (≈ −19 m at 10 km, −34 m at 18 km).
const EARTH_RADIUS = 6371000.0

@kernel function _slice_at_pressure!(slice, field, pressure, p★, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(slice)
    value = zero(FT)
    found = false

    @inbounds for k in 1:(Nz - 1)
        p_below = pressure[i, j, k]        # larger pressure, lower model level
        p_above = pressure[i, j, k + 1]    # smaller pressure, upper model level

        # Bracketing test written so a column is claimed exactly once (the first interval that
        # straddles the target), leaving `found` monotone.
        brackets = (p_below >= p★) & (p★ > p_above)
        take = brackets & !found

        # Guard the denominator: an inverted or degenerate pair would divide by zero, and `ifelse`
        # evaluates both branches, so a NaN here would be selected only if `take` were true.
        Δlog = log(p_below) - log(p_above)
        w = ifelse(abs(Δlog) > 0, (log(p_below) - log(p★)) / Δlog, zero(FT))
        interpolated = field[i, j, k] * (1 - w) + field[i, j, k + 1] * w

        value = ifelse(take, interpolated, value)
        found = found | brackets
    end

    @inbounds slice[i, j, 1] = ifelse(found, value, convert(FT, NaN))
end

# Geopotential height of every cell center. Static — the terrain-following coordinate is fixed once
# the terrain is materialized — so this is filled once at construction and then sampled like any
# other 3-D field. `znode` on a `TerrainFollowingGrid` returns the physical altitude z = r + h·b(r).
@kernel function _set_geopotential_height!(Z, grid)
    i, j, k = @index(Global, NTuple)
    FT = eltype(Z)
    a = convert(FT, EARTH_RADIUS)
    z = convert(FT, znode(i, j, k, grid, Center(), Center(), Center()))
    @inbounds Z[i, j, k] = a * z / (a + z)
end

"""
    geopotential_height_field(grid)

A `Field` of geopotential height (m) at cell centers, filled from the grid's terrain-following
vertical coordinate.
"""
function geopotential_height_field(grid)
    Z = CenterField(grid)
    launch!(architecture(grid), grid, :xyz, _set_geopotential_height!, Z, grid)
    return Z
end

# Surface pressure by log-linear extrapolation from the lowest two cell centers down to the terrain.
# A layer of constant scale height has log p linear in z, so extrapolating the slope the model's own
# two lowest levels define is the hydrostatic result with the *local* scale height — no assumed
# virtual temperature, and consistent with the log-p interpolation used for the isobaric slices.
@kernel function _surface_pressure!(pₛ, pressure, grid)
    i, j = @index(Global, NTuple)

    @inbounds begin
        p₁ = pressure[i, j, 1]
        p₂ = pressure[i, j, 2]

        z₁ = znode(i, j, 1, grid, Center(), Center(), Center())
        z₂ = znode(i, j, 2, grid, Center(), Center(), Center())
        h  = znode(i, j, 1, grid, Center(), Center(), Face())   # bottom face = terrain elevation

        pₛ[i, j, 1] = exp(log(p₁) + (z₁ - h) / (z₂ - z₁) * (log(p₁) - log(p₂)))
    end
end

"""
    surface_pressure_field(child)

A 2-D `Field` to hold surface pressure (Pa); fill it with [`fill_surface_pressure!`](@ref).
"""
surface_pressure_field(child) = Field{Center, Center, Nothing}(child.grid)

function fill_surface_pressure!(pₛ, child)
    grid = child.grid
    pressure = dynamics_pressure(child.dynamics)
    launch!(architecture(grid), grid, :xy, _surface_pressure!, pₛ, pressure, grid)
    return nothing
end

"""
    PressureLevelDiagnostics(child, pressures_hPa, variables)

Preallocate one 2-D field per (variable, pressure level) and return the bundle a callback fills.

`variables` is a `NamedTuple` mapping an output prefix to a center-located 3-D `Field` or
`AbstractOperation`. Operations are wrapped as computed fields that all share a *single* 3-D scratch
array: `fill_pressure_levels!` computes each one immediately before sampling it, so N derived
variables cost one 3-D buffer instead of N (at 3 km each is 0.7 GB, which is the difference between
fitting on the GPU and not).
"""
struct PressureLevelDiagnostics{P, V, S, W}
    pressures :: P     # target pressures (Pa)
    variables :: V     # NamedTuple of 3-D fields to sample (operations already wrapped)
    slices :: S        # NamedTuple of 2-D output fields, keyed <var>_<hPa>
    scratch :: W       # shared backing store for the wrapped operations
end

function PressureLevelDiagnostics(child, pressures_hPa, variables)
    grid = child.grid
    pressures = [convert(eltype(grid), p * 100) for p in pressures_hPa]   # hPa → Pa

    scratch = CenterField(grid)

    # `compute = false`: the model state need not be valid at construction. Passing `data` also sets
    # `recompute_safely = true`, so `compute!` never skips a recomputation on a stale status flag —
    # essential when the buffer underneath is overwritten by the previous variable.
    sampled = NamedTuple(name => (v isa Field ? v : Field(v; data = scratch.data, compute = false))
                         for (name, v) in Base.pairs(variables))

    # Every sampled variable must share the pressure field's location, both so the shared scratch
    # buffer has the right size and so the interpolation samples the column it thinks it does.
    for (name, field) in Base.pairs(sampled)
        location(field) === (Center, Center, Center) ||
            error("Isobaric variable $name is at $(location(field)); it must be (Center, Center, Center).")
    end

    slice_pairs = Pair{Symbol, Any}[]
    for name in keys(sampled), hPa in pressures_hPa
        push!(slice_pairs, Symbol(name, "_", Int(hPa)) => Field{Center, Center, Nothing}(grid))
    end
    slices = NamedTuple(slice_pairs)

    # Call the struct's own four-field constructor explicitly: this builder shares its arity, so an
    # unqualified call would recurse into this method and silently shift the arguments.
    return PressureLevelDiagnostics{typeof(pressures), typeof(sampled), typeof(slices), typeof(scratch)}(
        pressures, sampled, slices, scratch)
end

"""
    fill_pressure_levels!(diagnostics, child)

Fill every (variable, level) slice from the model's current state.
"""
function fill_pressure_levels!(diagnostics, child)
    grid = child.grid
    arch = architecture(grid)
    Nz = size(grid, 3)
    pressure = dynamics_pressure(child.dynamics)

    for (name, field) in Base.pairs(diagnostics.variables)
        # A no-op for plain `Field`s; recomputes the shared scratch buffer for wrapped operations.
        compute!(field)

        for p★ in diagnostics.pressures
            hPa = Int(round(p★ / 100))
            slice = diagnostics.slices[Symbol(name, "_", hPa)]
            launch!(arch, grid, :xy, _slice_at_pressure!, slice, field, pressure, p★, Nz)
        end
    end

    return nothing
end
