include("runtests_setup.jl")

using ConservativeRegridding: ConservativeRegridding # loads NumericalEarthConservativeRegriddingExt
using NumericalEarth.YinYangRegridding
using NumericalEarth.YinYangRegridding: frame_unit_vectors
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnode, φnode
using Oceananigans.MultiRegion: owned_weights
using Oceananigans.Operators: rotation_angle
using Oceananigans.Utils: getregion
using Random

#####
##### Tolerances, declared before the measurements they gate.
#####
##### Those marked (design) are the gates of YY_SURFACE_EXCHANGE_DESIGN.md (G0, T2, T3, T5, T7) for the 1° grid;
##### the 2° values scale them by 4, the ratio of a second-order error. The others are a-priori bounds from the
##### order of the operators. Smoke runs of the operators preceded this file; no tolerance was set or changed
##### to fit a measurement.
#####

# Radian measure of the cell size of each grid
const one_degree = deg2rad(1)
cell_size(Nφ) = deg2rad(90 / Nφ)

# T2 (design): a constant regrids to the same constant. The operators are convex combinations with tens of
# weights each, so the deviation is a few units in the last place.
const constant_tolerance = 1e-13

# T3 (design): the conservation residual of the uncorrected row-normalized averages for a smooth field, relative
# to Σ A |F|. It comes from the cut cells (3 % of cells at 1°) and from great-circle cell edges.
smooth_residual_tolerance(Nφ) = 1e-5 * (90 / Nφ)^2

# T3, rough field: a field that is random at the grid scale has no cancellation across a cut cell, so allow
# ten times the smooth-field gate.
rough_residual_tolerance(Nφ) = 10 * smooth_residual_tolerance(Nφ)

# Momentum, uncorrected: besides the scalar residual, the average of a tangent vector over a cell of angular
# size Δ has a radial part of relative size Δ² / 8 that the projection removes, with one sign everywhere
# (4e-5 at 1°, 1.5e-4 at 2°). Gate the force and torque residuals at the rough-field level.
momentum_residual_tolerance(Nφ) = rough_residual_tolerance(Nφ)

# T3 and the equal-and-opposite budgets, corrected: the integrals agree to rounding. Each integral sums
# 1e4 to 1e5 terms, so rounding is a few 1e-14 relative to Σ A |F|.
const closure_tolerance = 1e-13

# T7 (design): bilinear interpolation of the 20 K wavenumber-2 field from the 1° data grid. The bilinear bound
# Δ² (|f_λλ| + |f_φφ|) / 8 with second derivatives up to 80 + 50 K rad⁻² is 5e-3 K; allow twice that.
const interpolation_tolerance = 1e-2 # K

# Convexity: bilinear and area-weighted weights are non-negative and sum to one, so bounds are kept to rounding.
const bound_tolerance = 1e-13

# First-order bound for the piecewise-constant area-weighted average: every source value is taken at most half
# a source-cell diagonal away, and the destination value is a cell average, so the error is at most
# |∇f| (Δ_source + Δ_destination) / √2.
first_order_bound(gradient, Δ_source, Δ_destination) = gradient * (Δ_source + Δ_destination) / √2

# T5 (design): a uniform flow straight over a geographic pole keeps its direction through the round trip to
# within 0.1° at 1°.
pole_direction_tolerance(Nφ) = 0.1 * (90 / Nφ)^2 # degrees

# T5: bilinear vector regridding onto the C-grid, then the Yin-Yang pair fill. Bilinear error Δ_ll² U / 4 (two
# second derivatives of size U), face averaging Δ_yy² U / 8, times 2 for the cubic halo interpolation of both.
staggered_tolerance(U, Nφ) = 2 * U * (one_degree^2 / 4 + cell_size(Nφ)^2 / 8)

# T5: the frame unit vectors against Oceananigans' closed-form panel rotation angle: rounding.
const basis_tolerance = 1e-12

# Convergence between 2° and 1° (both grids refined): second order gives 4 and first order 2; allow 25 %.
const second_order_ratio = 3
const first_order_ratio = 1.5

#####
##### Analytic fields
#####

design_temperature(λ, φ) = 280 + 20 * cosd(φ)^2 * sind(2λ) + 10 * sind(φ)                 # K, the design's field

# W m⁻², signed. The design's pattern alone integrates to zero residual by the symmetry of the Yin-Yang grid, so
# two terms without that symmetry are added.
heat_flux(λ, φ) = 100 * cosd(φ)^2 * sind(2λ) + 50 * sind(φ) + 20 +
                  150 * cosd(φ) * cosd(λ - 40) + 80 * sind(3φ) * sind(2λ + 25)

water_flux(λ, φ) = 5e-5 * (1 + cosd(φ)^2 * sind(2λ + 60)) +
                   2e-4 * exp(-(φ / 10)^2) * (1 + cosd(3λ) / 2)                           # kg m⁻² s⁻¹, positive
ice_concentration(λ, φ) = (1 + tanh((abs(φ) - 70) / 2)) / 2                               # in [0, 1]

cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
dot3(a, b) = sum(a .* b)
unit_position(λ, φ) = (cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))
geographic_east(λ, φ) = (-sind(λ), cosd(λ), 0.0)
geographic_north(λ, φ) = (-sind(φ) * cosd(λ), -sind(φ) * sind(λ), cosd(φ))

const wind_speed = 10.0                       # m s⁻¹ at the equator of the rotation
const tilted_axis = (sind(45), 0.0, cosd(45)) # crosses the seam and the poles
const polar_axis = (1.0, 0.0, 0.0)            # uniform flow straight over both geographic poles

# Solid-body rotation about `axis` as a Cartesian vector, and as geographic components
solid_body(axis, λ, φ) = wind_speed .* cross3(axis, unit_position(λ, φ))
geographic_components(U, λ, φ) = (dot3(U, geographic_east(λ, φ)), dot3(U, geographic_north(λ, φ)))

# A surface stress with net force and torque: drag on the tilted rotation plus a zonal jet
function stress(λ, φ)
    U = solid_body(tilted_axis, λ, φ) .+ 15 * cosd(φ) * sind(2φ)^2 .* geographic_east(λ, φ)
    return -1.2 * 1.2e-3 * sqrt(dot3(U, U)) .* U
end

#####
##### Helpers
#####

c, f = Center(), Face()

# Numbers for the log
scientific(x) = string(round(x; sigdigits = 4))
fixed(x, digits) = string(round(x; digits))

node(i, j, grid, ℓx=c, ℓy=c) = (λnode(i, j, 1, grid, ℓx, ℓy, c), φnode(i, j, 1, grid, ℓx, ℓy, c))

# The components in a panel's basis of the Cartesian vector function `U(λ, φ)` at a node, with Oceananigans'
# rotation angle: independent of the unit vectors that the regridder stores
function panel_components(U, i, j, panel, ℓx, ℓy)
    λ, φ = node(i, j, panel, ℓx, ℓy)
    u, v = geographic_components(U(λ, φ), λ, φ)
    θ = rotation_angle(i, j, panel, ℓx, ℓy)
    return u * cos(θ) - v * sin(θ), u * sin(θ) + v * cos(θ)
end

region_array(field, r) = Array(interior(getregion(field, r), :, :, 1))
latlon_array(field) = Array(interior(field, :, :, 1))

function set_region!(field, r, values)
    region = getregion(field, r)
    copyto!(interior(region, :, :, 1), on_architecture(architecture(region), values))
    return nothing
end

# Set a Yin-Yang field at (ℓx, ℓy) to `func(i, j, panel)` on both panels
function set_panels!(field, grid, ℓx, ℓy, func)
    Nx, Ny, _ = size(grid)
    for r in 1:2
        panel = getregion(grid, r)
        set_region!(field, r, [func(i, j, panel) for i in 1:Nx, j in 1:Ny])
    end
    return nothing
end

set_panels!(field, grid, func) = set_panels!(field, grid, c, c, (i, j, panel) -> func(node(i, j, panel)...))

# Maximum over the cells of both panels of `error(i, j, panel, r)`
panel_maximum(error, grid) = maximum(maximum(error(i, j, getregion(grid, r), r) for i in 1:size(grid, 1), j in 1:size(grid, 2)) for r in 1:2)

direction_error(a, b) = rad2deg(atan(abs(a[1] * b[2] - a[2] * b[1]), a[1] * b[1] + a[2] * b[2]))

# Σ A |F| on the latitude-longitude grid, the scale of every integral residual
flux_scale(regridder, field) = sum(Array(regridder.latitude_longitude_geometry[1].area)[:, :, 1] .* abs.(latlon_array(field)))

center_field(grid) = Field{Center, Center, Nothing}(grid)

set_values!(field, values::Function) = set!(field, values)
set_values!(field, values::AbstractMatrix) = copyto!(interior(field, :, :, 1), on_architecture(architecture(field), values))

# Cells whose owned fraction is neither zero nor one, to rounding
cut_cells(owned) = @. 1e-9 < owned < 1 - 1e-9

# Cells within `n` cells of a cut cell of the same panel
function seam_band(owned, n)
    Nx, Ny = size(owned)
    cut = cut_cells(owned)
    band = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        band[i, j] = any(cut[max(1, i - n):min(Nx, i + n), max(1, j - n):min(Ny, j + n)])
    end
    return band
end

# The two operators onto the Yin-Yang grid
operators = (("interpolate_to_yin_yang!", interpolate_to_yin_yang!), ("transfer_flux!", transfer_flux!))

arch = first(test_architectures)
resolutions = ((45, "2°"), (90, "1°"))

exchange_grid(Nλ = 360) = LatitudeLongitudeGrid(arch; size = (Nλ, Nλ ÷ 2, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1))
yin_yang_grid(Nφ) = YinYangGrid(arch; size = (3Nφ, Nφ, 1), z = (0, 1))

latlon = exchange_grid()
cpu_latlon = on_architecture(CPU(), latlon)

@testset "YinYangRegridder construction" begin
    grid = yin_yang_grid(45)
    regridder = YinYangRegridder(grid, latlon)
    @test occursin("state interpolation and flux transfer", summary(regridder))

    interpolation_only = YinYangRegridder(grid, latlon; flux_transfer = false)
    @test_throws ArgumentError transfer_flux!(center_field(grid), interpolation_only, center_field(latlon))
    @test interpolate_to_yin_yang!(center_field(grid), interpolation_only, center_field(latlon)) isa Field

    regional = LatitudeLongitudeGrid(arch; size = (10, 10, 1), longitude = (0, 10), latitude = (0, 10), z = (0, 1))
    @test_throws ArgumentError YinYangRegridder(grid, regional)
    @test_throws ArgumentError transfer_flux!(Field{Face, Center, Nothing}(latlon), regridder, center_field(grid))
    @test_throws ArgumentError transfer_flux!((center_field(latlon), center_field(latlon)), regridder,
                                             (center_field(grid), center_field(grid)); correction = GlobalRescaling())

    # Interpolation takes no correction, and state does not go through the flux transfer by accident
    @test_throws MethodError interpolate_to_yin_yang!(center_field(grid), regridder, center_field(latlon); correction = UniformRedistribution())
    @test_throws MethodError interpolate_to_yin_yang!(center_field(latlon), regridder, center_field(grid))
end

@testset "Regression: fields on a grid the regridder was not built for are refused" begin
    grid = yin_yang_grid(45)
    regridder = YinYangRegridder(grid, latlon)

    # An equal grid built separately is accepted
    @test transfer_flux!(center_field(grid), regridder, center_field(exchange_grid())) isa Field
    @test transfer_flux!(center_field(latlon), regridder, center_field(yin_yang_grid(45))) isa Field

    # Another size
    @test_throws ArgumentError transfer_flux!(center_field(exchange_grid(180)), regridder, center_field(grid))
    @test_throws ArgumentError transfer_flux!(center_field(yin_yang_grid(30)), regridder, center_field(latlon))

    # The same size with shifted longitudes, or with another radius: the weights would be silently wrong
    shifted = LatitudeLongitudeGrid(arch; size = (360, 180, 1), longitude = (-180, 180), latitude = (-90, 90), z = (0, 1))
    small = LatitudeLongitudeGrid(arch; size = (360, 180, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = 3e6)
    small_panels = YinYangGrid(arch; size = (135, 45, 1), z = (0, 1), radius = 3e6)
    wide_overlap = YinYangGrid(arch; size = (135, 45, 1), z = (0, 1), overlap = 4)
    @test_throws ArgumentError transfer_flux!(center_field(grid), regridder, center_field(shifted))
    @test_throws ArgumentError interpolate_to_yin_yang!(center_field(grid), regridder, center_field(shifted))
    @test_throws ArgumentError transfer_flux!(center_field(grid), regridder, center_field(small))
    @test_throws ArgumentError transfer_flux!(center_field(small_panels), regridder, center_field(latlon))
    @test_throws ArgumentError transfer_flux!(center_field(wide_overlap), regridder, center_field(latlon))
    @test_throws ArgumentError YinYangRegridder(grid, small)
end

@testset "Regression: GlobalRescaling is refused where it is not defined" begin
    grid = yin_yang_grid(45)
    cpu_grid = on_architecture(CPU(), grid)
    regridder = YinYangRegridder(grid, latlon)
    panels, exchange = center_field(grid), center_field(latlon)

    # A signed flux with zero net integral, like net radiation in balance
    zero_net(λ, φ) = 100 * sind(φ) + 60 * cosd(φ) * cosd(λ - 40)
    source = center_field(latlon)
    set!(source, zero_net)
    scale = flux_scale(regridder, source)
    @test abs(surface_integral(regridder, source)) / scale ≤ closure_tolerance

    @test_throws ArgumentError transfer_flux!(panels, regridder, source; correction = GlobalRescaling())
    set_panels!(panels, cpu_grid, zero_net)
    @test_throws ArgumentError transfer_flux!(exchange, regridder, panels; correction = GlobalRescaling())

    # The additive redistribution closes it, without erasing the field
    transfer_flux!(panels, regridder, source; correction = UniformRedistribution())
    residual = (surface_integral(regridder, panels) - surface_integral(regridder, source)) / scale
    amplitude = maximum(r -> maximum(abs, region_array(panels, r)), 1:2)
    @info "Zero-net signed flux, UniformRedistribution: residual = $(scientific(residual)), max |F| after transfer = $(scientific(amplitude)) of 116.6"
    @test abs(residual) ≤ closure_tolerance
    @test amplitude > 100

    # A nonzero target cannot be closed on a field that integrates to zero
    set!(source, 1)
    set!(exchange, 0)
    @test_throws ArgumentError NumericalEarth.YinYangRegridding.correct_integral!(exchange, regridder, source, GlobalRescaling())

    # An identically zero flux stays zero
    set!(source, 0)
    transfer_flux!(panels, regridder, source; correction = GlobalRescaling())
    @test all(r -> all(==(0), region_array(panels, r)), 1:2)
end

@testset "Regression: GlobalRescaling checks the source before it writes the destination" begin
    grid = yin_yang_grid(45)
    regridder = YinYangRegridder(grid, latlon)
    source, panels = center_field(latlon), center_field(grid)
    Nλ, Nφ = size(latlon)

    # The destination as a failed transfer must leave it, halos included
    sentinel = 7
    panel_data() = map(r -> Array(parent(getregion(panels, r))), 1:2)
    function reset_panels!()
        for r in 1:2
            fill!(parent(getregion(panels, r)), sentinel)
        end
        return panel_data()
    end

    # A signed source that the fine-to-coarse averages hide. From the 1° exchange grid to the 2° panels, take a
    # source cell next to the north pole, where cells are smallest, and its largest weight α in any panel cell.
    # With the source 1 everywhere but -ε there, ε = (1 / α - 1) / 2, a panel cell with weight w of that cell
    # holds 1 - (1 + ε) w ≥ (1 - α) / 2 > 0: the transferred field is positive everywhere, and so is the integral.
    i★, j★ = 17, Nφ
    tables = map(table -> (i = Array(table.source_i), j = Array(table.source_j), weight = Array(table.weight)), regridder.to_yin_yang)
    α = maximum(table -> maximum(table.weight[(table.i .== i★) .& (table.j .== j★)]; init = 0.0), tables)
    @test 0 < α < 1
    ε = (1 / α - 1) / 2

    values = ones(Nλ, Nφ)
    values[i★, j★] = -ε
    set_values!(source, values)
    @test surface_integral(regridder, source) > 0

    # Uncorrected, nothing in the transferred field shows the sign of the source
    transfer_flux!(panels, regridder, source)
    lowest = minimum(r -> minimum(region_array(panels, r)), 1:2)
    @info "Signed fine-to-coarse source: α = $(scientific(α)), ε = $(scientific(ε)), minimum of the transferred field = $(scientific(lowest)), bound (1 - α) / 2 = $(scientific((1 - α) / 2))"
    @test lowest ≥ (1 - α) / 2 - bound_tolerance
    @test surface_integral(regridder, panels) > 0

    before = reset_panels!()
    @test_throws ArgumentError transfer_flux!(panels, regridder, source; correction = GlobalRescaling())
    @test panel_data() == before

    # NaN and infinite sources fail the same way, instead of scaling by a NaN factor
    for bad in (NaN, Inf)
        values = ones(Nλ, Nφ)
        values[i★, j★] = bad
        set_values!(source, values)
        before = reset_panels!()
        @test_throws ArgumentError transfer_flux!(panels, regridder, source; correction = GlobalRescaling())
        @test panel_data() == before
    end

    # From the panels: a negative, NaN or infinite cell of either panel is refused before the exchange field is written
    exchange = center_field(latlon)
    for bad in (-1e-3, NaN, Inf), r in 1:2
        set_panels!(panels, on_architecture(CPU(), grid), (λ, φ) -> 1.0)
        values = region_array(panels, r)
        values[40, 20] = bad
        set_region!(panels, r, values)
        fill!(parent(exchange), sentinel)
        @test_throws ArgumentError transfer_flux!(exchange, regridder, panels; correction = GlobalRescaling())
        @test all(==(sentinel), Array(parent(exchange)))
    end

    # A vector is refused before its destination is written, on the C-grid too
    uᵉ, vⁿ = center_field(latlon), center_field(latlon)
    set!(uᵉ, 1)
    applied = (Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid))
    for component in applied, r in 1:2
        fill!(parent(getregion(component, r)), sentinel)
    end
    @test_throws ArgumentError transfer_flux!(applied, regridder, (uᵉ, vⁿ); correction = GlobalRescaling())
    @test all(component -> all(r -> all(==(sentinel), Array(parent(getregion(component, r)))), 1:2), applied)

    # A non-negative source is still rescaled
    set!(source, water_flux)
    transfer_flux!(panels, regridder, source; correction = GlobalRescaling())
    @test abs(surface_integral(regridder, panels) - surface_integral(regridder, source)) / flux_scale(regridder, source) ≤ closure_tolerance
end

#####
##### Gating bug (4a): the pole node. The cause is upstream; the regridder does not depend on it.
#####

@testset "Regression: north polar halo of a latitude-longitude field (Oceananigans)" begin
    grid = LatitudeLongitudeGrid(arch; size = (8, 4, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1))
    field = CenterField(grid)
    set!(field, (λ, φ, z) -> 280 + 10 * sind(φ))
    fill_halo_regions!(field)

    north = @allowscalar field[1, 4, 1]
    south = @allowscalar field[1, 1, 1]

    # The halo across a pole holds 2 c̄ - c, with c̄ the zonal mean of the adjacent row, here equal to c
    @test @allowscalar(field[1, 0, 1]) ≈ south

    # Upstream bug: `fill_halo_regions!` adapts the boundary conditions, `Adapt.adapt_structure(to, ::PolarValue)`
    # drops `side`, and `update_pole_value!` then averages row 1 for the north pole as well, so the north halo
    # holds 2 c̄ˢ - c. `interpolate!` reads that halo for every node poleward of the last cell centers.
    # When this passes, Oceananigans is fixed and `bilinear_table` can defer to `interpolate!`.
    @test_broken @allowscalar(field[1, 5, 1]) ≈ north
end

for (Nφ, name) in resolutions
    grid = yin_yang_grid(Nφ)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, _ = size(grid)
    regridder = YinYangRegridder(grid, latlon)
    owned = map(r -> region_array(owned_weights(grid), r), 1:2)
    Δ = cell_size(Nφ)

    @testset "T2 constant preservation, both directions [$name]" begin
        constant = 287.15
        source = center_field(latlon)
        set!(source, constant)

        cases = (("interpolate_to_yin_yang!", panels -> interpolate_to_yin_yang!(panels, regridder, source)),
                 ("transfer_flux!", panels -> transfer_flux!(panels, regridder, source)),
                 ("transfer_flux! with GlobalRescaling", panels -> transfer_flux!(panels, regridder, source; correction = GlobalRescaling())),
                 ("transfer_flux! with UniformRedistribution", panels -> transfer_flux!(panels, regridder, source; correction = UniformRedistribution())))

        for (label, operator!) in cases
            panels = center_field(grid)
            operator!(panels)
            deviation = maximum(r -> maximum(abs, Array(parent(getregion(panels, r))) .- constant), 1:2) / constant
            @info "T2 [$(name)] latitude-longitude → Yin-Yang, $(label): max |c - c₀| / c₀ = $(scientific(deviation)) (halos included)"
            @test deviation ≤ constant_tolerance
        end

        panels = center_field(grid)
        set_panels!(panels, cpu_grid, (λ, φ) -> constant)

        for correction in (nothing, GlobalRescaling(), UniformRedistribution())
            destination = center_field(latlon)
            transfer_flux!(destination, regridder, panels; correction)
            deviation = maximum(abs, latlon_array(destination) .- constant) / constant
            @info "T2 [$(name)] Yin-Yang → latitude-longitude, transfer_flux!, $(nameof(typeof(correction))): max |c - c₀| / c₀ = $(scientific(deviation))"
            @test deviation ≤ constant_tolerance
        end
    end

    @testset "T3 global integral against the ownership quadrature [$name]" begin
        Random.seed!(1234)
        rough = rand(size(latlon, 1), size(latlon, 2))

        fields = (("heat (smooth, signed)", heat_flux, smooth_residual_tolerance(Nφ)),
                  ("water (smooth, positive)", water_flux, smooth_residual_tolerance(Nφ)),
                  ("random (rough, positive)", rough, rough_residual_tolerance(Nφ)))

        # The ownership quadrature itself: Σ w Az = 4π R², and the sampled water flux against the same field
        # sampled on the 1° grid (midpoint rule on both)
        sphere = 4π * latlon.radius^2
        ones_field = center_field(grid)
        set_panels!(ones_field, cpu_grid, (λ, φ) -> 1)
        area_error = surface_integral(regridder, ones_field) / sphere - 1
        sampled = center_field(grid)
        set_panels!(sampled, cpu_grid, water_flux)
        reference = center_field(latlon)
        set!(reference, water_flux)
        quadrature_error = surface_integral(regridder, sampled) / surface_integral(regridder, reference) - 1
        @info "T3 [$(name)] ownership quadrature: Σ w Az / 4πR² - 1 = $(scientific(area_error)); water flux sampled on the panels / sampled on the 1° grid - 1 = $(scientific(quadrature_error))"
        @test abs(area_error) ≤ 1e-12

        for (label, values, tolerance) in fields
            source = center_field(latlon)
            set_values!(source, values)
            scale = flux_scale(regridder, source)
            I_source = surface_integral(regridder, source)

            # Latitude-longitude → Yin-Yang: a surface flux applied to the atmosphere
            panels = center_field(grid)
            transfer_flux!(panels, regridder, source)
            residual = (surface_integral(regridder, panels) - I_source) / scale
            @info "T3 [$(name)] $(label), latitude-longitude → Yin-Yang: uncorrected residual = $(scientific(residual)) (gate $(scientific(tolerance)))"
            @test abs(residual) ≤ tolerance

            corrections = minimum(values isa Function ? latlon_array(source) : values) > 0 ?
                          (GlobalRescaling(), UniformRedistribution()) : (UniformRedistribution(),)

            for correction in corrections
                transfer_flux!(panels, regridder, source; correction)
                corrected = (surface_integral(regridder, panels) - I_source) / scale
                @info "T3 [$(name)] $(label), latitude-longitude → Yin-Yang: $(nameof(typeof(correction))) residual = $(scientific(corrected))"
                @test abs(corrected) ≤ closure_tolerance
            end

            # Yin-Yang → latitude-longitude: an atmosphere flux (precipitation, radiation) applied to the surface
            transfer_flux!(panels, regridder, source)
            I_panels = surface_integral(regridder, panels)
            destination = center_field(latlon)
            transfer_flux!(destination, regridder, panels)
            residual = (surface_integral(regridder, destination) - I_panels) / scale
            @info "T3 [$(name)] $(label), Yin-Yang → latitude-longitude: uncorrected residual = $(scientific(residual)) (gate $(scientific(tolerance)))"
            @test abs(residual) ≤ tolerance

            for correction in corrections
                transfer_flux!(destination, regridder, panels; correction)
                corrected = (surface_integral(regridder, destination) - I_panels) / scale
                @info "T3 [$(name)] $(label), Yin-Yang → latitude-longitude: $(nameof(typeof(correction))) residual = $(scientific(corrected))"
                @test abs(corrected) ≤ closure_tolerance
            end
        end
    end

    @testset "Equal and opposite integrals through a round trip [$name]" begin
        # Each exchange hands a flux from one grid to the other. With the correction, what one side loses the
        # other gains, in the ownership quadrature, and the round trip returns the integral it started from.
        for (label, flux, correction) in (("heat", heat_flux, UniformRedistribution()),
                                          ("water", water_flux, GlobalRescaling()))
            surface = center_field(latlon)
            set!(surface, flux)
            scale = flux_scale(regridder, surface)
            atmosphere = center_field(grid)
            returned = center_field(latlon)

            transfer_flux!(atmosphere, regridder, surface; correction)
            transfer_flux!(returned, regridder, atmosphere; correction)

            I_surface = surface_integral(regridder, surface)
            I_atmosphere = surface_integral(regridder, atmosphere)
            I_returned = surface_integral(regridder, returned)

            # The surface loses I_surface and the atmosphere gains I_atmosphere, then the reverse
            down = (I_atmosphere - I_surface) / scale
            up = (I_returned - I_atmosphere) / scale
            @info "Round trip [$(name)] $(label) with $(nameof(typeof(correction))): surface → atmosphere imbalance = $(scientific(down)), atmosphere → surface = $(scientific(up)), round trip = $(scientific((I_returned - I_surface) / scale))"
            @test abs(down) ≤ closure_tolerance
            @test abs(up) ≤ closure_tolerance
            @test abs(I_returned - I_surface) / scale ≤ 2 * closure_tolerance
        end

        # Momentum: the net force and the net torque of the surface stress
        τˣ, τʸ = center_field(latlon), center_field(latlon)
        set!(τˣ, (λ, φ) -> geographic_components(stress(λ, φ), λ, φ)[1])
        set!(τʸ, (λ, φ) -> geographic_components(stress(λ, φ), λ, φ)[2])
        magnitude = center_field(latlon)
        set!(magnitude, (λ, φ) -> sqrt(dot3(stress(λ, φ), stress(λ, φ))))
        scale = flux_scale(regridder, magnitude)

        atmosphere = (center_field(grid), center_field(grid))
        returned = (center_field(latlon), center_field(latlon))
        F_surface, L_surface = surface_force_and_torque(regridder, (τˣ, τʸ))

        for correction in (nothing, UniformRedistribution())
            transfer_flux!(atmosphere, regridder, (τˣ, τʸ); correction)
            F_atmosphere, L_atmosphere = surface_force_and_torque(regridder, atmosphere)
            transfer_flux!(returned, regridder, atmosphere; correction)
            F_returned, L_returned = surface_force_and_torque(regridder, returned)

            down = max(maximum(abs, F_atmosphere - F_surface), maximum(abs, L_atmosphere - L_surface)) / scale
            up = max(maximum(abs, F_returned - F_atmosphere), maximum(abs, L_returned - L_atmosphere)) / scale
            round_trip = max(maximum(abs, F_returned - F_surface), maximum(abs, L_returned - L_surface)) / scale
            @info "Round trip [$(name)] momentum with $(nameof(typeof(correction))): surface → atmosphere force/torque imbalance = $(scientific(down)), atmosphere → surface = $(scientific(up)), round trip = $(scientific(round_trip)) (net torque / Σ A|τ| = $(scientific(maximum(abs, L_surface) / scale)))"

            if correction isa UniformRedistribution
                @test down ≤ closure_tolerance
                @test up ≤ closure_tolerance
                @test round_trip ≤ 2 * closure_tolerance
            else
                @test down ≤ momentum_residual_tolerance(Nφ)
                @test up ≤ momentum_residual_tolerance(Nφ)
            end
        end

        # The stress that the atmosphere applies is on the C-grid. Its force and torque are measured at the faces,
        # after the conversion from the cell centers, with the lumped owned areas there; the correction closes them
        # in that representation. Uncorrected values are reported without a gate.
        applied = (Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid))

        for correction in (nothing, UniformRedistribution())
            transfer_flux!(applied, regridder, (τˣ, τʸ); correction)
            F_applied, L_applied = surface_force_and_torque(regridder, applied)
            transfer_flux!(returned, regridder, applied; correction)
            F_returned, L_returned = surface_force_and_torque(regridder, returned)

            down = max(maximum(abs, F_applied - F_surface), maximum(abs, L_applied - L_surface)) / scale
            up = max(maximum(abs, F_returned - F_applied), maximum(abs, L_returned - L_applied)) / scale
            @info "Round trip [$(name)] momentum on the C-grid faces with $(nameof(typeof(correction))): surface → atmosphere force/torque imbalance at the faces = $(scientific(down)), atmosphere → surface = $(scientific(up))"

            if correction isa UniformRedistribution
                @test down ≤ closure_tolerance
                @test up ≤ closure_tolerance

                # The halos were filled again after the correction: another pair fill changes neither component,
                # and the force and torque measured after it are the ones that were closed
                before = map(component -> map(r -> Array(parent(getregion(component, r))), 1:2), applied)
                fill_halo_regions!(applied)
                for (index, label) in enumerate(("u", "v"))
                    after = map(r -> Array(parent(getregion(applied[index], r))), 1:2)
                    change = maximum(r -> maximum(abs, after[r] .- before[index][r]), 1:2)
                    change == 0 || @info "Repeat halo fill changes $label by up to $(scientific(change))"
                    @test all(r -> after[r] == before[index][r], 1:2)
                end

                F_refilled, L_refilled = surface_force_and_torque(regridder, applied)
                refilled = max(maximum(abs, F_refilled - F_surface), maximum(abs, L_refilled - L_surface)) / scale
                @info "Round trip [$(name)] momentum on the C-grid faces after a repeat halo fill: surface → atmosphere force/torque imbalance = $(scientific(refilled))"
                @test F_refilled == F_applied
                @test L_refilled == L_applied
                @test refilled ≤ closure_tolerance
            end
        end
    end

    @testset "T5 vector alignment [$name]" begin
        # The stored unit vectors are the basis of Oceananigans' C-grid pair rotation
        easts = map(geometry -> map(Array, geometry.east), regridder.yin_yang_geometry)
        norths = map(geometry -> map(Array, geometry.north), regridder.yin_yang_geometry)
        basis_error = panel_maximum(cpu_grid) do i, j, panel, r
            λ, φ = node(i, j, panel)
            abs(φ) > 90 - 1e-6 && return 0.0 # the geographic basis is undefined on the pole
            east = ntuple(d -> easts[r][d][i, j, 1], 3)
            north = ntuple(d -> norths[r][d][i, j, 1], 3)
            θ = rotation_angle(i, j, panel, c, c)
            eᵍ, nᵍ = geographic_east(λ, φ), geographic_north(λ, φ)
            max(abs(dot3(east, eᵍ) - cos(θ)), abs(dot3(east, nᵍ) + sin(θ)),
                abs(dot3(north, eᵍ) - sin(θ)), abs(dot3(north, nᵍ) - cos(θ)))
        end
        @info "T5 [$(name)] unit vectors against Oceananigans' panel rotation angle: max difference = $(scientific(basis_error))"
        @test basis_error ≤ basis_tolerance

        tilted(λ, φ) = solid_body(tilted_axis, λ, φ)
        uᵉ, vⁿ = center_field(latlon), center_field(latlon)
        set!(uᵉ, (λ, φ) -> geographic_components(tilted(λ, φ), λ, φ)[1])
        set!(vⁿ, (λ, φ) -> geographic_components(tilted(λ, φ), λ, φ)[2])

        # Latitude-longitude → Yin-Yang cell centers, against the analytic panel components
        u, v = center_field(grid), center_field(grid)
        for (operator_name, operator!, tolerance) in (("interpolate_to_yin_yang!", interpolate_to_yin_yang!, staggered_tolerance(wind_speed, Nφ)),
                                                      ("transfer_flux!", transfer_flux!, first_order_bound(wind_speed, one_degree, Δ)))
            operator!((u, v), regridder, (uᵉ, vⁿ))
            uʳ, vʳ = map(r -> region_array(u, r), 1:2), map(r -> region_array(v, r), 1:2)

            vector_error = panel_maximum(cpu_grid) do i, j, panel, r
                hypot(((uʳ[r][i, j], vʳ[r][i, j]) .- panel_components(tilted, i, j, panel, c, c))...)
            end

            angle = panel_maximum(cpu_grid) do i, j, panel, r
                exact = panel_components(tilted, i, j, panel, c, c)
                hypot(exact...) < wind_speed / 10 ? 0.0 : direction_error((uʳ[r][i, j], vʳ[r][i, j]), exact)
            end

            @info "T5 [$(name)] tilted solid body, latitude-longitude → Yin-Yang, $(operator_name): max |ΔU| = $(scientific(vector_error)) m/s of $(wind_speed) (gate $(scientific(tolerance))), max direction error = $(fixed(angle, 4))°"
            @test vector_error ≤ tolerance
        end

        # Onto the C-grid, then Oceananigans' pair fill: the halos hold the analytic panel components
        uᶠ, vᶠ = Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid)
        interpolate_to_yin_yang!((uᶠ, vᶠ), regridder, (uᵉ, vⁿ))
        H = 2
        halo_error = maximum(1:2) do r
            panel = getregion(cpu_grid, r)
            uʳ = Array(parent(getregion(uᶠ, r)))
            vʳ = Array(parent(getregion(vᶠ, r)))
            Hx, Hy = panel.Hx, panel.Hy
            maximum(max(abs(uʳ[i + Hx, j + Hy, 1] - panel_components(tilted, i, j, panel, f, c)[1]),
                        abs(vʳ[i + Hx, j + Hy, 1] - panel_components(tilted, i, j, panel, c, f)[2]))
                    for i in 1-H:Nx+H, j in 1-H:Ny+H)
        end
        @info "T5 [$(name)] C-grid components after fill_halo_regions!((u, v)): max error over interior and $(H) halo cells = $(scientific(halo_error)) m/s (gate $(scientific(staggered_tolerance(wind_speed, Nφ))))"
        @test halo_error ≤ staggered_tolerance(wind_speed, Nφ)

        # Yin-Yang C-grid (analytic panel components) → latitude-longitude
        set_panels!(uᶠ, cpu_grid, f, c, (i, j, panel) -> panel_components(tilted, i, j, panel, f, c)[1])
        set_panels!(vᶠ, cpu_grid, c, f, (i, j, panel) -> panel_components(tilted, i, j, panel, c, f)[2])
        fill_halo_regions!((uᶠ, vᶠ))
        uᵍ, vᵍ = center_field(latlon), center_field(latlon)
        transfer_flux!((uᵍ, vᵍ), regridder, (uᶠ, vᶠ))
        vector_error = maximum(hypot.(latlon_array(uᵍ) .- latlon_array(uᵉ), latlon_array(vᵍ) .- latlon_array(vⁿ)))
        tolerance = first_order_bound(wind_speed, Δ, one_degree)
        @info "T5 [$(name)] tilted solid body, Yin-Yang C-grid → latitude-longitude, transfer_flux!: max |ΔU| = $(scientific(vector_error)) m/s of $(wind_speed) (gate $(scientific(tolerance)))"
        @test vector_error ≤ tolerance

        # Uniform flow straight over both geographic poles, through the round trip Yin-Yang → latitude-longitude →
        # Yin-Yang, at the cells of the Yang panel within 5° of a pole
        polar(λ, φ) = solid_body(polar_axis, λ, φ)
        set_panels!(uᶠ, cpu_grid, f, c, (i, j, panel) -> panel_components(polar, i, j, panel, f, c)[1])
        set_panels!(vᶠ, cpu_grid, c, f, (i, j, panel) -> panel_components(polar, i, j, panel, c, f)[2])
        fill_halo_regions!((uᶠ, vᶠ))
        transfer_flux!((uᵍ, vᵍ), regridder, (uᶠ, vᶠ))
        transfer_flux!((u, v), regridder, (uᵍ, vᵍ))

        yang = getregion(cpu_grid, 2)
        polar_cells = [(i, j) for i in 1:Nx, j in 1:Ny if abs(node(i, j, yang)[2]) > 85]
        uʸ, vʸ = region_array(u, 2), region_array(v, 2)
        pole_angle = maximum(direction_error((uʸ[i, j], vʸ[i, j]), panel_components(polar, i, j, yang, c, c)) for (i, j) in polar_cells)
        pole_speed = maximum(abs(hypot(uʸ[i, j], vʸ[i, j]) / hypot(panel_components(polar, i, j, yang, c, c)...) - 1) for (i, j) in polar_cells)

        # The same round trip with the geographic components regridded as two scalars
        sᵘ, sᵛ = center_field(grid), center_field(grid)
        set_panels!(sᵘ, cpu_grid, (λ, φ) -> geographic_components(polar(λ, φ), λ, φ)[1])
        set_panels!(sᵛ, cpu_grid, (λ, φ) -> geographic_components(polar(λ, φ), λ, φ)[2])
        transfer_flux!(uᵍ, regridder, sᵘ)
        transfer_flux!(vᵍ, regridder, sᵛ)
        transfer_flux!(sᵘ, regridder, uᵍ)
        transfer_flux!(sᵛ, regridder, vᵍ)
        nᵘ, nᵛ = region_array(sᵘ, 2), region_array(sᵛ, 2)
        naive_angle = maximum(polar_cells) do (i, j)
            λ, φ = node(i, j, yang)
            abs(φ) > 90 - 1e-6 && return 0.0
            direction_error((nᵘ[i, j], nᵛ[i, j]), geographic_components(polar(λ, φ), λ, φ))
        end

        @info "T5 [$(name)] flow over the poles, round trip, $(length(polar_cells)) Yang cells poleward of 85°: max direction error = $(fixed(pole_angle, 4))° (gate $(fixed(pole_direction_tolerance(Nφ), 3))°), max speed error = $(scientific(pole_speed)); geographic components as scalars: $(fixed(naive_angle, 2))°"
        @test pole_angle ≤ pole_direction_tolerance(Nφ)
        @test naive_angle > pole_angle
    end

    @testset "T7 pole and data nodes [$name]" begin
        source = center_field(latlon)
        set!(source, design_temperature)
        panels = center_field(grid)
        interpolate_to_yin_yang!(panels, regridder, source)
        values = map(r -> region_array(panels, r), 1:2)

        interpolation_error = panel_maximum((i, j, panel, r) -> abs(values[r][i, j] - design_temperature(node(i, j, panel)...)), cpu_grid)
        @info "T7 [$(name)] bilinear, 20 K wavenumber-2 field from the 1° grid: max error over both panels = $(scientific(interpolation_error)) K"
        @test interpolation_error ≤ interpolation_tolerance

        # The cells of the Yang panel nearest the geographic poles; with odd Nφ their centers are the poles
        yang = getregion(cpu_grid, 2)
        latitudes = [node(i, j, yang)[2] for i in 1:Nx, j in 1:Ny]
        for (pole, cell, exact) in (("north", argmax(latitudes), 290), ("south", argmin(latitudes), 270))
            @info "T7 [$(name)] Yang cell nearest the $(pole) pole at φ = $(fixed(latitudes[cell], 6))°: value = $(fixed(values[2][cell], 6)) K, pole value $(exact) K, analytic $(fixed(design_temperature(node(Tuple(cell)..., yang)...), 6)) K"
            isodd(Nφ) && @test abs(latitudes[cell]) == 90
            @test abs(values[2][cell] - design_temperature(node(Tuple(cell)..., yang)...)) ≤ interpolation_tolerance
        end

        # Fractions stay in [0, 1], and a sea surface temperature clipped at freezing stays above freezing
        freezing = 271.38
        for (operator_name, operator!) in operators
            set!(source, ice_concentration)
            operator!(panels, regridder, source)
            lowest, highest = extrema(vcat(map(r -> vec(region_array(panels, r)), 1:2)...))
            @test lowest ≥ minimum(latlon_array(source)) - bound_tolerance
            @test highest ≤ maximum(latlon_array(source)) + bound_tolerance

            # Halos: Oceananigans' cubic interpolation between panels overshoots a 0/1 mask to about [-0.13, 1.13];
            # both operators clamp the halos to the range of the interior, so a fraction stays a fraction there
            set!(source, (λ, φ) -> sind(6λ) * sind(6φ) > 0)
            operator!(panels, regridder, source)
            halo_lowest, halo_highest = extrema(vcat(map(r -> vec(Array(parent(getregion(panels, r)))), 1:2)...))
            unclamped = center_field(grid)
            for r in 1:2
                copyto!(interior(getregion(unclamped, r)), interior(getregion(panels, r)))
            end
            fill_halo_regions!(unclamped)
            raw_lowest, raw_highest = extrema(vcat(map(r -> vec(Array(parent(getregion(unclamped, r)))), 1:2)...))
            @info "T7 [$(name)] $(operator_name) of a 0/1 pattern of 30° blocks: with halos in [$(fixed(halo_lowest, 3)), $(fixed(halo_highest, 3))]; before clamping the halos [$(fixed(raw_lowest, 3)), $(fixed(raw_highest, 3))]"
            @test halo_lowest ≥ -bound_tolerance
            @test halo_highest ≤ 1 + bound_tolerance

            set!(source, (λ, φ) -> max(freezing, 300 - 35 * sind(φ)^2))
            operator!(panels, regridder, source)
            coldest = minimum(r -> minimum(region_array(panels, r)), 1:2)
            @info "T7 [$(name)] $(operator_name): ice concentration in [$(scientific(lowest)), 1 - $(scientific(1 - highest))], coldest clipped temperature = $(fixed(coldest, 10)) K"
            @test coldest ≥ freezing - freezing * bound_tolerance
        end

        # Data nodes: destination nodes that coincide with nodes of the data grid take the data bit for bit.
        # Every Yin cell center of the 1° grid is one.
        Random.seed!(4321)
        data = rand(size(latlon, 1), size(latlon, 2))
        set_values!(source, data)
        interpolate_to_yin_yang!(panels, regridder, source)
        yin = getregion(cpu_grid, 1)
        yin_values = region_array(panels, 1)
        coincident = [(i, j) for i in 1:Nx, j in 1:Ny if all(x -> abs(x - round(x)) < 1e-9, node(i, j, yin) .+ 0.5)]
        identical = count(coincident) do (i, j)
            λ, φ = node(i, j, yin)
            yin_values[i, j] == data[round(Int, mod(λ, 360) + 0.5), round(Int, φ + 90.5)]
        end
        @info "T7 [$(name)] $(length(coincident)) of $(Nx * Ny) Yin cell centers coincide with data nodes; $(identical) hold the data bit for bit"
        Nφ == 90 && @test length(coincident) == Nx * Ny
        @test identical == length(coincident)
    end

    @testset "T6 round-trip damping: measured, not gated [$name]" begin
        # MEASURED, not gated: no tolerance is set here, and nothing below says that the seam passed. Grid-scale
        # structure on the panels goes through transfer_flux! Yin-Yang → latitude-longitude → Yin-Yang. The number
        # is the unweighted projection Σ input × returned / Σ input² over the cells of each group: an amplitude
        # ratio, not an energy ratio, and without area or ownership weights.
        interior_cells = map(w -> (w .> 1 - 1e-9) .& .!seam_band(w, 8), owned)
        seam_cells = map(w -> (w .> 1e-9) .& seam_band(w, 2), owned)
        @info "T6 [$(name)] cells: $(sum(count ∘ cut_cells, owned)) cut, $(sum(count, seam_cells)) owned within 2 cells of a cut cell, $(sum(count, interior_cells)) owned farther than 8 cells, of $(2 * Nx * Ny)"

        patterns = (("checkerboard (2Δ)", (i, j) -> (-1.0)^(i + j)),
                    ("4Δ wave", (i, j) -> sin(π * i / 2 + π / 4) * sin(π * j / 2 + π / 4)),
                    ("8Δ wave", (i, j) -> sin(π * i / 4) * sin(π * j / 4)))

        for (label, pattern) in patterns
            input = center_field(grid)
            set_panels!(input, cpu_grid, c, c, (i, j, panel) -> pattern(i, j))
            exchange = center_field(latlon)
            output = center_field(grid)
            transfer_flux!(exchange, regridder, input)
            transfer_flux!(output, regridder, exchange)

            a, b = map(r -> region_array(input, r), 1:2), map(r -> region_array(output, r), 1:2)
            returned(cells) = sum(sum(a[r][cells[r]] .* b[r][cells[r]]) for r in 1:2) / sum(sum(abs2, a[r][cells[r]]) for r in 1:2)
            @info "T6 [$(name)] MEASURED, no gate: $(label): unweighted projection of the returned field on the input = $(fixed(returned(interior_cells), 3)) in the interior, $(fixed(returned(seam_cells), 3)) within 2 cells of a cut cell"
        end
    end
end

@testset "Convergence between 2° and 1°, both grids refined" begin
    errors = map(((45, 180), (90, 360))) do (Nφ, Nλ)
        grid = yin_yang_grid(Nφ)
        cpu_grid = on_architecture(CPU(), grid)
        exchange = exchange_grid(Nλ)
        regridder = YinYangRegridder(grid, exchange)

        source = center_field(exchange)
        set!(source, design_temperature)
        panels = center_field(grid)

        map(operators) do (operator_name, operator!)
            operator!(panels, regridder, source)
            values = map(r -> region_array(panels, r), 1:2)
            panel_maximum((i, j, panel, r) -> abs(values[r][i, j] - design_temperature(node(i, j, panel)...)), cpu_grid)
        end
    end

    bilinear_ratio = errors[1][1] / errors[2][1]
    area_weighted_ratio = errors[1][2] / errors[2][2]
    @info "Convergence, latitude-longitude → Yin-Yang, max error (K) at 2° and 1°: bilinear $(scientific(errors[1][1])), $(scientific(errors[2][1])) (ratio $(fixed(bilinear_ratio, 2))); transfer_flux! $(scientific(errors[1][2])), $(scientific(errors[2][2])) (ratio $(fixed(area_weighted_ratio, 2)))"
    @test bilinear_ratio ≥ second_order_ratio
    @test area_weighted_ratio ≥ first_order_ratio
end
