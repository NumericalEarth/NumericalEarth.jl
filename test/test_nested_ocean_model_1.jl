include("nested_ocean_test_utils.jl")

@testset "Open boundary sides follow the coastline" begin
    for arch in test_architectures
        open_ocean = synthetic_regional_grid(arch; longitude = (100, 140), latitude = (-40, 0))
        @test open_boundary_sides(open_ocean) == (:west, :east, :south, :north)
        @test open_boundary_sides(coastal_grid(arch)) == (:east, :south, :north)

        periodic = LatitudeLongitudeGrid(arch; size = (16, 12, 6), longitude = (0, 360), latitude = (-40, 0),
                                         z = (-4000, 0), topology = (Periodic, Bounded, Bounded))
        @test open_boundary_sides(periodic) == (:south, :north)
    end
end

@testset "Volumetric PrescribedOcean from a dataset on $(arch)" for arch in test_architectures
    region = BoundingBox(longitude = (0, 60), latitude = (-35, 45))
    ocean = PrescribedOcean(region, synthetic_dates, SyntheticOcean(); architecture = arch)
    @test ocean isa PrescribedOcean
    @test size(ocean.temperature, 3) == 4
    @test location(ocean.free_surface) == (Center, Center, Nothing)
    @test length(ocean.times) == 3

    time_step!(ocean, 86400)
    @test ocean.clock.time == 86400
end

@testset "nested_ocean_model in a synthetic dataset on $(arch)" for arch in test_architectures
    grid = coastal_grid(arch)
    nested = nested_ocean_model(grid, SyntheticOcean(); dates = synthetic_dates, parent_padding = 10, relaxation_width = 3)

    @test nested isa NestedModel
    @test nested.parent isa BoundaryPrescribedComponent
    @test keys(nested.exchanger.bands) == (:east, :south, :north)
    check_nested_ocean(nested, grid)

    # The interior starts from the parent's full-domain snapshot
    @test wet_maximum(nested.child.velocities.u) ≈ 0.1 rtol = 1e-6
    @test wet_maximum(nested.child.free_surface.displacement) ≈ 0.2 rtol = 1e-6

    # The parent steps first, and the bands then bracket the child's step
    time_step!(nested, 60)
    @test nested.parent.clock.time == nested.child.clock.time == 60
    @test (nested.exchanger.times.previous, nested.exchanger.times.current) == (0, 60)
    time_step!(nested, 60)
    @test !any(isnan, interior(nested.child.velocities.u))
    @test !any(isnan, interior(nested.child.tracers.T))
end

@testset "nested_ocean_simulation wraps the nested ocean in a Simulation on $(arch)" for arch in test_architectures
    grid = coastal_grid(arch)
    ocean = nested_ocean_simulation(grid, SyntheticOcean(); dates = synthetic_dates, parent_padding = 10,
                                    relaxation_width = 3, Δt = 60)

    @test ocean isa NestedSimulation
    @test ocean.Δt == 60
    @test keys(ocean.model.exchanger.bands) == (:east, :south, :north)

    ocean.stop_iteration = 2
    run!(ocean)
    @test iteration(ocean) == 2
    @test ocean.model.parent.clock.time == ocean.model.child.clock.time == 120
    @test !any(isnan, interior(ocean.model.child.tracers.T))
end

@testset "nested_ocean_model in a full-domain PrescribedOcean on $(arch)" for arch in test_architectures
    child_grid = coastal_grid(arch)
    parent_region = BoundingBox(longitude = (0, 60), latitude = (-35, 45))

    # A full-domain prescribed parent also initializes the child
    prescribed = PrescribedOcean(parent_region, synthetic_dates, SyntheticOcean(); architecture = arch)
    nested = nested_ocean_model(prescribed, child_grid; relaxation_width = 3)
    @test nested.parent === prescribed
    check_nested_ocean(nested, child_grid)

    # The open sides are `Open` on the child's grid; the grid passed in keeps its walls
    @test topology(nested.child.grid)[1:2] == (TwoSided{Closed, Open}, TwoSided{Open, Open})
    @test topology(child_grid)[1:2] == (Bounded, Bounded)
    @test wet_maximum(nested.child.velocities.u) ≈ 0.1 rtol = 1e-6

    time_step!(nested, 60)
    time_step!(nested, 60)
    @test prescribed.clock.time == nested.child.clock.time == 120
    @test nested.exchanger.times.current == 120
    @test !any(isnan, interior(nested.child.velocities.u))
end

@testset "A nested ocean couples like an ocean on $(arch)" for arch in test_architectures
    grid = coastal_grid(arch)
    nested = nested_ocean_model(grid, SyntheticOcean(); dates = synthetic_dates, parent_padding = 10, relaxation_width = 3)
    ocean = Simulation(nested; Δt = 60)

    atmosphere = synthetic_prescribed_atmosphere(arch)
    radiation = synthetic_prescribed_radiation(arch)
    coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

    for _ in 1:2
        time_step!(coupled_model, 60)
    end

    @test nested.child.clock.time == nested.parent.clock.time == 120
    @test !any(isnan, interior(nested.child.tracers.T))
end

@testset "PrescribedOcean: grid vertical size selects surface vs volumetric fields" begin
    surface_grid = LatitudeLongitudeGrid(size = (8, 8, 1), longitude = (-1, 1), latitude = (-1, 1),
                                         z = (-1, 0), topology = (Bounded, Bounded, Bounded))
    surface_ocean = PrescribedOcean(surface_grid)

    @test location(surface_ocean.temperature)  == (Center, Center, Nothing)
    @test location(surface_ocean.salinity)     == (Center, Center, Nothing)
    @test location(surface_ocean.velocities.u) == (Center, Center, Nothing)
    @test isnothing(surface_ocean.free_surface)

    volumetric_grid = LatitudeLongitudeGrid(size = (8, 8, 4), longitude = (-1, 1), latitude = (-1, 1),
                                            z = (-100, 0), topology = (Bounded, Bounded, Bounded))
    volumetric_ocean = PrescribedOcean(volumetric_grid)

    @test location(volumetric_ocean.temperature)  == (Center, Center, Center)
    @test location(volumetric_ocean.velocities.v) == (Center, Center, Center)
    @test location(volumetric_ocean.free_surface) == (Center, Center, Nothing)
    @test all(interior(volumetric_ocean.salinity) .== 35)
    @test keys(parent_ocean_sources(volumetric_ocean)) == (:u, :v, :η, :T, :S)

    @test size(ocean_temperature(volumetric_ocean)) == (8, 8, 4, 1)
    @test size(ocean_surface_temperature(surface_ocean)) == (8, 8, 1, 1)
end

@testset "nested_ocean_model: boundary conditions land where the nest needs them on $(arch)" for arch in test_architectures
    model = nested_ocean_model(nesting_test_parent(arch), nesting_test_child_grid(arch))
    child = model.child

    free_surface = child.free_surface
    U_bcs = free_surface.barotropic_velocities.U.boundary_conditions
    V_bcs = free_surface.barotropic_velocities.V.boundary_conditions

    # Flather on the barotropic transport, on the sides each component is normal to
    for side in (:west, :east)
        @test getproperty(U_bcs, side).classification isa NormalFlow
        @test getproperty(U_bcs, side).classification.scheme isa GravityWaveRadiation
    end

    for side in (:south, :north)
        @test getproperty(V_bcs, side).classification isa NormalFlow
        @test getproperty(V_bcs, side).classification.scheme isa GravityWaveRadiation
    end

    # Chapman on the free surface at the Flather sides
    η_bcs = free_surface.displacement.boundary_conditions
    for side in lateral_sides
        @test getproperty(η_bcs, side).classification isa Value
        @test getproperty(η_bcs, side).classification.scheme isa SurfaceWaveRadiation
    end

    # Tracers are reservoirs (Value); momentum radiates obliquely, NormalFlow on its own side and Value on the
    # tangential ones
    for side in lateral_sides
        for name in (:T, :S)
            bc = getproperty(child.tracers[name].boundary_conditions, side)
            @test bc.classification isa Value
            @test bc.classification.scheme isa TracerReservoir
        end

        u_bc = getproperty(child.velocities.u.boundary_conditions, side)
        v_bc = getproperty(child.velocities.v.boundary_conditions, side)
        @test u_bc.classification isa (side in (:west, :east) ? NormalFlow : Value)
        @test v_bc.classification isa (side in (:south, :north) ? NormalFlow : Value)
        @test u_bc.classification.scheme isa ObliqueRadiation
        @test v_bc.classification.scheme isa ObliqueRadiation
    end

    # The coupled interface reaches the child through the wrapper
    @test prognostic_model(model) === child
end

@testset "nested_ocean_model: schemes can be set per side on $(arch)" for arch in test_architectures
    narrow = TracerReservoir(inflow_length_scale = 5e3, outflow_length_scale = 0)
    wide   = TracerReservoir(inflow_length_scale = 5e4, outflow_length_scale = 0)
    tracer_scheme = (west = narrow, east = wide, south = narrow, north = wide)
    child = nested_ocean_model(nesting_test_parent(arch), nesting_test_child_grid(arch); tracer_scheme).child

    for side in lateral_sides, name in (:T, :S)
        scheme = getproperty(child.tracers[name].boundary_conditions, side).classification.scheme
        @test scheme.inflow_length_scale == tracer_scheme[side].inflow_length_scale
    end

    @test_throws ArgumentError nested_ocean_model(nesting_test_parent(arch), nesting_test_child_grid(arch);
                                                  tracer_scheme = (west = narrow, east = wide))
end

@testset "nested_ocean_model: a quiescent parent leaves the child unchanged on $(arch)" for arch in test_architectures
    T₀, S₀ = 15.0, 35.0
    model = nested_ocean_model(nesting_test_parent(arch; u = 0, T = T₀, S = S₀), nesting_test_child_grid(arch))
    set!(model, T = T₀, S = S₀, u = 0, v = 0)
    update_state!(model)

    for n in 1:20
        time_step!(model, 60.0)
    end

    T = Array(interior(model.child.tracers.T))
    S = Array(interior(model.child.tracers.S))

    # Radiating boundaries on all four sides must not manufacture a gradient out of a uniform state
    @test maximum(abs, T .- T₀) < 1e-10
    @test maximum(abs, S .- S₀) < 1e-10
    @test model.parent.clock.time == model.clock.time == 20 * 60.0
end

# A child on a rotated grid, nested in a parent that covers it with uniform eastward and northward velocities
function rotated_nested_ocean(arch; north_pole, longitude, latitude, uₑ, vₑ)
    grid = RotatedLatitudeLongitudeGrid(arch; size = (16, 16, 6), halo = (7, 7, 7), north_pole, longitude, latitude,
                                        z = (-1000, 0))

    box = BoundingBox(grid; padding = 1)
    parent_grid = LatitudeLongitudeGrid(arch; size = (24, 24, 6), longitude = box.longitude, latitude = box.latitude,
                                        z = (-1000, 0), topology = (Bounded, Bounded, Bounded))

    parent_ocean = PrescribedOcean(parent_grid, [0.0, 10days])
    set!(parent_ocean; T = 15, S = 35, u = uₑ, v = vₑ, η = 0)

    return nested_ocean_model(parent_ocean, grid)
end

# The components of an eastward `uₑ` and northward `vₑ` along the axes of `grid` at its `u` and `v` points, from the
# cell corners that straddle each: the `y` line through a `u` point and the `x` line through a `v` point.
function velocities_along_grid_axes(grid, uₑ, vₑ)
    Nx, Ny, _ = size(grid)
    λ = Array(grid.λᶠᶠᵃ[1:Nx+1, 1:Ny+1])
    φ = Array(grid.φᶠᶠᵃ[1:Nx+1, 1:Ny+1])

    function east_north(i₁, j₁, i₂, j₂)
        δe = cosd((φ[i₁, j₁] + φ[i₂, j₂]) / 2) * (λ[i₂, j₂] - λ[i₁, j₁])
        δn = φ[i₂, j₂] - φ[i₁, j₁]
        return δe / hypot(δe, δn), δn / hypot(δe, δn)
    end

    y_lines = [east_north(i, j, i, j + 1) for i in 1:Nx+1, j in 1:Ny]
    x_lines = [east_north(i, j, i + 1, j) for i in 1:Nx, j in 1:Ny+1]

    u = [uₑ * n - vₑ * e for (e, n) in y_lines]
    v = [vₑ * e - uₑ * n for (e, n) in x_lines]

    return u, v
end

@testset "A nested ocean on a rotated grid takes the parent's velocities along its own axes on $(arch)" for arch in test_architectures
    uₑ, vₑ = 0.1, -0.05

    nested = rotated_nested_ocean(arch; north_pole = (-150, 40), longitude = (118, 122), latitude = (-2, 2), uₑ, vₑ)
    grid = nested.child.grid
    Nx, Ny, Nz = size(grid)
    @test topology(grid, 1) == topology(grid, 2) == TwoSided{Open, Open}
    @test keys(nested.exchanger.bands) == lateral_sides

    u★, v★ = velocities_along_grid_axes(grid, uₑ, vₑ)
    @test !(u★[1, 1] ≈ uₑ) && !(v★[1, 1] ≈ vₑ)

    u = Array(interior(nested.child.velocities.u))
    v = Array(interior(nested.child.velocities.v))
    @test all(isapprox(u[:, :, k], u★; atol = 1e-4) for k in 1:Nz)
    @test all(isapprox(v[:, :, k], v★; atol = 1e-4) for k in 1:Nz)

    bands = nested.exchanger.bands
    @test Array(interior(bands.west.u.current))[1, :, 1]    ≈ u★[1, :]    atol = 1e-4
    @test Array(interior(bands.east.u.current))[end, :, 1]  ≈ u★[Nx+1, :] atol = 1e-4
    @test Array(interior(bands.south.v.current))[:, 1, 1]   ≈ v★[:, 1]    atol = 1e-4
    @test Array(interior(bands.north.v.current))[:, end, 1] ≈ v★[:, Ny+1] atol = 1e-4
    @test Array(interior(bands.west.v.current))[1, :, 1]    ≈ v★[1, :]    atol = 1e-4

    for n in 1:3
        time_step!(nested, 60)
    end

    @test !any(isnan, Array(interior(nested.child.velocities.u)))
    @test maximum(abs, Array(interior(nested.child.tracers.T)) .- 15) < 1e-10

    # A few degrees from the pole the axes turn by degrees per cell, so a face needs its own angle, not a centre's
    nested = rotated_nested_ocean(arch; north_pole = (180, 0), longitude = (172, 176), latitude = (-2, 2), uₑ, vₑ)
    grid = on_architecture(CPU(), nested.child.grid)
    u★, v★ = velocities_along_grid_axes(grid, uₑ, vₑ)
    u = Array(interior(nested.child.velocities.u))[:, :, Nz]
    v = Array(interior(nested.child.velocities.v))[:, :, Nz]

    θ = [rotation_angle(i, j, grid, Center(), Center()) for i in 1:Nx+1, j in 1:Ny+1]
    uᶜ = [uₑ * cos(θ[i, j]) - vₑ * sin(θ[i, j]) for i in 1:Nx+1, j in 1:Ny]
    vᶜ = [uₑ * sin(θ[i, j]) + vₑ * cos(θ[i, j]) for i in 1:Nx, j in 1:Ny+1]
    @info "max error of face velocities rotated by face / centre angles near the pole" maximum(abs, u .- u★) maximum(abs, uᶜ .- u★) maximum(abs, v .- v★) maximum(abs, vᶜ .- v★)
    @test maximum(abs, u .- u★) < maximum(abs, uᶜ .- u★) / 10
    @test maximum(abs, v .- v★) < maximum(abs, vᶜ .- v★) / 10

    # The x axis of this grid points south at its center, and its y axis east
    nested = rotated_nested_ocean(arch; north_pole = (90, 0), longitude = (-122, -118), latitude = (-2, 2), uₑ, vₑ)
    u = Array(interior(nested.child.velocities.u))
    v = Array(interior(nested.child.velocities.v))
    @test u[Nx÷2 + 1, Ny÷2, Nz] ≈ -vₑ atol = 1e-3
    @test v[Nx÷2, Ny÷2 + 1, Nz] ≈ uₑ atol = 1e-3
    time_step!(nested, 60)
    @test !any(isnan, Array(interior(nested.child.velocities.v)))
end

@testset "Split-explicit substeps past the child step hold the parent, not extrapolate it, on $(arch)" for arch in test_architectures
    # A parent free surface rising from 0 to 1 m over its 10 days
    parent_ocean = nesting_test_parent(arch)
    interior(parent_ocean.free_surface)[:, :, 1, 2] .= 1
    fill_prescribed_halo_regions!(parent_ocean)

    grid = nesting_test_child_grid(arch)
    model = nested_ocean_model(parent_ocean, grid)
    Δt = 600.0
    time_step!(model, Δt)

    U_condition = on_architecture(CPU(), model.child.free_surface.barotropic_velocities.U.boundary_conditions.west.condition)
    η_parent(t) = t / 10days
    η_boundary(t) = getbc(U_condition, 10, 1, on_architecture(CPU(), grid), (; time = t))[2]

    @test η_boundary(Δt / 2) ≈ η_parent(Δt / 2)
    @test η_boundary(Δt) ≈ η_parent(Δt)
    @test η_boundary(3Δt / 2) ≈ η_parent(Δt)
end

@testset "Flather adds the child's inverse-barometer sea level to the parent's on $(arch)" for arch in test_architectures
    η₀ = 0.2
    @test nested_ocean_model(nesting_test_parent(arch; η = η₀), nesting_test_child_grid(arch)).child_callbacks === ()

    model = nested_ocean_model(nesting_test_parent(arch; η = η₀), nesting_test_child_grid(arch); inverse_barometer = true)
    child = model.child
    grid = child.grid
    Nx, Ny, Nz = size(grid)

    # A pressure potential pᵃ / ρ₀ rising northward by 10 m² s⁻² per degree, as the atmosphere would fill it
    pᵃ = Field{Center, Center, Nothing}(grid)
    set!(pᵃ, (λ, φ) -> 10 * (φ - 29) + λ)
    parent(forcing_barotropic_potential(child)) .= parent(pᵃ)
    η_before = Array(interior(child.free_surface.displacement, :, :, 1))
    update_state!(model)

    cpu_grid = on_architecture(CPU(), grid)
    g = child.free_surface.gravitational_acceleration
    p = Array(interior(pᵃ, :, :, 1))
    Az = [Azᶜᶜᶜ(i, j, Nz, cpu_grid) for i in 1:Nx, j in 1:Ny]
    p̄ = sum(p .* Az) / sum(Az)
    inverse_barometer = only(model.child_callbacks).parameters
    ηᴵᴮ = Array(interior(inverse_barometer.sea_level, :, :, 1))
    wet = Array(interior(inverse_barometer.wet_surface_area, :, :, 1)) .> 0

    @test ηᴵᴮ ≈ (p̄ .- p) ./ g
    @test abs(sum(ηᴵᴮ .* Az)) < 1e-12 * sum(abs, ηᴵᴮ .* Az)

    bcs = child.free_surface.barotropic_velocities
    η_boundary(bc, s) = getbc(on_architecture(CPU(), bc.condition), s, 1, cpu_grid, (; time = 0.0))[2]

    @test [η_boundary(bcs.U.boundary_conditions.west, j) for j in 1:Ny] ≈ η₀ .+ ηᴵᴮ[1, :]
    @test [η_boundary(bcs.U.boundary_conditions.east, j) for j in 1:Ny] ≈ η₀ .+ ηᴵᴮ[Nx, :]
    @test [η_boundary(bcs.V.boundary_conditions.south, i) for i in 1:Nx] ≈ η₀ .+ ηᴵᴮ[:, 1]
    @test [η_boundary(bcs.V.boundary_conditions.north, i) for i in 1:Nx] ≈ η₀ .+ ηᴵᴮ[:, Ny]

    η_initial = Array(interior(child.free_surface.displacement, :, :, 1))
    @test any(wet)
    @test η_initial ≈ η_before .+ ηᴵᴮ .* wet

    update_state!(model)
    @test Array(interior(child.free_surface.displacement, :, :, 1)) == η_initial

    time_step!(model, 600)
    η_stepped = Array(interior(child.free_surface.displacement, :, :, 1))
    @test all(isfinite, η_stepped)

    parent(forcing_barotropic_potential(child)) .*= 2
    update_state!(model)
    @test Array(interior(child.free_surface.displacement, :, :, 1)) == η_stepped
end

@testset "The atmosphere's surface pressure reaches a nested ocean's child on $(arch)" for arch in test_architectures
    model = nested_ocean_model(nesting_test_parent(arch), nesting_test_child_grid(arch); inverse_barometer = true)
    ocean = Simulation(model; Δt = 600)

    atmosphere_grid = LatitudeLongitudeGrid(arch; size = (8, 8), longitude = (-2, 2), latitude = (28, 32),
                                            topology = (Bounded, Bounded, Flat))
    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0, 10days])
    for n in 1:2
        set!(atmosphere.pressure[n], (λ, φ) -> 1e5 + 100 * (φ - 30))
    end

    coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation = nothing)
    update_state!(coupled_model)

    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
    Nx, Ny, _ = size(model.child.grid)
    Hx, Hy, _ = halo_size(model.child.grid)
    potential = Array(parent(forcing_barotropic_potential(model.child)))[Hx+1:Hx+Nx, Hy+1:Hy+Ny, 1]
    @test forcing_barotropic_potential(model) === forcing_barotropic_potential(model.child)
    @test all(potential .> 9e4 / ρᵒᶜ)

    time_step!(coupled_model, 600)
    @test all(isfinite, Array(interior(model.child.free_surface.displacement)))
end

@testset "Open boundary conditions move to the CPU with the child's fields on $(arch)" for arch in test_architectures
    # As when output written on `arch` is read back on the CPU
    model = nested_ocean_model(nesting_test_parent(arch), nesting_test_child_grid(arch))
    child = model.child
    fields = (u = child.velocities.u, v = child.velocities.v, T = child.tracers.T)

    for name in keys(fields)
        field = fields[name]
        cpu_field = on_architecture(CPU(), field)
        fill_halo_regions!(field)
        fill_halo_regions!(cpu_field)
        @test Array(parent(cpu_field)) ≈ Array(parent(field))
    end
end

@testset "A z★ child samples the parent at its reference depths on $(arch)" for arch in test_architectures
    parent_ocean = nesting_test_parent(arch; T = (λ, φ, z, t) -> 20 + z / 100)
    T_parent = parent_ocean_sources(parent_ocean).T

    zstar_grid(arch) = LatitudeLongitudeGrid(arch; size = (20, 20, 6), halo = (7, 7, 7),
                                             longitude = (-1, 1), latitude = (29, 31),
                                             z = MutableVerticalDiscretization((-1000, 0)),
                                             topology = (Bounded, Bounded, Bounded))

    # A 50 m rise of the free surface lifts every z★ level, and the top cell above the parent's top level
    grid = zstar_grid(arch)
    fill!(grid.z.ηⁿ, 50)

    T_static = CenterField(nesting_test_child_grid(arch))
    T_zstar = CenterField(grid)
    sample_parent!(T_static, (Center(), Center(), Center()), T_parent, 0)
    sample_parent!(T_zstar, (Center(), Center(), Center()), T_parent, 0)
    @test Array(interior(T_zstar)) ≈ Array(interior(T_static))

    model = nested_ocean_model(parent_ocean, zstar_grid(arch))
    time_step!(model, 60.0)
    @test all(isfinite, Array(interior(model.child.tracers.T)))
end
