include("runtests_setup.jl")

using Breeze
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Models: boundary_condition_args
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Units
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

using NumericalEarth.EarthSystemModels: thermodynamics_parameters, surface_layer_height
using NumericalEarth.EarthSystemModels.InterfaceComputations: ImpureSaturationSpecificHumidity, net_fluxes

# The surface exchange of Breeze's `HydrostaticDynamics`, whose grid labels coordinate surfaces of the
# pressure p = A + B pˢ. Breeze versions without that dynamical core have nothing to test.
if !isdefined(Breeze, :HydrostaticDynamics)
    @info "Breeze $(pkgversion(Breeze)) has no HydrostaticDynamics: the hydrostatic coupling tests do not run."
else

using Breeze.HydrostaticEquations: VerletSubsteps

# Hybrid coefficients on the z-faces, from the ground to the model top at 2 hPa. The lowest layer is 14 hPa
# thick at a surface pressure of 1000 hPa.
const A = [0.0, 100.0, 4000.0, 9000.0, 6000.0, 200.0]
const B = [1.0, 0.985, 0.8, 0.45, 0.1, 0.0]
const Nz = length(A) - 1
const Nx = 8
const Lx = 8e5
const pˢᵗ = 1e5

# Two labelings of the same coordinate surfaces: the unit interval, and unevenly spaced labels up to 30 km
const unit_labels = collect(range(0, 1, length = Nz + 1))
const uneven_labels = [0.0, 40.0, 900.0, 5000.0, 14000.0, 30000.0]

surface_values(field) = Array(interior(field))[:, 1, 1]

#####
##### An oracle of the hydrostatic column, independent of Breeze: Phillips' layer Exner function, the
##### mass-weighted layer mean of (p / pˢᵗ)^κ, and the geopotential of a layer of uniform potential temperature
#####

interface_pressure(k, pˢ) = A[k] + B[k] * pˢ
interface_exner(k, pˢ, κ) = (interface_pressure(k, pˢ) / pˢᵗ)^κ

function layer_exner(k, pˢ, κ)
    p⁻, p⁺ = interface_pressure(k, pˢ), interface_pressure(k + 1, pˢ)
    return (p⁻ * interface_exner(k, pˢ, κ) - p⁺ * interface_exner(k + 1, pˢ, κ)) / ((1 + κ) * (p⁻ - p⁺))
end

layer_pressure(k, pˢ, κ) = pˢᵗ * layer_exner(k, pˢ, κ)^(1 / κ)

function thermodynamic_numbers(constants)
    Rᵈ = Breeze.dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    return (; Rᵈ, cᵖᵈ, κ = Rᵈ / cᵖᵈ, g = constants.gravitational_acceleration)
end

# Height of the lowest level above the ground: Φ rises by cᵖᵈ θ (Πᶠ - Π) from the ground to the level
function lowest_level_height(pˢ, θ₁, constants)
    (; cᵖᵈ, κ, g) = thermodynamic_numbers(constants)
    return cᵖᵈ * θ₁ * (interface_exner(1, pˢ, κ) - layer_exner(1, pˢ, κ)) / g
end

# ... and of the top of the lowest layer
function lowest_layer_top_height(pˢ, θ₁, constants)
    (; cᵖᵈ, κ, g) = thermodynamic_numbers(constants)
    return cᵖᵈ * θ₁ * (interface_exner(1, pˢ, κ) - interface_exner(2, pˢ, κ)) / g
end

#####
##### Models
#####

layer_index(labels, z) = clamp(searchsortedlast(labels, z), 1, Nz)

# A hydrostatic Breeze atmosphere in an x-z plane, whose columns differ in surface pressure `pˢ(x)`, in the
# geopotential of the ground `Φˢ(x)` and in temperature: `temperature(x, k)` is the temperature of layer `k`.
function hydrostatic_atmosphere(arch; labels = unit_labels, pˢ = x -> 1e5, Φˢ = x -> 0, temperature = (x, k) -> 280,
                                u = 0, constants = ThermodynamicConstants(Float64), kw...)
    grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5), x = (0, Lx), z = labels,
                           topology = (Periodic, Flat, Bounded))

    dynamics = HydrostaticDynamics(surface_pressure = pˢ, surface_geopotential = Φˢ,
                                   pressure_coefficient = A, surface_pressure_coefficient = B)

    atmosphere = atmosphere_simulation(grid; dynamics, thermodynamic_constants = constants,
                                       scalar_advection = Centered(order = 2), kw...)

    κ = thermodynamic_numbers(constants).κ
    θ(x, z) = temperature(x, layer_index(labels, z)) / layer_exner(layer_index(labels, z), pˢ(x), κ)
    set!(atmosphere.model; θ, u)

    return atmosphere
end

surface_grid(arch) = RectilinearGrid(arch; size = Nx, halo = 5, x = (0, Lx), topology = (Periodic, Flat, Flat))

# Surfaces that do not evaporate
const dry_ocean_humidity = ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Liquid(), 0)

function coupled_model(arch, atmosphere; sea_surface_temperature = 284, kw...)
    ocean = PrescribedOcean(surface_grid(arch))
    set!(ocean, T = sea_surface_temperature)
    return EarthSystemModel(; atmosphere, ocean, atmosphere_ocean_interface_specific_humidity = dry_ocean_humidity, kw...)
end

exchanged_height(model) = surface_values(model.interfaces.properties.surface_layer_height)

function interface_fluxes(model)
    fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
    return (sensible_heat = surface_values(fluxes.sensible_heat),
            x_momentum = surface_values(fluxes.x_momentum),
            net_energy = surface_values(model.interfaces.net_fluxes.atmosphere.ρE),
            net_x_momentum = surface_values(model.interfaces.net_fluxes.atmosphere.ρu))
end

x_centers(model) = Array(xnodes(model.interfaces.exchanger.grid, Center()))

# Columns that differ in surface pressure and temperature
varied_pressure(x) = 1e5 - 3e4 * (1 + cos(2π * x / Lx)) / 2
varied_temperature(x, k) = 285 - 10 * (k - 1) + 15 * sin(2π * x / Lx)

#####
##### (a) Surface-layer height
#####

@testset "The exchange height of HydrostaticDynamics is the geopotential height of the lowest level" begin
    for arch in test_architectures
        constants = ThermodynamicConstants(Float64)
        (; Rᵈ, κ, g) = thermodynamic_numbers(constants)
        atmosphere = hydrostatic_atmosphere(arch; pˢ = varied_pressure, temperature = varied_temperature, constants)
        model = coupled_model(arch, atmosphere)
        x = x_centers(model)
        z₁ = exchanged_height(model)

        # (Φ₁ - Φˢ) / g, where the layer Exner function is the mass-weighted mean: the placement of the core
        θ₁ = [varied_temperature(xᵢ, 1) / layer_exner(1, varied_pressure(xᵢ), κ) for xᵢ in x]
        oracle = lowest_level_height.(varied_pressure.(x), θ₁, Ref(constants))
        # The difference Πᶠ - Π of the lowest layer, 2e-3, is formed from products that cancel to 1.4e-2 of
        # their size: the two evaluations agree to 1e-16 / (2e-3 × 1.4e-2), a few 1e-12
        @test z₁ ≈ oracle rtol=1e-10
        @test z₁ == surface_values(surface_layer_height(atmosphere.model, model.interfaces.exchanger.grid))

        # Supported bounds: above the ground and below the top of the lowest layer, in every column
        top = lowest_layer_top_height.(varied_pressure.(x), θ₁, Ref(constants))
        @test all(0 .< z₁ .< top)
        @test all(0.45 .< z₁ ./ top .< 0.55)
        @info "Lowest-level heights of the test columns: $(round.(z₁, digits=2)) m"

        # Not half the spacing of the labels, which is what the grid's z would give
        @test all(abs.(z₁ .- unit_labels[2] / 2) .> 30)

        # The lowest level of a column that has no temperature yet is at the ground: refused
        cold = hydrostatic_atmosphere(arch; constants)
        fill!(parent(cold.model.formulation.potential_temperature_density), 0)
        update_state!(cold.model)
        @test_throws DomainError surface_layer_height(cold.model, model.interfaces.exchanger.grid)

        # A grid whose z is a height keeps the half spacing of its lowest cell
        height_grid = RectilinearGrid(arch; size = (Nx, 10), halo = (5, 5), x = (0, Lx), z = (0, 1000),
                                      topology = (Periodic, Flat, Bounded))
        @test surface_layer_height(atmosphere_model(height_grid), height_grid) == 50
    end
end

@testset "Isothermal columns: the exchange height against the barometric formula" begin
    for arch in test_architectures
        constants = ThermodynamicConstants(Float64)
        (; Rᵈ, κ, g) = thermodynamic_numbers(constants)

        heights = map((230, 260, 300)) do T₀
            atmosphere = hydrostatic_atmosphere(arch; pˢ = varied_pressure, temperature = (x, k) -> T₀, constants)
            model = coupled_model(arch, atmosphere; sea_surface_temperature = T₀)
            x = x_centers(model)
            z₁ = exchanged_height(model)

            # Barometric height of the pressure of the lowest level. The core integrates a layer of uniform
            # potential temperature, which in a 14 hPa layer differs from an isothermal one by κ ln(pˢ / p₁) / 2.
            p₁ = layer_pressure.(1, varied_pressure.(x), κ)
            barometric = @. Rᵈ * T₀ / g * log(varied_pressure(x) / p₁)
            @test surface_values(atmosphere.model.temperature) ≈ fill(T₀, Nx) rtol=1e-13
            @test z₁ ≈ barometric rtol=2e-3
            @test z₁ ≈ @.(barometric * (1 + κ * log(varied_pressure(x) / p₁) / 2)) rtol=1e-5
            z₁
        end

        # The height is proportional to the temperature. It follows the fraction of the surface pressure that
        # the lowest layer holds, Δp / pˢ = ΔB + ΔA / pˢ = 0.015 - 100 Pa / pˢ, which here grows with pˢ.
        @test heights[1] ./ 230 ≈ heights[2] ./ 260 rtol=1e-10
        @test heights[1] ./ 230 ≈ heights[3] ./ 300 rtol=1e-10
        column_pressure = varied_pressure.(range(Lx / 2Nx, step = Lx / Nx, length = Nx))
        @test sortperm(heights[2]) == sortperm(column_pressure .+ 1e-6 .* (1:Nx))
        @info "Isothermal 260 K, pˢ from $(round(Int, minimum(varied_pressure, 0:1e4:Lx))) to 100000 Pa: lowest level at $(round.(extrema(heights[2]), digits=2)) m"
    end
end

@testset "The exchange height and the fluxes do not depend on the labels or on the height of the terrain" begin
    for arch in test_architectures
        kw = (; pˢ = varied_pressure, temperature = varied_temperature, u = 6)
        build(; atmosphere_kw...) = coupled_model(arch, hydrostatic_atmosphere(arch; kw..., atmosphere_kw...))

        reference = build()
        relabeled = build(labels = uneven_labels)
        raised = build(Φˢ = x -> 9.81 * 2500)
        mountains = build(Φˢ = x -> 9.81 * 1500 * (1 + sin(4π * x / Lx)))

        # The labels are different, and so is the coordinate density Δp / (g Δz)
        @test Array(interior(relabeled.atmosphere.model.dynamics.density)) != Array(interior(reference.atmosphere.model.dynamics.density))
        @test Array(interior(mountains.atmosphere.model.dynamics.geopotential)) != Array(interior(reference.atmosphere.model.dynamics.geopotential))

        z₁ = exchanged_height(reference)
        fluxes = interface_fluxes(reference)
        @test all(abs.(fluxes.sensible_heat) .> 1)
        @test all(abs.(fluxes.x_momentum) .> 1e-3)

        for (name, model) in pairs((; relabeled, raised, mountains))
            @test exchanged_height(model) ≈ z₁ rtol=1e-10
            other = interface_fluxes(model)
            for flux in keys(fluxes)
                @test other[flux] ≈ fluxes[flux] rtol=1e-9
            end
            @info "$name: max relative difference of the height $(maximum(abs, exchanged_height(model) ./ z₁ .- 1)), of the sensible heat flux $(maximum(abs, other.sensible_heat ./ fluxes.sensible_heat .- 1))"
        end

        # The tendencies the fluxes give the lowest layer are the same under either labeling: the acceleration
        # and the rate of change of the potential temperature
        function lowest_layer_rates(model)
            atmosphere = model.atmosphere.model
            grid = atmosphere.grid
            args = boundary_condition_args(atmosphere)
            Δz = @allowscalar Oceananigans.zspacing(1, 1, 1, grid, Center(), Center(), Center())
            ρ = surface_values(atmosphere.dynamics.density)
            u_bc = atmosphere.momentum.u.boundary_conditions.bottom
            ρθ_bc = atmosphere.formulation.potential_temperature_density.boundary_conditions.bottom
            acceleration = [getbc(u_bc, i, 1, grid, args...) / Δz for i in 1:Nx]
            warming = [getbc(ρθ_bc, i, 1, grid, args...) / Δz for i in 1:Nx] ./ ρ
            return (; acceleration, warming)
        end

        rates = lowest_layer_rates(reference)
        relabeled_rates = lowest_layer_rates(relabeled)
        @test all(abs.(rates.acceleration) .> 1e-6)
        @test relabeled_rates.acceleration ≈ rates.acceleration rtol=1e-9
        @test relabeled_rates.warming ≈ rates.warming rtol=1e-9
    end
end

@testset "The exchange height follows the state: after a step, a set! and a restore" begin
    for arch in test_architectures
        constants = ThermodynamicConstants(Float64)
        κ = thermodynamic_numbers(constants).κ
        atmosphere = hydrostatic_atmosphere(arch; pˢ = varied_pressure, temperature = varied_temperature, u = 6, constants)
        model = coupled_model(arch, atmosphere)
        before = exchanged_height(model)

        # The height published with the exchanged state is that of the state at the end of the step
        function height_of_state(model)
            dynamics = model.atmosphere.model.dynamics
            θ = surface_values(model.atmosphere.model.formulation.potential_temperature)
            return lowest_level_height.(surface_values(dynamics.surface_pressure), θ, Ref(constants))
        end

        for _ in 1:3
            time_step!(model, 30)
        end

        after = exchanged_height(model)
        @test after != before
        @test after ≈ height_of_state(model) rtol=1e-10
        @test surface_values(model.interfaces.exchanger.atmosphere.state.T) == surface_values(model.atmosphere.model.temperature)

        # A new state set into the atmosphere: the height is that of the new state once it is exchanged.
        # Breeze's `set!` leaves the tendencies of the old state for the next stage once the model has
        # stepped, so the atmosphere's own `update_state!` recomputes them first.
        warmer(x, k) = varied_temperature(x, k) + 12
        set!(model.atmosphere.model; θ = (x, z) -> warmer(x, layer_index(unit_labels, z)) /
                                                    layer_exner(layer_index(unit_labels, z), varied_pressure(x), κ))
        update_state!(model.atmosphere.model)
        update_state!(model)
        at_save = exchanged_height(model)
        @test at_save ≈ height_of_state(model) rtol=1e-10
        @test all(at_save .> after .+ 1)

        # A restored atmosphere: its prognostic state is restored, its diagnostics and then the exchanged
        # state and height are recomputed, in that order
        saved = deepcopy(Oceananigans.prognostic_state(model))
        fluxes = interface_fluxes(model)

        restored = coupled_model(arch, hydrostatic_atmosphere(arch; constants))
        @test exchanged_height(restored) != at_save
        Oceananigans.restore_prognostic_state!(restored, saved)
        @test exchanged_height(restored) == at_save
        restored_fluxes = interface_fluxes(restored)
        for flux in keys(fluxes)
            @test restored_fluxes[flux] == fluxes[flux]
        end

        time_step!(model, 30)
        time_step!(restored, 30)
        @test exchanged_height(restored) == exchanged_height(model)
        @test Array(interior(restored.atmosphere.model.velocities.u)) == Array(interior(model.atmosphere.model.velocities.u))
    end
end

#####
##### (b) and (d) Density and constants
#####

@testset "The interfaces take their constants from the Breeze atmosphere" begin
    for arch in test_architectures
        default = ThermodynamicConstants(Float64)
        modified = ThermodynamicConstants(Float64; gravitational_acceleration = 9.80665,
                                                   dry_air_molar_mass = 8.314462618 / 287,
                                                   dry_air_heat_capacity = 1004.5)

        parameters = map((default, modified)) do constants
            atmosphere = hydrostatic_atmosphere(arch; constants, u = 6)
            ℂ = thermodynamics_parameters(atmosphere)
            P = AtmosphericThermodynamics.Parameters

            @test ℂ === thermodynamics_parameters(atmosphere.model)
            @test eltype(ℂ) == Float64
            @test P.R_d(ℂ) == Breeze.dry_air_gas_constant(constants)
            @test P.R_v(ℂ) == Breeze.vapor_gas_constant(constants)
            @test P.cp_d(ℂ) ≈ constants.dry_air.heat_capacity rtol=1e-15
            @test P.kappa_d(ℂ) == Breeze.dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
            @test P.cp_v(ℂ) == constants.vapor.heat_capacity
            @test P.cp_l(ℂ) == constants.liquid.heat_capacity
            @test P.cp_i(ℂ) == constants.ice.heat_capacity
            @test P.LH_v0(ℂ) == constants.liquid.reference_latent_heat
            @test P.LH_s0(ℂ) == constants.ice.reference_latent_heat
            @test P.T_0(ℂ) == constants.energy_reference_temperature
            @test P.T_triple(ℂ) == constants.triple_point_temperature
            @test P.press_triple(ℂ) == constants.triple_point_pressure

            # The air density of the interfaces is that of the equation of state with these constants
            T, p = 281.5, 9.3e4
            @test AtmosphericThermodynamics.air_density(ℂ, T, p, 0.0) ≈ p / (Breeze.dry_air_gas_constant(constants) * T) rtol=1e-14

            model = coupled_model(arch, atmosphere)
            @test model.interfaces.atmosphere_properties === ℂ
            @test model.interfaces.properties.gravitational_acceleration == constants.gravitational_acceleration
            (; ℂ, model)
        end

        # Changing Breeze's constants changes those of the interfaces, and the fluxes
        P = AtmosphericThermodynamics.Parameters
        @test P.R_d(parameters[2].ℂ) == 287
        @test P.R_d(parameters[2].ℂ) != P.R_d(parameters[1].ℂ)
        @test P.cp_d(parameters[2].ℂ) ≈ 1004.5 rtol=1e-15
        @test P.cp_d(parameters[2].ℂ) != P.cp_d(parameters[1].ℂ)
        @test parameters[2].model.interfaces.properties.gravitational_acceleration == 9.80665
        @test parameters[1].model.interfaces.properties.gravitational_acceleration == 9.81
        @test interface_fluxes(parameters[2].model).sensible_heat != interface_fluxes(parameters[1].model).sensible_heat

        # The float type of the constants is kept
        @test eltype(NumericalEarth.Atmospheres.AtmosphereThermodynamicsParameters(ThermodynamicConstants(Float32))) == Float32

        # Other Breeze dynamics read the same source
        height_grid = RectilinearGrid(arch; size = (Nx, 10), halo = (5, 5), x = (0, Lx), z = (0, 1000),
                                      topology = (Periodic, Flat, Bounded))
        compressible = atmosphere_model(height_grid; thermodynamic_constants = modified)
        @test P.R_d(thermodynamics_parameters(compressible)) == 287
    end
end

#####
##### (c) Flux insertion
#####

@testset "A surface stress changes the momentum of the column by stress × time × area" begin
    for arch in test_architectures
        schemes = (; SSPRungeKutta3 = (; timestepper = :SSPRungeKutta3),
                     MIS3C_Verlet = (; timestepper = :GravityWaveMultirateInfinitesimalStep,
                                       timestepper_kwargs = (; tableau = :MIS3C, inner_solver = VerletSubsteps())))

        @testset "$name with $(labels === unit_labels ? "unit" : "uneven") labels" for (name, scheme) in pairs(schemes), labels in (unit_labels, uneven_labels)
            constants = ThermodynamicConstants(Float64)
            (; cᵖᵈ, κ, g) = thermodynamic_numbers(constants)
            pˢ = 9.7e4

            # A resting atmosphere of uniform potential temperature under a uniform stress and heat flux stays
            # horizontally uniform, so nothing but the fluxes changes it
            atmosphere = hydrostatic_atmosphere(arch; labels, pˢ = x -> pˢ, constants, scheme...,
                                                temperature = (x, k) -> 290 * layer_exner(k, pˢ, κ))
            model = atmosphere.model
            grid = model.grid
            net = net_fluxes(model)
            τ, 𝒬, Δt = -0.3, 40.0, 120.0
            fill!(parent(net.ρu), τ)
            fill!(parent(net.ρE), 𝒬)

            u⁰ = Array(interior(model.velocities.u))
            ρθ⁰ = Array(interior(model.formulation.potential_temperature_density))
            time_step!(model, Δt)
            Δu = Array(interior(model.velocities.u)) .- u⁰
            Δρθ = Array(interior(model.formulation.potential_temperature_density)) .- ρθ⁰

            # Mass of each layer per unit area, Δp / g, from the hybrid coefficients: not from the grid
            layer_mass = [(interface_pressure(k, pˢ) - interface_pressure(k + 1, pˢ)) / g for k in 1:Nz]
            area = Lx / Nx  # per unit length in y

            # Momentum of the column, integrated over the domain
            momentum_change = sum(Δu[i, 1, k] * layer_mass[k] * area for i in 1:Nx, k in 1:Nz)
            @test momentum_change ≈ τ * Δt * Lx rtol=1e-11
            @test all(Δu[:, 1, 1] .≈ τ * Δt / layer_mass[1])
            @test maximum(abs, Δu[:, 1, 2:end]) < 1e-12 * abs(τ * Δt / layer_mass[1])

            # Enthalpy of the column: cᵖᵈ T Δp / g, with T = θ Π and ρθ Δz = θ Δp / g
            Δz = [@allowscalar(Oceananigans.zspacing(1, 1, k, grid, Center(), Center(), Center())) for k in 1:Nz]
            enthalpy_change = sum(cᵖᵈ * layer_exner(k, pˢ, κ) * Δρθ[i, 1, k] * Δz[k] * area for i in 1:Nx, k in 1:Nz)
            @test enthalpy_change ≈ 𝒬 * Δt * Lx rtol=1e-11
            @test maximum(abs, Δρθ[:, 1, 2:end]) < 1e-12 * maximum(abs, Δρθ[:, 1, 1])
            @info "$name, $(labels === unit_labels ? "unit" : "uneven") labels: momentum change / (stress × Δt × area) - 1 = $(momentum_change / (τ * Δt * Lx) - 1), enthalpy change / (flux × Δt × area) - 1 = $(enthalpy_change / (𝒬 * Δt * Lx) - 1)"
        end

        # Where the mass of the lowest layer varies from column to column, the stress at each velocity point
        # is divided by the mass there
        constants = ThermodynamicConstants(Float64)
        g = constants.gravitational_acceleration
        atmosphere = hydrostatic_atmosphere(arch; labels = uneven_labels, pˢ = varied_pressure, temperature = varied_temperature, constants)
        model = atmosphere.model
        grid = model.grid
        net = net_fluxes(model)
        stress = [0.1 + 0.05 * sin(2π * i / Nx) for i in 1:Nx]
        copyto!(interior(net.ρu), on_architecture(arch, reshape(stress, Nx, 1, 1)))
        update_state!(model)

        x = Array(xnodes(grid, Center()))
        Δp = [interface_pressure(1, varied_pressure(xᵢ)) - interface_pressure(2, varied_pressure(xᵢ)) for xᵢ in x]
        Δp_at_u = (Δp .+ circshift(Δp, 1)) ./ 2
        Δz₁ = uneven_labels[2] - uneven_labels[1]
        u_bc = model.momentum.u.boundary_conditions.bottom
        acceleration = [getbc(u_bc, i, 1, grid, boundary_condition_args(model)...) / Δz₁ for i in 1:Nx]
        @test acceleration .* Δp_at_u ./ g ≈ stress rtol=1e-13
        @test maximum(Δp) / minimum(Δp) > 1.3
    end
end

@testset "A hydrostatic atmosphere takes no water vapor from the surfaces" begin
    for arch in test_architectures
        # Breeze's dry guard rejects a boundary condition that could be a source of moisture
        grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5), x = (0, Lx), z = (0, 1), topology = (Periodic, Flat, Bounded))
        vapor_flux = Field{Center, Center, Nothing}(grid)
        moist_boundary_conditions = (; ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(vapor_flux)))
        @test_throws ArgumentError atmosphere_model(grid; dynamics = HydrostaticDynamics(top_pressure = 200),
                                                    boundary_conditions = moist_boundary_conditions)

        # The coupling gives the moisture density no boundary condition, and the model no microphysics
        atmosphere = hydrostatic_atmosphere(arch; u = 6)
        @test isnothing(atmosphere.model.microphysics)
        @test isnothing(atmosphere.model.moisture_density.boundary_conditions.bottom.condition)
        @test keys(net_fluxes(atmosphere)) == (:ρu, :ρv, :ρE, :ρqᵛᵉ)

        # An evaporating ocean stops the coupled model when it is built ...
        ocean = PrescribedOcean(surface_grid(arch))
        set!(ocean, T = 290)
        @test_throws ArgumentError EarthSystemModel(; atmosphere, ocean)

        # ... and a surface that starts to evaporate stops it when its fluxes are next assembled
        model = coupled_model(arch, hydrostatic_atmosphere(arch; u = 6))
        @test all(iszero, Array(interior(model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor)))
        @test all(iszero, Array(interior(model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat)))
        @test all(iszero, Array(interior(model.interfaces.net_fluxes.atmosphere.ρqᵛᵉ)))
        @test all(abs.(surface_values(model.interfaces.net_fluxes.atmosphere.ρE)) .> 1)
        time_step!(model, 30)
        @test all(iszero, Array(interior(model.atmosphere.model.moisture_density)))

        fill!(parent(model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor), 1e-5)
        @test_throws ArgumentError NumericalEarth.EarthSystemModels.update_net_fluxes!(model, model.atmosphere)
    end
end

end # if isdefined(Breeze, :HydrostaticDynamics)
