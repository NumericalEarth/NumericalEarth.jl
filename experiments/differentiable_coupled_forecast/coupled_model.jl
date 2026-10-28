# The coupled land–ocean–atmosphere model of `coupled_forecast_gradient.jl` and its forecast, shared by
# that script and `compile_ladder.jl`.

using NumericalEarth
using NumericalEarth: FixedIterations # Breeze exports its own
using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: update_state!, time_step!
using Reactant, CUDA                 # CUDA loads the Reactant KernelAbstractions extension
using Reactant: @trace
using Enzyme
using Printf, Random, Statistics

include("patches.jl")

Reactant.set_default_backend("cpu")
arch = ReactantState()

# ## Domain
#
# Periodic in x, so the sea breeze has ocean on both sides of the island. The land is the central half
# of the domain, `|x| < Lx/4`, the columns `land_columns` of both the land and the atmosphere.

Nx = 64
Lx = 20kilometers
land_columns = (Nx ÷ 4 + 1):(3Nx ÷ 4)

z = PiecewiseStretchedDiscretization(z  = [0, 3000, 8000, 15000],
                                     Δz = [100,  100, 1000,  1000])

Nz = length(z) - 1
Nzᵒᶜ = 20
ocean_depth = 50
latitude = 15
U₀ = 5 # m/s

Δt = 2seconds
stop_time = 20minutes
Nt = round(Int, stop_time / Δt)

# ## The coupled model
#
# Built by a function so that fresh copies start from the same seeded initial state.

on_land(x) = abs(x) < Lx/4

θᵇᵍ(z) = 300 + 4e-3 * z
qᵗᵇᵍ(z) = 0.012 * exp(-z / 2500)

function coupled_model(arch)
    Random.seed!(2025)

    grid = RectilinearGrid(arch;
                           size = (Nx, Nz),
                           x = (-Lx/2, Lx/2),
                           z,
                           halo = (5, 5),
                           topology = (Periodic, Flat, Bounded))

    ## Ocean: 50 m deep, with the land columns immersed to the surface
    island(x) = ifelse(on_land(x), 0, -ocean_depth)

    ocean_grid = RectilinearGrid(arch;
                                 size = (Nx, Nzᵒᶜ),
                                 halo = (grid.Hx, 5),
                                 x = (-Lx/2, Lx/2),
                                 z = (-ocean_depth, 0),
                                 topology = (Periodic, Flat, Bounded))

    ocean_grid = ImmersedBoundaryGrid(ocean_grid, GridFittedBottom(island))

    ocean = ocean_simulation(ocean_grid; Δt,
                             coriolis = FPlane(; latitude),
                             closure = VerticalScalarDiffusivity(κ = 1e-4, ν = 1e-4),
                             free_surface = SplitExplicitFreeSurface(substeps = 10),
                             momentum_advection = nothing,
                             tracer_advection = nothing,
                             warn = false)

    Tᵒᶜ(x, z) = 25 + (z + 10) / 50 * (z < -10) # °C
    set!(ocean.model, T = Tᵒᶜ, S = 35)

    ## Land: the slab of `breeze_over_slab_land.jl`, warm and wet
    land_grid = RectilinearGrid(arch;
                                size = Nx,
                                x = (-Lx/2, Lx/2),
                                halo = grid.Hx,
                                topology = (Periodic, Flat, Flat))

    hydrology = VariablySaturatedHydrology(eltype(land_grid);
        slab_depth = 1.0,
        porosity = 0.4,
        residual_liquid_fraction = 0.05,
        storage_height = 1000,
        retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
        hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-7, pore_size_uniformity = 2),
        deep_liquid_flux = NoDeepLiquidFlux(),
        runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

    energy = WaterCoupledEnergy(eltype(land_grid);
        dry_heat_capacity = 1480 * 1500 * 0.10,
        liquid_heat_capacity = 4186,
        reference_temperature = 273.15,
        deep_temperature = 305.0,
        deep_time_scale = 12hours,
        advect_deep_liquid_energy = false,
        advect_surface_liquid_energy = false)

    slab_land = SlabLand(land_grid; hydrology, energy)

    Mˡᵃ⁺ = slab_land.hydrology.porosity * slab_land.hydrology.slab_depth * 1000
    set!(slab_land.temperature, 305)
    set!(slab_land.water_storage, 0.6 * Mˡᵃ⁺)
    update_state!(slab_land)

    ## Atmosphere: compressible, with a fixed number of acoustic substeps
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 8);
                                    base_pressure = 101325,
                                    reference_potential_temperature = θᵇᵍ)

    atmosphere = atmosphere_simulation(grid; dynamics, coriolis = FPlane(; latitude))

    ## On ReactantState the reference state is computed by `initialize!`, not on construction
    Reactant.@jit Oceananigans.initialize!(atmosphere.model)

    θᵢ(x, z) = θᵇᵍ(z) + 0.5 * (rand() - 0.5) * (z < 1000)
    qᵗᵢ(x, z) = qᵗᵇᵍ(z)
    set!(atmosphere.model; θ = θᵢ, u = U₀, qᵗ = qᵗᵢ, ρ = atmosphere.model.dynamics.reference_state.density)

    ## Coupling: the atmosphere takes its surface fluxes from the land over the island and from the ocean
    ## elsewhere, fixed-iteration flux solvers, and the dry-layer humidity of `breeze_over_slab_land.jl`
    ocean_fraction(x) = ifelse(on_land(x), 0, 1)
    FT = eltype(grid)
    atmosphere_ocean_fluxes = SimilarityTheoryFluxes(FT; solver_stop_criteria = FixedIterations(8))
    atmosphere_land_fluxes = NumericalEarth.EarthSystemModels.InterfaceComputations.default_atmosphere_land_fluxes(
        slab_land, FT; solver_stop_criteria = FixedIterations(8))

    atmosphere_land_interface_specific_humidity = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05,
            dry_layer_onset_saturation = 0.5,
            dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4,
            molecular_diffusivity = 2.5e-5,
            tortuosity = PowerLawTortuosity()),
        thermal_exchange_depth = 0.10,
        porosity = slab_land.hydrology.porosity)

    return AtmosphereOceanModel(atmosphere, ocean; land = slab_land,
                                atmosphere_ocean_fluxes,
                                atmosphere_land_fluxes,
                                atmosphere_land_interface_specific_humidity,
                                ocean_fraction,
                                clock = Clock(grid))
end

# ## The forecast
#
# The initial-condition perturbations `δ` are added to the prognostic state, the diagnostics and
# surface fluxes are refreshed from it, and the coupled model runs `Nt` steps in a checkpointed
# `@trace` loop. The forecast is the mean skin temperature over the land columns.

initial_perturbations(model) = (ρθ  = CenterField(model.atmosphere.model.grid),
                                Tᵒᶜ = CenterField(model.ocean.model.grid),
                                Tˡᵃ = CenterField(model.land.grid))

function forecast_land_temperature(model, δ, land_columns, Δt, Nt)
    atmosphere = model.atmosphere.model
    ocean = model.ocean.model

    parent(Oceananigans.prognostic_fields(atmosphere).ρθ) .+= parent(δ.ρθ)
    parent(ocean.tracers.T) .+= parent(δ.Tᵒᶜ)
    parent(model.land.temperature) .+= parent(δ.Tˡᵃ)

    update_state!(atmosphere; compute_tendencies = false)
    update_state!(ocean)
    update_state!(model)

    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nt
        time_step!(model, Δt)
    end

    return mean(interior(model.land.temperature, land_columns, 1, 1))
end
