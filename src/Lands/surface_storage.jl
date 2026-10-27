"""
$(TYPEDSIGNATURES)

Canopy-air heat and moisture storage for `SlabLand(surface_storage=...)`.
`layer_depth` is the air-layer depth in meters, supplied as a number or a horizontal field.
The initial `temperature` (K) and `specific_humidity` (kg kg⁻¹) default to 300 and 0.008.

```jldoctest
using NumericalEarth

PrognosticCanopyAir(layer_depth=12)

# output
PrognosticCanopyAir(hᵃᶜᵃⁿ=12.0)
```
"""
struct PrognosticCanopyAir{H, P, N, B}
    layer_depth :: H
    prognostic :: P
    tendencies :: N
    boundary_condition :: B
end

PrognosticCanopyAir(FT::Type = Oceananigans.defaults.FloatType;
                   layer_depth=10, temperature=300, specific_humidity=0.008) =
    PrognosticCanopyAir(convert_if_number(FT, layer_depth),
                       (; temperature=convert_if_number(FT, temperature), specific_humidity=convert_if_number(FT, specific_humidity)), nothing, nothing)

"""
$(TYPEDSIGNATURES)

Ground-skin heat storage for `SlabLand(surface_storage=...)`.
`heat_capacity` is the areal heat capacity in J m⁻² K⁻¹, supplied as a number or a
horizontal field. The bulk land heat capacity describes the soil below this skin.
The initial `temperature` defaults to 300 K.

```jldoctest
using NumericalEarth

PrognosticSkin()

# output
PrognosticSkin(C=100000.0)
```
"""
struct PrognosticSkin{C, P, N, B}
    heat_capacity :: C
    prognostic :: P
    tendencies :: N
    boundary_condition :: B
end

PrognosticSkin(FT::Type = Oceananigans.defaults.FloatType; heat_capacity=1e5, temperature=300) =
    PrognosticSkin(convert_if_number(FT, heat_capacity), (; temperature=convert_if_number(FT, temperature)), nothing, nothing)

const SurfaceStorage = Union{PrognosticCanopyAir, PrognosticSkin}

materialize_surface_storage(::Nothing, grid) = nothing
function materialize_surface_storage(storage::SurfaceStorage, grid)
    names = keys(storage.prognostic)
    prognostic = map(storage.prognostic) do initial
        field = Field{Center, Center, Nothing}(grid)
        Oceananigans.set!(field, initial)
        return field
    end
    tendencies = NamedTuple{names}(map(_ -> Field{Center, Center, Nothing}(grid), names))
    return materialize_surface_storage(storage, prognostic, tendencies)
end

materialize_surface_storage(s::PrognosticCanopyAir, prognostic, tendencies) =
    PrognosticCanopyAir(s.layer_depth, prognostic, tendencies, nothing)
materialize_surface_storage(s::PrognosticSkin, prognostic, tendencies) =
    PrognosticSkin(s.heat_capacity, prognostic, tendencies, nothing)

function EarthSystemModels.materialize_land_interface(land::SlabLand, interfaces, atmosphere, radiation)
    (isnothing(land.surface_storage) || isnothing(interfaces.atmosphere_land_interface)) && return land
    boundary = LandInterfaceBoundaryCondition(land, interfaces, atmosphere, radiation)
    storage = land.surface_storage
    storage = @set storage.boundary_condition = boundary
    return SlabLand{eltype(land)}(land.grid, land.clock, land.temperature, land.water_storage, land.saturation,
                                 land.fluxes, land.diagnostics, land.prognostic, land.energy, land.hydrology, storage)
end

step_surface_storage!(::Nothing, land, Δt) = nothing
function step_surface_storage!(storage::SurfaceStorage, land, Δt)
    boundary = storage.boundary_condition
    isnothing(boundary) && return nothing
    grid = land.grid
    arch = architecture(grid)
    time = land.clock.time
    leaf_area_index = canopy_leaf_area_index(specific_humidity_formulation(boundary.interface.formulation.state))
    vegetation, time_interpolator = kernel_surface_field(leaf_area_index, arch, time)
    launch!(arch, grid, :xy, _step_surface_storage!, storage.prognostic, storage.tendencies,
            boundary, vegetation, time_interpolator, grid, time, Δt)
    map(fill_halo_regions!, values(storage.prognostic))
    EarthSystemModels.update_net_fluxes!(land, boundary.interface, boundary.atmosphere_state)
    return nothing
end

@inline function trial_surface_problem(problem, T, q)
    storage = merge(problem.land.surface_storage, (; temperature=T, specific_humidity=q))
    land = merge(problem.land, (; surface_storage=storage))
    interface = rebuild_interface_state(problem.interface, problem.interface.fluxes, T, q)
    return merge(problem, (; interface, land))
end

@inline function surface_tendency(t::EnergyBalanceTemperature, problem, properties)
    Ψ = evaluate_land_interface(problem, properties)
    F, _ = skin_energy_imbalance(Ψ.temperature, t, Ψ,
                                 problem.atmosphere, problem.land, problem.radiation, problem.formulation, properties)
    return F / convert(typeof(F), problem.land.surface_storage.heat_capacity), zero(F)
end

@inline function surface_tendency(c::CanopyInterface, problem, properties)
    Ψ = evaluate_land_interface(problem, properties)
    solution = canopy_air_space_solve(c, Ψ, problem.atmosphere, problem.land, problem.radiation, properties)
    return solution.Σgᵀ * (solution.equilibrium_temperature - Ψ.temperature) / solution.Cᵀ,
           solution.Σgᵛ * (solution.equilibrium_humidity - Ψ.specific_humidity) / solution.Cᵛ
end

@kernel function _step_surface_storage!(prognostic, tendencies, boundary, vegetation, time_interpolator,
                                      grid, time, Δt)
    i, j = @index(Global, NTuple)
    interface = boundary.interface
    properties = boundary.atmosphere_properties
    problem = land_interface_problem(i, j, grid, Time(time), interface.formulation, boundary.land_state,
                                     vegetation, time_interpolator, boundary.atmosphere_state, properties,
                                     boundary.radiation_properties, boundary.radiation_state)
    state = temperature_formulation(problem.formulation.state)
    Tⁿ, qⁿ = problem.interface.temperature, problem.interface.specific_humidity
    T, q = Tⁿ, qⁿ
    FT = typeof(T)
    Δt = convert(FT, Δt)
    δT = sqrt(eps(FT)) * Tⁿ
    δq = sqrt(eps(FT)) * max(qⁿ, convert(FT, 1e-3))

    # Backward Euler: solve x − xⁿ − Δt f(x) = 0 at fixed atmospheric and bulk-land states.
    for _ in 1:8
        fᵀ, fᵛ = surface_tendency(state, trial_surface_problem(problem, T, q), properties)
        fᵀ⁺, fᵛ⁺ = surface_tendency(state, trial_surface_problem(problem, T + δT, q), properties)
        fᵀᵠ, fᵛᵠ = surface_tendency(state, trial_surface_problem(problem, T, q + δq), properties)
        A = 1 - Δt * (fᵀ⁺ - fᵀ) / δT
        B = -Δt * (fᵀᵠ - fᵀ) / δq
        C = -Δt * (fᵛ⁺ - fᵛ) / δT
        D = 1 - Δt * (fᵛᵠ - fᵛ) / δq
        Rᵀ = T - Tⁿ - Δt * fᵀ
        Rᵛ = q - qⁿ - Δt * fᵛ
        determinant = A * D - B * C
        T -= clamp((D * Rᵀ - B * Rᵛ) / determinant, -25, 25)
        q = max(0, q - (A * Rᵛ - C * Rᵀ) / determinant)
    end

    problem = trial_surface_problem(problem, T, q)
    store_land_interface!(interface.fluxes, interface.temperature, interface.specific_humidity,
                          tendencies, i, j, problem, properties)
    maybe_write!(prognostic.temperature, i, j, T)
    maybe_write!(get(prognostic, :specific_humidity, nothing), i, j, q)
end

Oceananigans.prognostic_state(storage::SurfaceStorage) = map(prognostic_state, storage.prognostic)

function Oceananigans.restore_prognostic_state!(storage::SurfaceStorage, state)
    for name in keys(storage.prognostic)
        restore_prognostic_state!(storage.prognostic[name], state[name])
    end
    return storage
end

Base.summary(s::PrognosticCanopyAir) = string("PrognosticCanopyAir(hᵃᶜᵃⁿ=", prettysummary(s.layer_depth), ")")
Base.summary(s::PrognosticSkin) = string("PrognosticSkin(C=", prettysummary(s.heat_capacity), ")")
Base.show(io::IO, s::SurfaceStorage) = print(io, summary(s))

Adapt.@adapt_structure PrognosticCanopyAir
Adapt.@adapt_structure PrognosticSkin

InterfaceComputations.land_surface_state(land::SlabLand) = surface_storage_state(land.surface_storage)

surface_storage_state(::Nothing) = (; state=nothing, tendencies=(;))
surface_storage_state(s::PrognosticCanopyAir) =
    (; state=merge(s.prognostic, (; s.layer_depth)), tendencies=s.tendencies)
surface_storage_state(s::PrognosticSkin) =
    (; state=merge(s.prognostic, (; s.heat_capacity)), tendencies=s.tendencies)

function InterfaceComputations.validate_surface_storage(land::SlabLand, state)
    storage = land.surface_storage
    compatible = isnothing(storage) ||
                 (storage isa PrognosticCanopyAir && state isa CanopyInterface) ||
                 (storage isa PrognosticSkin && state isa NamedTuple && state.temperature isa EnergyBalanceTemperature)
    compatible || throw(ArgumentError("$(summary(storage)) is incompatible with the interface state formulation."))
    return nothing
end

soil_hydrology(hydrology) = hydrology
soil_hydrology(hydrology::InterceptingHydrology) = hydrology.soil

# The plant-stress endpoints are fixed by the soil's retention curve, so they are evaluated once.
(s::InterfaceComputations.SaturationAtSuction)(land::SlabLand) =
    saturation_at_suction(soil_hydrology(land.hydrology), s.suction_head, land.grid)

function saturation_at_suction(hydrology::VariablySaturatedHydrology, suction_head, grid)
    saturation = KernelFunctionOperation{Center, Center, Nothing}(effective_saturation, grid, hydrology.retention_curve,
                                                                  convert(eltype(grid), suction_head))
    return compute!(Field(saturation))
end

saturation_at_suction(hydrology, suction_head, grid) =
    throw(ArgumentError("PlantAvailableWaterStress requires soil water retention physics."))

function (d::InterfaceComputations.SoilVaporDiffusivity)(land::SlabLand)
    vapor = d.vapor_exchange
    return soil_vapor_diffusivity(soil_hydrology(land.hydrology), vapor.tortuosity, vapor, land.saturation, land.grid)
end

soil_vapor_diffusivity(hydrology, ::ConstantTortuosity, vapor, saturation, grid) = vapor.molecular_diffusivity

function soil_vapor_diffusivity(hydrology::VariablySaturatedHydrology, ::PowerLawTortuosity, vapor, saturation, grid)
    return KernelFunctionOperation{Center, Center, Nothing}(surface_vapor_diffusivity, grid,
        saturation, hydrology.porosity, hydrology.residual_liquid_fraction, vapor)
end

@inline function surface_vapor_diffusivity(i, j, k, grid, saturation, porosity, residual, vapor)
    FT = eltype(grid)
    ν = convert(FT, stateindex(porosity, i, j, 1))
    θʳ = convert(FT, stateindex(residual, i, j, 1))
    θˡ = θʳ + saturation[i, j, 1] * (ν - θʳ)
    return effective_vapor_diffusivity(vapor, ν, θˡ)
end
