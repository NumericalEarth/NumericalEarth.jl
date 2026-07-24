using ..EarthSystemModels: EarthSystemModel, reference_density, heat_capacity

struct MeridionalFluxMethod end
struct TendencyMethod end
"""
    meridional_heat_transport(simulation::Simulation, method = TendencyMethod();
                              destination_grid = nothing)
    meridional_heat_transport(simulation::Simulation, MeridionalFluxMethod();
                              reference_temperature = 0)

Return the meridional heat transport for a coupled `simulation` using either direct
meridional heat fluxes or the ocean heat-content tendency. A `BudgetComputation` callback
must be registered on `simulation` before using the default `TendencyMethod()`.

!!! warning "The flux method only works on LatitudeLongitudeGrid"

    `MeridionalFluxMethod()` currently supports only `LatitudeLongitudeGrid`.
    For an `OrthogonalSphericalShellGrid`, use `TendencyMethod()` and provide a
    `LatitudeLongitudeGrid` via `destination_grid`.

Arguments
=========

* `simulation`: A `Simulation` of an `EarthSystemModel`. For `TendencyMethod()`, the
  simulation must contain exactly one registered `BudgetComputation` callback.

* The method for the computation. Available options are: `TendencyMethod()` (default)
  and `MeridionalFluxMethod()`.

  `MeridionalFluxMethod()` computes the meridional heat transport directly by summing
  the meridional heat flux; `TendencyMethod()` computes the meridional heat
  transport indirectly using the ocean heat content.

  1. For `MeridionalFluxMethod()`, the meridional heat transport is computed via:

     ```math
     \\mathrm{MHT} ≡ ρᵒᶜ cᵒᶜ ∫  v (T - T_{\\rm ref}) \\, \\mathrm{d}x \\, \\mathrm{d}z
     ```

     Above, ``T_{\\rm ref}`` is a reference temperature and ``ρᵒᶜ`` and ``cᵒᶜ`` are the
     ocean reference density and specific heat capacity respectively.

  2. For `TendencyMethod()` we have:

     Let ``T`` be three-dimensional (potential) temperature, ``ρᵒᶜ`` the ocean reference
     density, ``cᵒᶜ`` the specific heat capacity, ``H`` the resting depth, and ``η`` the
     free-surface elevation.

     The column heat content per unit horizontal area (units of J m⁻²) is:

     ```math
     ℋ = ρᵒᶜ cᵒᶜ ∫_{-H}^{η} T \\, \\mathrm{d}z
     ```

     and its time-derivative is a heat flux, ``𝒬 ≡ ∂ℋ/∂t``.

     The vertically-integrated heat budget is

     ```math
     𝒬 = \\frac{∂ℋ}{∂t} = - \\boldsymbol{∇}_h \\cdot \\boldsymbol{F}_h - 𝒬ᵃᵒ_{\\rm net} + 𝒬_{\\rm rad} + ℛ
     ```

     where

     * ``\\boldsymbol{F}_h`` is the depth-integrated horizontal heat flux vector (units W m⁻¹),
       that includes advection and any parameterized lateral/eddy fluxes,
     * ``𝒬ᵃᵒ_{\\rm net}`` is the [net ocean surface heat flux](@ref NumericalEarth.Diagnostics.net_ocean_heat_flux)
       (units W m⁻²), and
     * ``𝒬_{\\rm rad}`` is the vertically integrated penetrating-radiation source
       (units W m⁻²), and
     * ``ℛ`` is the residual sources/sinks and non-closed terms (e.g. numerics, unaccounted
       physics, mass/volume effects).

     The total ocean heat content (OHC) South of latitude ``φ`` is:

     ```math
     \\mathrm{OHC}_S(φ, t) ≡ ∫_{A(φ)} ℋ \\, \\mathrm{d}A
     ```

     where ``A(φ)`` is the ocean area South of latitude ``φ``.

     Integrating the vertically-integrated heat budget over ``A(φ)`` and using the divergence
     theorem we get

     ```math
     \\frac{\\mathrm{d}}{\\mathrm{d}t} \\, \\mathrm{OHC}_S(φ, t) =
        - ∮_{∂A(φ)} \\boldsymbol{F}_h \\cdot \\hat{\\boldsymbol{n}} \\, \\mathrm{d}ℓ
        - ∫_{A(φ)} 𝒬ᵃᵒ_{\\rm net} \\, \\mathrm{d}A
        + ∫_{A(φ)} 𝒬_{\\rm rad} \\, \\mathrm{d}A
        + ∫_{A(φ)} ℛ \\, \\mathrm{d}A
     ```

     where ``∂A(φ)`` is the boundary of ``A(φ)``, i.e., the circle at latitude ``φ``.

     The northward meridional heat transport across latitude ``φ`` is

     ```math
     \\mathrm{MHT}(φ, t) ≡ ∮_{\\mathrm{lat}=φ} \\boldsymbol{F}_h \\cdot \\hat{\\boldsymbol{n}} \\, \\mathrm{d}ℓ
     ```

     with the understanding that ``\\mathrm{MHT} > 0`` implies northward heat transport.

     Ignoring the residual ``ℛ``, the OHC-based diagnostic relation is

     ```math
     \\begin{align*}
         \\mathrm{MHT} & = - ∫_{A(φ)} 𝒬ᵃᵒ_{\\rm net} \\, \\mathrm{d}A
                           + ∫_{A(φ)} 𝒬_{\\rm rad} \\, \\mathrm{d}A
                           - \\frac{\\mathrm{d}}{\\mathrm{d}t} \\, \\mathrm{OHC}_S \\\\
                       & = - ∫_{A(φ)} \\left( 𝒬ᵃᵒ_{\\rm net} + 𝒬 - 𝒬_{\\rm rad} \\right) \\, \\mathrm{d}A
     \\end{align*}
     ```

     The tendency is evaluated as the finite difference in column heat content over
     the most recently completed coupled timestep. Surface and radiative fluxes are
     retained from the beginning of that timestep, so all budget terms refer to the
     same coupling interval. The completed budget is stored in a BudgetComputation callback after every
     coupled timestep. Oceananigans' RegriddedOperation recomputes the remapping whenever an
     output writer materializes the diagnostic. Thus `TimeInterval(interval)` writes
     the most recently completed timestep, while `AveragedTimeInterval(interval)`
     samples and time-averages the completed budget from every timestep when using the
     default `stride=1`.

Keyword Arguments
=================

* `reference_temperature`: The reference temperature (in ᵒC) used for `MeridionalFluxMethod()`; default: 0 ᵒC.

* `destination_grid`: A `LatitudeLongitudeGrid` onto which the two-dimensional column
  heat budget is conservatively regridded before zonal integration. This is required
  when using `TendencyMethod()` with an `OrthogonalSphericalShellGrid`.

  !!! info "Reference temperature"

      The reference temperature is only relevant when we compute the meridional heat transport over a section
      where there is a net volume transport. If we are computing the diagnostic globally, i.e., around a whole
      latitude circle, then by necessity there is no net volume transport and thus the reference temperature
      value is irrelevant. Section-averaged transport could also be considered as a reference temperature to
      remove residual barotropic volume fluxes in basin-scale/regional analyses where a net volume transport
      is present.

Example
=======

```julia
using ..NumericalEarth
using Oceananigans

grid = RectilinearGrid(size = (4, 5, 2), extent = (1, 1, 1),
                       topology = (Periodic, Bounded, Bounded))

ocean = ocean_simulation(grid;
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         closure = nothing,
                         coriolis = nothing)

sea_ice = sea_ice_simulation(grid, ocean)

atmosphere = PrescribedAtmosphere(grid, [0.0])
radiation = PrescribedRadiation(grid)

esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(esm; Δt=1)
budget = BudgetComputation(:temperature, esm)
add_callback!(simulation, budget)

mht = meridional_heat_transport(simulation)
```
"""
function meridional_heat_transport(simulation::Simulation,
                                   ::MeridionalFluxMethod;
                                   reference_temperature=0)
    esm = simulation.model
    esm isa EarthSystemModel ||
        throw(ArgumentError("Meridional heat transport requires a Simulation of an EarthSystemModel."))

    grid = underlying_grid(esm.ocean.model.grid)
    validate_meridional_flux_grid(grid)

    FT = eltype(esm)
    reference_temperature = convert(FT, reference_temperature)
    return meridional_heat_transport_via_meridional_heat_flux(esm; reference_temperature)
end

function meridional_heat_transport(simulation::Simulation,
                                   ::TendencyMethod=TendencyMethod();
                                   destination_grid=nothing)
    budgets = [callback.func for callback in values(simulation.callbacks)
               if callback.func isa BudgetComputation]

    if isempty(budgets)
        throw(ArgumentError("TendencyMethod() requires a BudgetComputation callback. " *
                            "Create a budget and register it with " *
                            "add_callback!(simulation, budget) first."))
    elseif length(budgets) > 1
        throw(ArgumentError("TendencyMethod() requires exactly one BudgetComputation callback, " *
                            "but the simulation has $(length(budgets))."))
    end

    budget = only(budgets)
    grid = underlying_grid(budget.residual.grid)
    validate_tendency_destination(grid, destination_grid)
    return meridional_heat_transport_via_ocean_heat_content(budget; destination_grid)
end

function meridional_heat_transport(::BudgetComputation, ::TendencyMethod=TendencyMethod(); kwargs...)
    message = """
    meridional_heat_transport does not accept a BudgetComputation directly.

    Add the budget to the simulation as a callback first, then pass the
    simulation to meridional_heat_transport. For example:

        budget = BudgetComputation(:temperature, esm)
        add_callback!(simulation, budget)
        mht = meridional_heat_transport(simulation; destination_grid = latlon_grid)

    BudgetComputation stores heat-budget history while the simulation runs.
    The MHT diagnostic needs the simulation so it can find that registered
    callback and use the completed budget at the right time.
    """

    throw(ArgumentError(message))
end

function meridional_heat_transport(::Simulation, method; kwargs...)
    throw(ArgumentError(string("Unknown method ", method, "; choose either MeridionalFluxMethod() or TendencyMethod().")))
end

underlying_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid
underlying_grid(grid) = grid

validate_meridional_flux_grid(::OrthogonalSphericalShellGrid) =
    throw(ArgumentError("MeridionalFluxMethod() diagnostic does not work on OrthogonalSphericalShellGrid at the moment. Use TendencyMethod() instead."))

validate_meridional_flux_grid(grid) = nothing

validate_tendency_destination(::OrthogonalSphericalShellGrid, ::Nothing) =
    throw(ArgumentError("TendencyMethod() on an OrthogonalSphericalShellGrid requires a `destination_grid`."))

validate_tendency_destination(grid, ::Nothing) = nothing
validate_tendency_destination(grid, destination_grid) =
    validate_tendency_destination_grid(underlying_grid(destination_grid))

validate_tendency_destination_grid(::LatitudeLongitudeGrid) = nothing
validate_tendency_destination_grid(::RectilinearGrid) = nothing

validate_tendency_destination_grid(destination_grid) =
    throw(ArgumentError("The `destination_grid` must be a LatitudeLongitudeGrid, " *
                        "a RectilinearGrid, or an ImmersedBoundaryGrid wrapped around one of those grids."))

function meridional_heat_transport_via_meridional_heat_flux(esm; reference_temperature)
    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)
    T = esm.ocean.model.tracers.T
    v = esm.ocean.model.velocities.v
    MHT = Integral(ρᵒᶜ * cᵒᶜ * v * (T - reference_temperature), dims=(1, 3))
    return MHT
end

function meridional_heat_transport_via_ocean_heat_content(budget; destination_grid=nothing)
    column_budget = budget.residual

    if destination_grid !== nothing
        column_budget = RegriddedOperation(column_budget, destination_grid)
    end

    zonal_budget = Field(Integral(column_budget, dims=1))
    MHT = CumulativeIntegral(-zonal_budget, dims=2) |> Field
    return MHT
end
