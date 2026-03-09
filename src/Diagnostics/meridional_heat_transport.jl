using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, QuasiAdamsBashforth2TimeStepper
using ..EarthSystemModels: EarthSystemModel, reference_density, heat_capacity

import ..EarthSystemModels: checkpoint_auxiliary_state, restore_auxiliary_state!

struct OceanHeatContentTendencyMethod end
struct MeridionalHeatFluxMethod end

"""
    meridional_heat_transport(esm::EarthSystemModel, MeridionalHeatFluxMethod();
                              reference_temperature = 0)

Return the meridional heat transport for the coupled `esm::EarthSystemModel` using either
two methods: either by directly computing the meridional heat flux or indirectly using
the total ocean heat content and the ocean heat uptake.

!!! warning "Only works on LatitudeLongitudeGrid"

    The `meridional_heat_transport` diagnostic currently is only supported only on
    `LongitudeLatitudeGrid`s.

Arguments
=========

* `esm`: An EarthSystemModel.

* The method for the computation. Available options are: `MeridionalHeatFluxMethod()` (default)
  and `OceanHeatContentTendencyMethod()`.

  `MeridionalHeatFluxMethod()` computes the meridional heat transport directly by summing
  the meridional heat flux; `OceanHeatContentTendencyMethod()` computes the meridional heat
  transport indirectly using the ocean heat content.

  1. For `MeridionalHeatFluxMethod()`, the meridional heat transport is computed via:

     ```math
     \\mathrm{MHT} ≡ ρᵒᶜ cᵒᶜ ∫  v (T - T_{\\rm ref}) \\, \\mathrm{d}x \\, \\mathrm{d}z
     ```

     Above, ``T_{\\rm ref}`` is a reference temperature and ``ρᵒᶜ`` and ``cᵒᶜ`` are the
     ocean reference density and heat capacity respectively.

  2. For `OceanHeatContentTendencyMethod()` we have:

     Let ``T`` be three-dimensional (potential) temperature, ``ρᵒᶜ`` a reference density,
     ``cᵒᶜ`` the heat capacity, ``H`` the resting depth, and ``η`` the free-surface elevation.

     The column heat content per unit horizontal area (units of J m⁻²) is:

     ```math
     ℋ = ρᵒᶜ cᵒᶜ ∫_{-H}^{η} T \\, \\mathrm{d}z
     ```

     The vertically-integrated heat budget is

     ```math
     \\frac{∂ℋ}{∂t} = - \\boldsymbol{∇}_h \\cdot \\boldsymbol{F}_h - 𝒬_{\\rm net} + ℛ
     ```

     where

     * ``\\boldsymbol{F}_h`` is the depth-integrated horizontal heat flux vector (units W m⁻¹),
       that includes advection and any parameterized lateral/eddy fluxes,
     * ``𝒬_{\\rm net}`` is the [net ocean surface heat flux](@ref NumericalEarth.net_ocean_heat_flux)
       (units W m⁻²), and
     * ``ℛ`` is the residual sources/sinks and non-closed terms (e.g. numerics, unaccounted
       physics, mass/volume effects).

     The total ocean heat content (OHC) South of latitude ``φ`` is:

     ```math
     \\mathrm{OHC}_S(φ, t) ≡ ∫_{A(φ)} ℋ \\, \\mathrm{d}A
     ```

     where ``A(φ)`` is the ocean area South of latitude ``φ``.

     Integrating the vertically-integrated budget over ``A(φ)`` and using the divergence
     theorem we get

     ```math
     \\frac{d}{dt} \\, \\mathrm{OHC}_S(φ, t) =
         - ∮_{∂A(φ)} \\boldsymbol{F}_h \\cdot \\hat{\\boldsymbol{n}} \\, \\mathrm{d}ℓ
         - ∫_{A(φ)} 𝒬_{\\rm net} \\, \\mathrm{d}A
         + ∫_{A(φ)} ℛ \\, \\mathrm{d}A
     ```

     The northward meridional heat transport across latitude ``φ`` is

     ```math
     \\mathrm{MHT}(φ, t) ≡ ∮_{\\mathrm{lat}=φ} \\boldsymbol{F}_h \\cdot \\hat{\\boldsymbol{n}} \\, \\mathrm{d}ℓ
     ```

     with the sign convention that ``\\mathrm{MHT} > 0`` is northward.

     Ignoring the residual ``ℛ``, the OHC-based diagnostic relation is

     ```math
     \\mathrm{MHT} = - ∫_{A(φ)} 𝒬_{\\rm net} \\, \\mathrm{d}A - \\frac{\\mathrm{d}}{\\mathrm{d}t} \\, \\mathrm{OHC}_S
     ```

Keyword Arguments
=================

* `reference_temperature`: The reference temperature (in ᵒC) used for `MeridionalHeatFluxMethod()`; default: 0 ᵒC.
"""
function meridional_heat_transport(esm::EarthSystemModel, method=MeridionalHeatFluxMethod(); reference_temperature=0)

    esm.ocean.model.grid.underlying_grid isa OrthogonalSphericalShellGrid &&
        throw(ArgumentError("meridional_heat_transport diagnostic does not work on OrthogonalSphericalShellGrid at the moment; use LatitudeLongitudeGrid."))

    if method isa MeridionalHeatFluxMethod
        FT = eltype(esm)
        reference_temperature = convert(FT, reference_temperature)
        return meridional_heat_transport_via_meridional_heat_flux(esm; reference_temperature)
    elseif method isa OceanHeatContentTendencyMethod
        return meridional_heat_transport_via_ocean_heat_content(esm)
    else
        throw(ArgumentError("Unknown method $(method); choose either MeridionalHeatFluxMethod() or OceanHeatContentTendencyMethod()."))
    end
end

function meridional_heat_transport_via_meridional_heat_flux(esm; reference_temperature)
    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)
    T = esm.ocean.model.tracers.T
    v = esm.ocean.model.velocities.v
    MHT = Integral(ρᵒᶜ * cᵒᶜ * v * (T - reference_temperature), dims=(1, 3))
    return MHT
end

function meridional_heat_transport_via_ocean_heat_content(esm)
    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)
    ∂T_∂t = temperature_evolution_tendency(esm.ocean.model.timestepper)
    heat_flux = net_ocean_heat_flux(esm) |> Field

    ∂ℋ_∂t = Integral(ρᵒᶜ * cᵒᶜ * ∂T_∂t, dims=3) |> Field
    ∫sum_dx = Integral(heat_flux + ∂ℋ_∂t, dims=1) |> Field
    MHT = CumulativeIntegral(- ∫sum_dx, dims=2)

    return MHT
end

temperature_evolution_tendency(timestepper::SplitRungeKuttaTimeStepper) = timestepper.Gⁿ.T

function temperature_evolution_tendency(timestepper::QuasiAdamsBashforth2TimeStepper)
    Gⁿ = timestepper.Gⁿ.T
    G⁻ = timestepper.G⁻.T
    χ  = timestepper.χ
    return (3/2 + χ) * Gⁿ - (1/2 + χ) * G⁻
end
