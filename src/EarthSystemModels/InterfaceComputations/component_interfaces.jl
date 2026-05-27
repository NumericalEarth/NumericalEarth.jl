using KernelAbstractions: @kernel, @index
using Oceananigans: initialize!
using Oceananigans.Architectures: architecture
using Oceananigans.Units: Time
using Oceananigans.Grids: inactive_node, topology
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using Oceananigans.Units: Time

using ..EarthSystemModels: reference_density,
                           heat_capacity,
                           thermodynamics_parameters,
                           ocean_surface_temperature,
                           ocean_surface_salinity

#####
##### Container for organizing information related to fluxes
#####

mutable struct AtmosphereInterface{J, F, ST, P}
    fluxes :: J
    flux_formulation :: F
    temperature :: ST
    properties :: P
end

"""
    SeaIceOceanInterface{J, F, T, S, P}

Container for sea ice-ocean interface data including fluxes, formulation, and interface state.

Fields
======

- `fluxes::J`: `SeaIceOceanFluxes` containing interface_heat, frazil_heat, salt, x_momentum, y_momentum
- `flux_formulation::F`: heat flux formulation (`IceBathHeatFlux` or `ThreeEquationHeatFlux`)
- `temperature::T`: interface temperature field (ocean surface view or computed field)
- `salinity::S`: interface salinity field (ocean surface view or computed field)
"""
mutable struct SeaIceOceanInterface{J, F, T, S}
    fluxes :: J
    flux_formulation :: F
    temperature :: T
    salinity :: S
end

# Utilities to get the computed fluxes
@inline computed_fluxes(interface::AtmosphereInterface)  = interface.fluxes
@inline computed_fluxes(interface::SeaIceOceanInterface) = interface.fluxes

"""
    AtmosphereSurfaceFluxes{F}

Atmosphere↔surface turbulent flux container, shared by the
atmosphere–ocean and atmosphere–land interfaces (both produce the same
8 quantities). Atmosphere–sea-ice uses a smaller container
([`AtmosphereSeaIceFluxes`](@ref)) because it does not emit the
characteristic scales.
"""
struct AtmosphereSurfaceFluxes{F}
    latent_heat       :: F
    sensible_heat     :: F
    water_vapor       :: F
    x_momentum        :: F
    y_momentum        :: F
    friction_velocity :: F
    temperature_scale :: F
    water_vapor_scale :: F
end

function AtmosphereSurfaceFluxes(grid)
    F = Field{Center, Center, Nothing}
    return AtmosphereSurfaceFluxes(F(grid), F(grid), F(grid),
                                   F(grid), F(grid), F(grid),
                                   F(grid), F(grid))
end

AtmosphereSurfaceFluxes(::Nothing) = AtmosphereSurfaceFluxes(ntuple(_ -> ZeroField(), 8)...)

Adapt.adapt_structure(to, fluxes::AtmosphereSurfaceFluxes) =
    AtmosphereSurfaceFluxes(Adapt.adapt(to, fluxes.latent_heat),
                            Adapt.adapt(to, fluxes.sensible_heat),
                            Adapt.adapt(to, fluxes.water_vapor),
                            Adapt.adapt(to, fluxes.x_momentum),
                            Adapt.adapt(to, fluxes.y_momentum),
                            Adapt.adapt(to, fluxes.friction_velocity),
                            Adapt.adapt(to, fluxes.temperature_scale),
                            Adapt.adapt(to, fluxes.water_vapor_scale))

Oceananigans.Architectures.on_architecture(arch, fluxes::AtmosphereSurfaceFluxes) =
    AtmosphereSurfaceFluxes(on_architecture(arch, fluxes.latent_heat),
                            on_architecture(arch, fluxes.sensible_heat),
                            on_architecture(arch, fluxes.water_vapor),
                            on_architecture(arch, fluxes.x_momentum),
                            on_architecture(arch, fluxes.y_momentum),
                            on_architecture(arch, fluxes.friction_velocity),
                            on_architecture(arch, fluxes.temperature_scale),
                            on_architecture(arch, fluxes.water_vapor_scale))

struct AtmosphereSeaIceFluxes{F}
    latent_heat   :: F
    sensible_heat :: F
    water_vapor   :: F
    x_momentum    :: F
    y_momentum    :: F
end

function AtmosphereSeaIceFluxes(grid)
    F = Field{Center, Center, Nothing}
    return AtmosphereSeaIceFluxes(F(grid), F(grid), F(grid), F(grid), F(grid))
end

AtmosphereSeaIceFluxes(::Nothing) = AtmosphereSeaIceFluxes(ntuple(_ -> ZeroField(), 5)...)

Adapt.adapt_structure(to, fluxes::AtmosphereSeaIceFluxes) =
    AtmosphereSeaIceFluxes(Adapt.adapt(to, fluxes.latent_heat),
                           Adapt.adapt(to, fluxes.sensible_heat),
                           Adapt.adapt(to, fluxes.water_vapor),
                           Adapt.adapt(to, fluxes.x_momentum),
                           Adapt.adapt(to, fluxes.y_momentum))

Oceananigans.Architectures.on_architecture(arch, fluxes::AtmosphereSeaIceFluxes) =
    AtmosphereSeaIceFluxes(on_architecture(arch, fluxes.latent_heat),
                           on_architecture(arch, fluxes.sensible_heat),
                           on_architecture(arch, fluxes.water_vapor),
                           on_architecture(arch, fluxes.x_momentum),
                           on_architecture(arch, fluxes.y_momentum))

struct SeaIceOceanFluxes{C, FX, FY}
    interface_heat :: C
    frazil_heat    :: C
    salt           :: C
    x_momentum     :: FX
    y_momentum     :: FY
end

function SeaIceOceanFluxes(grid)
    C  = Field{Center, Center, Nothing}
    return SeaIceOceanFluxes(C(grid), C(grid), C(grid),
                             Field{Face, Center, Nothing}(grid),
                             Field{Center, Face, Nothing}(grid))
end

SeaIceOceanFluxes(::Nothing) = SeaIceOceanFluxes(ntuple(_ -> ZeroField(), 5)...)

Adapt.adapt_structure(to, fluxes::SeaIceOceanFluxes) =
    SeaIceOceanFluxes(Adapt.adapt(to, fluxes.interface_heat),
                      Adapt.adapt(to, fluxes.frazil_heat),
                      Adapt.adapt(to, fluxes.salt),
                      Adapt.adapt(to, fluxes.x_momentum),
                      Adapt.adapt(to, fluxes.y_momentum))

Oceananigans.Architectures.on_architecture(arch, fluxes::SeaIceOceanFluxes) =
    SeaIceOceanFluxes(on_architecture(arch, fluxes.interface_heat),
                      on_architecture(arch, fluxes.frazil_heat),
                      on_architecture(arch, fluxes.salt),
                      on_architecture(arch, fluxes.x_momentum),
                      on_architecture(arch, fluxes.y_momentum))

# ZeroFluxes is returned by computed_fluxes(::Nothing) for absent interfaces.
# It contains the union of all flux field names across interface types.
struct ZeroFluxes{Z}
    # Atmosphere-ocean and atmosphere-sea-ice flux fields (turbulent only;
    # radiative diagnostic fields live on the radiation component)
    latent_heat           :: Z
    sensible_heat         :: Z
    water_vapor           :: Z
    x_momentum            :: Z
    y_momentum            :: Z
    friction_velocity     :: Z
    temperature_scale     :: Z
    water_vapor_scale     :: Z
    # Sea ice-ocean flux fields
    interface_heat        :: Z
    frazil_heat           :: Z
    salt                  :: Z
end

ZeroFluxes() = ZeroFluxes(ntuple(_ -> ZeroField(), 11)...)

@inline computed_fluxes(::Nothing) = ZeroFluxes()

mutable struct ComponentInterfaces{AO, ASI, SIO, AL, C, AP, OP, SIP, EX, P}
    atmosphere_ocean_interface :: AO
    atmosphere_sea_ice_interface :: ASI
    sea_ice_ocean_interface :: SIO
    atmosphere_land_interface :: AL
    atmosphere_properties :: AP
    ocean_properties :: OP
    sea_ice_properties :: SIP
    exchanger :: EX
    net_fluxes :: C
    properties :: P
end

using ..EarthSystemModels: DegreesCelsius, temperature_units, exchange_grid,
                           celsius_to_kelvin, convert_to_kelvin, convert_from_kelvin

Base.summary(crf::ComponentInterfaces) = "ComponentInterfaces"
Base.show(io::IO, crf::ComponentInterfaces) = print(io, summary(crf))

#####
##### Atmosphere-Ocean Interface
#####

atmosphere_ocean_interface(grid, ::Nothing,   ocean,    args...) = nothing
atmosphere_ocean_interface(grid, ::Nothing,  ::Nothing, args...) = nothing
atmosphere_ocean_interface(grid, atmosphere, ::Nothing, args...) = nothing

function atmosphere_ocean_interface(grid,
                                    atmosphere,
                                    ocean,
                                    ao_flux_formulation,
                                    temperature_formulation,
                                    velocity_formulation,
                                    specific_humidity_formulation)

    ao_fluxes = AtmosphereSurfaceFluxes(grid)

    ao_properties = InterfaceProperties(specific_humidity_formulation,
                                        temperature_formulation,
                                        velocity_formulation)

    interface_temperature = Field{Center, Center, Nothing}(grid)

    return AtmosphereInterface(ao_fluxes, ao_flux_formulation, interface_temperature, ao_properties)
end

#####
##### Atmosphere-Sea Ice Interface
#####

atmosphere_sea_ice_interface(grid, atmos, ::Nothing,     args...) = nothing
atmosphere_sea_ice_interface(grid, ::Nothing, sea_ice,   args...) = nothing
atmosphere_sea_ice_interface(grid, ::Nothing, ::Nothing, args...) = nothing

function atmosphere_sea_ice_interface(grid,
                                      atmosphere,
                                      sea_ice,
                                      ai_flux_formulation,
                                      temperature_formulation,
                                      velocity_formulation)

    fluxes = AtmosphereSeaIceFluxes(grid)

    phase = AtmosphericThermodynamics.Ice()
    specific_humidity_formulation = ImpureSaturationSpecificHumidity(phase)

    properties = InterfaceProperties(specific_humidity_formulation,
                                     temperature_formulation,
                                     velocity_formulation)

    snow_thermo = sea_ice.model.snow_thermodynamics
    interface_temperature = if isnothing(snow_thermo)
        sea_ice.model.ice_thermodynamics.top_surface_temperature
    else
        snow_thermo.top_surface_temperature
    end

    return AtmosphereInterface(fluxes, ai_flux_formulation, interface_temperature, properties)
end

#####
##### Sea Ice-Ocean Interface
#####

sea_ice_ocean_interface(grid, ::Nothing, ocean,     flux_formulation; kwargs...) = nothing
sea_ice_ocean_interface(grid, ::Nothing, ::Nothing, flux_formulation; kwargs...) = nothing
sea_ice_ocean_interface(grid, sea_ice,   ::Nothing, flux_formulation; kwargs...) = nothing

# Disambiguation
sea_ice_ocean_interface(grid, ::Nothing,     ocean, ::ThreeEquationHeatFlux; kwargs...) = nothing
sea_ice_ocean_interface(grid, sea_ice,   ::Nothing, ::ThreeEquationHeatFlux; kwargs...) = nothing
sea_ice_ocean_interface(grid, ::Nothing, ::Nothing, ::ThreeEquationHeatFlux; kwargs...) = nothing

"""
    sea_ice_ocean_interface(grid, sea_ice, ocean, flux_formulation)

Construct a `SeaIceOceanInterface` with the specified flux formulation.

For `IceBathHeatFlux`, the interface temperature and salinity
point to the ocean surface values. For `ThreeEquationHeatFlux`, dedicated fields are
created to store the computed interface values.

Arguments
=========

- `grid`: the computational grid
- `sea_ice`: sea ice simulation
- `ocean`: ocean simulation
- `flux_formulation`: heat flux formulation (`IceBathHeatFlux` or `ThreeEquationHeatFlux`)
"""
function sea_ice_ocean_interface(grid, sea_ice, ocean, flux_formulation)
    io_fluxes = SeaIceOceanFluxes(grid)

    # For default flux formulations, interface temperature and salinity point to ocean surface
    Tⁱⁿᵗ = ocean_surface_temperature(ocean)
    Sⁱⁿᵗ = ocean_surface_salinity(ocean)

    return SeaIceOceanInterface(io_fluxes, flux_formulation, Tⁱⁿᵗ, Sⁱⁿᵗ)
end

function sea_ice_ocean_interface(grid, sea_ice, ocean, flux_formulation::ThreeEquationHeatFlux)
    io_fluxes = SeaIceOceanFluxes(grid)

    # Interface temperature and salinity are computed fields
    Tⁱⁿᵗ = Field{Center, Center, Nothing}(grid)
    Sⁱⁿᵗ = Field{Center, Center, Nothing}(grid)

    return SeaIceOceanInterface(io_fluxes, flux_formulation, Tⁱⁿᵗ, Sⁱⁿᵗ)
end

#####
##### Component Interfaces
#####

default_ai_temperature(::Nothing) = nothing

function default_ao_specific_humidity(ocean)
    FT    = eltype(ocean)
    phase = AtmosphericThermodynamics.Liquid()
    x_H₂O = convert(FT, 0.98)
    return ImpureSaturationSpecificHumidity(phase, x_H₂O)
end

"""
    ComponentInterfaces(atmosphere, ocean, sea_ice=nothing; kwargs...)

Construct component interfaces for atmosphere-ocean-sea ice coupling.

Keyword Arguments
=================

- `sea_ice_ocean_heat_flux`: formulation for sea ice-ocean heat flux. Options are:
  - `IceBathHeatFlux()`: bulk heat flux with interface at freezing point
  - `ThreeEquationHeatFlux()`: coupled heat/salt/freezing point system (default)

- `radiation`: radiation component. Default: `nothing`.
- `freshwater_density`: reference density of freshwater. Default: `default_freshwater_density`.
- `atmosphere_ocean_fluxes`: flux formulation for atmosphere-ocean interface. Default: `SimilarityTheoryFluxes()`.
- `atmosphere_sea_ice_fluxes`: flux formulation for atmosphere-sea ice interface. Default: `SimilarityTheoryFluxes()`.
- `atmosphere_ocean_interface_temperature`: temperature formulation for atmosphere-ocean interface. Default: `BulkTemperature()`.
- `atmosphere_ocean_interface_specific_humidity`: specific humidity formulation. Default: `default_ao_specific_humidity(ocean)`.
- `atmosphere_sea_ice_interface_temperature`: temperature formulation for atmosphere-sea ice interface. Default: `default_ai_temperature(sea_ice)`.
- `ocean_reference_density`: reference density for the ocean. Default: `reference_density(ocean)`.
- `ocean_heat_capacity`: heat capacity for the ocean. Default: `heat_capacity(ocean)`.
- `ocean_temperature_units`: temperature units for the ocean. Default: `DegreesCelsius()`.
- `sea_ice_temperature_units`: temperature units for sea ice. Default: `DegreesCelsius()`.
- `sea_ice_reference_density`: reference density for sea ice. Default: `reference_density(sea_ice)`.
- `sea_ice_heat_capacity`: heat capacity for sea ice. Default: `heat_capacity(sea_ice)`.
- `gravitational_acceleration`: gravitational acceleration. Default: `default_gravitational_acceleration`.
"""
function ComponentInterfaces(atmosphere, ocean, sea_ice=nothing;
                             radiation = nothing,
                             land = nothing,
                             exchange_grid = exchange_grid(atmosphere, ocean, sea_ice, land),
                             freshwater_density = default_freshwater_density,
                             atmosphere_ocean_fluxes = SimilarityTheoryFluxes(eltype(exchange_grid)),
                             atmosphere_sea_ice_fluxes = atmosphere_sea_ice_similarity_theory(eltype(exchange_grid)),
                             atmosphere_land_fluxes = default_atmosphere_land_fluxes(land, eltype(exchange_grid)),
                             sea_ice_ocean_heat_flux = ThreeEquationHeatFlux(sea_ice),
                             atmosphere_ocean_interface_temperature = BulkTemperature(),
                             atmosphere_ocean_velocity_difference = RelativeVelocity(),
                             atmosphere_ocean_interface_specific_humidity = default_ao_specific_humidity(ocean),
                             atmosphere_sea_ice_interface_temperature = default_ai_temperature(sea_ice),
                             atmosphere_sea_ice_velocity_difference = RelativeVelocity(),
                             atmosphere_land_interface_temperature = BulkTemperature(),
                             atmosphere_land_velocity_difference = RelativeVelocity(),
                             atmosphere_land_interface_specific_humidity = default_al_specific_humidity(land),
                             ocean_reference_density = reference_density(ocean),
                             ocean_heat_capacity = heat_capacity(ocean),
                             ocean_temperature_units = temperature_units(ocean),
                             sea_ice_temperature_units = DegreesCelsius(),
                             sea_ice_reference_density = reference_density(sea_ice),
                             sea_ice_heat_capacity = heat_capacity(sea_ice),
                             gravitational_acceleration = default_gravitational_acceleration,
                             exchanger_correction = nothing)

    FT = eltype(exchange_grid)

    ocean_reference_density    = convert(FT, ocean_reference_density)
    ocean_heat_capacity        = convert(FT, ocean_heat_capacity)
    sea_ice_reference_density  = convert(FT, sea_ice_reference_density)
    sea_ice_heat_capacity      = convert(FT, sea_ice_heat_capacity)
    freshwater_density         = convert(FT, freshwater_density)
    gravitational_acceleration = convert(FT, gravitational_acceleration)

    # Component properties
    atmosphere_properties = thermodynamics_parameters(atmosphere)

    ocean_properties = (reference_density  = ocean_reference_density,
                        heat_capacity      = ocean_heat_capacity,
                        freshwater_density = freshwater_density,
                        temperature_units  = ocean_temperature_units)

    # Only build sea_ice_properties if sea_ice is an actual Simulation with a model
    if sea_ice isa Simulation
        sea_ice_properties = (reference_density  = sea_ice_reference_density,
                              heat_capacity      = sea_ice_heat_capacity,
                              freshwater_density = freshwater_density,
                              liquidus           = sea_ice.model.phase_transitions.liquidus,
                              temperature_units  = sea_ice_temperature_units)
    else
        sea_ice_properties = nothing
    end

    # Component interfaces
    ao_interface = atmosphere_ocean_interface(exchange_grid,
                                              atmosphere,
                                              ocean,
                                              atmosphere_ocean_fluxes,
                                              atmosphere_ocean_interface_temperature,
                                              atmosphere_ocean_velocity_difference,
                                              atmosphere_ocean_interface_specific_humidity)

    io_interface = sea_ice_ocean_interface(exchange_grid, sea_ice, ocean, sea_ice_ocean_heat_flux)

    ai_interface = atmosphere_sea_ice_interface(exchange_grid,
                                                atmosphere,
                                                sea_ice,
                                                atmosphere_sea_ice_fluxes,
                                                atmosphere_sea_ice_interface_temperature,
                                                atmosphere_sea_ice_velocity_difference)

    al_interface = atmosphere_land_interface(exchange_grid,
                                             atmosphere,
                                             land,
                                             atmosphere_land_fluxes,
                                             atmosphere_land_interface_temperature,
                                             atmosphere_land_velocity_difference,
                                             atmosphere_land_interface_specific_humidity)
    # Total interface fluxes
    total_fluxes = (ocean      = net_fluxes(ocean),
                    sea_ice    = net_fluxes(sea_ice),
                    atmosphere = net_fluxes(atmosphere))

    exchanger = StateExchanger(exchange_grid, radiation, atmosphere, land, ocean, sea_ice;
                               correction = exchanger_correction)

    properties = (; gravitational_acceleration)

    return ComponentInterfaces(ao_interface,
                               ai_interface,
                               io_interface,
                               al_interface,
                               atmosphere_properties,
                               ocean_properties,
                               sea_ice_properties,
                               exchanger,
                               total_fluxes,
                               properties)
end

# Default land surface humidity formulation: bulk (saturated where wet, dry
# otherwise). The binary saturation is read from `saturation` per cell
# by the flux kernel and threaded through the iteration's `S` slot.
default_al_specific_humidity(::Nothing) = nothing
default_al_specific_humidity(land) =
    BulkHumidity(AtmosphericThermodynamics.Liquid())

# Default atmosphere--land flux formulation. Aerodynamic roughness lengths are a
# property of the flux closure, not the land model: the defaults below are
# uniform constants (0.1 m momentum, 0.01 m scalar). Override per-domain by
# passing `atmosphere_land_fluxes = SimilarityTheoryFluxes(...)` with explicit
# roughness lengths (constants, `Field`s, or roughness-length models such as
# `LandRoughnessLength`) to `ComponentInterfaces` / `AtmosphereLandModel`.
default_atmosphere_land_fluxes(::Nothing, FT) = nothing

function default_atmosphere_land_fluxes(land, FT)
    return SimilarityTheoryFluxes(FT;
                                   momentum_roughness_length    = convert(FT, 0.1),
                                   temperature_roughness_length = convert(FT, 0.01),
                                   water_vapor_roughness_length = convert(FT, 0.01))
end

#####
##### Chekpointing (not needed for ComponentInterfaces)
#####

Oceananigans.prognostic_state(::ComponentInterfaces) = nothing
Oceananigans.restore_prognostic_state!(ci::ComponentInterfaces, state) = ci
Oceananigans.restore_prognostic_state!(ci::ComponentInterfaces, ::Nothing) = ci
