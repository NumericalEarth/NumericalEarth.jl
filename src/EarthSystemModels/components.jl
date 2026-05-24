#####
##### Temperature units
#####

struct DegreesCelsius end
struct DegreesKelvin end

const celsius_to_kelvin = 273.15
@inline convert_to_kelvin(::DegreesCelsius, T::FT) where FT = T + convert(FT, celsius_to_kelvin)
@inline convert_to_kelvin(::DegreesKelvin, T) = T

@inline convert_from_kelvin(::DegreesCelsius, T::FT) where FT = T - convert(FT, celsius_to_kelvin)
@inline convert_from_kelvin(::DegreesKelvin, T) = T

#####
##### generic defaults
#####

# Default: build the exchange grid from the ocean. When the model has no
# ocean / sea ice, fall back to the land grid (used by AtmosphereLandModel).
exchange_grid(atmosphere, ocean, sea_ice, land=nothing) = ocean.model.grid
exchange_grid(atmosphere, ::Nothing, ::Nothing, land) = land.grid

#####
##### Functions extended by sea-ice and ocean models
#####

reference_density(::Nothing) = 0
heat_capacity(::Nothing) = 0
ocean_temperature(ocean) = ZeroField()
ocean_salinity(ocean) = ZeroField()
ocean_surface_temperature(ocean) = ZeroField()
ocean_surface_salinity(ocean) = ZeroField()
ocean_surface_velocities(ocean) = ZeroField(), ZeroField()
temperature_units(ocean) = DegreesCelsius()

#####
##### Functions extended by sea-ice models
#####

sea_ice_thickness(::Nothing) = ZeroField()
sea_ice_concentration(::Nothing) = ZeroField()
function default_sea_ice end

#####
##### Functions extended by atmosphere models
#####

function thermodynamics_parameters end
function surface_layer_height end
function boundary_layer_height end

#####
##### Functions extended by all component models
#####

function interpolate_state! end
function update_net_fluxes! end

# Fallbacks for a  generic component model
update_net_fluxes!(coupled_model, component) = nothing
interpolate_state!(exchanger, grid, component, coupled_model) = nothing

# Fallback for radiative coupling when no radiation is configured.
apply_air_land_radiative_fluxes!(::Any) = nothing

#####
##### Functions extended by land models
#####

# Surface skin temperature exposed by a land component. Default is `nothing`
# so consumers (e.g. the default NaN checker) can fall back when the land
# component doesn't track a skin temperature.
function surface_temperature end
surface_temperature(::Any) = nothing
