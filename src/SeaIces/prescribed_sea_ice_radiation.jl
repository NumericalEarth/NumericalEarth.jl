using ..Radiations: Radiations, SeaIceAlbedo, SurfaceRadiationProperties, default_sea_ice_emissivity

#####
##### Radiative properties of prescribed sea ice
#####

# The CCSM3 broadband albedo of `SeaIceAlbedo` for snow-free ice of the prescribed thickness, reduced within
# 1 K of the melting point: prescribed ice carries no snow, so the cold albedo is that of bare ice (0.54).
# The skin temperature is in Kelvin, and so is the melting point the reduction is measured from.
function Radiations.default_sea_ice_surface_radiation(sea_ice::PrescribedSeaIce, grid)
    FT = eltype(grid)
    melting_temperature = sea_ice.liquidus.freshwater_melting_temperature
    albedo = SeaIceAlbedo(sea_ice.ice_thickness, nothing, sea_ice.top_surface_temperature; FT, melting_temperature)
    return SurfaceRadiationProperties(albedo, convert(FT, default_sea_ice_emissivity))
end
