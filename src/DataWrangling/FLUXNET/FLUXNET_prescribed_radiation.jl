"""
    FLUXNETPrescribedRadiation(site::FLUXNETSite, architecture = CPU(), FT = Float64;
                               start_date = first_date(site, :incoming_shortwave_radiation),
                               end_date = last_date(site, :incoming_shortwave_radiation),
                               max_gap = 48,
                               land_surface = SurfaceRadiationProperties(0.2, 0.97),
                               stefan_boltzmann_constant = default_stefan_boltzmann_constant)

Construct a [`PrescribedRadiation`](@ref) from a FLUXNET tower's downwelling
shortwave (`SW_IN_F`) and longwave (`LW_IN_F`) radiation on a single-column grid.
`land_surface` sets the surface albedo and emissivity; ocean and sea-ice surfaces
are omitted.
"""
function FLUXNETPrescribedRadiation(site::FLUXNETSite, architecture = CPU(), FT = Float64;
                                    start_date = first_date(site, :incoming_shortwave_radiation),
                                    end_date = last_date(site, :incoming_shortwave_radiation),
                                    max_gap = 48,
                                    land_surface = SurfaceRadiationProperties(0.2, 0.97),
                                    stefan_boltzmann_constant = default_stefan_boltzmann_constant)

    grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))
    sw = fluxnet_field_time_series(site, :incoming_shortwave_radiation, grid; start_date, end_date, max_gap)
    lw = fluxnet_field_time_series(site, :incoming_longwave_radiation,  grid; start_date, end_date, max_gap)

    return PrescribedRadiation(sw, lw;
                               ocean_surface = nothing,
                               sea_ice_surface = nothing,
                               land_surface,
                               stefan_boltzmann_constant)
end
