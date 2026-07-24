using ...Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant

ERA5PrescribedRadiation(arch::Distributed; kw...) =
    ERA5PrescribedRadiation(child_architecture(arch); kw...)

"""
    ERA5PrescribedRadiation([architecture = CPU()];
                            dataset = ERA5HourlySingleLevel(),
                            start_date = first_date(dataset, :downwelling_shortwave_radiation),
                            end_date = last_date(dataset, :downwelling_shortwave_radiation),
                            dir = download_ERA5_cache,
                            time_indices_in_memory = 24,
                            time_indexing = Cyclical(),
                            ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                            sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                            snow_surface = nothing,
                            land_surface = nothing,
                            stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                            region = nothing,
                            other_kw...)

Return a [`PrescribedRadiation`](@ref NumericalEarth.Radiations.PrescribedRadiation)
backed by ERA5 downwelling shortwave and longwave `FieldTimeSeries`, suitable for
regional hindcast forcing. ERA5 stores these as energy accumulated over the previous
hour (J m⁻²); the load-time `conversion_units` divides by the accumulation interval to
recover the mean flux (W m⁻²).

`region` (a `BoundingBox`) restricts the download and the native grid to a sub-domain;
the coupled model interpolates the native-resolution radiation onto the exchange grid.
Surface radiative properties (albedo, emissivity) default to standard ocean/sea-ice
values; pass `*_surface = nothing` to omit a surface or supply your own
`SurfaceRadiationProperties` (e.g. `land_surface` for land-only runs).
"""
function ERA5PrescribedRadiation(architecture = CPU();
                                 dataset = ERA5HourlySingleLevel(),
                                 start_date = first_date(dataset, :downwelling_shortwave_radiation),
                                 end_date = last_date(dataset, :downwelling_shortwave_radiation),
                                 dir = download_ERA5_cache,
                                 time_indices_in_memory = 24,
                                 time_indexing = Cyclical(),
                                 ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                                 sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                                 snow_surface = nothing,
                                 land_surface = nothing,
                                 stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                                 region = nothing,
                                 other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    # Both bands ride one batched download (see the `MetadataSet` backends)
    mset = MetadataSet(:downwelling_shortwave_radiation, :downwelling_longwave_radiation;
                       dataset, start_date, end_date, dir, region)
    Downloads.download(mset)

    ℐꜜˢʷ_meta = Metadata(:downwelling_shortwave_radiation; dataset, start_date, end_date, dir, region)
    ℐꜜˡʷ_meta = Metadata(:downwelling_longwave_radiation;  dataset, start_date, end_date, dir, region)

    ℐꜜˢʷ = FieldTimeSeries(ℐꜜˢʷ_meta, architecture; kw...)
    ℐꜜˡʷ = FieldTimeSeries(ℐꜜˡʷ_meta, architecture; kw...)

    return PrescribedRadiation(ℐꜜˢʷ, ℐꜜˡʷ;
                               ocean_surface,
                               sea_ice_surface,
                               snow_surface,
                               land_surface,
                               stefan_boltzmann_constant)
end
