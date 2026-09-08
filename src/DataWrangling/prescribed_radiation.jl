using ..Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant,
                    default_water_emissivity

"""
$(TYPEDSIGNATURES)

Return a [`PrescribedRadiation`](@ref NumericalEarth.Radiations.PrescribedRadiation) backed by `dataset`'s
downwelling shortwave and longwave `FieldTimeSeries`.

`region` (a `BoundingBox`) restricts the download and the native grid to a sub-domain; the coupled model
interpolates the native-resolution radiation onto the exchange grid. Surface radiative properties (albedo,
emissivity) default to standard ocean/sea-ice values; pass `*_surface = nothing` to omit a surface or supply
your own `SurfaceRadiationProperties` (e.g. `land_surface` for land-only runs).
"""
function prescribed_radiation(dataset, architecture = CPU();
                              start_date = first_date(dataset, :downwelling_shortwave_radiation),
                              end_date = last_date(dataset, :downwelling_shortwave_radiation),
                              dir = default_download_directory(dataset),
                              time_indices_in_memory = 24,
                              time_indexing = Cyclical(),
                              ocean_surface = SurfaceRadiationProperties(0.05, default_water_emissivity),
                              sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                              snow_surface = nothing,
                              land_surface = nothing,
                              stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                              region = nothing,
                              other_kw...)

    kw = merge((; time_indexing, time_indices_in_memory), other_kw)

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
