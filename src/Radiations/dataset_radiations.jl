#####
##### Prescribed radiation built from the ERA5, JRA55, and ECCO datasets
#####

using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads
using Oceananigans.DistributedComputations: Distributed, child_architecture
using Oceananigans.OutputReaders: Cyclical

using ..DataWrangling: Metadata, MetadataSet, first_date, last_date, default_download_directory
using ..DataWrangling.ERA5: ERA5HourlySingleLevel
using ..DataWrangling.JRA55: RepeatYearJRA55
using ..DataWrangling.ECCO: ECCO4Monthly

                    default_water_emissivity

"""
$(TYPEDSIGNATURES)

Return a [`PrescribedRadiation`](@ref) backed by `dataset`'s
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

ERA5PrescribedRadiation(arch::Distributed; kw...) =
    ERA5PrescribedRadiation(child_architecture(arch); kw...)

"""
    ERA5PrescribedRadiation([architecture = CPU()]; dataset = ERA5HourlySingleLevel(),
                            time_indices_in_memory = 24, other_kw...)

ERA5 downwelling radiation, suitable for regional hindcast forcing. ERA5 stores these as energy
accumulated over the previous hour (J m⁻²); the load-time `conversion_units` divides by the accumulation
interval to recover the mean flux (W m⁻²). Remaining keyword arguments go to
[`prescribed_radiation`](@ref).
"""
ERA5PrescribedRadiation(architecture = CPU(); dataset = ERA5HourlySingleLevel(),
                        time_indices_in_memory = 24, kw...) =
    prescribed_radiation(dataset, architecture; time_indices_in_memory, kw...)

JRA55PrescribedRadiation(arch::Distributed; kw...) =
    JRA55PrescribedRadiation(child_architecture(arch); kw...)

"""
    JRA55PrescribedRadiation([architecture = CPU()]; dataset = RepeatYearJRA55(),
                             time_indices_in_memory = 10, other_kw...)

JRA55 downwelling radiation. Remaining keyword arguments go to [`prescribed_radiation`](@ref).
"""
JRA55PrescribedRadiation(architecture = CPU(); dataset = RepeatYearJRA55(),
                         time_indices_in_memory = 10, kw...) =
    prescribed_radiation(dataset, architecture; time_indices_in_memory, kw...)

                     default_water_emissivity

"""
    ECCOPrescribedRadiation([architecture = CPU()];
                            dataset = ECCO4Monthly(),
                            start_date = first_date(dataset, :downwelling_shortwave),
                            end_date = last_date(dataset, :downwelling_shortwave),
                            dir = default_download_directory(dataset),
                            time_indices_in_memory = 10,
                            time_indexing = Cyclical(),
                            ocean_surface = SurfaceRadiationProperties(0.05, default_water_emissivity),
                            sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                            stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                            other_kw...)

Return a [`PrescribedRadiation`](@ref) backed by ECCO downwelling shortwave
and longwave fields.
"""
function ECCOPrescribedRadiation(architecture = CPU();
                                 dataset = ECCO4Monthly(),
                                 start_date = first_date(dataset, :downwelling_shortwave),
                                 end_date = last_date(dataset, :downwelling_shortwave),
                                 dir = default_download_directory(dataset),
                                 time_indexing = Cyclical(),
                                 time_indices_in_memory = 10,
                                 ocean_surface = SurfaceRadiationProperties(0.05, default_water_emissivity),
                                 sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                                 stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                                 other_kw...)

    ℐꜜˢʷ_meta = Metadata(:downwelling_shortwave; dataset, start_date, end_date, dir)
    ℐꜜˡʷ_meta = Metadata(:downwelling_longwave;  dataset, start_date, end_date, dir)

    kw = (; time_indices_in_memory, time_indexing)
    kw = merge(kw, other_kw)

    ℐꜜˢʷ = FieldTimeSeries(ℐꜜˢʷ_meta, architecture; kw...)
    ℐꜜˡʷ = FieldTimeSeries(ℐꜜˡʷ_meta, architecture; kw...)

    return PrescribedRadiation(ℐꜜˢʷ, ℐꜜˡʷ;
                               ocean_surface,
                               sea_ice_surface,
                               stefan_boltzmann_constant)
end
