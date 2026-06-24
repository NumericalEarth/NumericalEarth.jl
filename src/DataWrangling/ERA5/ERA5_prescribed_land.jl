using ...Lands: PrescribedLand

ERA5PrescribedLand(arch::Distributed; kw...) =
    ERA5PrescribedLand(child_architecture(arch); kw...)

"""
    ERA5PrescribedLand([architecture = CPU()];
                       dataset = ERA5HourlySingleLevel(),
                       start_date,
                       end_date,
                       dir = download_ERA5_cache,
                       time_indices_in_memory = 10,
                       time_indexing = Linear(),
                       region = nothing,
                       other_kw...)

Return a [`PrescribedLand`](@ref) representing ERA5 reanalysis land surface data.
For now, returns zero freshwater flux (rivers and icebergs not in ERA5 single-level).
"""
function ERA5PrescribedLand(architecture = CPU();
                            dataset = ERA5HourlySingleLevel(),
                            start_date = nothing,
                            end_date = nothing,
                            dir = download_ERA5_cache,
                            time_indices_in_memory = 10,
                            time_indexing = Linear(),
                            region = nothing,
                            other_kw...)

    # ERA5 single-level doesn't have river/iceberg flux
    # Return PrescribedLand with zero freshwater flux
    # Could be extended with ERA5-Land dataset in future

    freshwater_flux = nothing

    return PrescribedLand(freshwater_flux)
end
