"""
    GloFASReanalysis(; system_version="version_4_0")

The GloFAS-ERA5 historical river-discharge reanalysis (`cems-glofas-historical`,
`product_type="consolidated"`, `hydrological_model="lisflood"`). Daily river
discharge in m³ s⁻¹ on a global 0.05° grid (v4), 1979–present.
"""
struct GloFASReanalysis <: GloFASDataset
    system_version :: String
end

GloFASReanalysis(; system_version="version_4_0") = GloFASReanalysis(system_version)

dataset_name(::GloFASReanalysis) = "GloFASReanalysis"

# GloFAS v4 is 0.05° (7200 × 3000 covering -180:180, -60:90).
Base.size(::GloFASReanalysis, variable) = (7200, 3000, 1)

# Daily consolidated reanalysis. Available from 1979 to near-present; we use a
# practical upper bound matching the ERA5 forcing range.
DataWrangling.all_dates(::GloFASReanalysis, variable) =
    range(DateTime("1979-01-01"), stop=DateTime("2024-12-31"), step=Day(1))

#####
##### Variable name mappings
#####

# NumericalEarth name → CDS API variable name.
GloFAS_dataset_variable_names = Dict(
    :river_discharge => "river_discharge_in_the_last_24_hours",
)

# NumericalEarth name → NetCDF short variable name (as stored in downloaded files).
GloFAS_netcdf_variable_names = Dict(
    :river_discharge => "dis24",
)

DataWrangling.available_variables(::GloFASDataset) = GloFAS_dataset_variable_names
DataWrangling.dataset_variable_name(md::GloFASMetadata) = GloFAS_netcdf_variable_names[md.name]

# River discharge is already a volume flux (m³ s⁻¹); unit handling and the
# conversion to a per-area freshwater mass flux happen during routing (see
# `build_river_routing`), where the receiving ocean-cell area is known.
DataWrangling.conversion_units(md::GloFASMetadata) = nothing

# Discharge is undefined (missing) over the ocean. Do not inpaint: missing means
# "no river here", and the routing relies on the land/ocean boundary in the raw
# field to locate river mouths.
DataWrangling.default_inpainting(md::GloFASMetadata) = nothing
