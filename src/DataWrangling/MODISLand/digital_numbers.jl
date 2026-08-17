#####
##### Digital-number decode
#####

const MODIS_LAI_MAXIMUM_VALID = 100
const MODIS_LAI_SCALE  = 0.1
const MODIS_FPAR_SCALE = 0.01

const MODIS_LAI_LANDCOVER_CODES = 249:254

const modis_landcover_class_names = (unclassified = 249,
                                     urban = 250,
                                     wetland = 251,
                                     snow_and_ice = 252,
                                     barren = 253,
                                     water = 254)

"""
    mask_lai_fill(DN)

Return the MCD15 digital number `DN` as a `Float32`, or `NaN` when `DN` exceeds the valid
range of `0:$(MODIS_LAI_MAXIMUM_VALID)` — the fill value and the land-cover special codes.
The product's scale factor is applied downstream, so the rejection has to happen here.
"""
@inline mask_lai_fill(DN) = ifelse(DN > MODIS_LAI_MAXIMUM_VALID, NaN32, Float32(DN))

"""
    mask_lai_landcover(DN)

The complement of [`mask_lai_fill`](@ref): return the digital number `DN` as a `Float32` when
it is one of the land-cover codes `$(MODIS_LAI_LANDCOVER_CODES)` the product substitutes for a
retrieval it does not attempt, and `NaN` otherwise — for a valid retrieval, and for the fill
value 255, neither of which names a class.

| code | class |
|---|---|
| 249 | unclassified — not a usable class |
| 250 | urban / built-up |
| 251 | wetland / inundated marshland |
| 252 | perennial snow or ice |
| 253 | barren or sparsely vegetated |
| 254 | perennial salt or fresh water |

These are **class codes**, so read them on the product's own grid: interpolating them onto
another grid averages 250 against 254 into 252, which is a different class.
"""
@inline mask_lai_landcover(DN) =
    ifelse((DN ≥ first(MODIS_LAI_LANDCOVER_CODES)) & (DN ≤ last(MODIS_LAI_LANDCOVER_CODES)),
           Float32(DN), NaN32)

#####
##### Quality screening
#####
##### Two bit-packed bytes accompany every retrieval. `FparLai_QC` carries the retrieval
##### provenance — bit 0 MODLAND_QC, bit 1 SENSOR, bit 2 DEADDETECTOR, bits 3–4 CLOUDSTATE,
##### bits 5–7 SCF_QC — and `FparExtra_QC` the scene state — bits 0–1 LANDSEA, bit 2
##### SNOW_ICE, bit 3 AEROSOL, bit 4 CIRRUS, bit 5 INTERNAL_CLOUDMASK, bit 6 CLOUD_SHADOW,
##### bit 7 SCF_BIOME_MASK.
#####

@inline modland_quality_control(qc) = qc & 0x01
@inline dead_detector(qc)           = (qc >> 0x02) & 0x01
@inline cloud_state(qc)             = (qc >> 0x03) & 0x03
@inline scf_quality_control(qc)     = (qc >> 0x05) & 0x07

@inline snow_or_ice(extra_qc)        = (extra_qc >> 0x02) & 0x01
@inline high_aerosol(extra_qc)       = (extra_qc >> 0x03) & 0x01
@inline cirrus(extra_qc)             = (extra_qc >> 0x04) & 0x01
@inline internal_cloud_mask(extra_qc) = (extra_qc >> 0x05) & 0x01
@inline cloud_shadow(extra_qc)       = (extra_qc >> 0x06) & 0x01

# A private bit assignment used to combine criteria into one screening mask — unrelated to
# the product's own bit positions, which the accessors above decode.
const lai_screening_flags = (other_quality    = 0x0001,
                             backup_algorithm = 0x0002,
                             cloudy           = 0x0004,
                             dead_detector    = 0x0008,
                             snow_or_ice      = 0x0010,
                             high_aerosol     = 0x0020,
                             cirrus           = 0x0040,
                             internal_cloud   = 0x0080,
                             cloud_shadow     = 0x0100)

"""
    lai_rejection_flags(qc, extra_qc)

Return the [`lai_screening_mask`](@ref) bits of every quality criterion the pixel with
`FparLai_QC` byte `qc` and `FparExtra_QC` byte `extra_qc` fails:

| flag | meaning |
|---|---|
| `:other_quality` | `MODLAND_QC` reports other than good quality |
| `:backup_algorithm` | the radiative-transfer retrieval failed and the empirical back-up was used, or no value was produced |
| `:cloudy` | `CLOUDSTATE` reports significant or mixed cloud (the clear and assumed-clear states pass) |
| `:dead_detector` | dead detectors forced a retrieval from adjacent detectors |
| `:snow_or_ice` | snow or ice was detected |
| `:high_aerosol` | average or high aerosol was detected |
| `:cirrus` | cirrus was detected |
| `:internal_cloud` | the surface-reflectance internal cloud mask flagged cloud |
| `:cloud_shadow` | cloud shadow was detected |

A pixel that passes everything returns `0x0000`.
"""
@inline function lai_rejection_flags(qc, extra_qc)
    f = lai_screening_flags
    cloud = cloud_state(qc)

    rejected  = ifelse(modland_quality_control(qc) != 0, f.other_quality,    0x0000)
    rejected |= ifelse(scf_quality_control(qc) > 0x01,   f.backup_algorithm, 0x0000)
    rejected |= ifelse((cloud == 0x01) | (cloud == 0x02), f.cloudy,          0x0000)
    rejected |= ifelse(dead_detector(qc) != 0,           f.dead_detector,    0x0000)
    rejected |= ifelse(snow_or_ice(extra_qc) != 0,       f.snow_or_ice,      0x0000)
    rejected |= ifelse(high_aerosol(extra_qc) != 0,      f.high_aerosol,     0x0000)
    rejected |= ifelse(cirrus(extra_qc) != 0,            f.cirrus,           0x0000)
    rejected |= ifelse(internal_cloud_mask(extra_qc) != 0, f.internal_cloud, 0x0000)
    rejected |= ifelse(cloud_shadow(extra_qc) != 0,      f.cloud_shadow,     0x0000)
    return rejected
end

"""
    lai_screening_mask(names::Symbol...)

Combine the named quality criteria into one screening mask, for use as the
`screened_flags` of an [`MCD15A2H`](@ref) dataset. A pixel failing any criterion in the
mask is read as `NaN`. The criteria are listed under [`lai_rejection_flags`](@ref).

```jldoctest
julia> using NumericalEarth

julia> lai_screening_mask(:other_quality, :snow_or_ice)
0x0011
```
"""
function lai_screening_mask(names::Symbol...)
    mask = 0x0000
    for name in names
        haskey(lai_screening_flags, name) ||
            throw(ArgumentError("$name is not a quality criterion; valid names are " *
                                "$(keys(lai_screening_flags))."))
        mask |= lai_screening_flags[name]
    end
    return mask
end

"""
    recommended_lai_screening()

The screen the product's user guide recommends and the aerodynamic-roughness literature
applies: keep only good-quality pixels retrieved by the main radiative-transfer algorithm
under a clear (or assumed-clear) sky. This is the default `screened_flags` of
[`MCD15A2H`](@ref).

The scene-state detections — snow, aerosol, cirrus, internal cloud, cloud shadow — are
deliberately left in: snow in particular is a physical state rather than a retrieval failure,
and screening it out is opt-in via [`lai_screening_mask`](@ref).

```jldoctest
julia> using NumericalEarth

julia> recommended_lai_screening()
0x0007
```
"""
recommended_lai_screening() =
    lai_screening_mask(:other_quality, :backup_algorithm, :cloudy)

@inline lai_screened(qc, extra_qc, mask) = !iszero(lai_rejection_flags(qc, extra_qc) & mask)

