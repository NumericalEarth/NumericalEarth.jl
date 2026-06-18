abstract type AbstractGLORYSBGCDataset <: AbstractGLORYSDataset end 

struct GLORYSBGCDaily <: AbstractGLORYSBGCDataset end
struct GLORYSBGCMonthly <: AbstractGLORYSBGCDataset end

dataset_name(::GLORYSBGCDaily) = "GLORYSBGCDaily"
dataset_name(::GLORYSBGCMonthly) = "GLORYSBGCMonthly"

const GLORYSBGC = Union{GLORYSBGCDaily, GLORYSBGCMonthly}

const GLORYSBGCMetadatum = 
    Union{Metadatum{<:GLORYSBGCDaily}, Metadatum{<:GLORYSBGCMonthly}}

Base.size(::GLORYSBGC, variable) = (1140, 680, 75)
Base.size(::GLORYSBGCMetadatum) = (1140, 680, 75, 1)

all_dates(::AbstractGLORYSBGCDataset, var) = 
    range(DateTime("1993-01-01"), stop=DateTime(now(UTC)) + ifelse(hour(now(UTC)) >= 13, Day(10), Day(9)), step=Day(1))

copernicusmarine_dataset_id(::GLORYSBGCDaily) = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
copernicusmarine_dataset_id(::GLORYSBGCMonthly) = "cmems_mod_glo_bgc_my_0.25deg_P1M-m"

available_variables(::GLORYSBGCDaily) = 
    Dict(:chlorophyll => "chl",
         :nitrate => "no3",
         :primary_production => "nppv",
         :oxygen => "o2",
         :phosphate => "po4",
         :silicate => "si")

available_variables(::GLORYSBGCMonthly) = 
    Dict(:chlorophyll => "chl",
         :iron => "fe",
         :nitrate => "no3",
         :primary_production => "nppv",
         :oxygen => "o2",
         :pH => "ph",
         :phytoplankton => "phyc",
         :phosphate => "po4",
         :silicate => "si",
         :pCO₂ => "spco2")

is_three_dimensional(metadata::GLORYSBGCMetadatum) = metadata.name != :pCO₂

struct GLORYSAnalysisForecastBGCDaily   <: AbstractGLORYSBGCDataset end
struct GLORYSAnalysisForecastBGCMonthly <: AbstractGLORYSBGCDataset end

dataset_name(::GLORYSAnalysisForecastBGCDaily)   = "GLORYSAnalysisForecastBGCDaily"
dataset_name(::GLORYSAnalysisForecastBGCMonthly) = "GLORYSAnalysisForecastBGCMonthly"

const GLORYSAnalysisForecastBGC = Union{GLORYSAnalysisForecastBGCDaily, GLORYSAnalysisForecastBGCMonthly}

const GLORYSAnalysisForecastBGCMetadatum =
    Union{Metadatum{<:GLORYSAnalysisForecastBGCDaily}, Metadatum{<:GLORYSAnalysisForecastBGCMonthly}}

# Grid: 1440 lon × 681 lat × 50 depth
Base.size(::GLORYSAnalysisForecastBGC, variable)      = (1440, 681, 50)
Base.size(::GLORYSAnalysisForecastBGCMetadatum)       = (1440, 681, 50, 1)

# Running 2-year sliding window — update these bounds as the catalogue grows.
# The analysis product currently runs from 2022-10-17; forecast extends 10 days ahead.
all_dates(::GLORYSAnalysisForecastBGCDaily,   var) = range(DateTime("2022-10-17"), stop=DateTime("2026-06-28"), step=Day(1))
all_dates(::GLORYSAnalysisForecastBGCMonthly, var) = range(DateTime("2022-10-01"), stop=DateTime("2026-06-01"), step=Month(1))

# Each variable group lives in its own sub-dataset.
# The mapping below returns the correct dataset_id for a given variable symbol.
const _AnalysisForecast_BGC_DATASET_ID = Dict(
    # bio
    :primary_production => "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
    :oxygen             => "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
    # car
    :alkalinity         => "cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
    :dissolved_inorganic_carbon => "cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
    :pH                 => "cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
    # co2
    :pCO₂               => "cmems_mod_glo_bgc-co2_anfc_0.25deg_P1D-m",
    # nut
    :nitrate            => "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    :phosphate          => "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    :silicate           => "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    :iron               => "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    # optics
    :light_attenuation  => "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m",
    # pft
    :chlorophyll        => "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
    :phytoplankton      => "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
    # plankton
    :zooplankton        => "cmems_mod_glo_bgc-plankton_anfc_0.25deg_P1D-m",
)

const _AnalysisForecast_BGC_DATASET_ID_MONTHLY = Dict(
    k => replace(v, "P1D-m" => "P1M-m") for (k, v) in _AnalysisForecast_BGC_DATASET_ID
)

copernicusmarine_dataset_id(::GLORYSAnalysisForecastBGCDaily,   var) = _AnalysisForecast_BGC_DATASET_ID[var]
copernicusmarine_dataset_id(::GLORYSAnalysisForecastBGCMonthly, var) = _AnalysisForecast_BGC_DATASET_ID_MONTHLY[var]

available_variables(::GLORYSAnalysisForecastBGCDaily) =
    Dict(:chlorophyll               => "chl",
         :dissolved_inorganic_carbon => "dissic",
         :iron                      => "fe",
         :light_attenuation         => "kd",
         :nitrate                   => "no3",
         :oxygen                    => "o2",
         :pH                        => "ph",
         :phytoplankton             => "phyc",
         :phosphate                 => "po4",
         :primary_production        => "nppv",
         :silicate                  => "si",
         :pCO₂                      => "spco2",
         :alkalinity                => "talk",
         :zooplankton               => "zooc")

available_variables(::GLORYSAnalysisForecastBGCMonthly) = available_variables(GLORYSAnalysisForecastBGCDaily())

is_three_dimensional(metadata::GLORYSAnalysisForecastBGCMetadatum) = metadata.name != :pCO₂
