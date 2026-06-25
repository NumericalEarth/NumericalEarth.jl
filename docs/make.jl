using NumericalEarth
using Breeze
using PythonCall, CondaPkg
using SpeedyWeather, ConservativeRegridding
using CUDA
using Documenter
using DocumenterCitations
using Literate
using TOML

if CUDA.functional()
    CUDA.versioninfo()
end

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

bib_filepath = joinpath(dirname(@__FILE__), "src", "NumericalEarth.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

#####
##### Example definition and filtering
#####

struct Example
    title::String
    basename::String
    build_always::Bool
end

const EXAMPLES_DIR   = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR     = joinpath(@__DIR__, "src/literated")
const DEVELOPERS_DIR = joinpath(@__DIR__, "src/developers")

mkpath(OUTPUT_DIR)

# Examples from examples/ directory.
# Set `build_always = false` for long-running examples that should only be built
# on pushes to `main`/tags, or when the `build all examples` label is added to a PR.
examples = [
    Example("Single-column ocean simulation", "single_column_os_papa_simulation", true),
    Example("Coupled energy and freshwater conservation", "coupled_conservation", true),
    Example("One-degree ocean--sea ice simulation", "one_degree_simulation", false),
    Example("Near-global ocean simulation", "near_global_ocean_simulation", false),
    Example("Global climate simulation", "global_climate_simulation", false),
    Example("Veros ocean simulation", "veros_ocean_forced_simulation", false),
    Example("Breeze over four oceans", "breeze_over_four_oceans", false),
    Example("ERA5 and GloFAS reanalysis data", "exploring_era5_reanalysis_data", true),
    Example("ERA5-forced slab land", "era5_forced_slab_land", true),
    Example("Breeze over slab land", "breeze_over_slab_land", true),
    Example("ERA5 downscaling with Breeze and NestedSimulation", "era5_breeze", false),
]

# Developer examples from docs/src/developers/ directory
developer_examples = [
    # Example("EarthSystemModel interface", "slab_ocean", false),
]

# Filter out long-running examples unless NUMERICAL_EARTH_BUILD_ALL_EXAMPLES is set
build_all = get(ENV, "NUMERICAL_EARTH_BUILD_ALL_EXAMPLES", "false") == "true"
filter!(x -> x.build_always || build_all, examples)
filter!(x -> x.build_always || build_all, developer_examples)

#####
##### Generate examples using Literate (each in a subprocess for memory isolation)
#####

skip_literate = get(ENV, "NUMERICAL_EARTH_SKIP_LITERATE", "false") == "true"

if skip_literate
    @info "Skipping Literate generation because NUMERICAL_EARTH_SKIP_LITERATE=true."
    filter!(ex -> isfile(joinpath(OUTPUT_DIR, ex.basename * ".md")), examples)
    filter!(ex -> isfile(joinpath(OUTPUT_DIR, ex.basename * ".md")), developer_examples)
else
    for example in examples
        script_path = joinpath(EXAMPLES_DIR, example.basename * ".jl")
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "literate.jl")) $(script_path) $(OUTPUT_DIR)`)
        CUDA.functional() && CUDA.reclaim()
    end

    for example in developer_examples
        script_path = joinpath(DEVELOPERS_DIR, example.basename * ".jl")
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "literate.jl")) $(script_path) $(OUTPUT_DIR)`)
        CUDA.functional() && CUDA.reclaim()
    end
end

modules = Module[]
NumericalEarthBreezeExt = isdefined(Base, :get_extension) ? Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt) : NumericalEarth.NumericalEarthBreezeExt
NumericalEarthSpeedyWeatherExt = isdefined(Base, :get_extension) ? Base.get_extension(NumericalEarth, :NumericalEarthSpeedyWeatherExt) : NumericalEarth.NumericalEarthSpeedyWeatherExt
NumericalEarthVerosExt = isdefined(Base, :get_extension) ? Base.get_extension(NumericalEarth, :NumericalEarthVerosExt) : NumericalEarth.NumericalEarthVerosExt

for m in [NumericalEarth, NumericalEarthBreezeExt, NumericalEarthSpeedyWeatherExt, NumericalEarthVerosExt]
    if !isnothing(m)
        push!(modules, m)
    end
end


#####
##### Automatically generate files with docstrings for all modules
#####

function walk_submodules!(result, visited, mod::Module)
    for name in sort(names(mod; all=true, imported=false))
        isdefined(mod, name) || continue
        value = getproperty(mod, name)
        if value isa Module &&
            parentmodule(value) === mod &&
            !(value in visited) &&
            value !== mod

            push!(visited, value)
            push!(result, value)
            walk_submodules!(result, visited, value)
        end
    end
end

function get_submodules(mod::Module)
    result = Module[]
    visited = Set{Module}()

    walk_submodules!(result, visited, mod)
    return result
end

function write_api_md(filename; public)
    modules = get_submodules(NumericalEarth)
    append!(modules, [NumericalEarthBreezeExt, NumericalEarthSpeedyWeatherExt, NumericalEarthVerosExt])
    io = IOBuffer()

    title = public ? "Public API" : "Private API"
    privacy_keyword = public ? "Private = false" : "Public = false"

    println(io, "# ", title)
    println(io)
    println(io, "```@autodocs")
    println(io, "Modules = [NumericalEarth]")
    println(io, privacy_keyword)
    println(io, "```")
    println(io)

    for mod in modules
        println(io, "## ", chopprefix(string(mod), "NumericalEarth."))
        println(io)
        println(io, "```@autodocs")
        println(io, "Modules = [", mod, "]")
        println(io, privacy_keyword)
        println(io, "```")
        println(io)
    end

    # Remove multiple trailing whitespaces, but keep the final one.
    write(joinpath(@__DIR__, "src", "library", filename), strip(String(take!(io))) * "\n")
end

write_api_md("public_api.md"; public = true)
write_api_md("private_api.md"; public = false)

#####
##### Build docs
#####

examples_pages  = [ex.title => joinpath("literated", ex.basename * ".md") for ex in examples]
developer_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in developer_examples]

format = Documenter.HTML(collapselevel = 3,
                         size_threshold = nothing,
                         canonical = "https://numericalearth.github.io/NumericalEarthDocumentation/stable/")

pages = [
    "Home" => "index.md",

    "EarthSystemModel" => "earth_system_model.md",

    "Examples" => examples_pages,

    # "Developers" => developer_pages,

    "Vertical grids" => "vertical_grids.md",

    "Metadata" => [
        "Overview" => "Metadata/metadata_overview.md",
        "Supported variables" => "Metadata/supported_variables.md",
    ],
    "Interface fluxes" => "interface_fluxes.md",

    "Appendix" => [
        "Notation" => "appendix/notation.md",
    ],

    "Library" => [
        "Contents"       => "library/outline.md",
        "Public API"     => "library/public_api.md",
        "Private API"    => "library/private_api.md",
        "Function index" => "library/function_index.md",
    ],

    "References" => "references.md",
]

makedocs(; sitename = "NumericalEarth.jl",
         format,
         pages,
         modules,
         plugins = [bib],
         doctest = true,
         draft = false,
         doctestfilters = [
             r"┌ Warning:.*",  # remove standard warning lines
             r"│ Use at own risk",
             r"└ @ .*",        # remove the source location of warnings
             r"(?s)(└── dir:).*" => s"\1",
         ],
         clean = true,
         warnonly = [:cross_references, :missing_docs],
         checkdocs = :exports,
         linkcheck = true,
         linkcheck_ignore = [
             r"https://www\.ncei\.noaa\.gov/products/etopo-global-relief-model",
             r"https://www\.ncei\.noaa\.gov/products/world-ocean-atlas",
             r"https://www\.ncei\.noaa\.gov/data/sea-surface-temperature-optimum-interpolation/v2\.1/access/avhrr",
        ],)
