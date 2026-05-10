using NumericalEarth
using CUDA
using Documenter
using DocumenterCitations
using Literate

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# Documenter's `TextDiff.makediff!` is a recursive LCS reconstruction
# whose depth scales with the token count of the diff. When a doctest
# fails with a long output (e.g. a multi-thousand-frame error
# stacktrace), the recursion overflows the default ~4 MB task stack
# and aborts the build before the actual doctest error is reported.
# Replace with a behaviorally-equivalent iterative version so the
# diff completes and the underlying failure surfaces.
@eval Documenter.TextDiff function makediff!(out, weights, X, Y, i, j)
    actions = Vector{eltype(out)}()
    while i > 1 || j > 1
        if i > 1 && j > 1 && X[i - 1] == Y[j - 1]
            push!(actions, :normal => X[i - 1])
            i -= 1; j -= 1
        elseif j > 1 && (i == 1 || weights[i, j - 1] >= weights[i - 1, j])
            push!(actions, :green => Y[j - 1])
            j -= 1
        elseif i > 1 && (j == 1 || weights[i, j - 1] < weights[i - 1, j])
            push!(actions, :red => X[i - 1])
            i -= 1
        else
            break
        end
    end
    append!(out, reverse!(actions))
    return out
end

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
    Example("One-degree ocean--sea ice simulation", "one_degree_simulation", false),
    Example("Near-global ocean simulation", "near_global_ocean_simulation", false),
    Example("Global climate simulation", "global_climate_simulation", false),
    Example("Veros ocean simulation", "veros_ocean_forced_simulation", false),
    Example("Breeze over two oceans", "breeze_over_two_oceans", false),
    Example("ERA5 hourly data", "ERA5_hourly_data", true),
]

# Developer examples from docs/src/developers/ directory
developer_examples = [
    Example("EarthSystemModel interface", "slab_ocean", false),
]

# Filter out long-running examples unless NUMERICAL_EARTH_BUILD_ALL_EXAMPLES is set
build_all = get(ENV, "NUMERICAL_EARTH_BUILD_ALL_EXAMPLES", "false") == "true"
filter!(x -> x.build_always || build_all, examples)
filter!(x -> x.build_always || build_all, developer_examples)

#####
##### Generate examples using Literate (each in a subprocess for memory isolation)
#####

for example in examples
    script_path = joinpath(EXAMPLES_DIR, example.basename * ".jl")
    run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "literate.jl")) $(script_path) $(OUTPUT_DIR)`)
    # Reclaim GPU memory between subprocesses, but tolerate a non-
    # functional / unsupported GPU on the runner (e.g. CUDA 13+ with a
    # TITAN V): `CUDA.functional()` itself may throw on an unsupported
    # compute capability.
    try
        CUDA.functional() && CUDA.reclaim()
    catch err
        @debug "Skipping CUDA.reclaim()" exception=err
    end
end

for example in developer_examples
    script_path = joinpath(DEVELOPERS_DIR, example.basename * ".jl")
    run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "literate.jl")) $(script_path) $(OUTPUT_DIR)`)
    # Reclaim GPU memory between subprocesses, but tolerate a non-
    # functional / unsupported GPU on the runner (e.g. CUDA 13+ with a
    # TITAN V): `CUDA.functional()` itself may throw on an unsupported
    # compute capability.
    try
        CUDA.functional() && CUDA.reclaim()
    catch err
        @debug "Skipping CUDA.reclaim()" exception=err
    end
end

#####
##### Build docs
#####

examples_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in examples]
developer_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in developer_examples]

format = Documenter.HTML(collapselevel = 2,
                         size_threshold = nothing,
                         canonical = "https://numericalearth.github.io/NumericalEarthDocumentation/stable/")

pages = [
    "Home" => "index.md",

    "EarthSystemModel" => "earth_system_model.md",

    "Examples" => examples_pages,

    "Developers" => developer_pages,

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
        "Public"         => "library/public.md",
        "Private"        => "library/internals.md",
        "Function index" => "library/function_index.md",
    ],

    "References" => "references.md",
]

modules = Module[]
NumericalEarthSpeedyWeatherExt = isdefined(Base, :get_extension) ? Base.get_extension(NumericalEarth, :NumericalEarthSpeedyWeatherExt) : NumericalEarth.NumericalEarthSpeedyWeatherExt
NumericalEarthVerosExt = isdefined(Base, :get_extension) ? Base.get_extension(NumericalEarth, :NumericalEarthVerosExt) : NumericalEarth.NumericalEarthVerosExt
NumericalEarthBreezeExt = isdefined(Base, :get_extension) ? Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt) : nothing

for m in [NumericalEarth, NumericalEarthSpeedyWeatherExt, NumericalEarthBreezeExt, NumericalEarthVerosExt]
    if !isnothing(m)
        push!(modules, m)
    end
end

makedocs(; sitename = "NumericalEarth.jl",
         format, pages, modules,
         plugins = [bib],
         doctest = true,
         doctestfilters = [
             r"┌ Warning:.*",  # remove standard warning lines
             r"│ Use at own risk",
             r"└ @ .*",        # remove the source location of warnings
         ],
         clean = true,
         warnonly = [:cross_references, :missing_docs],
         checkdocs = :exports)
