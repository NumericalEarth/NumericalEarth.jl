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
    gpu::Bool
end

Example(title, basename; build_always, gpu) = Example(title, basename, build_always, gpu)

const EXAMPLES_DIR   = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR     = joinpath(@__DIR__, "src/literated")
const DEVELOPERS_DIR = joinpath(@__DIR__, "src/developers")

mkpath(OUTPUT_DIR)

# Examples from examples/ directory.
# Set `build_always = false` for long-running examples that should only be built
# on pushes to `main`/tags, or when the `build all examples` label is added to a PR.
examples = [
    Example("Single-column ocean simulation", "single_column_os_papa_simulation"; build_always=true, gpu=false),
    Example("Coupled conservation on a z-star grid", "coupled_conservation"; build_always=true, gpu=false),
    # Near-global is the heaviest example; disabled while docs build on the L4 runner
    # Example("Near-global ocean simulation", "near_global_ocean_simulation"; build_always=false, gpu=true),
    # One-degree and global-climate initialize from ECCO; disabled while
    # ecco.jpl.nasa.gov/drive returns 503 (down since at least 2026-08-10)
    # Example("One-degree ocean--sea ice simulation", "one_degree_simulation"; build_always=false, gpu=true),
    # Example("Global climate simulation", "global_climate_simulation"; build_always=false, gpu=true),
    Example("Veros ocean simulation", "veros_ocean_forced_simulation"; build_always=false, gpu=false),
    Example("Breeze over four oceans", "breeze_over_four_oceans"; build_always=false, gpu=false),
    Example("ERA5 and GloFAS reanalysis data", "exploring_era5_reanalysis_data"; build_always=true, gpu=false),
    Example("Breeze over slab land", "breeze_over_slab_land"; build_always=true, gpu=false),
    Example("Differentiable ERA5-forced slab land", "era5_forced_slab_land"; build_always=false, gpu=false),
    Example("ERA5 downscaling with Breeze", "breeze_downscaling_era5"; build_always=true, gpu=true),
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

# A failed example does not abort the build: the site is built from the
# examples that succeeded, and the failures are reported (and fail the build)
# at the very end, so one broken example cannot block every other page.
failed_examples = String[]
failure_lock = ReentrantLock()

# Each example is an independent subprocess and `run` yields while it executes,
# so tasks overlap them: GPU examples serialize on the single device while CPU
# examples run alongside, a couple at a time (each subprocess loads the full
# package stack, so memory — not cores — bounds CPU concurrency).
cpu_semaphore = Base.Semaphore(2)
gpu_semaphore = Base.Semaphore(1)

function generate_example!(example, dir)
    script_path = joinpath(dir, example.basename * ".jl")
    Base.acquire(example.gpu ? gpu_semaphore : cpu_semaphore) do
        try
            run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "literate.jl")) $(script_path) $(OUTPUT_DIR)`)
        catch
            @error "Example $(example.basename) failed to build; continuing with the remaining examples."
            lock(() -> push!(failed_examples, example.basename), failure_lock)
        end
        example.gpu && CUDA.functional() && CUDA.reclaim()
    end
end

if skip_literate
    @info "Skipping Literate generation because NUMERICAL_EARTH_SKIP_LITERATE=true."
else
    @time "Example generation" @sync begin
        foreach(ex -> @async(generate_example!(ex, EXAMPLES_DIR)), examples)
        foreach(ex -> @async(generate_example!(ex, DEVELOPERS_DIR)), developer_examples)
    end
end

# Build pages from the examples that produced markdown, whether because
# generation succeeded just now or (with skip_literate) on a previous build.
filter!(ex -> isfile(joinpath(OUTPUT_DIR, ex.basename * ".md")), examples)
filter!(ex -> isfile(joinpath(OUTPUT_DIR, ex.basename * ".md")), developer_examples)

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
         linkcheck_timeout = 30, # some hosts (e.g. JMA JRA-55) are slow; the default 10s times out
         linkcheck_ignore = [
             r"^https://ecco\.jpl\.nasa\.gov/.*",
             r"https://www\.ncei\.noaa\.gov/products/etopo-global-relief-model",
             r"https://www\.ncei\.noaa\.gov/products/world-ocean-atlas",
             r"https://www\.ncei\.noaa\.gov/data/sea-surface-temperature-optimum-interpolation/v2\.1/access/avhrr",
             # Self-referential links to this repo's own test files on `main`:
             # they 404 during a PR's CI (the files only land on `main` at merge).
             r"https://github\.com/NumericalEarth/NumericalEarth\.jl/blob/main/test/",
        ],)

# The site above is complete for every example that built; now surface the ones
# that didn't, and fail the build so CI reports them.
if !isempty(failed_examples)
    error("The following examples failed to build: ", join(failed_examples, ", "))
end
