using NumericalEarth
using CUDA
using Documenter
using DocumenterCitations
using Literate
using TOML

if CUDA.has_cuda()
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
    Example("One-degree ocean--sea ice simulation", "one_degree_simulation", false),
    Example("Near-global ocean simulation", "near_global_ocean_simulation", false),
    Example("Global climate simulation", "global_climate_simulation", false),
    # Example("Veros ocean simulation", "veros_ocean_forced_simulation", false),
    Example("Breeze over four oceans", "breeze_over_four_oceans", false),
    Example("ERA5 hourly data", "ERA5_hourly_data", true),
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

#####
##### Library autodocs generation
#####

const LIBRARY_DIR = joinpath(@__DIR__, "src", "library")

function direct_child_modules(parent::Module)
    children = Module[]

    for name in names(parent; all = true, imported = false)
        isdefined(parent, name) || continue
        child = getfield(parent, name)
        child isa Module || continue
        child === parent && continue
        parentmodule(child) === parent || continue
        push!(children, child)
    end

    sort!(children; by = string)
    return children
end

function collect_module_tree!(modules, parent::Module, seen)
    parent in seen && return modules
    push!(seen, parent)
    push!(modules, parent)

    for child in direct_child_modules(parent)
        collect_module_tree!(modules, child, seen)
    end

    return modules
end

function collect_module_tree(parent::Module)
    return collect_module_tree!(Module[], parent, IdSet{Module}())
end

function extension_trigger_package_names(project_toml_path)
    project = TOML.parsefile(project_toml_path)
    extensions = get(project, "extensions", Dict())
    trigger_names = String[]

    for triggers in values(extensions)
        if triggers isa AbstractString
            push!(trigger_names, triggers)
        else
            append!(trigger_names, triggers)
        end
    end

    sort!(unique!(trigger_names))
    return Symbol.(trigger_names)
end

function load_extension_trigger_packages(package_project_toml_path, docs_project_toml_path)
    package_trigger_names = extension_trigger_package_names(package_project_toml_path)
    docs_dependencies = keys(get(TOML.parsefile(docs_project_toml_path), "deps", Dict()))
    docs_dependency_names = Set(Symbol.(docs_dependencies))

    for trigger_name in package_trigger_names
        trigger_name in docs_dependency_names || continue
        Base.eval(Main, :(import $(trigger_name)))
    end

    return nothing
end

function loaded_extension_modules(package::Module)
    ext_dir = joinpath(dirname(@__DIR__), "ext")
    extension_symbols = Symbol[]

    for (root, _, files) in walkdir(ext_dir)
        for file in files
            endswith(file, "Ext.jl") || continue
            push!(extension_symbols, Symbol(first(splitext(file))))
        end
    end

    sort!(unique!(extension_symbols); by = string)

    extensions = Module[]
    for extension_symbol in extension_symbols
        extension = isdefined(Base, :get_extension) ? Base.get_extension(package, extension_symbol) : nothing
        isnothing(extension) || push!(extensions, extension)
    end

    return extensions
end

function bind_modules_in_main!(modules)
    for mod in modules
        name = nameof(mod)
        isdefined(Main, name) || Core.eval(Main, :(const $name = $mod))
    end

    return modules
end

function write_library_autodocs(path, title, description, modules; private)
    open(path, "w") do io
        println(io, "# ", title)
        println(io)
        println(io, description)
        println(io)
        println(io, "<!-- This file is autogenerated by docs/make.jl. -->")
        println(io)

        for mod in modules
            println(io, "## ", string(mod))
            println(io)
            println(io, "```@autodocs")
            println(io, "Modules = [", string(mod), "]")
            println(io, private ? "Public = false" : "Private = false")
            println(io, "```")
            println(io)
        end
    end

    return nothing
end

package_project_toml_path = joinpath(dirname(@__DIR__), "Project.toml")
docs_project_toml_path = joinpath(@__DIR__, "Project.toml")

load_extension_trigger_packages(package_project_toml_path, docs_project_toml_path)

core_modules = collect_module_tree(NumericalEarth)
extension_modules = loaded_extension_modules(NumericalEarth)
bind_modules_in_main!(extension_modules)
modules = vcat(core_modules, extension_modules)

write_library_autodocs(joinpath(LIBRARY_DIR, "public.md"),
                       "Public Documentation",
                       "Documentation for `NumericalEarth.jl`'s public interface. This page is generated from the loaded module tree, so new submodules and loaded extensions are included automatically.",
                       modules;
                       private = false)

write_library_autodocs(joinpath(LIBRARY_DIR, "internals.md"),
                       "Private types and functions",
                       "Documentation for `NumericalEarth.jl`'s internal interface. This page is generated from the loaded module tree, so new submodules and loaded extensions are included automatically.",
                       modules;
                       private = true)

#####
##### Build docs
#####

examples_pages  = [ex.title => joinpath("literated", ex.basename * ".md") for ex in examples]
developer_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in developer_examples]

format = Documenter.HTML(collapselevel = 2,
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
        "Public"         => "library/public.md",
        "Private"        => "library/internals.md",
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
         ],
         clean = true,
         warnonly = [:cross_references, :missing_docs],
         checkdocs = :exports)
