#####
##### Conformance test harness for third-party dataset implementations.
#####
#####
##### `test_dataset_contract(d)` walks every method of the dataset contract,
##### classifies it as :ok (user-implemented), :default (shipped fallback
##### applied), :missing (required method absent), or :error (raised on a
##### sample input), and prints a structured report. Intended for use in a
##### companion package's test suite.
#####

const CONTRACT_MODULE = DataWrangling

"""
    ContractCheck(name, status, detail; required=false, variable=nothing)

One entry in a [`ContractReport`](@ref). `status` is one of `:ok`, `:default`, `:missing`, `:error`. 
`detail` is a short human-readable summary of the method's result or the raised error. `required` flags whether 
absence makes the dataset non-functional.
"""
struct ContractCheck
    name :: Symbol
    status :: Symbol
    detail :: String
    required :: Bool
    variable :: Union{Symbol, Nothing}
end

ContractCheck(name, status, detail; required=false, variable=nothing) =
    ContractCheck(name, status, detail, required, variable)

"""
    ContractReport(dataset, sample_variable, sample_date, checks)

Full result of [`test_dataset_contract`](@ref). The `checks` field is a vector of [`ContractCheck`](@ref) entries in 
the order they were run. Use `is_conforming(report)` to ask whether there are any missing required methods or errors.
"""
struct ContractReport
    dataset
    sample_variable :: Union{Symbol, Nothing}
    sample_date
    checks :: Vector{ContractCheck}
end

"""
    is_conforming(report)

Return `true` when every required method is present and no check raised an error. Optional methods using shipped defaults do 
not count against conformance.
"""
is_conforming(r::ContractReport) = all(c -> !(c.status === :missing && c.required) && c.status !== :error, r.checks)

function Base.show(io::IO, ::MIME"text/plain", r::ContractReport)
    print(io, "Conformance report for ", r.dataset, ":\n\n")
    if !isnothing(r.sample_variable)
        print(io, "  sample_variable: :", r.sample_variable)
        !isnothing(r.sample_date) && print(io, ",  sample_date: ", r.sample_date)
        print(io, "\n\n")
    end

    for c in r.checks
        icon = c.status === :ok       ? "[✓]" :
               c.status === :default  ? "[·]" :
               c.status === :missing  ? (c.required ? "[✗]" : "[ ]") :
                                        "[!]"
        label = string(c.name)
        if !isnothing(c.variable)
            label = string(label, "(:", c.variable, ")")
        end
        print(io, "  ", icon, " ", rpad(label, 44), "  ", c.detail, "\n")
    end

    print(io, "\nLegend: [✓] implemented/overridden, [·] default shipped, [✗] missing (required), [ ] missing (optional), [!] error.\n")

    n_required_missing = count(c -> c.required && c.status === :missing, r.checks)
    n_errors = count(c -> c.status === :error, r.checks)
    n_defaults = count(c -> c.status === :default, r.checks)
    print(io, "\nStatus: ",
          n_required_missing, " required method",
          n_required_missing == 1 ? "" : "s",
          " missing, ",
          n_errors, " error", n_errors == 1 ? "" : "s", ", ",
          n_defaults, " optional method", n_defaults == 1 ? "" : "s",
          " using shipped defaults.")

    is_conforming(r) ? print(io, "  → CONFORMING.") : print(io, "  → NOT CONFORMING.")
end

# Compact summary string of any returned value for the report.
function _detail(x)
    s = try
        sprint(show, x, context=(:compact => true, :limit => true))
    catch
        string(typeof(x))
    end
    length(s) > 60 ? s[1:57] * "..." : s
end
_detail(::Nothing) = "nothing"

# Classify the result of calling a defaulted method: :default if the method
# that dispatched was defined inside DataWrangling (which hosts the trait
# declarations, identity fallbacks, and download orchestrator). Methods
# defined in any dataset submodule (ECCO, EN4, …) or in a third-party package
# count as overrides.
function _classify_overridden(f, argtypes)
    m = try
        which(f, argtypes)
    catch
        return :missing
    end
    if m.module === CONTRACT_MODULE
        return :default
    else
        return :ok
    end
end

# Invoke a method, catching MethodError as :missing and any other exception as :error.
function _invoke(name::Symbol, f, args::Tuple; required=false, variable=nothing, classify_default=false)
    argtypes = map(typeof, args)
    if !hasmethod(f, argtypes)
        return ContractCheck(name, :missing, "no method for ($(join(argtypes, ", ")))"; required, variable)
    end
    local result
    try
        result = f(args...)
    catch e
        if e isa MethodError
            return ContractCheck(name, :missing, "MethodError: $(sprint(showerror, e))"; required, variable)
        else
            return ContractCheck(name, :error, "$(typeof(e)): $(sprint(showerror, e))"; required, variable)
        end
    end
    status = classify_default ? _classify_overridden(f, argtypes) : :ok
    detail = status === :default ? string(_detail(result), " (default)") : _detail(result)
    return ContractCheck(name, status, detail; required, variable)
end

# Like _invoke but only checks method presence — used for methods whose execution
# has side effects (download_dataset, download_file!).
function _check_method(name::Symbol, f, argtypes::Tuple; required=false, classify_default=false)
    if !hasmethod(f, argtypes)
        return ContractCheck(name, :missing, "no method for ($(join(argtypes, ", ")))"; required)
    end
    status = classify_default ? _classify_overridden(f, argtypes) : :ok
    detail = status === :default ? "method present (default)" : "method present"
    return ContractCheck(name, status, detail)
end

"""
    test_dataset_contract(dataset::AbstractDataset;
                          sample_variable=nothing,
                          sample_date=nothing,
                          verbose=true) -> ContractReport

Walk the full Metadata extension contract on `dataset` and return a [`ContractReport`](@ref). Every method is tried in isolation:
missing methods and errors are collected and reported rather than raised. When `verbose` is true (the default) the report is also 
printed to `stdout`.

`sample_variable` and `sample_date` control the inputs for methods that take a variable or metadatum. If omitted, they are inferred from
`available_variables` and `all_dates` respectively; if inference fails the corresponding checks are reported as missing.
"""
function test_dataset_contract(dataset::AbstractDataset;
                               sample_variable=nothing,
                               sample_date=nothing,
                               verbose=true)

    # --- Infer sample variable ---
    sample_variable = if !isnothing(sample_variable)
        sample_variable
    else
        vars = try
            available_variables(dataset)
        catch
            nothing
        end
        if vars isa AbstractDict && !isempty(vars)
            first(keys(vars))
        else
            nothing
        end
    end

    # --- Infer sample date ---
    sample_date = if !isnothing(sample_date)
        sample_date
    elseif !isnothing(sample_variable)
        dates = try
            all_dates(dataset, sample_variable)
        catch
            nothing
        end
        if isnothing(dates)
            nothing
        elseif dates isa AnyDateTime
            dates
        else
            try; first(dates); catch; nothing; end
        end
    else
        nothing
    end

    # --- Build sample Metadatum (best-effort); capture reason for later ---
    md_build_error = nothing
    sample_md = try
        Metadatum(sample_variable; dataset=dataset, date=sample_date)
    catch e
        md_build_error = e
        nothing
    end

    checks = ContractCheck[]

    # --- Metadatum-construction prerequisites ---
    push!(checks, _check_method(:default_download_directory, default_download_directory, (typeof(dataset),); required=true))
    if !isnothing(sample_variable) && !isnothing(sample_date)
        push!(checks, _check_method(:metadata_filename, metadata_filename,
                                    (typeof(dataset), Symbol, typeof(sample_date), Nothing);
                                    required=true))
        push!(checks, _check_method(:size, Base.size, (typeof(dataset), Symbol); required=true))
    end

    # If Metadatum construction failed, surface the reason prominently.
    if !isnothing(md_build_error)
        push!(checks, ContractCheck(:Metadatum_constructor, :error,
                                    "Metadatum(:$sample_variable; dataset, date=$sample_date) raised: " *
                                    sprint(showerror, md_build_error),
                                    true, nothing))
    end

    # --- Required method set ---
    if !isnothing(sample_variable)
        push!(checks, _invoke(:dataset_variable_name, dataset_variable_name, (sample_md,); required=true, variable=sample_variable))
        push!(checks, _invoke(:all_dates, all_dates, (dataset, sample_variable); required=true, variable=sample_variable))
    else
        push!(checks, ContractCheck(:dataset_variable_name, :missing, "no sample variable available"; required=true))
        push!(checks, ContractCheck(:all_dates, :missing, "no sample variable available"; required=true))
    end

    # retrieve_data is risky to call directly (requires a downloaded file); only
    # check method presence.
    if !isnothing(sample_md)
        push!(checks, _check_method(:retrieve_data, retrieve_data, (typeof(sample_md),); required=true))
    else
        push!(checks, ContractCheck(:retrieve_data, :missing, "no sample metadatum available"; required=true))
    end

    # --- Grid interfaces: need one path of (three *_interfaces) or a native_grid override ---
    push!(checks, _invoke(:longitude_interfaces, longitude_interfaces, (dataset,); required=true))
    push!(checks, _invoke(:latitude_interfaces, latitude_interfaces, (dataset,); required=true))
    # `z_interfaces` is the one *_interfaces method that legitimately varies per
    # variable (e.g., station datasets where temperature and salinity sit on
    # different depth grids). Probe it on the sample metadatum, which is the
    # signature the runtime pipeline actually uses.
    if !isnothing(sample_md)
        push!(checks, _invoke(:z_interfaces, z_interfaces, (sample_md,); required=true, variable=sample_variable))
    else
        push!(checks, ContractCheck(:z_interfaces, :missing, "no sample metadatum available"; required=true))
    end

    # --- Download contract ---
    if !isnothing(sample_md)
        push!(checks, _invoke(:dataset_url, dataset_url, (sample_md,); classify_default=true))
    else
        push!(checks, ContractCheck(:dataset_url, :missing, "no sample metadatum available"))
    end
    push!(checks, _invoke(:authenticate, authenticate, (dataset,); classify_default=true))
    push!(checks, _check_method(:download_file!, download_file!, (String, String, typeof(dataset)); classify_default=true))
    if !isnothing(sample_md)
        push!(checks, _check_method(:download_dataset, download_dataset, (typeof(sample_md),); classify_default=true))
    end

    # --- Optional methods with defaults ---
    push!(checks, _invoke(:available_variables, available_variables, (dataset,)))
    if !isnothing(sample_md)
        push!(checks, _invoke(:conversion_units, conversion_units, (sample_md,); classify_default=true, variable=sample_variable))
        push!(checks, _invoke(:is_three_dimensional, is_three_dimensional, (sample_md,); variable=sample_variable))
        push!(checks, _invoke(:location, location, (sample_md,); variable=sample_variable))
    end
    push!(checks, _invoke(:reversed_vertical_axis, reversed_vertical_axis, (dataset,)))

    # --- Pipeline hook ---
    if !isnothing(sample_md)
        push!(checks, _check_method(:preprocess_data, preprocess_data, (Array, typeof(sample_md)); classify_default=true))
    end

    report = ContractReport(dataset, sample_variable, sample_date, checks)
    verbose && show(stdout, MIME"text/plain"(), report)
    verbose && println()
    return report
end
