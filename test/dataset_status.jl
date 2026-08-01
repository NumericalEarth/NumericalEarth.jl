#####
##### Per-dataset download-status recording for the nightly dashboard
#####
#
# The nightly `Data downloading` workflow publishes a page reporting, per dataset, whether
# it could still be fetched. Tests feed that page through `@dataset_check`, which records
# the outcome of a single dataset fetch and then asserts it.
#
# Recording is a no-op unless `DATASET_STATUS_DIR` is set, so the PR suite and local runs
# are unaffected. `@dataset_check` still runs the expression and still `@test`s the result.

using Test

# Records go to TSV rather than JSON deliberately: the renderer has to parse them, and a
# tab-delimited line needs no dependency at either end. `test/Project.toml` carries no JSON
# package and this is not worth adding one for.
const DATASET_STATUS_SEPARATOR = '\t'

dataset_status_directory() = get(ENV, "DATASET_STATUS_DIR", "")

sanitize_status_field(value) = replace(string(value), r"[\t\r\n]+" => " ")

# Size of whatever the check returned, when that is a downloaded path (or paths). Anything
# else contributes nothing -- the column is a convenience, not an assertion.
status_bytes(path::AbstractString) = isfile(path) ? filesize(path) : 0
status_bytes(paths::AbstractVector) = sum(status_bytes, paths; init=0)
status_bytes(::Any) = 0

function record_dataset_status(dataset_name, variable_name, succeeded, message, elapsed_seconds, bytes)
    directory = dataset_status_directory()
    isempty(directory) && return nothing
    mkpath(directory)

    # One file per process. ParallelTestRunner runs its workers concurrently and appends
    # from several of them to a single file interleave partial lines.
    filepath = joinpath(directory, string("status-", getpid(), ".tsv"))

    open(filepath, "a") do io
        println(io, join((sanitize_status_field(dataset_name),
                          sanitize_status_field(variable_name),
                          succeeded ? "ok" : "fail",
                          sanitize_status_field(message),
                          round(elapsed_seconds, digits=2),
                          bytes), DATASET_STATUS_SEPARATOR))
    end

    return nothing
end

"""
    @dataset_check dataset_name variable_name expr

Run `expr` as the download check for `dataset_name`/`variable_name`, record whether it
succeeded, and assert it with `@test`. Returns whatever `expr` returned, or `nothing` if it
threw.

A throwing `expr` is caught rather than propagated so that the record is written and the
remaining datasets still get checked: one dead server should not hide the state of every
dataset behind it. The `@test` keeps the run red.
"""
macro dataset_check(dataset_name, variable_name, expr)
    quote
        local recorded_dataset = $(esc(dataset_name))
        local recorded_variable = $(esc(variable_name))
        local started = time()
        local succeeded = false
        local message = ""
        local result = nothing

        try
            result = $(esc(expr))
            succeeded = true
        catch exception
            message = sprint(showerror, exception)
            @warn "Download check failed for $(recorded_dataset) $(recorded_variable)" exception=(exception, catch_backtrace())
        end

        record_dataset_status(recorded_dataset, recorded_variable, succeeded, message,
                              time() - started, status_bytes(result))
        @test succeeded

        result
    end
end
