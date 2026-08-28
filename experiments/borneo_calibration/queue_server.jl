# A warm Julia session that `include`s every script dropped into a queue directory, in
# name order, logging each to `logs/<script>.log`, so iterating on a compiled model does
# not pay the package load each time.
#
#   julia --project=<docs> queue_server.jl <queue directory>
#
# Drop `STOP` in the directory to exit.

import Dates
using Logging

queue = abspath(ARGS[1])
logs = joinpath(queue, "logs")
done = joinpath(queue, "done")
mkpath(logs); mkpath(done)

@info "loading packages"
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CUDA
using Reactant
using Enzyme

using CopernicusClimateDataStore
using ArchGDAL
using CairoMakie
using JLD2
using Statistics
using Printf
@info "queue server ready on $queue"

# A console logger that flushes after every message, so the log is readable while the
# main thread is busy compiling.
struct FlushingLogger{L} <: Logging.AbstractLogger
    logger :: L
end
Logging.handle_message(l::FlushingLogger, args...; kw...) = (Logging.handle_message(l.logger, args...; kw...); flush(l.logger.stream))
Logging.shouldlog(l::FlushingLogger, args...) = Logging.shouldlog(l.logger, args...)
Logging.min_enabled_level(l::FlushingLogger) = Logging.min_enabled_level(l.logger)
Logging.catch_exceptions(l::FlushingLogger) = Logging.catch_exceptions(l.logger)

function run_script(path)
    name = splitext(basename(path))[1]
    open(joinpath(logs, "$name.log"), "w") do io
        with_logger(FlushingLogger(ConsoleLogger(io))) do
            redirect_stdout(io) do
                redirect_stderr(io) do
                    @info "start $(Dates.now())"
                    try
                        Base.invokelatest(Base.include, Main, path)
                        @info "done $(Dates.now())"
                        println(io, "SCRIPT_OK")
                    catch err
                        showerror(io, err, catch_backtrace())
                        println(io)
                        println(io, "SCRIPT_FAILED")
                    end
                    flush(io)
                end
            end
        end
    end
end

while !isfile(joinpath(queue, "STOP"))
    scripts = sort(filter(f -> endswith(f, ".jl"), readdir(queue)))
    if isempty(scripts)
        sleep(2)
        continue
    end
    for f in scripts
        src = joinpath(queue, f)
        dst = joinpath(done, f)
        mv(src, dst; force = true)
        @info "running $f"
        run_script(dst)
        GC.gc()
    end
end
@info "queue server stopped"
