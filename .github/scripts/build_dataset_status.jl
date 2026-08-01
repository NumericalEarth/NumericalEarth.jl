#####
##### Render the nightly dataset-download dashboard
#####
#
#   julia build_dataset_status.jl <status-root> <output-directory>
#
# `<status-root>` holds one subdirectory per matrix leg (as downloaded by
# actions/download-artifact), each containing the TSV records written by
# `test/dataset_status.jl`. The subdirectory name is used as the platform label.
#
# Emits `index.html` (the dashboard) and `status.json` (a shields.io endpoint badge, so the
# summary can be embedded anywhere without scraping the page).
#
# Base-only on purpose: this runs in whatever Julia the workflow has to hand, before any
# project is instantiated.

using Dates: Dates, now, UTC

struct DatasetRecord
    platform :: String
    dataset :: String
    variable :: String
    succeeded :: Bool
    message :: String
    seconds :: Float64
    bytes :: Int
end

function parse_records(root)
    records = DatasetRecord[]
    isdir(root) || return records

    for (directory, _, files) in walkdir(root)
        for file in files
            endswith(file, ".tsv") || continue

            # Platform label is the first path component below the root: one artifact per
            # matrix leg, unpacked into a directory named after it.
            relative = relpath(directory, root)
            platform = relative == "." ? "unknown" : first(splitpath(relative))
            platform = chopprefix(platform, "dataset-status-")

            for line in eachline(joinpath(directory, file))
                isempty(strip(line)) && continue
                fields = split(line, '\t')
                length(fields) == 6 || continue
                push!(records, DatasetRecord(platform, fields[1], fields[2],
                                             fields[3] == "ok", fields[4],
                                             something(tryparse(Float64, fields[5]), 0.0),
                                             something(tryparse(Int, fields[6]), 0)))
            end
        end
    end

    return records
end

escape_html(text) = foldl(replace, ("&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;"), init = string(text))
escape_json(text) = foldl(replace, ("\\" => "\\\\", "\"" => "\\\"", "\n" => " ", "\r" => " ", "\t" => " "), init = string(text))

function format_bytes(bytes)
    bytes <= 0 && return "-"
    units = ("B", "KB", "MB", "GB", "TB")
    magnitude = min(floor(Int, log(1024, max(bytes, 1))), length(units) - 1)
    return string(round(bytes / 1024^magnitude, digits = magnitude == 0 ? 0 : 1), " ", units[magnitude + 1])
end

format_duration(seconds) = seconds <= 0 ? "-" : string(round(seconds, digits = 1), " s")

function render_page(records, repository, run_url, generated)
    platforms = sort(unique(record.platform for record in records))
    keys_in_order = sort(unique((record.dataset, record.variable) for record in records))

    passing = count(record -> record.succeeded, records)
    failing = length(records) - passing

    io = IOBuffer()

    println(io, """
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>NumericalEarth.jl — dataset download status</title>
    <style>
      :root { color-scheme: light dark; --fg: #1a1a1a; --bg: #ffffff; --muted: #666; --line: #e3e3e3; --ok: #1a7f37; --fail: #cf222e; }
      @media (prefers-color-scheme: dark) {
        :root { --fg: #e6e6e6; --bg: #0d1117; --muted: #9aa0a6; --line: #30363d; --ok: #3fb950; --fail: #f85149; }
      }
      body { margin: 0 auto; padding: 2rem 1rem; max-width: 70rem; background: var(--bg); color: var(--fg);
             font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
      h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
      .meta { color: var(--muted); font-size: .875rem; margin-bottom: 1.5rem; }
      .summary { display: inline-block; padding: .35rem .7rem; border-radius: 6px; font-weight: 600;
                 background: color-mix(in srgb, var(--ok) 15%, transparent); color: var(--ok); }
      .summary.bad { background: color-mix(in srgb, var(--fail) 15%, transparent); color: var(--fail); }
      .scroll { overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; font-size: .9rem; }
      th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); white-space: nowrap; }
      th { font-weight: 600; color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }
      td.ok { color: var(--ok); font-weight: 600; }
      td.fail { color: var(--fail); font-weight: 600; }
      td.msg { white-space: normal; color: var(--muted); font-size: .8rem; max-width: 28rem; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }
      a { color: inherit; }
    </style>
    </head>
    <body>
    <h1>Dataset download status</h1>
    <div class="meta">
      <a href="https://github.com/$(escape_html(repository))">$(escape_html(repository))</a> —
      generated $(escape_html(generated)) from the nightly
      <a href="$(escape_html(run_url))">Data downloading</a> run.
    </div>""")

    if isempty(records)
        println(io, """<p class="summary bad">No status records were produced — the run failed before any dataset was checked.</p>""")
    else
        css = failing == 0 ? "summary" : "summary bad"
        label = failing == 0 ? "All $(passing) dataset checks passed" : "$(failing) of $(length(records)) dataset checks failed"
        println(io, """<p class="$(css)">$(escape_html(label))</p>""")

        println(io, """<div class="scroll"><table><thead><tr><th>Dataset</th><th>Variable</th>""")
        for platform in platforms
            println(io, "<th>", escape_html(platform), "</th>")
        end
        println(io, "<th>Time</th><th>Size</th><th>Error</th></tr></thead><tbody>")

        for (dataset, variable) in keys_in_order
            matching = filter(record -> record.dataset == dataset && record.variable == variable, records)
            println(io, "<tr><td><code>", escape_html(dataset), "</code></td><td><code>", escape_html(variable), "</code></td>")

            for platform in platforms
                index = findfirst(record -> record.platform == platform, matching)
                if isnothing(index)
                    println(io, """<td class="msg">—</td>""")
                else
                    record = matching[index]
                    println(io, """<td class="$(record.succeeded ? "ok" : "fail")">$(record.succeeded ? "pass" : "fail")</td>""")
                end
            end

            slowest = maximum(record.seconds for record in matching)
            largest = maximum(record.bytes for record in matching)
            failures = filter(record -> !record.succeeded, matching)
            message = isempty(failures) ? "" : first(failures).message

            println(io, "<td>", format_duration(slowest), "</td><td>", format_bytes(largest),
                        """</td><td class="msg">""", escape_html(message), "</td></tr>")
        end

        println(io, "</tbody></table></div>")
    end

    println(io, "</body></html>")

    return String(take!(io))
end

function render_badge(records)
    passing = count(record -> record.succeeded, records)
    total = length(records)

    message, color = if total == 0
        "no data", "lightgrey"
    elseif passing == total
        "$(total)/$(total) ok", "brightgreen"
    else
        "$(passing)/$(total) ok", "red"
    end

    return """{"schemaVersion":1,"label":"datasets","message":"$(escape_json(message))","color":"$(color)"}"""
end

function render_details(records)
    entries = map(records) do record
        string("{\"platform\":\"", escape_json(record.platform),
               "\",\"dataset\":\"", escape_json(record.dataset),
               "\",\"variable\":\"", escape_json(record.variable),
               "\",\"ok\":", record.succeeded,
               ",\"seconds\":", record.seconds,
               ",\"bytes\":", record.bytes,
               ",\"message\":\"", escape_json(record.message), "\"}")
    end

    return string("[", join(entries, ","), "]")
end

function main(arguments)
    length(arguments) == 2 || error("usage: build_dataset_status.jl <status-root> <output-directory>")
    root, output = arguments

    records = parse_records(root)
    mkpath(output)

    repository = get(ENV, "GITHUB_REPOSITORY", "NumericalEarth/NumericalEarth.jl")
    server = get(ENV, "GITHUB_SERVER_URL", "https://github.com")
    run_id = get(ENV, "GITHUB_RUN_ID", "")
    run_url = isempty(run_id) ? "$(server)/$(repository)/actions" : "$(server)/$(repository)/actions/runs/$(run_id)"
    generated = Dates.format(now(UTC), "yyyy-mm-dd HH:MM \\U\\T\\C")

    write(joinpath(output, "index.html"), render_page(records, repository, run_url, generated))
    write(joinpath(output, "status.json"), render_badge(records))
    write(joinpath(output, "details.json"), render_details(records))

    @info "Rendered dataset status" records=length(records) failing=count(r -> !r.succeeded, records) output
end

main(ARGS)
