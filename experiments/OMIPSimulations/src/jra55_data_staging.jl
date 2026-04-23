using Dates

# JRA55 shortnames used in filenames (subset actually loaded by OMIP)
const JRA55_SHORTNAMES = ["tas", "huss", "psl", "uas", "vas", "rlds", "rsds", "prra", "prsn", "friver", "licalvf"]

"""
    setup_staging_directory(source_dir, staging_dir)

Populate `staging_dir` with symlinks to every `.nc` file in `source_dir`.
Reads go through symlinks (slow) until files are staged with `stage_jra55_year!`.
Also sweeps any leftover `*.tmp` from a killed prior run, since those are
half-written staging artefacts and would otherwise be mistaken for real copies.
"""
function setup_staging_directory(source_dir, staging_dir)
    mkpath(staging_dir)
    for leftover in filter(f -> endswith(f, ".nc.tmp"), readdir(staging_dir; join=true))
        rm(leftover; force=true)
    end
    for src in filter(f -> endswith(f, ".nc"), readdir(source_dir; join=true))
        dst = joinpath(staging_dir, basename(src))
        isfile(dst) || ispath(dst) || symlink(src, dst)
    end
    return staging_dir
end

# Replace `dst` atomically with whatever `make_new!(tmp)` produces at `tmp`.
# `rename(2)` is atomic on the same filesystem, so concurrent readers either
# see the old `dst` (symlink or previous real copy) or the new one — never a
# half-written file. Readers holding an fd to the old inode keep reading it
# correctly; the kernel keeps the inode alive until they close.
function atomic_replace!(dst, make_new!)
    tmp = dst * ".tmp"
    isfile(tmp) && rm(tmp; force=true)     # stale tmp from a crash — drop it
    make_new!(tmp)
    mv(tmp, dst; force=true)
    return dst
end

"""
    stage_jra55_year!(source_dir, staging_dir, year)

Replace symlinks in `staging_dir` with real copies for all JRA55 files
matching `year`. Skips files that are already real copies. The replacement
is atomic — a partial copy is never visible at `dst`, so concurrent
`PrefetchingBackend` readers on background threads cannot race with the `cp`.
"""
function stage_jra55_year!(source_dir, staging_dir, year)
    year_str = string(year)
    for name in JRA55_SHORTNAMES
        for dst in filter(f -> contains(f, name) && contains(f, year_str) && endswith(f, ".nc"), readdir(staging_dir; join=true))
            if islink(dst)
                src = joinpath(source_dir, basename(dst))
                atomic_replace!(dst, tmp -> cp(src, tmp))
                @debug "Staged $(basename(dst)) to scratch"
            end
        end
    end
end

"""
    unstage_jra55_year!(source_dir, staging_dir, year)

Remove real copies for `year` from `staging_dir` and restore symlinks
to `source_dir`. Uses the same atomic swap as staging so a concurrent
reader never sees a missing file at the swap point.
"""
function unstage_jra55_year!(source_dir, staging_dir, year)
    year_str = string(year)
    for name in JRA55_SHORTNAMES
        for dst in filter(f -> contains(f, name) && contains(f, year_str) && endswith(f, ".nc"), readdir(staging_dir; join=true))
            if isfile(dst) && !islink(dst)
                src = joinpath(source_dir, basename(dst))
                atomic_replace!(dst, tmp -> symlink(src, tmp))
                @debug "Unstaged $(basename(dst)) from scratch"
            end
        end
    end
end

"""
    JRA55DataStagingCallback(; source_dir, staging_dir, start_date)

Return a simulation callback that dynamically stages JRA55 yearly files
from `source_dir` (slow disk) to `staging_dir` (fast scratch).

At each invocation the callback:
  1. Copies the current and next year's files to scratch (if not already there)
  2. Removes files from two or more years ago to free space

Each year of JRA55 data is ~15–25 GB, so scratch holds at most ~50 GB at any time.
"""
function JRA55DataStagingCallback(; source_dir, staging_dir, start_date)

    staged_years = Set{Int}()

    function stage_forcing_data!(sim)
        current_year = year(start_date + Second(round(Int, sim.model.clock.time)))
        needed_years = Set([current_year, current_year + 1])

        # Stage upcoming years
        for y in needed_years
            if y ∉ staged_years
                @info "Staging JRA55 data for year $y to $staging_dir"
                stage_jra55_year!(source_dir, staging_dir, y)
                push!(staged_years, y)
            end
        end

        # Unstage old years
        for y in collect(staged_years)
            if y < current_year - 1
                @info "Unstaging JRA55 data for year $y from $staging_dir"
                unstage_jra55_year!(source_dir, staging_dir, y)
                delete!(staged_years, y)
            end
        end
    end

    return stage_forcing_data!
end
