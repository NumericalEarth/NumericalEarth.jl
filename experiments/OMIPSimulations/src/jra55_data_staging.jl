using Dates

# JRA55 shortnames used in filenames (subset actually loaded by OMIP)
const JRA55_SHORTNAMES = ["tas", "huss", "psl", "uas", "vas", "rlds", "rsds", "prra", "prsn", "friver", "licalvf"]

"""
    setup_staging_directory(source_dir, staging_dir)

Populate `staging_dir` with symlinks to every `.nc` file in `source_dir`.
Reads go through symlinks (slow) until files are staged with `stage_jra55_year!`.
"""
function setup_staging_directory(source_dir, staging_dir)
    mkpath(staging_dir)
    for src in filter(f -> endswith(f, ".nc"), readdir(source_dir; join=true))
        dst = joinpath(staging_dir, basename(src))
        isfile(dst) || symlink(src, dst)
    end
    return staging_dir
end

"""
    stage_jra55_year!(source_dir, staging_dir, year)

Replace symlinks in `staging_dir` with real copies for all JRA55 files
matching `year`.  Skips files that are already real copies.
"""
function stage_jra55_year!(source_dir, staging_dir, year)
    year_str = string(year)
    for name in JRA55_SHORTNAMES
        for dst in filter(f -> contains(f, name) && contains(f, year_str) && endswith(f, ".nc"), readdir(staging_dir; join=true))
            if islink(dst)
                src = joinpath(source_dir, basename(dst))
                rm(dst)
                cp(src, dst)
                @debug "Staged $(basename(dst)) to scratch"
            end
        end
    end
end

"""
    unstage_jra55_year!(source_dir, staging_dir, year)

Remove real copies for `year` from `staging_dir` and restore symlinks
to `source_dir`.
"""
function unstage_jra55_year!(source_dir, staging_dir, year)
    year_str = string(year)
    for name in JRA55_SHORTNAMES
        for dst in filter(f -> contains(f, name) && contains(f, year_str) && endswith(f, ".nc"), readdir(staging_dir; join=true))
            if isfile(dst) && !islink(dst)
                src = joinpath(source_dir, basename(dst))
                rm(dst)
                symlink(src, dst)
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
