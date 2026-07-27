#####
##### FLUXNET Shuttle download (https://data.fluxnet.org)
#####
##### Data is fetched through the official `fluxnet-shuttle` command-line tool, which
##### federates the ONEFlux products of the regional networks (AmeriFlux, ICOS, TERN, …).
##### Install it with:  pip install git+https://github.com/fluxnet/shuttle.git
#####

const FLUXNET_SHUTTLE_EXECUTABLE = "fluxnet-shuttle"
const FLUXNET_SHUTTLE_INSTALL = "pip install git+https://github.com/fluxnet/shuttle.git"

function fluxnet_shuttle_executable()
    exe = Sys.which(FLUXNET_SHUTTLE_EXECUTABLE)
    isnothing(exe) &&
        error("""
              The `$(FLUXNET_SHUTTLE_EXECUTABLE)` tool was not found on the PATH, so FLUXNET
              data cannot be downloaded. Either install it with

                  $(FLUXNET_SHUTTLE_INSTALL)

              or download the site's FLUXNET data product from https://data.fluxnet.org and
              place the unzipped CSV in the dataset's `dir`.
              """)
    return exe
end

# `listall` writes a timestamped catalog `fluxnet_shuttle_snapshot_<...>.csv`; reuse the
# newest existing one (rebuilding it queries every hub) or create it in `dir`.
function fluxnet_shuttle_snapshot(dir, exe)
    existing = glob("fluxnet_shuttle_snapshot_*.csv", dir)
    isempty(existing) || return first(sort(existing; rev=true))

    @info "Building the FLUXNET Shuttle catalog (fluxnet-shuttle listall)..."
    cd(dir) do
        run(`$exe listall`)
    end

    snapshots = glob("fluxnet_shuttle_snapshot_*.csv", dir)
    isempty(snapshots) &&
        error("`$(FLUXNET_SHUTTLE_EXECUTABLE) listall` did not produce a snapshot in \"$dir\".")
    return first(sort(snapshots; rev=true))
end

# The Shuttle delivers one archive zip per site; extract its CSVs alongside it.
function extract_shuttle_archives(ds::FLUXNETSite)
    for zip_path in glob(string("*_", ds.site, "_*.zip"), ds.dir)
        reader = ZipFile.Reader(zip_path)
        for file in reader.files
            endswith(file.name, ".csv") || continue
            out = joinpath(ds.dir, basename(file.name))
            isfile(out) || open(io -> write(io, read(file)), out, "w")
        end
        close(reader)
    end
    return nothing
end

"""
    download_from_shuttle(ds::FLUXNETSite)

Download the FLUXNET data product for `ds.site` from the FLUXNET Shuttle into
`ds.dir`, using the `fluxnet-shuttle` command-line tool, and unpack the resulting
archive. Errors with installation instructions if the tool is not available.
"""
function download_from_shuttle(ds::FLUXNETSite)
    exe = fluxnet_shuttle_executable()
    mkpath(ds.dir)

    snapshot = fluxnet_shuttle_snapshot(ds.dir, exe)

    @info "Downloading FLUXNET site $(ds.site) from the FLUXNET Shuttle..."
    @root run(`$exe download -f $snapshot -s $(ds.site) -o $(ds.dir) --quiet`)

    extract_shuttle_archives(ds)
    return nothing
end
