#####
##### Copernicus Climate Data Store clients
#####
##### The CDS request construction and NetCDF splitting live in `src`; a client only supplies the
##### function that fetches one request into a file. Each client's package extension defines its
##### `cds_retrieve` method, so `download` picks a client explicitly, never by extension load order.
#####

abstract type CDSClient end

"""
$(TYPEDEF)

Fetch Copernicus Climate Data Store requests with CDSAPI.jl. Requires `using CDSAPI`.
"""
struct CDSAPIClient <: CDSClient end

"""
$(TYPEDEF)

Fetch Copernicus Climate Data Store requests with CopernicusClimateDataStore.jl.
Requires `using CopernicusClimateDataStore`.
"""
struct CopernicusClimateDataStoreClient <: CDSClient end

"""
    cds_retrieve(client)

Return the function `(product, request, path)` with which `client` fetches one CDS request into
`path`, keeping a multi-member ZIP delivery intact. Methods are defined by each client's extension.
"""
function cds_retrieve end

missing_cds_client(product, request, path) =
    throw(ArgumentError("Downloading $product from the Copernicus Climate Data Store needs a client: " *
                        "load `CopernicusClimateDataStore` or `CDSAPI`, or pass `client`."))

cds_retrieve(::Nothing) = missing_cds_client

"""
$(TYPEDSIGNATURES)

Return the CDS client `download` uses when none is given: `CopernicusClimateDataStoreClient()` when
CopernicusClimateDataStore is loaded, otherwise `CDSAPIClient()` when CDSAPI is loaded, otherwise
`nothing`. The choice depends only on which packages are loaded, not on the order they were loaded in.
"""
function default_cds_client()
    hasmethod(cds_retrieve, Tuple{CopernicusClimateDataStoreClient}) && return CopernicusClimateDataStoreClient()
    hasmethod(cds_retrieve, Tuple{CDSAPIClient}) && return CDSAPIClient()
    return nothing
end

is_transient_cds_error(e) = true
is_transient_cds_error(::ArgumentError) = false

"""
$(TYPEDSIGNATURES)

Fetch `request` for `product` into `path` with `retrieve(product, request, path)`, retrying with
backoff on the transient errors the CDS/EWDS gateway answers with (e.g. 502 Bad Gateway).
"""
function retrieve_with_retries(product, request, path; retrieve, max_retries = 3)
    for attempt in 1:max_retries
        try
            return retrieve(product, request, path)
        catch e
            (attempt < max_retries && is_transient_cds_error(e)) || rethrow(e)
            @warn "CDS retrieve attempt $attempt/$max_retries failed for $product; retrying..." exception=(e, catch_backtrace())
            sleep(5.0 * attempt)
        end
    end
end
