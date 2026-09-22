const BBOX = NumericalEarth.DataWrangling.BoundingBox
const COL  = NumericalEarth.DataWrangling.Column
const LIN  = NumericalEarth.DataWrangling.Linear
const NR   = NumericalEarth.DataWrangling.Nearest

"""
$(TYPEDSIGNATURES)

Fetch `request` for `product` into `path` with `retrieve(product, request, path)`, retrying with
backoff on the transient errors the CDS/EWDS gateway answers with (e.g. 502 Bad Gateway).
"""
function retrieve_with_retries(product, request, path; retrieve = CDSAPI.retrieve, max_retries = 3)
    for attempt in 1:max_retries
        try
            return retrieve(product, request, path)
        catch e
            attempt < max_retries || rethrow(e)
            @warn "CDS retrieve attempt $attempt/$max_retries failed for $product; retrying..." exception=(e, catch_backtrace())
            sleep(5.0 * attempt)
        end
    end
end
