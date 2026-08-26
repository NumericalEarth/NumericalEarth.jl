const BBOX = NumericalEarth.DataWrangling.BoundingBox
const COL  = NumericalEarth.DataWrangling.Column
const LIN  = NumericalEarth.DataWrangling.Linear
const NR   = NumericalEarth.DataWrangling.Nearest

#####
##### Retry wrapper — the CDS/EWDS gateway intermittently answers valid requests
##### with a transient error (e.g. 502 Bad Gateway); retry with backoff instead of
##### failing on the first hiccup. Every `CDSAPI.retrieve` call site below goes
##### through this one wrapper.
#####

function retrieve_with_retries(product, request, path; max_retries = 3)
    for attempt in 1:max_retries
        try
            return CDSAPI.retrieve(product, request, path)
        catch e
            attempt < max_retries || rethrow(e)
            @warn "CDS retrieve attempt $attempt/$max_retries failed for $product; retrying..." exception=(e, catch_backtrace())
            sleep(5.0 * attempt)
        end
    end
end
