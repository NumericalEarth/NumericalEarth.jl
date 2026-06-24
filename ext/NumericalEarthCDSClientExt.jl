module NumericalEarthCDSClientExt

# Extension that loads ERA5 CDS client when HTTP and JSON3 are available
# This allows pure Julia ERA5 downloads without Python dependencies

using NumericalEarth
using NumericalEarth.DataWrangling.ERA5
using HTTP
using JSON3

# Include the CDS client implementation
include("../src/DataWrangling/ERA5/ERA5_variables.jl")
include("../src/DataWrangling/ERA5/ERA5_cds_client.jl")

# Export the CDS client functions
export download_era5, read_cds_credentials, cds_variable_name

end # module
