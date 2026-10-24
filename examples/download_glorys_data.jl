# # Downloading GLORYS data
#
# This script downloads a daily GLORYS temperature snapshot from the Copernicus Marine Service over a
# 20° × 20° patch of the northeast Pacific and loads it into a `Field`. Loading `CopernicusMarine`
# activates NumericalEarth's Copernicus Marine extension, which performs the download; it requires a
# Copernicus Marine account.

using NumericalEarth
using Oceananigans
using CopernicusMarine

# The download region is the horizontal extent of the grid we intend to initialize: a 1/12° grid with
# 50 exponentially stretched levels down to 6000 m.

arch = CPU()
resolution = 1/12 # degrees
Nx = 20 * Int(1 / resolution)
Ny = 20 * Int(1 / resolution)
Nz = 50

depth = 6000
z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4.5)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z,
                             latitude  = (35, 55),
                             longitude = (200, 220))

region = BoundingBox(grid)

# We download the temperature and load it, without inpainting, on the native GLORYS grid.
# Salinity (`:salinity`) and velocities (`:u_velocity`, `:v_velocity`) are downloaded the same way.

dataset = GLORYSDaily()
temperature_metadatum = Metadatum(:temperature; dataset, region)
download(temperature_metadatum)

T = Field(temperature_metadatum, inpainting=nothing)
