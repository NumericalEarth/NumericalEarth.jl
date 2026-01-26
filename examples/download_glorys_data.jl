using NumericalEarth
using Oceananigans
using CopernicusMarine

arch = CPU()
Nx = 20 * 12
Ny = 20 * 12
Nz = 50

depth = 6000
z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4.5)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z,
                             latitude  = (35, 55),
                             longitude = (200, 220))

bounding_box = NumericalEarth.DataWrangling.BoundingBox(longitude=(200, 220), latitude=(35, 55))

# dataset = NumericalEarth.DataWrangling.Copernicus.GLORYSStatic()
# static_meta = NumericalEarth.DataWrangling.Metadatum(:depth; dataset, bounding_box)
# coords_path = NumericalEarth.DataWrangling.download_dataset(static_meta)
# @info "Downloaded coordinates data to $coords_path"

# T_ecco = NumericalEarth.DataWrangling.ECCOMetadatum(:temperature; dataset, bounding_box)
# T_en4_meta = NumericalEarth.DataWrangling.EN4Metadatum(:temperature)
# T_en4_path = NumericalEarth.DataWrangling.download_dataset(T_en4_meta)
# T_en4 = Field(T_en4_meta)

dataset = NumericalEarth.DataWrangling.Copernicus.GLORYSDaily()
T_meta = NumericalEarth.DataWrangling.Metadatum(:temperature; dataset, bounding_box)
T_path = NumericalEarth.DataWrangling.download_dataset(T_meta)
@info "Downloaded temperature data to $T_path"
T = Field(T_meta, inpainting=nothing)

#=
u_meta = NumericalEarth.DataWrangling.Metadatum(:u_velocity; dataset)
u_path = NumericalEarth.DataWrangling.download_dataset(u_meta)
@info "Downloaded u velocity data to $u_path"
u = Field(u_meta)

v_meta = NumericalEarth.DataWrangling.Metadatum(:v_velocity; dataset)
v_path = NumericalEarth.DataWrangling.download_dataset(v_meta)
@info "Downloaded data to $v_path"
v = Field(v_meta)

S_meta = NumericalEarth.DataWrangling.Metadatum(:salinity; dataset)
S_path = NumericalEarth.DataWrangling.download_dataset(S_meta)
@info "Downloaded data to $S_path"
S = Field(S_meta)
=#

#=
# FOR ERA5:
# account: https://cds.climate.copernicus.eu/how-to-api
CondaPkg.add("cdsapi"; channel = "conda-forge")
cds = pyimport("cdsapi")          # should succeed instantly
client = cds.Client()
=#
