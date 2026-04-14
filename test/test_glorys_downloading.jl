include("runtests_setup.jl")

using CopernicusMarine

@testset "Downloading GLORYS data" begin
    variables = (:temperature, :salinity, :u_velocity, :v_velocity)
    bounding_box = NumericalEarth.DataWrangling.BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = NumericalEarth.DataWrangling.GLORYS.GLORYSDaily()
    for variable in variables
        metadatum = Metadatum(variable; dataset, bounding_box)
        filepath = NumericalEarth.DataWrangling.metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        NumericalEarth.DataWrangling.download_dataset(metadatum)
    end
end
