include("runtests_setup.jl")

using NumericalEarth.DataWrangling.GLODAP
using NumericalEarth.DataWrangling.GLODAP: glodap_z_interfaces_from_centers
using NumericalEarth.DataWrangling: Metadatum, metadata_path, inpainted_metadata_path,
                                    longitude_interfaces, latitude_interfaces,
                                    reversed_vertical_axis, available_variables,
                                    is_three_dimensional

@testset "GLODAP dataset traits" begin
    dataset = GLODAPClimatology()
    @test dataset.product_year == 2016
    @test longitude_interfaces(dataset) == (-180, 180)
    @test latitude_interfaces(dataset) == (-90, 90)
    @test reversed_vertical_axis(dataset)
    @test :dic ∈ keys(available_variables(dataset))
end

@testset "GLODAP metadatum paths" begin
    dir = mktempdir()
    metadatum = Metadatum(:temperature; dataset=GLODAPClimatology(), dir)

    @test metadatum.filename == "GLODAPv2.theta.nc"
    @test metadata_path(metadatum) == joinpath(dir, "GLODAPv2_Mapped_Climatologies", "GLODAPv2.theta.nc")
    @test inpainted_metadata_path(metadatum) == joinpath(dir, "GLODAPv2.theta_temperature_inpainted.jld2")
    @test is_three_dimensional(metadatum)
end

@testset "GLODAP z interfaces from depth centers" begin
    z = glodap_z_interfaces_from_centers([10, 30, 50])
    @test z == [-60, -40, -20, 0]
end
